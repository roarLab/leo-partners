#!/usr/bin/env python3
# ==============================================================================
# rosbag_process_imu_v3.py
#
# Ego D435i IMU extraction for a recorded ROS 2 bag from the multicam rig.
#
# SCOPE: the ego D435i IMU, in two parts, both written into the shared metadata.json:
#   1. EXTRINSICS (latched): depth_to_gyro and depth_to_accel, recorded VERBATIM into
#      camera_extrinsics. (rosbag_process_depth_v3 records the third leg, depth_to_color,
#      the same way; between the two, metadata.json carries all three legs so a VIO
#      front-end can compose imu->color itself.)
#   2. SAMPLES (~200 Hz): the /imu angular_velocity + linear_acceleration stream,
#      written to imu/<cam>_imu.csv and registered as a steps.streams entry. Each sample
#      is tagged with the ego COLOR frame it falls under in time (see COLOR-FRAME
#      MATCHING) so a consumer can group IMU per frame for per-frame camera pose.
#
# UNITED /imu STREAM: this rig records a single sensor_msgs/Imu on .../imu (realsense
# unite_imu_method: gyro+accel merged, published at the gyro rate, in the gyro's frame
# camera_imu_optical_frame). So there is exactly ONE frame_id and the imu->color leg is
# ALWAYS depth_to_gyro (depth_to_accel is identical on this device). The CSV keeps only
# the live measurements and DROPS orientation + all covariances: the driver leaves
# orientation unpopulated (quaternion 0,0,0,0; covariance[0]=-1) and the covariances are
# a fixed 0.01 config default, not device noise — both are constant regardless of motion,
# so they carry no information (verified constant across stationary and moving recordings).
#
# WHY VERBATIM EXTRINSICS, NO COMPOSITION: a wrong convention fails VIO SILENTLY, so this
# script composes nothing — it copies the device's own numbers (float64[9] column-major
# rotation + float64[3] translation, meters) straight from realsense2_camera_msgs/
# Extrinsics into metadata.json, tagged with the source topic + convention string. The
# consumer composes imu->color = depth_to_color ∘ inverse(depth_to_gyro) itself, with full
# knowledge of the convention. (See pipeline_metadata.upsert_extrinsic.)
#
# COLOR-FRAME MATCHING (preceding rule): each IMU sample's bag timestamp is bucketed into
# the ego color frame interval it falls in — color_frame_index = the last cam_ego color
# frame at or before the sample (-1 before the first frame). That is the interval a VIO
# preintegrator consumes (all IMU in [frame k, frame k+1)). Matched against the WRITTEN
# cam_ego timestamps (timestamps/cam_ego.csv) so the index lines up with the video/frames
# even if color dropped one — hence color must run first (samples no-op if metadata absent).
#
# OWNERSHIP: this script owns the depth_to_gyro / depth_to_accel camera_extrinsics entries
# and the cam_ego "imu" stream. Owner-scoped + idempotent (find-or-append / replace-in-
# place), so re-runs update in place and every other key of metadata.json is preserved.
#
# USAGE (standalone; the wrapper drives main() the house way):
#   python rosbag_process_imu_v3.py --bag /path/to/session_bag --camera ego
# ==============================================================================
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from pipeline_metadata import add_error, upsert_extrinsic

# =============================================================================
# CONFIG — a wrapper sets these (or passes bag/out_dir/camera to main()); the CLI
# shim overwrites them for one-off shell runs. Defaults suit the D435i ego rig.
# =============================================================================
BAG_PATH = None           # path to the .mcap/.db3 bag (file or bag dir)
OUT_DIR = None            # output root (holds metadata.json); None -> bag's parent
CAMERA = "ego"            # substring to disambiguate topics when >1 camera present

# The extrinsic legs this script owns, matched by topic SUFFIX (prefix/namespace
# does not matter). name -> the camera_extrinsics entry name written for it.
EXTRINSICS: List[Tuple[str, str]] = [
    ("depth_to_gyro", "extrinsics/depth_to_gyro"),
    ("depth_to_accel", "extrinsics/depth_to_accel"),
]
# explicit full-topic overrides per name (None = auto-discover via the suffix above)
EXTRINSICS_TOPIC_OVERRIDES: Dict[str, Optional[str]] = {
    "depth_to_gyro": None,
    "depth_to_accel": None,
}

METADATA_FILENAME = "metadata.json"
# metadata schema label for the ego D435i (color/depth use it too); the "imu" stream and
# every color_frame_index attach to this camera. CAMERA above is a topic-discovery
# substring ("ego"); this is the metadata label ("cam_ego") — a separate knob.
METADATA_CAMERA_LABEL = "cam_ego"
# An absent OR empty (0 msgs) /imu stream is a PRESENCE failure, recorded as the generic
# "missing_stream" token — exactly like a missing depth topic or a missing camera. It is
# NOT a data-quality/validation error (there is no imu validator yet); a future
# validate_imu would own separate imu-quality tokens. See record_missing_imu.

# --- IMU SAMPLES (united gyro+accel /imu stream) -----------------------------
IMU_SUFFIX = "/imu"             # sensor_msgs/Imu, matched by suffix (prefix/namespace agnostic)
IMU_TOPIC_OVERRIDE = None       # explicit /imu topic (None = auto-discover via IMU_SUFFIX)
# imu->color leg for the united stream (frame camera_imu_optical_frame == gyro frame):
# a consumer forms imu->color = depth_to_color ∘ inverse(depth_to_gyro).
COMPOSE_LEG = "extrinsics/depth_to_gyro"

typestore = get_typestore(Stores.ROS2_HUMBLE)


# ==============================================================================
# Bag reading helpers (standalone so this script runs on its own, like the others)
# ==============================================================================
def matching_topics(conns, suffix: str, camera: Optional[str],
                    override: Optional[str]) -> List[str]:
    """ALL topics ending in `suffix` (optionally containing `camera`), sorted; an
    override forces a single explicit topic. Returns [] if none. Unlike the old
    single-topic finder, this preserves multiplicity so the CALLER can decide presence
    against the expected count (exactly one per extrinsic / per /imu): [] -> missing,
    len > 1 -> extra. A missing extrinsic is flagged + skipped, not fatal."""
    if override:
        return [override]
    cands = sorted({c.topic for c in conns if c.topic.endswith(suffix)})
    if camera:
        cands = [t for t in cands if camera in t]
    return cands


def read_extrinsics(reader, topic: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read one realsense2_camera_msgs/Extrinsics message VERBATIM -> (rotation[9]
    column-major, translation[3] meters). The message type comes from the bag's own
    embedded definitions (same as the depth aligner reads depth_to_color)."""
    conns = [c for c in reader.connections if c.topic == topic]
    for conn, _t, raw in reader.messages(connections=conns):
        try:
            msg = reader.deserialize(raw, conn.msgtype)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(
                f"Could not deserialize {conn.msgtype} on {topic}: {e}\n"
                "If reading a .db3 without embedded type defs, the Extrinsics type "
                "is unavailable — record from an .mcap or register the type.")
        rot = np.asarray(list(msg.rotation), np.float64)
        trans = np.asarray(list(msg.translation), np.float64)
        return rot, trans
    raise SystemExit(f"No Extrinsics message on {topic}")


# ==============================================================================
# IMU sample extraction (united /imu stream -> CSV + color-frame match)
# ==============================================================================
def _relpath(path, root) -> str:
    """Path relative to out_root for the metadata (falls back to abs if unrelated)."""
    p = Path(path)
    try:
        return str(p.relative_to(Path(root)))
    except ValueError:
        return str(p)


def load_color_frame_times(out_root, camera_label: str) -> Optional[np.ndarray]:
    """The WRITTEN color frame times (seconds) for `camera_label`, read from the color
    stream's timestamps CSV that rosbag_process_color_v3 produced — so a color_frame_index
    lines up with the video/frames even if color dropped one. None (color_frame_index then
    stays -1) if metadata.json, the color stream, or its CSV is unavailable."""
    meta_path = Path(out_root) / METADATA_FILENAME
    if not meta_path.is_file():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    streams = meta.get("steps", {}).get("streams", [])
    color = next((s for s in streams
                  if s.get("camera") == camera_label and s.get("kind") == "color"), None)
    if color is None or not color.get("timestamps"):
        print(f"[imu] no color stream for '{camera_label}' in metadata; "
              "color_frame_index will be -1 for all samples.")
        return None
    csv_path = Path(out_root) / color["timestamps"]
    if not csv_path.is_file():
        print(f"[imu] color timestamps {csv_path} missing; color_frame_index = -1.")
        return None
    df = pd.read_csv(csv_path)
    col = "ros_time_s" if "ros_time_s" in df.columns else df.columns[-1]
    return df[col].to_numpy(dtype=np.float64)


def export_imu_samples(reader, topic: str, out_root, color_times: Optional[np.ndarray],  # noqa: E501
                       cam_label: str = METADATA_CAMERA_LABEL
                       ) -> Optional[dict]:
    """Extract the united /imu stream to imu/<label>_imu.csv and return its metadata
    stream entry. Columns: index, ros_time_s (bag clock, seconds, same clock as the color
    timestamps), color_frame_index (the cam_ego color frame this sample falls under by the
    PRECEDING rule; -1 before the first frame), then wx,wy,wz (angular_velocity rad/s) and
    ax,ay,az (linear_acceleration m/s^2). Orientation + covariances are intentionally
    dropped (constant driver config, no information). Returns None if the topic is empty."""
    out_root = Path(out_root)
    conns = [c for c in reader.connections if c.topic == topic]
    t_ns: List[int] = []
    gx: List[float] = []; gy: List[float] = []; gz: List[float] = []
    ax: List[float] = []; ay: List[float] = []; az: List[float] = []
    frame_id: Optional[str] = None
    for conn, t, raw in reader.messages(connections=conns):
        msg = reader.deserialize(raw, conn.msgtype)
        if frame_id is None:
            frame_id = getattr(getattr(msg, "header", None), "frame_id", None)
        t_ns.append(int(t))                                   # bag record time (same clock as color)
        gx.append(msg.angular_velocity.x); gy.append(msg.angular_velocity.y); gz.append(msg.angular_velocity.z)
        ax.append(msg.linear_acceleration.x); ay.append(msg.linear_acceleration.y); az.append(msg.linear_acceleration.z)

    n = len(t_ns)
    if n == 0:
        print(f"[imu] {topic} has no messages; skipping sample extraction.")
        return None
    ts_s = np.asarray(t_ns, dtype=np.int64) / 1e9

    # preceding-frame match: color_frame_index = last color frame at or before the sample.
    color_idx = np.full(n, -1, dtype=np.int64)
    if color_times is not None and len(color_times):
        ct = np.asarray(color_times, dtype=np.float64)
        order = np.argsort(ct, kind="stable")                 # map sorted-time pos -> csv frame index
        pos = np.searchsorted(ct[order], ts_s, side="right") - 1
        ok = pos >= 0
        color_idx[ok] = order[pos[ok]]
    n_unmatched = int((color_idx < 0).sum())

    (out_root / "imu").mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "imu" / f"{cam_label}_imu.csv"
    pd.DataFrame({
        "index": np.arange(n, dtype=np.int64),
        "ros_time_s": ts_s,
        "color_frame_index": color_idx,
        "wx": gx, "wy": gy, "wz": gz,
        "ax": ax, "ay": ay, "az": az,
    }).to_csv(csv_path, index=False)

    dt = np.diff(ts_s)
    dt = dt[dt > 0]
    rate_hz = float(1.0 / np.median(dt)) if dt.size else None  # median: robust to startup gaps
    max_gap_s = float(dt.max()) if dt.size else None

    rate_txt = f"{rate_hz:.1f} Hz" if rate_hz is not None else "n/a"
    print(f"[imu] {n} sample(s) -> {_relpath(csv_path, out_root)}  "
          f"(rate~{rate_txt}, {n_unmatched} before first color frame)")

    return {
        "camera": cam_label,
        "kind": "imu",
        "topic": topic,
        "file": _relpath(csv_path, out_root),
        "frame_id": frame_id,
        "compose_leg": COMPOSE_LEG,
        "units": {"angular_velocity": "rad/s", "linear_acceleration": "m/s^2"},
        "matched_color_stream": METADATA_CAMERA_LABEL,
        "color_frame_match_rule": "preceding",
        "rate_hz": rate_hz,
        "num_samples": int(n),
        "n_unmatched_leading": n_unmatched,
        "ts_min": float(ts_s.min()),
        "ts_max": float(ts_s.max()),
        "max_gap_s": max_gap_s,
        "found": True,
    }


# ==============================================================================
# metadata.json integration (owner-scoped; no-op if metadata absent)
# ==============================================================================
def write_to_metadata(out_root, found: Dict[str, dict], imu_streams: Optional[list]) -> int:
    """Upsert the IMU extrinsics AND (if given) append the imu sample stream(s) into the
    EXISTING metadata.json in ONE read-modify-write. Extrinsics go through the shared
    owner-scoped upsert_extrinsic; each imu stream REPLACES any prior (same-camera, imu)
    entry (idempotent) and leaves every other stream/key untouched. Returns the number of
    extrinsics written; no-op returning 0 if metadata.json is absent (run color first)."""
    meta_path = Path(out_root) / METADATA_FILENAME
    if not meta_path.is_file():
        print(f"[imu] {meta_path} not found; skipping metadata update "
              "(run rosbag_process_color_v3 first).")
        return 0
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    for name, e in found.items():
        upsert_extrinsic(meta, name, e["topic"], e["rotation"], e["translation"])
    for imu_stream in (imu_streams or []):
        streams = meta.setdefault("steps", {}).setdefault("streams", [])
        streams[:] = [s for s in streams
                      if not (s.get("camera") == imu_stream["camera"]
                              and s.get("kind") == "imu")]
        streams.append(imu_stream)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    msg = f"+{len(found)} extrinsic(s): {sorted(found)}"
    for imu_stream in (imu_streams or []):
        msg += f", imu stream {imu_stream['camera']} ({imu_stream['num_samples']} samples)"
    print(f"[imu] updated {meta_path} ({msg})")
    return len(found)


# ==============================================================================
# Main
# ==============================================================================
def main(bag=None, out_dir=None, camera=None) -> dict:
    """Read the ego IMU extrinsics (depth_to_gyro, depth_to_accel) AND the united /imu
    sample stream from one bag, recording both into the shared metadata.json (extrinsics
    verbatim into camera_extrinsics; samples to imu/<cam>_imu.csv + a steps.streams entry,
    each sample tagged with the ego color frame it falls under). Wrapper usage (no shell):
    main(bag=<path>, out_dir=<root>, camera="ego"). Returns a summary dict."""
    bag = Path(bag if bag is not None else BAG_PATH)
    if not bag.exists():
        raise SystemExit(f"Bag not found: {bag}")
    _out = out_dir if out_dir is not None else OUT_DIR
    out_root = Path(_out) if _out else bag.parent
    camera = camera if camera is not None else CAMERA
    meta_present = (out_root / METADATA_FILENAME).is_file()

    found: Dict[str, dict] = {}
    # FACTS ONLY: streams + extrinsics, never error tokens. Presence (missing/extra)
    # is validate_imu's — diffed from the output. Per-UNIT isolation: each /imu
    # candidate is one unit (extract-all); a crashed unit is recorded here and the
    # survivors still commit before the failure re-raises for the wrapper.
    imu_streams: List[dict] = []
    unit_failures: List[tuple] = []
    with AnyReader([bag], default_typestore=typestore) as reader:
        conns = list(reader.connections)
        for name, suffix in EXTRINSICS:
            cands = matching_topics(conns, suffix, camera, EXTRINSICS_TOPIC_OVERRIDES.get(name))
            if not cands:
                # console note only — absent extrinsics in the OUTPUT is validate_imu's
                # imu_info verdict
                print(f"[imu] extrinsic '{name}' not found (*{suffix})")
                continue
            if len(cands) > 1:
                print(f"[imu] WARN multiple '{name}' topics {cands}; using {cands[0]}")
            topic = cands[0]
            rot, trans = read_extrinsics(reader, topic)
            found[name] = {"topic": topic, "rotation": rot, "translation": trans}
            print(f"[imu] {name}: R(col-major)={np.round(rot, 6).tolist()} "
                  f"t(m)={np.round(trans, 6).tolist()}  <- {topic}")

        # --- /imu SAMPLES (united gyro+accel). Needs the color-written metadata for the
        # frame-time match, so it no-ops when metadata.json is absent (run color first). ---
        imu_cands = matching_topics(conns, IMU_SUFFIX, camera, IMU_TOPIC_OVERRIDE)
        if not imu_cands:
            print(f"[imu] no /imu topic (*{IMU_SUFFIX}); nothing extracted "
                  "(the MISSING verdict is validate_imu's).")
        elif not meta_present:
            print("[imu] metadata.json absent; skipping sample extraction "
                  "(run rosbag_process_color_v3 first).")
        else:
            color_times = load_color_frame_times(out_root, METADATA_CAMERA_LABEL)
            # EXTRACT-ALL: every /imu candidate becomes its own stream; the first keeps
            # the canonical cam_ego label, a surplus one gets cam_ego_extraN so the
            # presence diff can flag it EXTRA from the output.
            for i, topic in enumerate(imu_cands):
                label = (METADATA_CAMERA_LABEL if i == 0
                         else f"{METADATA_CAMERA_LABEL}_extra{i + 1}")
                if i > 0:
                    print(f"[imu] *** SURPLUS /imu topic {topic} — extracting as {label} ***")
                try:
                    stream = export_imu_samples(reader, topic, out_root, color_times,
                                                cam_label=label)
                except Exception as e:  # noqa: BLE001 — unit isolation
                    unit_failures.append((label, topic, traceback.format_exc()))
                    print(f"[FAIL] {label} ({topic}) imu extraction crashed: {e} — continuing")
                    continue
                if stream is None:
                    print(f"[imu] {topic}: topic present but streamed no messages")
                else:
                    imu_streams.append(stream)

    if not found:
        print(f"[imu] no IMU extrinsics found in {bag} "
              f"(looked for {[s for _, s in EXTRINSICS]}).")

    n_written = write_to_metadata(out_root, found, imu_streams)
    # Per-unit isolation, part 2: survivors are committed above; a failed unit now
    # surfaces (wrapper records step_errors + the traceback naming each unit).
    if unit_failures:
        failed = ", ".join(f"{lbl} ({top})" for lbl, top, _ in unit_failures)
        tails = "\n".join(tb for _, _, tb in unit_failures)
        raise RuntimeError(
            f"imu extraction failed for unit(s): {failed} — surviving streams "
            f"committed to metadata.json\n{tails}")
    primary = imu_streams[0] if imu_streams else None
    return {
        "bag": str(bag),
        "camera": camera,
        "extrinsics_found": sorted(found),
        "extrinsics_written": int(n_written),
        "imu_samples": int(primary["num_samples"]) if primary else 0,
        "imu_missing": primary is None,         # data-plane absent/empty (console tail)
    }


def _cli() -> None:
    """Thin shell shim for one-off runs. The wrapper does NOT use this — it sets the
    CONFIG globals and calls main() directly."""
    ap = argparse.ArgumentParser(
        description="Extract ego IMU extrinsics (depth_to_gyro/accel) into metadata.json.")
    ap.add_argument("--bag", required=True, help="path to the .mcap/.db3 bag (file or dir)")
    ap.add_argument("--out-dir", default=None, help="output root holding metadata.json "
                    "(default: bag's parent)")
    ap.add_argument("--camera", default=CAMERA, help="substring to disambiguate topics")
    ap.add_argument("--gyro-topic", default=None, help="explicit depth_to_gyro topic")
    ap.add_argument("--accel-topic", default=None, help="explicit depth_to_accel topic")
    ap.add_argument("--imu-topic", default=None, help="explicit /imu topic (else auto-discover)")
    a = ap.parse_args()

    g = globals()
    g["EXTRINSICS_TOPIC_OVERRIDES"] = {
        "depth_to_gyro": a.gyro_topic,
        "depth_to_accel": a.accel_topic,
    }
    g["IMU_TOPIC_OVERRIDE"] = a.imu_topic
    main(bag=a.bag, out_dir=a.out_dir, camera=a.camera)


if __name__ == "__main__":
    try:
        _cli()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(130)
