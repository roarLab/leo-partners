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


def export_imu_samples(reader, topic: str, out_root, color_times: Optional[np.ndarray]
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
    csv_path = out_root / "imu" / f"{METADATA_CAMERA_LABEL}_imu.csv"
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
        "camera": METADATA_CAMERA_LABEL,
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
def record_imu_presence(out_root, missing_data: List[str], missing_info: List[str],
                        extra_data: List[str], extra_info: List[str]) -> bool:
    """Record IMU presence deviations in the EXISTING metadata.json, mirroring colour's
    data/info split. Two planes:
      - data-plane (the /imu sample stream) miss/extra   -> termination reason 'imu_presence_err'
      - info-plane (depth_to_gyro/accel extrinsics) miss/extra -> reason 'imu_info'
    Both planes share the steps.missing_stream_error / extra_stream_error keys (like a
    missing camera vs its camera_info in colour); only the reason token distinguishes
    them. Everything is append-only via add_error, so re-runs don't duplicate and other
    writers' signals are preserved; is_successful is recomputed. No-op (False) if
    metadata.json is absent or there is nothing to record.

    FLAG-AND-CONTINUE: unlike missing depth (which aborts — there is nothing to align),
    a missing/extra IMU is only flagged; colour/depth outputs remain usable."""
    if not (missing_data or missing_info or extra_data or extra_info):
        return False
    meta_path = Path(out_root) / METADATA_FILENAME
    if not meta_path.is_file():
        return False
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    steps = meta.setdefault("steps", {})
    add_error(steps, "missing_stream_error", missing_data + missing_info)
    add_error(steps, "extra_stream_error", extra_data + extra_info)
    term = meta.setdefault("termination", {"is_successful": True, "reason": []})
    if missing_data or extra_data:
        # Data-plane PRESENCE (missing/empty /imu, or a surplus topic) is its own token,
        # mirroring colour's color_presence_err — NOT imu_data (validate_imu's per-frame
        # QUALITY token). Kept distinct so validate_imu, which drops-then-re-adds imu_data,
        # can never strip a presence flag (the extra-topic case reaches the validator, so a
        # shared token would be clobbered on an otherwise-clean stream).
        add_error(term, "reason", ["imu_presence_err"])
    if missing_info or extra_info:
        add_error(term, "reason", ["imu_info"])
    term["is_successful"] = not term.get("reason")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[imu] recorded IMU presence deviations in {meta_path}")
    return True


def write_to_metadata(out_root, found: Dict[str, dict], imu_stream: Optional[dict]) -> int:
    """Upsert the IMU extrinsics AND (if given) append the imu sample stream into the
    EXISTING metadata.json in ONE read-modify-write. Extrinsics go through the shared
    owner-scoped upsert_extrinsic; the imu stream REPLACES any prior (cam_ego, imu) entry
    (idempotent) and leaves every other stream/key untouched. Returns the number of
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
    if imu_stream is not None:
        streams = meta.setdefault("steps", {}).setdefault("streams", [])
        streams[:] = [s for s in streams
                      if not (s.get("camera") == imu_stream["camera"]
                              and s.get("kind") == "imu")]
        streams.append(imu_stream)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    msg = f"+{len(found)} extrinsic(s): {sorted(found)}"
    if imu_stream is not None:
        msg += f", imu stream ({imu_stream['num_samples']} samples)"
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
    imu_stream: Optional[dict] = None
    # Presence buckets, split by plane so each maps to its own termination reason:
    #   data-plane (/imu samples) -> imu_data ; info-plane (extrinsics) -> imu_info.
    missing_info: List[str] = []   # gyro/accel extrinsic absent
    extra_info: List[str] = []     # >1 topic for one extrinsic
    missing_data: List[str] = []   # /imu absent OR present-but-empty
    extra_data: List[str] = []     # >1 /imu topic
    with AnyReader([bag], default_typestore=typestore) as reader:
        conns = list(reader.connections)
        for name, suffix in EXTRINSICS:
            cands = matching_topics(conns, suffix, camera, EXTRINSICS_TOPIC_OVERRIDES.get(name))
            if not cands:
                print(f"[imu] extrinsic '{name}' not found (*{suffix}); recording missing (imu_info).")
                missing_info.append(f"{METADATA_CAMERA_LABEL} {name}: expected but not found (*{suffix})")
                continue
            if len(cands) > 1:
                print(f"[imu] WARN multiple '{name}' topics {cands}; using {cands[0]}, flagging extra.")
                extra_info.append(f"{METADATA_CAMERA_LABEL} {name}: expected 1, found {len(cands)} ({cands})")
            topic = cands[0]
            rot, trans = read_extrinsics(reader, topic)
            found[name] = {"topic": topic, "rotation": rot, "translation": trans}
            print(f"[imu] {name}: R(col-major)={np.round(rot, 6).tolist()} "
                  f"t(m)={np.round(trans, 6).tolist()}  <- {topic}")

        # --- /imu SAMPLES (united gyro+accel). Needs the color-written metadata for the
        # frame-time match, so it no-ops when metadata.json is absent (run color first). ---
        imu_cands = matching_topics(conns, IMU_SUFFIX, camera, IMU_TOPIC_OVERRIDE)
        if not imu_cands:
            missing_data.append(f"{METADATA_CAMERA_LABEL} imu: expected but not found (*{IMU_SUFFIX})")
            print(f"[imu] no /imu topic (*{IMU_SUFFIX}); recording missing imu (imu_data).")
        elif not meta_present:
            print("[imu] metadata.json absent; skipping sample extraction "
                  "(run rosbag_process_color_v3 first).")
        else:
            if len(imu_cands) > 1:
                print(f"[imu] WARN multiple /imu topics {imu_cands}; using {imu_cands[0]}, flagging extra.")
                extra_data.append(f"{METADATA_CAMERA_LABEL} imu: expected 1, found {len(imu_cands)} ({imu_cands})")
            color_times = load_color_frame_times(out_root, METADATA_CAMERA_LABEL)
            imu_stream = export_imu_samples(reader, imu_cands[0], out_root, color_times)
            if imu_stream is None:                       # topic present but 0 messages streamed
                missing_data.append(f"{METADATA_CAMERA_LABEL} imu: topic present but streamed no messages")

    if not found:
        print(f"[imu] no IMU extrinsics found in {bag} "
              f"(looked for {[s for _, s in EXTRINSICS]}).")

    n_written = write_to_metadata(out_root, found, imu_stream)
    # Presence deviations -> imu_data (samples) / imu_info (extrinsics). No-op if
    # metadata.json is absent; flag-and-continue (never aborts the bag).
    record_imu_presence(out_root, missing_data, missing_info, extra_data, extra_info)
    return {
        "bag": str(bag),
        "camera": camera,
        "extrinsics_found": sorted(found),
        "extrinsics_written": int(n_written),
        "imu_samples": int(imu_stream["num_samples"]) if imu_stream else 0,
        "imu_missing": bool(missing_data),      # data-plane absent/empty (console tail)
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
