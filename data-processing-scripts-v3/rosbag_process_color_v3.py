#!/usr/bin/env python3
# EXPECTED INPUT
# BAG_PATH - a ROS2 bag DIRECTORY (containing metadata.yaml + a .mcap file), e.g.
#            .../2026-07-30_run4/   which holds  2026-07-30_run4_0.mcap + metadata.yaml
#            IMPORTANT: point BAG_PATH at the bag folder, not at the .mcap file itself.

# OUTPUT FOLDER STRUCTURE
# OUT_DIR/
# - videos/               (mp4 previews: cam_ego color + exo_cam<N> compressed streams)
# - timestamps/            (per-stream timestamp CSVs)
# - metadata.json

# DESCRIPTION
# Reads one ROS2 mcap bag and exports its color streams, then writes a metadata.json
# matching the dataset's standard schema (metadata / camera_intrinsics / steps /
# termination). Topics are DISCOVERED by suffix (see DEFAULT_CAMERAS), not hardcoded:
# every connection in the bag whose topic ends with a group's `suffix` is extracted,
# so any number of c922 webcams (c922_1..N) are picked up with no per-camera edit.
#   - ego group    (singleton):  *d435i_ego/color/image_raw   Image           -> videos/cam_ego.mp4
#                                 *d435i_ego/color/camera_info CameraInfo      -> camera_intrinsics[cam_ego].color
#   - exo group    (0..N):       *image_raw/compressed        CompressedImage -> videos/exo_cam<N>.mp4
# Labels: the ego is always `cam_ego`; each exo webcam is `exo_cam<N>`, where <N> is
# the trailing number of its device segment (e.g. /c922_4/... -> exo_cam4).
#
# NOTE: the c922 units are webcams with no meaningful calibration — their
# /c922_N/camera_info (published as zeros) is intentionally NOT discovered, so the
# exo_cam streams do NOT appear in camera_intrinsics. Only cam_ego carries intrinsics.
# NOTE: depth (image_rect_raw / depth camera_info) and the depth->color extrinsics
# topic are intentionally SKIPPED — not extracted, not written to metadata.json.
#
# STEP 0 — run this script once with INSPECT_ONLY = True first. It lists every
# connection (topic, msgtype, message count) found in the bag AND a discovery preview
# showing which topics each group's suffix matches, so you can confirm the suffixes
# pick up what you expect before the full extraction runs.

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json, sys, re, time, dataclasses, subprocess, shutil, tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import cv2
from rosbags.typesys import Stores, get_typestore

from pipeline_metadata import add_error
import bag_integrity                              # spine creator (step 0); its build_initial_spine is reused as a standalone fallback

try:
    from rosbags.highlevel import AnyReader
    from rosbags.image import message_to_cvimage
except Exception as e:
    raise SystemExit(
        "This script requires the 'rosbags[image]' package. Install with:\n"
        "  python -m pip install 'rosbags[image]' opencv-python numpy pandas\n"
        f"Import error: {e}"
    )

# ----------------------------
# CONFIG — EDIT THESE VALUES
# ----------------------------
BAG_PATH = r"D:\SUTD\Robot learning\leo\2026-08-03_color-err"      # bag DIRECTORY, not the .mcap file
OUT_DIR  = r"D:\SUTD\Robot learning\leo\output\run1_test_color_err_v1"          # output root directory for all extracted data
STRIDE   = 1                                 # keep every Nth message per topic (1 = every frame)
# Video encoding. VIDEO_CODEC is the single knob. "h264"/"avc1" streams frames through
# ffmpeg + libx264 (~3-6x smaller than mp4v, CPU-only friendly); the ffmpeg binary is
# resolved from the pip package imageio-ffmpeg first (portable, bundles libx264, travels
# with requirements.txt), then a system ffmpeg on PATH. Any other value is used directly
# as an OpenCV VideoWriter fourcc (e.g. "mp4v", "XVID"). NOTE: OpenCV's own VideoWriter
# cannot emit H.264 here — the opencv-python wheel bundles an ffmpeg without libx264 — so
# H.264 must go through the ffmpeg pipe; if VIDEO_CODEC is "h264" but no ffmpeg is found,
# it falls back to OpenCV "mp4v".
VIDEO_CODEC = "h264"                         # "h264"/"avc1" → ffmpeg libx264; else an OpenCV fourcc ("mp4v", "XVID", …)
CRF      = 23                                # h264 quality/size knob: lower=better/bigger (18≈visually lossless, 28≈small)
PRESET   = "veryfast"                        # h264 speed/size: ultrafast..veryslow (slower=smaller at same CRF)
FORCE_FPS = 0.0                              # 0.0 = infer FPS from median inter-frame interval
INSPECT_ONLY = False                         # True: just print topics/types/counts and exit. Set False to extract.

# metadata.json["metadata"] — hardcoded per-episode fields, edit per recording
DATASET_NAME    = "leo"
DATASET_VERSION = "3.0"
ROBOT_MODEL     = "human"
ENVIRONMENT     = "hotel/building/room name"
SETUP           = "v6"
SUBJECT         = "d4998223-fcab-49d7-999d-c15766b534cc"
BED_TYPE        = "single"

# Camera GROUPS, discovered by topic suffix (not exact topic strings). Every bag
# connection whose topic ends with `suffix` is extracted. This is the STANDALONE
# fallback, used only when main() is called with cameras=None (running this script
# on its own, or its unit tests). For BATCH runs the canonical layout lives in
# wrapper.py (wrapper.DEFAULT_CAMERAS), which passes it in per session — keep the two
# in sync. main(cameras=<dict of this shape>) overrides; None falls back to here.
#   singleton=True  -> at most one topic expected; matched once, labelled `label`
#                      verbatim (cam_ego). >1 match -> warn + use the first.
#   otherwise       -> 0..N topics; each labelled `label` + the trailing number of
#                      its device segment (/c922_4/... -> exo_cam4).
#   info_suffix     -> optional camera_info suffix carrying this group's intrinsics.
DEFAULT_CAMERAS = {
    "ego": {"present": True, "suffix": "d435i_ego/color/image_raw",
            "info_suffix": "d435i_ego/color/camera_info",
            "compressed": False, "label": "cam_ego", "singleton": True},
    "exo": {"present": True, "suffix": "image_raw/compressed",
            "compressed": True,  "label": "exo_cam", "ids": [1, 2, 3, 4]},
}
# `present` (default True) = the SAME opt-out idiom depth/imu use in wrapper.DEFAULT_CAMERAS.
# present:False on a color group skips its discovery/extraction/flagging; main() still
# writes metadata.json so depth/imu have a spine. An absent `present` key means present.
# `ids` is the EXACT SET of exo device numbers expected (labelled exo_cam<id>). Discovery
# still extracts whatever is present (never capped); ids only drives the missing/extra
# check, as a SET difference: a declared id not found -> "missing_stream", a found id not
# declared -> "extra_stream" (both can fire at once — a dropped cam AND a surprise cam).
# Omit an id to intentionally skip that webcam (e.g. [1, 2, 4] runs without cam3). A
# singleton group (ego) expects exactly 1. `ids: None`/absent -> no deviation check.

typestore = get_typestore(Stores.ROS2_HUMBLE)
# ----------------------------


# ---------- helpers ----------

def ensure_dirs(root: Path) -> None:
    (root / 'videos').mkdir(parents=True, exist_ok=True)
    (root / 'timestamps').mkdir(parents=True, exist_ok=True)


def sanitize_topic(topic: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", topic).strip("_")[:120]


def discover_topics(reader: AnyReader, suffix: str) -> List[str]:
    """Every distinct topic in the bag whose path ends with `suffix` (leading
    slashes ignored), sorted. Suffix match, not exact string: 'image_raw/compressed'
    finds '/c922_1/...' through '/c922_N/...' regardless of how many exist."""
    suf = suffix.strip("/")
    return sorted(t for t in {c.topic for c in reader.connections}
                  if t.strip("/").endswith(suf))


def exo_device_id(topic: str) -> Optional[int]:
    """Trailing device number of an exo topic's first path segment
    (/c922_4/image_raw/compressed -> 4), or None when that segment carries no number.
    Drives the declared-vs-found id-set check (which SPECIFIC webcams to expect), so a
    session can skip one webcam by omitting its id. Assumes exo topics are numbered
    (the c922 rig); an unnumbered topic is still extracted but not id-checked."""
    seg = topic.strip("/").split("/")[0]                      # 'c922_4'
    m = re.search(r"(\d+)$", seg)
    return int(m.group(1)) if m else None


def label_for(topic: str, group: Dict[str, Any]) -> str:
    """Output label for a discovered topic. Singleton groups use `label` verbatim
    (cam_ego); otherwise `label` + the trailing number of the device segment
    (/c922_4/image_raw/compressed -> exo_cam4), falling back to the sanitized
    segment when it carries no trailing number."""
    if group.get("singleton"):
        return group["label"]
    eid = exo_device_id(topic)
    if eid is not None:
        return f'{group["label"]}{eid}'
    seg = topic.strip("/").split("/")[0]
    return f'{group["label"]}_{sanitize_topic(seg)}'


def compute_fps(timestamps: np.ndarray) -> float:
    if timestamps.size < 3:
        return 20.0
    dt = np.diff(timestamps)
    dt = dt[dt > 0]
    if dt.size == 0:
        return 20.0
    med = float(np.mean(dt))
    return float(np.clip(1.0 / med, 1.0, 240.0))


def connections_for(reader: AnyReader, topic: str):
    return [c for c in reader.connections if c.topic == topic]


def inspect_bag(reader: AnyReader, cams: Dict[str, Any]) -> None:
    print(f"\n{'TOPIC':45s} {'MSGTYPE':45s} {'COUNT':>8s}")
    print("-" * 100)
    for c in sorted(reader.connections, key=lambda c: c.topic):
        print(f"{c.topic:45s} {str(c.msgtype):45s} {c.msgcount:8d}")

    print("\nDiscovery preview — topics each group's suffix will pick up:")
    for name, g in cams.items():
        matched = discover_topics(reader, g["suffix"])
        if matched:
            for t in matched:
                print(f"  [{name}] {t}  ->  {label_for(t, g)}")
        else:
            print(f"  [{name}] (no topic matches suffix '*{g['suffix']}')")
        for t in discover_topics(reader, g["info_suffix"]) if g.get("info_suffix") else []:
            print(f"  [{name}] {t}  ->  intrinsics")


# ---------- image / video export ----------

def _decode_raw_frame(reader: AnyReader, conn, raw) -> np.ndarray:
    """sensor_msgs/Image -> single BGR uint8 frame."""
    msg = reader.deserialize(raw, conn.msgtype)
    cv_img = message_to_cvimage(msg)
    if cv_img.dtype != np.uint8:
        cv_img = np.clip(cv_img, 0, 255).astype(np.uint8)
    if cv_img.ndim == 2:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
    elif cv_img.ndim == 3 and cv_img.shape[2] == 4:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2BGR)
    else:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    return cv_img


def _decode_compressed_frame(reader: AnyReader, conn, raw) -> np.ndarray:
    """sensor_msgs/CompressedImage (jpeg/png bytes) -> single BGR uint8 frame."""
    msg = reader.deserialize(raw, conn.msgtype)
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    cv_img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # already BGR
    if cv_img is None:
        raise RuntimeError("cv2.imdecode returned None")
    return cv_img


# ---------- video writer backends ----------

def _resolve_ffmpeg() -> Optional[str]:
    """Path to an ffmpeg binary that can encode H.264. Prefers the pip package
    imageio-ffmpeg (portable, bundles libx264, installs with requirements.txt) so the
    encoder travels with the Python env; falls back to a system ffmpeg on PATH. Returns
    None if neither is available (caller then uses the OpenCV mp4v writer)."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe:
            return exe
    except Exception:
        pass
    return shutil.which("ffmpeg")


class _FfmpegWriter:
    """Streams BGR frames into an ffmpeg subprocess encoding H.264 (libx264). Used
    instead of cv2.VideoWriter because the opencv-python wheel's bundled ffmpeg has no
    libx264, so avc1/H264 can't open on a CPU-only box; a real libx264 gives ~3-6x
    smaller files than mp4v with a CRF quality knob. Exposes write()/close()."""

    def __init__(self, exe: str, vid_path: Path, fps: float, w: int, h: int):
        self.vid_path = vid_path
        self._err = tempfile.TemporaryFile()          # ffmpeg stderr -> file, avoids pipe deadlock
        cmd = [
            exe, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", f"{max(fps, 1.0):.6f}", "-i", "pipe:0",
            "-an",                                     # no audio
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",    # yuv420p requires even dimensions
            "-c:v", "libx264", "-preset", PRESET, "-crf", str(CRF),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",                 # seek-friendly moov atom
            str(vid_path),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self._err)

    def write(self, frame_bgr: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame_bgr, dtype=np.uint8).tobytes())

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        ret = self.proc.wait()
        self._err.seek(0)
        err = self._err.read().decode("utf-8", "replace").strip()
        self._err.close()
        if ret != 0:
            raise RuntimeError(f"ffmpeg failed to encode {self.vid_path.name} (exit {ret}): {err}")


class _Cv2Writer:
    """OpenCV VideoWriter fallback (fourcc from VIDEO_CODEC, e.g. mp4v) for when no
    ffmpeg is available. Larger files than libx264 but needs zero extra dependencies."""

    def __init__(self, vid_path: Path, fps: float, w: int, h: int, fourcc: str):
        self.vw = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h), isColor=True)
        if not self.vw.isOpened():
            raise RuntimeError(f"Could not open VideoWriter (fourcc={fourcc!r}) for {vid_path}. "
                               f"Try VIDEO_CODEC='mp4v' or 'h264'.")

    def write(self, frame_bgr: np.ndarray) -> None:
        self.vw.write(frame_bgr)

    def close(self) -> None:
        self.vw.release()


def _open_video_writer(vid_path: Path, fps: float, w: int, h: int):
    """Open a writer for (w, h) BGR frames, chosen by VIDEO_CODEC. 'h264'/'avc1' pipe to
    ffmpeg+libx264 (small, portable, CPU-only friendly); any other value is used directly
    as an OpenCV fourcc (e.g. 'mp4v'). H.264 falls back to the OpenCV 'mp4v' writer only
    if no ffmpeg is found. Both writers expose write(frame)/close()."""
    if VIDEO_CODEC.lower() in ("h264", "avc1", "x264", "libx264"):
        exe = _resolve_ffmpeg()
        if exe:
            return _FfmpegWriter(exe, vid_path, fps, w, h)
        print(f"[WARN] No ffmpeg found (pip install imageio-ffmpeg for portable H.264); "
              f"falling back to OpenCV 'mp4v' for {vid_path.name}.")
        return _Cv2Writer(vid_path, fps, w, h, "mp4v")
    return _Cv2Writer(vid_path, fps, w, h, VIDEO_CODEC)


def peek_timestamps(reader: AnyReader, topic: str) -> np.ndarray:
    """Cheap first pass: just grab message timestamps (no deserialize/decode) so we
    can pick an fps before opening the VideoWriter, without holding decoded frames."""
    conns = connections_for(reader, topic)
    ts_list = [t_ns / 1e9 for i, (conn, t_ns, raw) in enumerate(reader.messages(connections=conns))
               if not (STRIDE > 1 and (i % STRIDE) != 0)]
    return np.asarray(ts_list, dtype=float)


def export_video_stream(reader: AnyReader, topic: str, out_root: Path, compressed: bool,
                         camera: str, label: Optional[str] = None) -> Dict[str, Any]:
    """Streams frames straight into the VideoWriter one at a time instead of buffering
    every decoded frame in a Python list first — buffer-then-write ran the process out
    of memory on longer recordings."""
    conns = connections_for(reader, topic)
    base = label or sanitize_topic(topic)
    vid_path = out_root / 'videos' / f'{base}.mp4'
    csv_path = out_root / 'timestamps' / f'{base}.csv'
    decode = _decode_compressed_frame if compressed else _decode_raw_frame

    if not conns:
        print(f"[WARN] Topic not found: {topic}")
        return {'camera': camera, 'kind': 'color', 'topic': topic, 'num_frames': 0, 'found': False}

    ts_peek = peek_timestamps(reader, topic)
    fps = FORCE_FPS if FORCE_FPS > 0 else compute_fps(ts_peek)

    writer = None
    ts_written: List[float] = []
    n_failed = 0
    for i, (conn, t_ns, raw) in enumerate(reader.messages(connections=conns)):
        if STRIDE > 1 and (i % STRIDE) != 0:
            continue
        try:
            cv_img = decode(reader, conn, raw)
        except Exception as e:
            n_failed += 1
            print(f"[WARN] Failed to decode {topic} frame {i}: {e}")
            continue
        if writer is None:
            h, w = cv_img.shape[:2]
            writer = _open_video_writer(vid_path, fps, w, h)
        writer.write(cv_img)
        ts_written.append(t_ns / 1e9)
    if writer is not None:
        writer.close()
        print(f"Saved {vid_path.name} ({len(ts_written)} frames @ {fps:.2f} fps"
              + (f", {n_failed} failed" if n_failed else "") + ")")
    else:
        print(f"[WARN] No frames decoded for {topic}; skipping video.")

    ts = np.asarray(ts_written, dtype=float)
    pd.DataFrame({'index': np.arange(ts.size), 'ros_time_s': ts}).to_csv(csv_path, index=False)
    return {
        'camera': camera, 'kind': 'color', 'topic': topic,
        'video': str(vid_path.relative_to(out_root)) if writer is not None else None,
        'timestamps': str(csv_path.relative_to(out_root)),
        'fps': fps, 'num_frames': int(ts.size), 'decode_failures': n_failed, 'found': True,
        'ts_min': float(ts.min()) if ts.size else None, 'ts_max': float(ts.max()) if ts.size else None,
    }


# ---------- camera info ----------

def read_first_camerainfo(reader: AnyReader, topic: str) -> Optional[Dict[str, Any]]:
    conns = connections_for(reader, topic)
    for conn, t_ns, raw in reader.messages(connections=conns):
        try:
            msg = reader.deserialize(raw, conn.msgtype)
            return {
                'camera_info_topic': topic,
                'width': getattr(msg, 'width', None),
                'height': getattr(msg, 'height', None),
                'distortion_model': getattr(msg, 'distortion_model', None),
                'K': list(getattr(msg, 'k', getattr(msg, 'K', []))),
                'D': list(getattr(msg, 'd', getattr(msg, 'D', []))),
                'R': list(getattr(msg, 'r', getattr(msg, 'R', []))),
                'P': list(getattr(msg, 'p', getattr(msg, 'P', []))),
                'frame_id': getattr(getattr(msg, 'header', None), 'frame_id', None),
            }
        except Exception as e:
            print(f"[WARN] Failed to decode camera_info {topic}: {e}")
            continue
    print(f"[WARN] No camera_info messages found on {topic}")
    return None


# ---------- msg -> JSON-able + ISO-time helpers ----------

def _to_jsonable(val: Any) -> Any:
    if dataclasses.is_dataclass(val):
        return {f.name: _to_jsonable(getattr(val, f.name)) for f in dataclasses.fields(val)}
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (list, tuple)):
        return [_to_jsonable(v) for v in val]
    if isinstance(val, (bytes, bytearray)):
        return {'__bytes_len__': len(val)}
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.integer,)):
        return int(val)
    return val


def convert_ros_to_iso(timestamp_s: float) -> str:
    """Converts a floating-point ROS epoch timestamp (in seconds) to ISO 8601 UTC string."""
    try:
        dt = datetime.fromtimestamp(timestamp_s, tz=timezone.utc)
        return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    except Exception:
        return str(timestamp_s)


# ---------- main ----------

def _resolve_meta(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge a caller-supplied metadata override over the module defaults.
    Any field the caller omits falls back to the module constant, so the
    script still runs standalone with no `meta` argument."""
    defaults = {
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "robot_model": ROBOT_MODEL,
        "environment": ENVIRONMENT,
        "setup": SETUP,
        "subject": SUBJECT,
        "bed_type": BED_TYPE,
    }
    return {**defaults, **(meta or {})}


def main(bag=None, out_dir=None, meta=None, inspect_only=None, cameras=None) -> Dict[str, Any]:
    """Extract one bag's color streams and write metadata.json.

    Wrapper usage (no shell): call main(bag=<bag dir>, out_dir=<root>,
    meta=<dict of the descriptive metadata fields>, cameras=<camera-group dict>).
    The call args override BAG_PATH / OUT_DIR / INSPECT_ONLY; every field omitted
    from `meta` falls back to the module constant (see _resolve_meta); `cameras`
    None falls back to DEFAULT_CAMERAS. Returns the metadata dict that was written
    (an empty dict in INSPECT_ONLY mode).
    """
    bagpath = Path(bag if bag is not None else BAG_PATH)
    out_root = Path(out_dir if out_dir is not None else OUT_DIR)
    inspect = INSPECT_ONLY if inspect_only is None else inspect_only
    cams = cameras if cameras is not None else DEFAULT_CAMERAS
    m = _resolve_meta(meta)

    if not bagpath.exists():
        raise SystemExit(f"Bag path not found: {bagpath}")
    if bagpath.is_file():
        raise SystemExit(
            f"BAG_PATH points at a file ({bagpath.name}). Point it at the bag DIRECTORY that "
            f"contains metadata.yaml and this .mcap file instead, e.g. BAG_PATH = '{bagpath.parent}'"
        )
    if not (bagpath / 'metadata.yaml').exists():
        raise SystemExit(f"No metadata.yaml found in {bagpath} — is this really a rosbag2 folder?")

    with AnyReader([bagpath], default_typestore=typestore) as reader:
        if inspect:
            inspect_bag(reader, cams)
            print("\nSet INSPECT_ONLY = False once the topic list above looks right.")
            return {}

        ensure_dirs(out_root)
        streams_meta: List[Dict[str, Any]] = []
        # Intrinsics-bearing cameras only (the ego RealSense); keyed by output label.
        cam_blocks: Dict[str, Dict[str, Any]] = {}
        global_min_ts: Optional[float] = None
        global_max_ts: Optional[float] = None

        def bump_range(ts_min, ts_max):
            nonlocal global_min_ts, global_max_ts
            if ts_min is None:
                return
            global_min_ts = ts_min if global_min_ts is None else min(global_min_ts, ts_min)
            global_max_ts = ts_max if global_max_ts is None else max(global_max_ts, ts_max)

        # Presence-deviation entries collected during extraction. Split by sub-asset
        # so each maps to its own termination reason (both still land in the same
        # steps.*_error key; only the reason distinguishes them):
        #   data-plane (image stream) miss/extra -> color_presence_err
        #   info-plane (camera_info / intrinsics) miss -> color_info
        missing_entries: List[str] = []       # data-plane -> color_presence_err
        extra_entries: List[str] = []         # data-plane -> color_presence_err
        missing_info_entries: List[str] = []  # camera_info -> color_info

        # --- ego (singleton): raw color + intrinsics, depth intentionally skipped ---
        g = cams["ego"]
        ego_present = g.get("present", True)
        # Declared absent (present:False) -> skip discovery/extraction/flagging entirely.
        # main() still writes metadata.json (the spine depth/imu append to). Absent-by-
        # DECLARATION is NOT a missing_stream (that reason is for a DECLARED-but-not-found
        # stream). The wrapper's validate_cameras guard already ensured depth isn't also
        # on, since depth aligns onto the ego color stream.
        ego_topics = discover_topics(reader, g["suffix"]) if ego_present else []
        if not ego_present:
            print("[info] ego color declared present:False — skipping ego extraction")
        elif not ego_topics:
            print(f"[WARN] no ego color topic matches suffix '*{g['suffix']}'")
            missing_entries.append(f"{g['label']}: expected but not found (*{g['suffix']})")
        else:
            # First match = the canonical ego (carries intrinsics + is the depth
            # anchor). Any further match is an unexpected EXTRA — still extracted,
            # under a distinct label so it neither overwrites cam_ego.mp4 nor is lost.
            for i, topic in enumerate(ego_topics):
                label = g["label"] if i == 0 else f"{g['label']}_{i + 1}"   # cam_ego, cam_ego_2, …
                rec = export_video_stream(reader, topic, out_root, compressed=g["compressed"],
                                          camera=label, label=label)
                streams_meta.append(rec); bump_range(rec.get('ts_min'), rec.get('ts_max'))
                if i == 0:
                    cam_blocks[label] = {'camera': label}
                    info_topics = discover_topics(reader, g["info_suffix"]) if g.get("info_suffix") else []
                    ci = read_first_camerainfo(reader, info_topics[0]) if info_topics else None
                    if ci:
                        cam_blocks[label]['color'] = ci
                    elif g.get("info_suffix"):
                        # The ego DECLARES intrinsics (info_suffix); an absent topic OR a
                        # topic with no decodable camera_info message is a presence failure.
                        # The RGB frames still extract, but without K the ego loses its
                        # geometric role — depth align / VIO can't register to color. It
                        # lands in steps.missing_stream_error like any presence miss, but
                        # under the color_info reason (intrinsics), NOT color_presence_err (frames).
                        # Exo webcams declare no info_suffix, so this never fires for them.
                        why = ("camera_info topic not found" if not info_topics
                               else "camera_info topic present but no decodable message")
                        missing_info_entries.append(
                            f"{label} camera_info: {why} (*{g['info_suffix']})")
                else:
                    print(f"[WARN] *** UNEXPECTED EXTRA EGO STREAM {label} ({topic}) — "
                          f"expected exactly 1; extracting anyway ***")
                    extra_entries.append(f"{label}: unexpected extra ego stream ({topic})")

        # --- exo webcams (0..N): compressed color only, no camera_info / no intrinsics ---
        g = cams["exo"]
        exo_present = g.get("present", True)                  # present:False -> skip the group
        if not exo_present:
            print("[info] exo color declared present:False — skipping exo extraction")
        exo_topics = discover_topics(reader, g["suffix"]) if exo_present else []   # c922_1..N, any count
        found_ids: List[int] = []
        for topic in exo_topics:
            label = label_for(topic, g)                       # 'exo_cam1' .. 'exo_camN'
            rec = export_video_stream(reader, topic, out_root, compressed=g["compressed"],
                                      camera=label, label=label)
            streams_meta.append(rec); bump_range(rec.get('ts_min'), rec.get('ts_max'))
            eid = exo_device_id(topic)
            if eid is not None:
                found_ids.append(eid)
        # id-set check: extraction already took everything; this only CLASSIFIES the
        # deviation as a SET difference against the declared ids. A declared id not found
        # -> missing; a found id not declared -> extra (both can fire together). Never
        # dropped. Skipped when the group is absent or `ids` is undeclared.
        ids = g.get("ids")
        if exo_present and ids is not None:
            declared = set(ids)
            found = set(found_ids)
            missing_ids = sorted(declared - found)
            extra_ids = sorted(found - declared)
            if missing_ids:
                print(f"[WARN] exo: declared {sorted(declared)}, missing {missing_ids}")
                missing_entries.append(
                    f"{g['label']}: declared {sorted(declared)}, missing {missing_ids} "
                    f"(present: {sorted(found)})")
            if extra_ids:
                print(f"[WARN] *** exo: undeclared webcam(s) {extra_ids} present — extracting all ***")
                extra_entries.append(
                    f"{g['label']}: undeclared {extra_ids} "
                    f"(declared {sorted(declared)}, present: {sorted(found)})")

    # ---- append into the spine (created by bag_integrity, step 0) ----
    # Colour extraction is an APPENDER now, not the spine creator. bag_integrity wrote
    # metadata.json first (the `metadata` block, index-derived date_recorded /
    # steps.timestamp_range, empty spine). We READ it and add ONLY colour's slices:
    # camera_intrinsics (ego), the colour streams, and the presence tokens. We do NOT
    # write the metadata block, date_recorded, timestamp_range, or any top-level fps
    # (fps is a per-stream stat -> steps.streams[].fps).
    #   Standalone / no bag_integrity step: fall back to building a minimal spine ourselves
    # (build_initial_spine is bag_integrity's pure core) so the script still runs on its own.
    meta_path = out_root / 'metadata.json'
    if meta_path.is_file():
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        date_recorded = time.strftime('%Y-%m-%d', time.gmtime(global_min_ts)) if global_min_ts is not None else None
        ts_range = [global_min_ts, global_max_ts] if global_min_ts is not None else None
        metadata = bag_integrity.build_initial_spine(
            m, corrupt=False, detail=[], date_recorded=date_recorded, timestamp_range=ts_range)

    # camera_intrinsics: find-or-create the ego camera entry and set its colour sub-block.
    # A cam whose camera_info was not decodable gets a bare {"camera": label} entry so the
    # depth extractor can still attach its own `depth` sub-block. Owner-scoped + idempotent
    # (re-run updates in place), mirroring pipeline_metadata.upsert_intrinsic.
    cams = metadata.setdefault('camera_intrinsics', [])
    for label, block in cam_blocks.items():
        entry = next((c for c in cams if c.get('camera') == label), None)
        if entry is None:
            entry = {'camera': label}
            cams.append(entry)
        if block.get('color') is not None:
            entry['color'] = block['color']

    # steps.streams: colour OWNS the kind=="color" entries — drop any prior colour streams
    # (idempotent re-run) then append this run's; depth/imu streams are left untouched.
    steps = metadata.setdefault('steps', {})
    streams = steps.setdefault('streams', [])
    streams[:] = [s for s in streams if s.get('kind') != 'color'] + streams_meta

    # ---- termination + presence keys (colour-owned, append-only via add_error) ----
    # OWNS the colour PRESENCE tokens: color_presence_err (image stream missing/extra) and
    # color_info (camera_info / intrinsics missing). Per-frame QUALITY (frame loss +
    # undecodable frames -> color_data, timestamp gaps -> color_timestamps) is validate_color's;
    # structural bag corruption (rosbag_corruption) is bag_integrity's — we touch neither.
    # Tokens are DISJOINT, so validate_color never strips color_presence_err.
    add_error(steps, 'missing_stream_error', missing_entries + missing_info_entries)
    add_error(steps, 'extra_stream_error', extra_entries)
    if missing_entries or extra_entries:
        add_error(metadata['termination'], 'reason', ['color_presence_err'])
    if missing_info_entries:
        add_error(metadata['termination'], 'reason', ['color_info'])
    metadata['termination']['is_successful'] = not metadata['termination']['reason']

    with open(out_root / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata to {out_root / 'metadata.json'}")
    if missing_entries or extra_entries or missing_info_entries:
        n_missing = len(missing_entries) + len(missing_info_entries)
        print(f"[WARN] termination.is_successful = False — "
              f"{n_missing} missing / {len(extra_entries)} extra stream issue(s):")
        for r in missing_entries + missing_info_entries + extra_entries:
            print(f"  - {r}")

    return metadata


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted by user.')
        sys.exit(130)