#!/usr/bin/env python3
# ==============================================================================
# rosbag_process_depth_v3.py
#
# Offline depth->color alignment for a recorded ROS 2 bag from the D435i
# multicam rig (captured with align_depth.enable:=False). Produces the SAME
# result rs2::align would have produced live, because it is a faithful
# transcription of librealsense's own alignment code applied to the SAME
# calibration and the SAME (in-camera-rectified) depth pixels the bag stores.
#
# WHY OFFLINE == LIVE (researched, with sources):
#   - Live rs2::align IS src/proc/align.cpp. The routine is purely geometric:
#     zero-fill output, then per depth pixel: deproject -> transform ->
#     project -> fill the pixel's projected rect -> keep nearest (min-z).
#     No filtering, no occlusion pass, no hidden state.
#   - The bag's camera_info + extrinsics/depth_to_color ARE the librealsense
#     stream-profile calibration (identical numbers live align reads via
#     get_intrinsics()/get_extrinsics_to()). Depth is already rectified in the
#     camera's D4 ASIC at capture, so the pixels are identical live or offline.
#   - Intel/Open3D explicitly recommend deferring alignment to bag-reading for
#     high-res capture (align_depth_to_color=false at capture, align on read).
#   - D400 color/depth distortion coeffs are all-zero by design, so deproject
#     and project both reduce to pinhole -> the distortion-model tag that ROS
#     camera_info drops is irrelevant here (guarded by an assert below).
#
# The ONLY places offline can differ from live:
#   1. Frame pairing on dropped frames (surfaced honestly as blank frames +
#      counts; live align would also have produced nothing for a missing depth).
#   2. Bit-exactness: align.cpp picks CUDA/SSE/NEON/generic at runtime, so even
#      live-vs-live is not bit-identical. We match the generic path by computing
#      in float32 with the same round-half-via-truncation, static_cast<int>(x+0.5f).
#
# ALGORITHM CITATION (transcribed verbatim; pin these when reviewing):
#   IntelRealSense/librealsense (Apache-2.0)
#   - src/proc/align.cpp @ master
#       align_images()      : corner-projection + rect-fill hole filling
#       align_z_to_other()  : memset(0); std::min z-buffer; stores RAW uint16,
#                             uses z_scale only to get metric depth for deproject
#   - include/librealsense2/rsutil.h @ v2.44.0 (last tag with inline impls)
#       rs2_deproject_pixel_to_point()  : pixel+depth -> 3D (depth camera)
#       rs2_transform_point_to_point()  : column-major R (9) + t (3)
#       rs2_project_point_to_pixel()    : 3D -> pixel (color camera)
#   - include/librealsense2/h/rs_types.h : rs2_distortion enum values below
#
# OUTPUT (mirrors rosbag_process_v2.py house convention):
#   <out-dir>/depth_frames/<cam>_aligned_depth_to_color.h5   dataset 'data'
#       (N_color, Hc, Wc) uint16 mm, gzip-4, per-frame chunks, indexed 1:1 to
#       the color stream (aligned[i] <-> color frame i). Unpaired color frames
#       stay all-zero (0 = "no depth", same as librealsense). Calibration and
#       provenance stored as HDF5 attrs so the file is self-describing.
#   <out-dir>/timestamps/<cam>_aligned_depth_to_color.csv
#       index, color_stamp_s, depth_stamp_s, pair_dt_ms, has_depth
#
# DEPENDENCIES: none new. Uses rosbags, rosbags-image, numpy, h5py (all already
# in requirements.txt). opencv-python only for the optional --overlay-png mode.
#
# USAGE:
#   python rosbag_process_depth_v3.py --bag /path/to/session.mcap --camera ego
#   python rosbag_process_depth_v3.py --bag ... --overlay-png 8
#
# Ground-truth A/B validation against the SDK's own aligned topic lives in the
# separate one-off check data-processing-scripts-v3/groundtruth_check.py -- it is
# NOT part of this module, since production bags carry no ground-truth topic.
# ==============================================================================
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import h5py

from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage
from rosbags.typesys import Stores, get_typestore

from pipeline_metadata import add_error, upsert_extrinsic

# --- rs2_distortion enum (include/librealsense2/h/rs_types.h) -----------------
RS2_DISTORTION_NONE = 0
RS2_DISTORTION_MODIFIED_BROWN_CONRADY = 1
RS2_DISTORTION_INVERSE_BROWN_CONRADY = 2
RS2_DISTORTION_FTHETA = 3
RS2_DISTORTION_BROWN_CONRADY = 4
RS2_DISTORTION_KANNALA_BRANDT4 = 5
_MODEL_NAMES = {
    0: "NONE", 1: "MODIFIED_BROWN_CONRADY", 2: "INVERSE_BROWN_CONRADY",
    3: "FTHETA", 4: "BROWN_CONRADY", 5: "KANNALA_BRANDT4",
}
# Best-effort ROS distortion_model string -> rs2 model. The ROS string cannot
# distinguish forward vs inverse Brown-Conrady, but with zero coeffs it does not
# matter (pinhole either way). Override with --color-model / --depth-model.
_ROS_MODEL_MAP = {
    "": RS2_DISTORTION_NONE,
    "none": RS2_DISTORTION_NONE,
    "plumb_bob": RS2_DISTORTION_BROWN_CONRADY,
    "rational_polynomial": RS2_DISTORTION_BROWN_CONRADY,
    "brown_conrady": RS2_DISTORTION_BROWN_CONRADY,
    "inverse_brown_conrady": RS2_DISTORTION_INVERSE_BROWN_CONRADY,
    "modified_brown_conrady": RS2_DISTORTION_MODIFIED_BROWN_CONRADY,
    "equidistant": RS2_DISTORTION_KANNALA_BRANDT4,
    "kannala_brandt": RS2_DISTORTION_KANNALA_BRANDT4,
    "ftheta": RS2_DISTORTION_FTHETA,
}

# =============================================================================
# CONFIG
# A wrapper drives this module the house way: set these constants (or pass
# bag/out_dir/camera to main()) and call main(). You normally only touch the
# first three; everything below has a sane default for the D435i ego rig.
# The CLI (_cli, for one-off shell runs) just overwrites these before calling.
# =============================================================================
BAG_PATH = None           # path to the .mcap/.db3 bag (file or bag dir)
OUT_DIR = None            # output root; None -> the bag's parent dir
CAMERA = "ego"            # substring to disambiguate topics when >1 camera present

PAIR_TOLERANCE_MS = 16.0  # nearest depth<->color match window (~half a frame @30fps)
HOLE_FILL = True          # True = librealsense corner-rect fill; False = sparse centers
Z_SCALE = 0.001           # depth unit -> meters (D435i verified 0.001 m/unit)
STRIDE = 1                # keep every Nth color frame (1 = all)
LIMIT = 0                 # cap color frames processed (0 = all); for quick tests
OVERLAY_PNG = 0           # write N color+depth overlay PNGs for eyeballing (needs cv2)

# topic discovery: matched by suffix, so namespace/prefix does not matter
DEPTH_SUFFIX = "/depth/image_rect_raw"
COLOR_SUFFIX = "/color/image_raw"
DEPTH_INFO_SUFFIX = "depth/camera_info"
COLOR_INFO_SUFFIX = "color/camera_info"
EXTRINSICS_SUFFIX = "extrinsics/depth_to_color"
# explicit full-topic overrides (None = auto-discover via the suffixes above)
DEPTH_TOPIC = None
COLOR_TOPIC = None
DEPTH_INFO_TOPIC = None
COLOR_INFO_TOPIC = None
EXTRINSICS_TOPIC = None
# calibration overrides (None = read from the bag's camera_info / extrinsics)
ROTATION = None           # 9 floats, column-major, depth->color
TRANSLATION = None        # 3 floats, meters
COLOR_MODEL = None        # force rs2 distortion model int (else inferred from camera_info)
DEPTH_MODEL = None

# metadata.json integration: append this aligned-depth stream to the metadata.json
# that rosbag_process_color_v3 already wrote in OUT_DIR (shared out_dir). CAMERA
# above is a topic-discovery substring ("ego"); the metadata schema labels the
# same camera "cam_ego", so it is a separate knob.
METADATA_FILENAME = "metadata.json"
METADATA_CAMERA_LABEL = "cam_ego"

# --- internal constants (do not change) ---
RECT_CAP = 40             # safety cap on a single pixel's fill-rect extent (px); realistic <=3
INVALID_SENTINEL = 65535  # z-buffer "empty" marker; > any real D435i depth (mm), reset to 0

typestore = get_typestore(Stores.ROS2_HUMBLE)


# ==============================================================================
# Intrinsics container
# ==============================================================================
class Intrinsics:
    """Mirror of rs2_intrinsics for the fields the projection math needs."""
    def __init__(self, width, height, fx, fy, cx, cy, model, coeffs, frame_id=None):
        self.width = int(width)
        self.height = int(height)
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)   # ppx
        self.cy = float(cy)   # ppy
        self.model = int(model)
        self.coeffs = np.asarray(coeffs, dtype=np.float32).ravel()
        if self.coeffs.size < 5:
            self.coeffs = np.pad(self.coeffs, (0, 5 - self.coeffs.size))
        self.frame_id = frame_id

    @property
    def has_distortion(self) -> bool:
        return bool(np.any(self.coeffs[:5] != 0.0))

    def as_dict(self) -> dict:
        return {
            "width": self.width, "height": self.height,
            "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
            "model": self.model, "model_name": _MODEL_NAMES.get(self.model, "?"),
            "coeffs": self.coeffs[:5].tolist(), "frame_id": self.frame_id,
        }


# ==============================================================================
# Faithful transcription of rsutil.h (v2.44.0), vectorized over pixels.
# ==============================================================================
def deproject_normalized(u: np.ndarray, v: np.ndarray, intr: Intrinsics
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """rs2_deproject_pixel_to_point, minus the depth scaling.

    Returns the distortion-corrected normalized coords (x, y) such that the 3D
    point is depth * [x, y, 1]. Independent of depth, so precomputable per pixel.
    """
    if intr.model == RS2_DISTORTION_MODIFIED_BROWN_CONRADY:
        # librealsense asserts this is not deprojectable (forward-distorted).
        raise ValueError("Cannot deproject from MODIFIED_BROWN_CONRADY intrinsics")
    x = (u.astype(np.float32) - np.float32(intr.cx)) / np.float32(intr.fx)
    y = (v.astype(np.float32) - np.float32(intr.cy)) / np.float32(intr.fy)
    c = intr.coeffs
    if intr.model == RS2_DISTORTION_INVERSE_BROWN_CONRADY and intr.has_distortion:
        r2 = x * x + y * y
        f = 1 + c[0] * r2 + c[1] * r2 * r2 + c[4] * r2 * r2 * r2
        ux = x * f + 2 * c[2] * x * y + c[3] * (r2 + 2 * x * x)
        uy = y * f + 2 * c[3] * x * y + c[2] * (r2 + 2 * y * y)
        x, y = ux, uy
    elif intr.model in (RS2_DISTORTION_KANNALA_BRANDT4, RS2_DISTORTION_FTHETA) and intr.has_distortion:
        raise NotImplementedError(
            f"deproject for {_MODEL_NAMES[intr.model]} with non-zero coeffs not "
            "implemented (D435i uses zero coeffs; add if a fisheye stream appears)")
    return x.astype(np.float32), y.astype(np.float32)


def project_pixel(Xc: np.ndarray, Yc: np.ndarray, Zc: np.ndarray, intr: Intrinsics
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """rs2_project_point_to_pixel: 3D point in the target (color) frame -> pixel."""
    with np.errstate(divide="ignore", invalid="ignore"):
        x = Xc / Zc
        y = Yc / Zc
    c = intr.coeffs
    if intr.has_distortion and intr.model in (
            RS2_DISTORTION_MODIFIED_BROWN_CONRADY,
            RS2_DISTORTION_INVERSE_BROWN_CONRADY,
            RS2_DISTORTION_BROWN_CONRADY):
        # v2.44.0 applies the same forward Brown-Conrady for all three tags.
        r2 = x * x + y * y
        f = 1 + c[0] * r2 + c[1] * r2 * r2 + c[4] * r2 * r2 * r2
        x = x * f
        y = y * f
        dx = x + 2 * c[2] * x * y + c[3] * (r2 + 2 * x * x)
        dy = y + 2 * c[3] * x * y + c[2] * (r2 + 2 * y * y)
        x, y = dx, dy
    elif intr.has_distortion and intr.model in (RS2_DISTORTION_FTHETA, RS2_DISTORTION_KANNALA_BRANDT4):
        raise NotImplementedError(
            f"project for {_MODEL_NAMES[intr.model]} with non-zero coeffs not implemented")
    u = x * np.float32(intr.fx) + np.float32(intr.cx)
    v = y * np.float32(intr.fy) + np.float32(intr.cy)
    return u, v


def _round_pixel(vals: np.ndarray) -> np.ndarray:
    """static_cast<int>(v + 0.5f): +0.5 in float32, truncate toward zero."""
    return (vals.astype(np.float32) + np.float32(0.5)).astype(np.int32)


# ==============================================================================
# Aligner: precompute per-pixel constants, then align each depth frame.
# ==============================================================================
class DepthToColorAligner:
    def __init__(self, depth_intr: Intrinsics, color_intr: Intrinsics,
                 rotation9: np.ndarray, translation3: np.ndarray,
                 z_scale: float = Z_SCALE, hole_fill: bool = True):
        self.depth = depth_intr
        self.color = color_intr
        self.z_scale = np.float32(z_scale)
        self.hole_fill = hole_fill
        self.rect_clamped = 0  # count of pixels whose fill-rect hit RECT_CAP

        # Column-major rotation (rsutil.h): to[i] = R[i] f0 + R[i+3] f1 + R[i+6] f2.
        # Equivalent matrix M with to = M @ f + t is reshape(3,3).T of the flat array.
        # Kept unfused (X=d*xn first, then rotate) so the float32 op-order matches
        # rs2_deproject_pixel_to_point + rs2_transform_point_to_point exactly.
        self.M = np.asarray(rotation9, dtype=np.float32).reshape(3, 3).T
        self.t = np.asarray(translation3, dtype=np.float32).ravel()

        Hd, Wd = depth_intr.height, depth_intr.width
        uu, vv = np.meshgrid(np.arange(Wd, dtype=np.float32),
                             np.arange(Hd, dtype=np.float32))
        uu, vv = uu.ravel(), vv.ravel()

        # Distortion-corrected normalized coords per corner (depth-independent).
        self.A = deproject_normalized(uu - 0.5, vv - 0.5, depth_intr)  # top-left
        self.B = deproject_normalized(uu + 0.5, vv + 0.5, depth_intr)  # bottom-right
        self.C = deproject_normalized(uu, vv, depth_intr)              # center
        self.n_out = color_intr.height * color_intr.width

    def _project_corner(self, d: np.ndarray, corner) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xn, yn = corner
        # point = depth * [xn, yn, 1]  (rs2_deproject_pixel_to_point)
        X = d * xn
        Y = d * yn
        Z = d
        M, t = self.M, self.t
        # rs2_transform_point_to_point, same operand order as the scalar SDK code.
        Xc = M[0, 0] * X + M[0, 1] * Y + M[0, 2] * Z + t[0]
        Yc = M[1, 0] * X + M[1, 1] * Y + M[1, 2] * Z + t[1]
        Zc = M[2, 0] * X + M[2, 1] * Y + M[2, 2] * Z + t[2]
        u, v = project_pixel(Xc, Yc, Zc, self.color)
        return u, v, Zc

    def align(self, raw: np.ndarray) -> np.ndarray:
        """Align one raw depth frame (Hd,Wd uint16 mm) -> (Hc,Wc) uint16 mm."""
        Hc, Wc = self.color.height, self.color.width
        rawflat = raw.reshape(-1)
        d = rawflat.astype(np.float32) * self.z_scale        # metric depth (m)
        valid = rawflat > 0                                   # get_depth()==0 -> skip

        zbuf = np.full(self.n_out, INVALID_SENTINEL, dtype=np.uint16)

        if self.hole_fill:
            uA, vA, ZcA = self._project_corner(d, self.A)
            uB, vB, ZcB = self._project_corner(d, self.B)
            x0, y0 = _round_pixel(uA), _round_pixel(vA)
            x1, y1 = _round_pixel(uB), _round_pixel(vB)
            inb = (valid & (ZcA > 0) & (ZcB > 0)
                   & (x0 >= 0) & (y0 >= 0) & (x1 < Wc) & (y1 < Hc))
            idx = np.nonzero(inb)[0]
            if idx.size:
                x0v, y0v = x0[idx], y0[idx]
                x1v, y1v = x1[idx], y1[idx]
                # Clamp pathological rects (never triggers on sane calibration).
                x1c = np.minimum(x1v, x0v + RECT_CAP)
                y1c = np.minimum(y1v, y0v + RECT_CAP)
                self.rect_clamped += int(np.count_nonzero((x1v > x1c) | (y1v > y1c)))
                rv = rawflat[idx]
                maxw = int((x1c - x0v).max()) + 1 if idx.size else 0
                maxh = int((y1c - y0v).max()) + 1 if idx.size else 0
                for dy in range(max(maxh, 1)):
                    ty = y0v + dy
                    row_ok = ty <= y1c
                    for dx in range(max(maxw, 1)):
                        tx = x0v + dx
                        m = row_ok & (tx <= x1c)
                        if not m.any():
                            continue
                        flat = ty[m] * Wc + tx[m]
                        np.minimum.at(zbuf, flat, rv[m])   # std::min z-buffer
        else:
            uC, vC, ZcC = self._project_corner(d, self.C)
            xc, yc = _round_pixel(uC), _round_pixel(vC)
            inb = valid & (ZcC > 0) & (xc >= 0) & (yc >= 0) & (xc < Wc) & (yc < Hc)
            idx = np.nonzero(inb)[0]
            if idx.size:
                flat = yc[idx] * Wc + xc[idx]
                np.minimum.at(zbuf, flat, rawflat[idx])

        zbuf[zbuf == INVALID_SENTINEL] = 0                   # empties -> 0 (no depth)
        return zbuf.reshape(Hc, Wc)


# ==============================================================================
# Bag reading helpers
# ==============================================================================
def _stamp_s(header) -> float:
    return float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9


def pick_topic(conns, suffix: str, camera: Optional[str], override: Optional[str],
               kind: str) -> str:
    if override:
        return override
    cands = [c.topic for c in conns if c.topic.endswith(suffix)]
    if camera:
        cands = [t for t in cands if camera in t]
    cands = sorted(set(cands))
    if not cands:
        raise SystemExit(
            f"[{kind}] no topic ending in '{suffix}'"
            + (f" containing '{camera}'" if camera else "")
            + ". Pass an explicit --{kind}-topic.".replace("{kind}", kind))
    if len(cands) > 1:
        raise SystemExit(
            f"[{kind}] ambiguous, matched {cands}. Disambiguate with --camera or "
            f"--{kind}-topic.")
    return cands[0]


def topics_matching(conns, suffix: str, camera: Optional[str],
                    override: Optional[str]) -> List[str]:
    """ALL topics ending in `suffix` (optionally containing `camera`), sorted; an
    override forces a single explicit topic. [] if none. Unlike pick_topic (which
    raises on 0 or >1), this preserves multiplicity so main() can FLAG-AND-CONTINUE:
    [] -> missing (depth_data / depth_info), len > 1 -> extra."""
    if override:
        return [override]
    cands = sorted({c.topic for c in conns if c.topic.endswith(suffix)})
    if camera:
        cands = [t for t in cands if camera in t]
    return cands


def read_camera_info(reader, topic: str, model_override: Optional[int]) -> Intrinsics:
    conns = [c for c in reader.connections if c.topic == topic]
    for conn, _t, raw in reader.messages(connections=conns):
        msg = reader.deserialize(raw, conn.msgtype)
        K = list(getattr(msg, "k", getattr(msg, "K", [])))
        D = list(getattr(msg, "d", getattr(msg, "D", [])))
        model_str = (getattr(msg, "distortion_model", "") or "").lower()
        model = model_override if model_override is not None else _ROS_MODEL_MAP.get(
            model_str, RS2_DISTORTION_BROWN_CONRADY)
        return Intrinsics(
            width=msg.width, height=msg.height,
            fx=K[0], fy=K[4], cx=K[2], cy=K[5],
            model=model, coeffs=D,
            frame_id=getattr(getattr(msg, "header", None), "frame_id", None))
    raise SystemExit(f"No CameraInfo message on {topic}")


def read_extrinsics(reader, topic: str,
                    rot_override: Optional[List[float]],
                    trans_override: Optional[List[float]]
                    ) -> Tuple[np.ndarray, np.ndarray]:
    if rot_override is not None and trans_override is not None:
        return np.asarray(rot_override, np.float64), np.asarray(trans_override, np.float64)
    conns = [c for c in reader.connections if c.topic == topic]
    if not conns:
        raise SystemExit(
            f"Extrinsics topic '{topic}' not found. Provide --rotation (9 vals, "
            "column-major) and --translation (3 vals, meters) instead.")
    for conn, _t, raw in reader.messages(connections=conns):
        try:
            msg = reader.deserialize(raw, conn.msgtype)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(
                f"Could not deserialize {conn.msgtype} on {topic}: {e}\n"
                "If reading a .db3 without embedded type defs, pass --rotation/"
                "--translation explicitly.")
        rot = np.asarray(list(msg.rotation), np.float64)
        trans = np.asarray(list(msg.translation), np.float64)
        return rot, trans
    raise SystemExit(f"No Extrinsics message on {topic}")


def collect_stamps(reader, topic: str, stride: int) -> np.ndarray:
    """One pass over an image topic collecting header stamps (seconds)."""
    conns = [c for c in reader.connections if c.topic == topic]
    stamps: List[float] = []
    for i, (conn, _t, raw) in enumerate(reader.messages(connections=conns)):
        if stride > 1 and (i % stride):
            continue
        msg = reader.deserialize(raw, conn.msgtype)
        stamps.append(_stamp_s(msg.header))
    return np.asarray(stamps, dtype=np.float64)


def read_camera_info_block(reader, topic: str) -> Optional[dict]:
    """Full ROS camera_info as a plain dict (same shape rosbag_process_color_v3
    writes for the color blocks), so the depth block we insert into
    camera_intrinsics matches the color blocks byte-for-byte. Unlike
    read_camera_info (which collapses to fx/fy/cx/cy) this keeps K/D/R/P and the
    raw distortion_model string. Returns None if the topic has no message."""
    conns = [c for c in reader.connections if c.topic == topic]
    for conn, _t, raw in reader.messages(connections=conns):
        msg = reader.deserialize(raw, conn.msgtype)
        return {
            "camera_info_topic": topic,
            "width": getattr(msg, "width", None),
            "height": getattr(msg, "height", None),
            "distortion_model": getattr(msg, "distortion_model", None),
            "K": list(getattr(msg, "k", getattr(msg, "K", []))),
            "D": list(getattr(msg, "d", getattr(msg, "D", []))),
            "R": list(getattr(msg, "r", getattr(msg, "R", []))),
            "P": list(getattr(msg, "p", getattr(msg, "P", []))),
            "frame_id": getattr(getattr(msg, "header", None), "frame_id", None),
        }
    print(f"[metadata] no camera_info on {topic}; depth intrinsics block skipped.")
    return None


def _relpath(path, root) -> str:
    """Path relative to out_root for the metadata (falls back to abs if unrelated)."""
    p = Path(path)
    try:
        return str(p.relative_to(Path(root)))
    except ValueError:
        return str(p)


def _fps_from_stamps(stamps: np.ndarray) -> Optional[float]:
    """fps as 1 / median(inter-frame interval). Median (not mean) is robust to the
    dropped-frame gaps the paired-depth stamps carry. None if too few stamps."""
    s = np.asarray(stamps, dtype=np.float64)
    if s.size < 3:
        return None
    dt = np.diff(np.sort(s))
    dt = dt[dt > 0]
    if dt.size == 0:
        return None
    return float(1.0 / np.median(dt))


def record_depth_presence(out_root, missing_data: List[str], missing_info: List[str],
                          extra_data: List[str], extra_info: List[str]) -> bool:
    """Record depth presence deviations in the EXISTING metadata.json (written by color),
    mirroring colour's / imu's data/info split. Two planes:
      - data-plane (the depth image topic) miss/extra           -> reason 'depth_presence_err'
      - info-plane (depth camera_info + depth->color extrinsics) -> reason 'depth_info'
    Both planes share the steps.missing_stream_error / extra_stream_error keys; only the
    termination reason distinguishes them. Everything is append-only via add_error, so
    re-runs don't duplicate and other writers' signals are preserved; is_successful is
    recomputed. No-op (False) if metadata.json is absent or there is nothing to record."""
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
        # Data-plane PRESENCE (declared depth topic absent, or a surplus topic) is its own
        # token, mirroring colour's color_presence_err — NOT depth_data (validate_depth's
        # per-frame QUALITY token). Kept distinct so validate_depth, which drops-then-re-adds
        # depth_data, can never strip a presence flag (the extra-topic case reaches the
        # validator, so a shared token would be clobbered on an otherwise-clean stream).
        add_error(term, "reason", ["depth_presence_err"])
    if missing_info or extra_info:
        add_error(term, "reason", ["depth_info"])
    term["is_successful"] = not term.get("reason")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[metadata] recorded depth presence deviations in {meta_path}")
    return True


def append_aligned_depth_to_metadata(out_root, summary: dict, depth_info_block: Optional[dict],
                                     depth_stamps_paired: np.ndarray, color_intr: Intrinsics,
                                     depth_topic: str,
                                     metadata_camera: str = METADATA_CAMERA_LABEL,
                                     depth_to_color_rot=None,
                                     depth_to_color_trans=None,
                                     extrinsics_topic: Optional[str] = None) -> bool:
    """Append the aligned-depth stream (and the depth intrinsics block) into an
    EXISTING metadata.json written by rosbag_process_color_v3, touching no other
    keys. Read-modify-write: only steps.streams gets one new entry and the
    matching camera_intrinsics[camera].depth gets filled; every other key
    (metadata, termination, color streams, timestamp_range, ...) is preserved
    verbatim. Idempotent: a prior (camera, aligned_depth_to_color) entry is
    replaced, not duplicated.

    When depth_to_color_rot/trans are given (from read_extrinsics), the same
    read-modify-write also records the depth->color extrinsic VERBATIM into
    camera_extrinsics (via pipeline_metadata.upsert_extrinsic) — the exact transform
    this alignment consumed — so metadata.json is self-contained for geometry. The
    numbers are copied as-is (no composition); the imu step fills depth_to_gyro /
    depth_to_accel the same owner-scoped way.

    No-op returning False if metadata.json is absent (run rosbag_process first).

    h5_index is the FULL h5 length (summary["n_color"]) -- every color-indexed
    slot including blank/unpaired frames -- so it matches the h5 dataset shape;
    the timestamps CSV has_depth column marks which slots actually carry depth.
    """
    out_root = Path(out_root)
    meta_path = out_root / METADATA_FILENAME
    if not meta_path.is_file():
        print(f"[metadata] {meta_path} not found; skipping metadata update "
              "(run rosbag_process_color_v3 first).")
        return False

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    stream = {
        "camera": metadata_camera,
        "kind": "aligned_depth_to_color",
        "topic": depth_topic,
        "frames_dir": _relpath(summary["h5"], out_root),
        "frame_dtype": "uint16",
        "depth_unit": "mm",
        "width": int(color_intr.width),
        "height": int(color_intr.height),
        "timestamps": _relpath(summary["csv"], out_root),
        "fps_estimate": _fps_from_stamps(depth_stamps_paired),
        # frame accounting (validate_depth_v3 reads these for the two-way check):
        #   h5_index              frames in the h5, one per color frame (blanks incl.)
        #   n_depth_frames        raw depth frames in the bag, before alignment
        #   n_paired              h5 frames carrying real depth
        #   n_pair_missing_depth  blank h5 frames: color had no depth partner
        #   n_pair_missing_color  depth frames dropped: no color partner (exists ONLY here)
        "h5_index": int(summary["n_color"]),
        "n_depth_frames": int(summary.get("n_depth_msgs", 0)),
        "n_paired": int(summary.get("n_paired", 0)),
        "n_pair_missing_depth": int(summary.get("n_blank_color", 0)),
        "n_pair_missing_color": int(summary.get("n_depth_no_partner", 0)),
        # span of the color-indexed timeline (same basis as the color streams' ts_min/ts_max)
        "ts_min": summary.get("ts_min"),
        "ts_max": summary.get("ts_max"),
    }

    steps = meta.setdefault("steps", {})
    streams = steps.setdefault("streams", [])
    streams[:] = [s for s in streams
                  if not (s.get("camera") == metadata_camera
                          and s.get("kind") == "aligned_depth_to_color")]
    streams.append(stream)

    if depth_info_block is not None:
        cams = meta.get("camera_intrinsics", [])
        target = next((c for c in cams if c.get("camera") == metadata_camera), None)
        if target is None:
            print(f"[metadata] no camera_intrinsics entry for '{metadata_camera}'; "
                  "appended stream but skipped depth intrinsics block.")
        else:
            target["depth"] = depth_info_block

    # depth->color extrinsic, verbatim (owner-scoped: depth owns only this name).
    if depth_to_color_rot is not None and depth_to_color_trans is not None:
        upsert_extrinsic(meta, "depth_to_color",
                         extrinsics_topic or EXTRINSICS_SUFFIX,
                         depth_to_color_rot, depth_to_color_trans)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[metadata] updated {meta_path} "
          f"(+aligned_depth_to_color stream for {metadata_camera})")
    return True


# ==============================================================================
# Main
# ==============================================================================
def main(bag=None, out_dir=None, camera=None) -> dict:
    """Align one bag's depth to its color and write the HDF5 + integrity CSV.

    Wrapper usage (no shell): set the module CONFIG constants above as needed,
    then call main(bag=<path>, out_dir=<root>, camera="ego"). The three call
    args override BAG_PATH/OUT_DIR/CAMERA; every other knob is read from CONFIG.
    Returns a summary dict (paired/blank/dropped counts) for batch reporting.
    """
    bag = Path(bag if bag is not None else BAG_PATH)
    if not bag.exists():
        raise SystemExit(f"Bag not found: {bag}")
    _out = out_dir if out_dir is not None else OUT_DIR
    out_root = Path(_out) if _out else bag.parent
    camera = camera if camera is not None else CAMERA
    (out_root / "depth_frames").mkdir(parents=True, exist_ok=True)
    (out_root / "timestamps").mkdir(parents=True, exist_ok=True)

    # Presence buckets (data plane = depth image topic -> depth_data; info plane =
    # depth camera_info + depth->color extrinsics -> depth_info). FLAG-AND-CONTINUE:
    # any missing declared input records these and returns early (no h5), which keeps
    # the bag completed=true because sanity is metadata-driven (no aligned_depth stream
    # is declared, so no h5 is expected).
    missing_data: List[str] = []; extra_data: List[str] = []
    missing_info: List[str] = []; extra_info: List[str] = []

    def _abort(reason: str) -> dict:
        record_depth_presence(out_root, missing_data, missing_info, extra_data, extra_info)
        return {"bag": str(bag), "camera": camera, "aligned": False, "reason": reason}

    with AnyReader([bag], default_typestore=typestore) as reader:
        conns = list(reader.connections)

        # --- depth image topic (data plane -> depth_data) ---
        depth_cands = topics_matching(conns, DEPTH_SUFFIX, camera, DEPTH_TOPIC)
        if not depth_cands:
            missing_data.append(f"{METADATA_CAMERA_LABEL} depth: expected but not found (*{DEPTH_SUFFIX})")
            return _abort("missing_depth_topic")
        if len(depth_cands) > 1:
            extra_data.append(f"{METADATA_CAMERA_LABEL} depth: expected 1, found {len(depth_cands)} ({depth_cands})")
        depth_topic = depth_cands[0]

        # --- depth camera_info (info plane -> depth_info) ---
        dinfo_cands = topics_matching(conns, DEPTH_INFO_SUFFIX, camera, DEPTH_INFO_TOPIC)
        if not dinfo_cands:
            missing_info.append(f"{METADATA_CAMERA_LABEL} depth camera_info: expected but not found (*{DEPTH_INFO_SUFFIX})")
            return _abort("missing_depth_info")
        if len(dinfo_cands) > 1:
            extra_info.append(f"{METADATA_CAMERA_LABEL} depth camera_info: expected 1, found {len(dinfo_cands)} ({dinfo_cands})")
        depth_info = dinfo_cands[0]

        # --- depth->color extrinsics (info plane -> depth_info) ---
        has_override = ROTATION is not None and TRANSLATION is not None
        extr_cands = topics_matching(conns, EXTRINSICS_SUFFIX, camera, EXTRINSICS_TOPIC)
        if not extr_cands and not has_override:
            missing_info.append(f"{METADATA_CAMERA_LABEL} depth->color extrinsics: expected but not found (*{EXTRINSICS_SUFFIX})")
            return _abort("missing_depth_extrinsics")
        if len(extr_cands) > 1:
            extra_info.append(f"{METADATA_CAMERA_LABEL} depth->color extrinsics: expected 1, found {len(extr_cands)} ({extr_cands})")
        extr_topic = extr_cands[0] if extr_cands else (EXTRINSICS_TOPIC or EXTRINSICS_SUFFIX)

        # colour side is OWNED by colour extraction (missing colour -> colour_data/_info,
        # completed=false per the colour slide); depth just consumes it via pick_topic.
        color_topic = pick_topic(conns, COLOR_SUFFIX, camera, COLOR_TOPIC, "color")
        color_info = pick_topic(conns, COLOR_INFO_SUFFIX, camera, COLOR_INFO_TOPIC, "color-info")

        print(f"[topics] depth  = {depth_topic}")
        print(f"[topics] color  = {color_topic}")
        print(f"[topics] d.info = {depth_info}")
        print(f"[topics] c.info = {color_info}")
        print(f"[topics] extrin = {extr_topic}")

        depth_intr = read_camera_info(reader, depth_info, DEPTH_MODEL)
        color_intr = read_camera_info(reader, color_info, COLOR_MODEL)
        rot, trans = read_extrinsics(reader, extr_topic, ROTATION, TRANSLATION)
        depth_info_block = read_camera_info_block(reader, depth_info)  # for metadata.json

        # --- fidelity guards ---------------------------------------------------
        print(f"[calib] depth  {depth_intr.as_dict()}")
        print(f"[calib] color  {color_intr.as_dict()}")
        print(f"[calib] R(col-major)={np.round(rot, 6).tolist()}")
        print(f"[calib] t(m)        ={np.round(trans, 6).tolist()}")
        # Capture each guard's outcome (True = passed) so it lands in the report,
        # not just stdout. Thresholds are unchanged from the prints below.
        depth_distortion_ok = not depth_intr.has_distortion
        color_distortion_ok = not color_intr.has_distortion
        if not depth_distortion_ok:
            print(f"[WARN] depth intrinsics have NON-ZERO distortion coeffs "
                  f"{depth_intr.coeffs[:5]} (model {_MODEL_NAMES[depth_intr.model]}); "
                  "verify the model tag is correct or this will stray from live align.")
        if not color_distortion_ok:
            print(f"[WARN] color intrinsics have NON-ZERO distortion coeffs "
                  f"{color_intr.coeffs[:5]} (model {_MODEL_NAMES[color_intr.model]}); "
                  "ROS camera_info cannot express forward-vs-inverse Brown-Conrady, "
                  "so force the correct rs2 model via --color-model if needed.")
        Rm = np.asarray(rot, np.float64).reshape(3, 3)
        rotation_ok = bool(np.allclose(Rm, np.eye(3), atol=0.05))
        if not rotation_ok:
            print(f"[WARN] rotation is far from identity; expected ~I for a D435i. "
                  "Check column-major order if alignment looks wrong.")
        translation_norm_m = float(np.linalg.norm(trans))
        translation_ok = bool(abs(translation_norm_m - 0.015) <= 0.02)
        if not translation_ok:
            print(f"[WARN] |translation|={translation_norm_m:.4f} m; expected "
                  "~0.015 m baseline for D435i depth->color.")

        aligner = DepthToColorAligner(
            depth_intr, color_intr, rot, trans,
            z_scale=Z_SCALE, hole_fill=HOLE_FILL)

        # --- pass 1: color stamps (define output length & pairing base) -------
        n_depth_msgs = sum(1 for c in conns if c.topic == depth_topic
                           for _ in reader.messages(connections=[c]))
        color_stamps = collect_stamps(reader, color_topic, STRIDE)
        n_color = color_stamps.size
        if LIMIT and LIMIT < n_color:
            color_stamps = color_stamps[:LIMIT]
            n_color = LIMIT
        if n_color == 0:
            raise SystemExit("No color frames to index against — depth aligns onto "
                             "the color stream, so an empty color topic leaves nothing "
                             "to align to.")
        # Monotonicity: searchsorted needs ascending; keep an order map if not.
        order = np.argsort(color_stamps, kind="stable")
        sorted_stamps = color_stamps[order]
        if not np.array_equal(order, np.arange(n_color)):
            print("[WARN] color stamps not monotonic; using a sorted index map.")
        Hc, Wc = color_intr.height, color_intr.width
        print(f"[info] depth msgs in bag={n_depth_msgs}  color frames (align target)={n_color}  "
              f"output={n_color}x{Hc}x{Wc}  hole_fill={HOLE_FILL}")

        # --- create HDF5 output (default fill 0 = blank/no depth) -------------
        stem = f"{camera or 'cam'}_aligned_depth_to_color"
        h5_path = out_root / "depth_frames" / f"{stem}.h5"
        csv_path = out_root / "timestamps" / f"{stem}.csv"

        depth_stamp_for = np.full(n_color, np.nan, np.float64)
        pair_dt_ms = np.full(n_color, np.nan, np.float64)
        best_dt = np.full(n_color, np.inf, np.float64)
        has_depth = np.zeros(n_color, bool)
        n_depth_no_partner = 0
        tol_s = PAIR_TOLERANCE_MS / 1000.0

        t_start = time.time()
        with h5py.File(h5_path, "w") as h5:
            dset = h5.create_dataset(
                "data", shape=(n_color, Hc, Wc), dtype=np.uint16,
                chunks=(1, Hc, Wc), compression="gzip", compression_opts=4)

            # --- pass 2: stream depth frames, pair, align, write --------------
            dconns = [c for c in reader.connections if c.topic == depth_topic]
            processed = 0
            for i, (conn, _t, raw) in enumerate(reader.messages(connections=dconns)):
                if STRIDE > 1 and (i % STRIDE):
                    continue
                msg = reader.deserialize(raw, conn.msgtype)
                ds = _stamp_s(msg.header)

                # nearest color frame by header stamp
                j = int(np.searchsorted(sorted_stamps, ds))
                best_k, best = -1, np.inf
                for k in (j - 1, j):
                    if 0 <= k < n_color:
                        dt = abs(sorted_stamps[k] - ds)
                        if dt < best:
                            best, best_k = dt, k
                if best > tol_s:
                    n_depth_no_partner += 1
                    continue
                cidx = int(order[best_k])              # map back to color frame index
                if best >= best_dt[cidx]:
                    continue                            # a closer depth already claimed it

                depth_img = message_to_cvimage(msg)
                if depth_img.dtype != np.uint16:
                    depth_img = np.clip(depth_img, 0, 65535).astype(np.uint16)
                aligned = aligner.align(depth_img)
                dset[cidx] = aligned
                best_dt[cidx] = best
                depth_stamp_for[cidx] = ds
                pair_dt_ms[cidx] = best * 1000.0
                has_depth[cidx] = True
                processed += 1
                if processed % 200 == 0:
                    print(f"  aligned {processed} frames "
                          f"({processed / max(time.time() - t_start, 1e-6):.1f} fps)")

            # --- provenance + calibration attrs (self-describing file) --------
            a = dset.attrs
            a["align_direction"] = "depth_to_color"
            a["depth_unit"] = "mm"
            a["z_scale"] = float(Z_SCALE)
            a["hole_fill"] = "corner_rect" if HOLE_FILL else "center_only"
            a["pair_tolerance_ms"] = float(PAIR_TOLERANCE_MS)
            a["indexed_to"] = "color"
            a["color_topic"] = color_topic
            a["depth_topic"] = depth_topic
            a["source_bag"] = str(bag)
            a["depth_intrinsics"] = json.dumps(depth_intr.as_dict())
            a["color_intrinsics"] = json.dumps(color_intr.as_dict())
            a["depth_to_color_rotation_colmajor"] = np.asarray(rot, np.float64)
            a["depth_to_color_translation_m"] = np.asarray(trans, np.float64)
            a["librealsense_ref"] = (
                "IntelRealSense/librealsense Apache-2.0; "
                "src/proc/align.cpp (align_images/align_z_to_other) @ master; "
                "include/librealsense2/rsutil.h (deproject/transform/project) @ v2.44.0")
            a["note"] = ("aligned[i] corresponds 1:1 to color frame i (same time "
                         "order as color_topic). 0 = no depth. Re-pair color by index.")

        # --- integrity CSV ----------------------------------------------------
        with open(csv_path, "w") as f:
            f.write("index,color_stamp_s,depth_stamp_s,pair_dt_ms,has_depth\n")
            for i in range(n_color):
                f.write(f"{i},{color_stamps[i]:.9f},"
                        f"{'' if np.isnan(depth_stamp_for[i]) else f'{depth_stamp_for[i]:.9f}'},"
                        f"{'' if np.isnan(pair_dt_ms[i]) else f'{pair_dt_ms[i]:.3f}'},"
                        f"{int(has_depth[i])}\n")

        # --- summary ----------------------------------------------------------
        n_paired = int(has_depth.sum())
        n_blank = n_color - n_paired
        maxdt = float(np.nanmax(pair_dt_ms)) if n_paired else 0.0
        print("\n===== SUMMARY =====")
        print(f"  output frames (color-indexed): {n_color}")
        print(f"  depth msgs in bag    : {n_depth_msgs}")
        print(f"  paired (has depth)   : {n_paired}")
        print(f"  blank color frames   : {n_blank}  (depth-side drops -> zero frame)")
        print(f"  depth w/o color pair : {n_depth_no_partner}  (color-side drops)")
        print(f"  max pair dt          : {maxdt:.2f} ms  (tolerance {PAIR_TOLERANCE_MS} ms)")
        if aligner.rect_clamped:
            print(f"  [WARN] fill-rect clamped on {aligner.rect_clamped} pixel-writes "
                  f"(RECT_CAP={RECT_CAP}); check calibration if large.")
        if n_depth_msgs == n_color and n_blank:
            print("  [note] depth/color counts match but some frames did not pair -> "
                  "timestamp skew, not a true drop. Try a larger PAIR_TOLERANCE_MS.")
        _report_gaps(has_depth, pair_dt_ms)
        print(f"  wrote {h5_path}")
        print(f"  wrote {csv_path}")

        # --- optional overlay -------------------------------------------------
        if OVERLAY_PNG:
            _write_overlays(reader, color_topic, h5_path, out_root / "overlays",
                            OVERLAY_PNG, STRIDE, n_color, order, color_stamps)

        summary = {
            "bag": str(bag), "camera": camera,
            "h5": str(h5_path), "csv": str(csv_path),
            "n_color": int(n_color), "n_depth_msgs": int(n_depth_msgs),
            "n_paired": int(n_paired), "n_blank_color": int(n_blank),
            "n_depth_no_partner": int(n_depth_no_partner),
            # span of the color-indexed timeline the aligned frames sit on
            "ts_min": float(color_stamps.min()), "ts_max": float(color_stamps.max()),
            "max_pair_dt_ms": round(float(maxdt), 3), "hole_fill": bool(HOLE_FILL),
            # calibration/geometry guards (True = passed; see header). calib_ok is
            # the single-glance gate: filter the report on calib_ok == False.
            "depth_distortion_ok": bool(depth_distortion_ok),
            "color_distortion_ok": bool(color_distortion_ok),
            "rotation_ok": bool(rotation_ok),
            "translation_norm_m": round(translation_norm_m, 5),
            "translation_ok": bool(translation_ok),
            "rect_clamped": int(aligner.rect_clamped),
            "calib_ok": bool(depth_distortion_ok and color_distortion_ok
                             and rotation_ok and translation_ok
                             and aligner.rect_clamped == 0),
        }

        # --- append this aligned-depth stream into the shared metadata.json ----
        # Own function (append_aligned_depth_to_metadata); a hiccup here warns but
        # never fails an otherwise-good alignment run.
        try:
            depth_stamps_paired = depth_stamp_for[has_depth]
            append_aligned_depth_to_metadata(
                out_root, summary, depth_info_block, depth_stamps_paired,
                color_intr, depth_topic,
                depth_to_color_rot=rot, depth_to_color_trans=trans,
                extrinsics_topic=extr_topic)
        except Exception as e:  # noqa: BLE001
            print(f"[metadata] WARN could not update metadata.json: {e}")

        return summary


def _report_gaps(has_depth: np.ndarray, pair_dt_ms: np.ndarray) -> None:
    runs = []
    i, n = 0, has_depth.size
    while i < n:
        if not has_depth[i]:
            j = i
            while j < n and not has_depth[j]:
                j += 1
            runs.append((i, j - 1))
            i = j
        else:
            i += 1
    if runs:
        print(f"  blank runs ({len(runs)}):")
        for s, e in runs[:20]:
            print(f"    color {s}..{e} ({e - s + 1} frame(s)) unpaired")
        if len(runs) > 20:
            print(f"    ... and {len(runs) - 20} more")


def _write_overlays(reader, color_topic, h5_path, out_dir, count, stride,
                    n_color, order, color_stamps) -> None:
    try:
        import cv2
    except Exception as e:  # noqa: BLE001
        print(f"[overlay] cv2 unavailable ({e}); skipping.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = set(np.linspace(0, n_color - 1, count, dtype=int).tolist())
    with h5py.File(h5_path, "r") as h5:
        dset = h5["data"]
        conns = [c for c in reader.connections if c.topic == color_topic]
        ci = -1
        for i, (conn, _t, raw) in enumerate(reader.messages(connections=conns)):
            if stride > 1 and (i % stride):
                continue
            ci += 1
            if ci not in targets or ci >= n_color:
                continue
            color = message_to_cvimage(reader.deserialize(raw, conn.msgtype))
            if color.ndim == 3 and color.shape[2] == 3:
                color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            depth = dset[ci]
            valid = depth > 0
            dn = np.zeros_like(depth, np.uint8)
            if valid.any():
                dv = depth[valid].astype(np.float32)
                lo, hi = np.percentile(dv, 2), np.percentile(dv, 98)
                dn[valid] = np.clip((depth[valid] - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
            cmap = cv2.applyColorMap(dn, cv2.COLORMAP_JET)
            cmap[~valid] = 0
            over = cv2.addWeighted(color, 0.6, cmap, 0.4, 0)
            cv2.imwrite(str(out_dir / f"overlay_{ci:06d}.png"), over)
    print(f"[overlay] wrote {len(targets)} overlays to {out_dir}")


def _cli() -> None:
    """Thin shell shim for one-off runs. The wrapper does NOT use this — it sets
    the CONFIG globals and calls main() directly. This just maps CLI flags onto
    those same globals so `python rosbag_process_depth_v3.py --bag ...` still works
    (handy for the --overlay-png QA mode)."""
    ap = argparse.ArgumentParser(description="Offline depth->color alignment (one bag).",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", required=True, help="path to the .mcap/.db3 bag (file or dir)")
    ap.add_argument("--out-dir", default=None, help="output root (default: bag's parent)")
    ap.add_argument("--camera", default=CAMERA, help="substring to disambiguate topics")
    ap.add_argument("--pair-tolerance-ms", type=float, default=PAIR_TOLERANCE_MS)
    ap.add_argument("--no-hole-fill", action="store_true", help="sparse centers, no rect fill")
    ap.add_argument("--z-scale", type=float, default=Z_SCALE)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--limit", type=int, default=LIMIT)
    ap.add_argument("--overlay-png", type=int, default=OVERLAY_PNG,
                    help="write N color+depth overlay PNGs (needs cv2)")
    ap.add_argument("--rotation", type=float, nargs=9, default=None)
    ap.add_argument("--translation", type=float, nargs=3, default=None)
    ap.add_argument("--color-model", type=int, default=None)
    ap.add_argument("--depth-model", type=int, default=None)
    a = ap.parse_args()

    g = globals()
    g["PAIR_TOLERANCE_MS"] = a.pair_tolerance_ms
    g["HOLE_FILL"] = not a.no_hole_fill
    g["Z_SCALE"] = a.z_scale
    g["STRIDE"] = a.stride
    g["LIMIT"] = a.limit
    g["OVERLAY_PNG"] = a.overlay_png
    g["ROTATION"] = a.rotation
    g["TRANSLATION"] = a.translation
    g["COLOR_MODEL"] = a.color_model
    g["DEPTH_MODEL"] = a.depth_model
    main(bag=a.bag, out_dir=a.out_dir, camera=a.camera)


if __name__ == "__main__":
    try:
        main()   # runs from the CONFIG constants above (BAG_PATH / OUT_DIR / CAMERA)
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(130)
