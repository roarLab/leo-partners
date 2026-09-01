"""Synthetic-capture factory shared by all layers.

The render/look_at/write_videos machinery draws the real printed board through a
real plane homography, so the genuine detector and solver run on it -- only image
noise and lens distortion are missing. L1 tests
render a single frame at a known pose; the L2 test builds a whole two-capture
session (intrinsics + walk) and runs the pipeline over it.
"""

import csv
import json
import os

import cv2
import numpy as np
import pytest

from board import DEFAULT, FLAT_ON_FLOOR, make_board, save_spec

W, H = 1920, 1080
PX_PER_M = 400
K_TRUE = np.array([[1350.0, 0, 960.0], [0, 1350.0, 540.0], [0, 0, 1]])
BED_CORNER = np.array([-1.0275, 0.15, 0.0])
BLANK = np.full((H, W, 3), 60, np.uint8)


def look_at(eye, target, up=np.array([0.0, 0.0, 1.0])):
    """T_world_cam for a camera at `eye` looking at `target` (OpenCV axes)."""
    f = target - eye
    f = f / np.linalg.norm(f)              # camera +z
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)              # camera +x
    d = np.cross(f, r)                     # camera +y, down-ish
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = r, d, f, eye
    return T


# The three ground-truth cameras every synthetic capture uses.
CAMS = {
    "exo_cam1": look_at(np.array([-1.8, -1.6, 2.3]), np.array([0.0, 0.6, 0.4])),
    "exo_cam2": look_at(np.array([2.2, -1.4, 2.1]), np.array([0.0, 0.6, 0.4])),
    "exo_cam3": look_at(np.array([1.9, 2.6, 2.4]), np.array([0.0, 0.6, 0.4])),
}


class Rig:
    """A board + renderer bound to one board.json, so tests can draw frames."""

    def __init__(self, root):
        spec = dict(DEFAULT)
        spec["nominal_square_mm"] = spec["square_mm"]
        spec["nominal_marker_mm"] = spec["marker_mm"]
        spec["print_scale"] = 1.0
        bdir = os.path.join(root, "board")
        os.makedirs(bdir, exist_ok=True)
        self.spec = spec
        self.board_json = os.path.join(bdir, "board.json")
        save_spec(self.board_json, spec)
        self.board, _ = make_board(spec)
        sq = spec["square_mm"] / 1000.0
        self.bw_m, self.bh_m = spec["squares_x"] * sq, spec["squares_y"] * sq
        bimg = self.board.generateImage(
            (int(self.bw_m * PX_PER_M), int(self.bh_m * PX_PER_M)), marginSize=0)
        self.bimg = (cv2.cvtColor(bimg, cv2.COLOR_GRAY2BGR)
                     if bimg.ndim == 2 else bimg)
        self.root = root

    def render(self, T_cam_board):
        """The board seen from a camera, via the plane homography, or None."""
        R, t = T_cam_board[:3, :3], T_cam_board[:3, 3]
        M = K_TRUE @ np.column_stack([R[:, 0], R[:, 1], t])
        Hm = M @ np.diag([1.0 / PX_PER_M, 1.0 / PX_PER_M, 1.0])
        if abs(Hm[2, 2]) < 1e-12:
            return None
        frame = np.full((H, W, 3), 60, np.uint8)
        return cv2.warpPerspective(self.bimg, Hm, (W, H), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_TRANSPARENT, dst=frame)

    def write_videos(self, name, frames_per_cam, times=None):
        """One mp4 per camera under <name>/videos/; if `times` given, also write
        matching <name>/timestamps/<cam>.csv (the layout exo_extrinsics expects)."""
        ep = os.path.join(self.root, name)
        os.makedirs(os.path.join(ep, "videos"), exist_ok=True)
        if times is not None:
            os.makedirs(os.path.join(ep, "timestamps"), exist_ok=True)
        for cam, frames in frames_per_cam.items():
            vw = cv2.VideoWriter(os.path.join(ep, "videos", f"{cam}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
            for f in frames:
                vw.write(f)
            vw.release()
            if times is not None:
                with open(os.path.join(ep, "timestamps", f"{cam}.csv"),
                          "w", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["index", "ros_time_s"])
                    for i, t in enumerate(times):
                        w.writerow([i, f"{t:.6f}"])
        return ep


def render_intrinsics(rig, rng, counts):
    """Per-camera close-up frames: the board thrown around each camera's view at
    random depth/tilt. `counts` maps camera name -> number of attempts; a camera
    given too few (< ~12 usable) ends up NOT calibrated == failed_intrinsics, which
    is how the starved-camera fixture forces that branch."""
    Kinv = np.linalg.inv(K_TRUE)
    centre = np.array([rig.bw_m / 2, rig.bh_m / 2, 0.0])
    intr = {c: [] for c in CAMS}
    for cam in CAMS:
        for _ in range(counts[cam]):
            u = rng.uniform(0.10, 0.90) * W
            v = rng.uniform(0.10, 0.90) * H
            z = rng.uniform(1.6, 3.4)
            C = z * (Kinv @ np.array([u, v, 1.0]))
            tilt, _ = cv2.Rodrigues(rng.uniform(-0.55, 0.55, 3))
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = tilt, C - tilt @ centre
            if T[2, 3] < 0.5:
                continue
            f = rig.render(T)
            if f is not None:
                intr[cam].append(f)
    return intr


@pytest.fixture
def rig(tmp_path):
    """A board + renderer under a fresh tmp dir (used by L1 and L2)."""
    return Rig(str(tmp_path))


@pytest.fixture
def one_frame(rig):
    """A single rendered frame with the board filling the view head-on, plus the
    T_cam_board it was rendered at -- the minimal input for L1 detect/pose tests."""
    centre = np.array([rig.bw_m / 2, rig.bh_m / 2, 0.0])
    T = np.eye(4)
    T[:3, 3] = np.array([0.0, 0.0, 2.2]) - centre    # board centred, 2.2 m out
    return rig.render(T), T, rig


@pytest.fixture
def synth_capture(rig):
    """A full two-capture session under tmp: an intrinsics capture (board filling
    each camera at its corners/tilts) and a walk (board carried across the shared
    floor, co-visible, timestamped), plus the tape-measured camera positions.

    Returns dict(board=, intrinsics=, walk=, camera_positions=, cams=, K_true=).
    Uses fixed parameters (seed 0, 70/60 frames) so calibration is known to
    converge."""
    rng = np.random.default_rng(0)
    intr = render_intrinsics(rig, rng, {c: 70 for c in CAMS})
    intr_ep = rig.write_videos("calib_intrinsics", intr)

    n_walk, t0 = 60, 1_000_000.0
    walk = {c: [] for c in CAMS}
    times = [t0 + i / 30.0 for i in range(n_walk)]
    for _ in range(n_walk):
        origin = BED_CORNER + np.array([rng.uniform(0.2, 1.7),
                                        rng.uniform(-0.5, 1.4), 0.0])
        tilt, _ = cv2.Rodrigues(rng.uniform(-0.30, 0.30, 3))
        T_world_board = np.eye(4)
        T_world_board[:3, :3] = tilt @ FLAT_ON_FLOOR
        T_world_board[:3, 3] = origin
        for cam, T_world_cam in CAMS.items():
            T_cam_board = np.linalg.inv(T_world_cam) @ T_world_board
            f = rig.render(T_cam_board) if T_cam_board[2, 3] > 0.5 else None
            walk[cam].append(f if f is not None else BLANK.copy())
    walk_ep = rig.write_videos("calib_walk", walk, times)

    cpjson = os.path.join(rig.root, "camera_positions.json")
    with open(cpjson, "w") as fh:
        json.dump({c: T[:3, 3].tolist() for c, T in CAMS.items()}, fh)

    return {"board": rig.board_json, "intrinsics": intr_ep, "walk": walk_ep,
            "camera_positions": cpjson, "cams": CAMS, "K_true": K_TRUE}


@pytest.fixture
def starved_intrinsics_capture(rig):
    """An intrinsics-only capture where one camera (exo_cam3) gets far too few
    frames, so its lens never calibrates. Used to prove the real
    exo_intrinsics.main writes it as failed_intrinsics (null lens) while the others
    solve. No walk/positions -- this exercises the intrinsics stage alone."""
    failed_cam = "exo_cam3"
    counts = {c: 70 for c in CAMS}
    counts[failed_cam] = 4          # < 12 usable views -> failed_intrinsics
    rng = np.random.default_rng(0)
    intr = render_intrinsics(rig, rng, counts)
    intr_ep = rig.write_videos("calib_intrinsics", intr)
    return {"board": rig.board_json, "intrinsics": intr_ep,
            "failed_cam": failed_cam,
            "ok_cams": [c for c in CAMS if c != failed_cam]}
