"""L0 -- the two JSON builders. Feed synthetic solved-pieces and assert the
output dict forms correctly, including the branches a real run rarely hits. The
per-camera `status` is a monotonic gate across the two stages -- each test is one
scenario in that state machine:

    lens fails to calibrate          -> status failed_intrinsics (both keys null)
    lens ok but not placed in rig    -> status failed_extrinsics (extrinsics null)
    camera passes both stages        -> status ok (both keys carry data)
    lens failure reaches extrinsics  -> stays failed_intrinsics, not overwritten

No pipeline, no OpenCV -- pure dict assembly."""

import cv2
import numpy as np

from exo_extrinsics import build_extrinsics_output, cameras_to_solve
from exo_intrinsics import build_intrinsics_output, calib_flags


# --------------------------------------------------------------- extrinsics


def _calib(names, failed=()):
    """Synthetic intrinsics-file input. Cameras in `failed` carry a null lens
    solve (status failed_intrinsics), as exo_intrinsics.py would have written."""
    K = [[1350.0, 0, 960.0], [0, 1350.0, 540.0], [0, 0, 1.0]]

    def entry(n):
        if n in failed:
            return {"intrinsics": None, "extrinsics": None,
                    "status": "failed_intrinsics"}
        return {"intrinsics": {"K": K, "dist": [0, 0, 0, 0, 0],
                               "image_size": [1920, 1080]}}

    return {
        "scene_details": {"board": {"squares_x": 7, "square_mm": 100.0}},
        "cameras": {n: entry(n) for n in names},
        "accuracy": {"intrinsics": {"fx_spread_pct": 0.5, "per_camera": {}}},
    }


def _pose(t):
    T = np.eye(4)
    T[:3, 3] = t
    return T


def test_camera_passes_both_stages():
    names = ["exo_cam1", "exo_cam2", "exo_cam3"]
    T_root = {0: _pose([0, 0, 0]), 1: _pose([1, 0, 0]), 2: _pose([0, 1, 0])}
    shared = list(names)
    out = build_extrinsics_output(
        calib=_calib(names), names=names, T_root=T_root, T_world_root=np.eye(4),
        rel_scatter=[0.01, 0.02],
        pair_scatter={"exo_cam1-exo_cam2": {"agree_m": 0.01}},
        residuals={n: 0.03 for n in names}, shared=shared, fit_rms_m=0.03)

    assert out["accuracy"]["cameras_solved"] == "3/3"
    for n in names:
        assert out["cameras"][n]["status"] == "ok"
        assert out["cameras"][n]["extrinsics"] is not None
    # 3 anchors -> no cross-check wording
    assert "no cross-check" in out["accuracy"]["absolute"]["describes"]
    assert out["cameras"]["exo_cam2"]["extrinsics"]["position_m"] == [1.0, 0.0, 0.0]


def test_extrinsics_four_anchors_gets_cross_check():
    names = ["exo_cam1", "exo_cam2", "exo_cam3", "exo_cam4"]
    T_root = {i: _pose([i, 0, 0]) for i in range(4)}
    out = build_extrinsics_output(
        calib=_calib(names), names=names, T_root=T_root, T_world_root=np.eye(4),
        rel_scatter=[0.01], pair_scatter={}, residuals={n: 0.02 for n in names},
        shared=names, fit_rms_m=0.02)
    assert out["accuracy"]["cameras_solved"] == "4/4"
    assert "cross-check" in out["accuracy"]["absolute"]["describes"]
    assert "no cross-check" not in out["accuracy"]["absolute"]["describes"]


def test_lens_ok_but_not_placed_in_rig():
    names = ["exo_cam1", "exo_cam2", "exo_cam3", "exo_cam4"]
    T_root = {0: _pose([0, 0, 0]), 1: _pose([1, 0, 0]), 2: _pose([0, 1, 0])}
    shared = ["exo_cam1", "exo_cam2", "exo_cam3"]
    out = build_extrinsics_output(
        calib=_calib(names), names=names, T_root=T_root, T_world_root=np.eye(4),
        rel_scatter=[0.01], pair_scatter={},
        residuals={n: 0.02 for n in shared}, shared=shared, fit_rms_m=0.02)

    failed = out["cameras"]["exo_cam4"]
    assert failed["status"] == "failed_extrinsics"
    assert failed["extrinsics"] is None
    assert failed["intrinsics"] is not None       # the lens solve survived
    assert "reason" not in failed                 # reason machinery removed
    assert out["accuracy"]["cameras_solved"] == "3/4"
    assert "unsolved_cameras" not in out["accuracy"]


def test_lens_failure_survives_the_extrinsics_stage():
    # exo_cam4's lens never calibrated (intrinsics null). With no lens it is also
    # absent from T_root, so a naive check would mislabel it failed_extrinsics.
    # The null-intrinsics branch must win.
    names = ["exo_cam1", "exo_cam2", "exo_cam3", "exo_cam4"]
    T_root = {0: _pose([0, 0, 0]), 1: _pose([1, 0, 0]), 2: _pose([0, 1, 0])}
    shared = ["exo_cam1", "exo_cam2", "exo_cam3"]
    out = build_extrinsics_output(
        calib=_calib(names, failed=["exo_cam4"]), names=names, T_root=T_root,
        T_world_root=np.eye(4), rel_scatter=[0.01], pair_scatter={},
        residuals={n: 0.02 for n in shared}, shared=shared, fit_rms_m=0.02)

    failed = out["cameras"]["exo_cam4"]
    assert failed["status"] == "failed_intrinsics"
    assert failed["intrinsics"] is None
    assert failed["extrinsics"] is None
    assert out["accuracy"]["cameras_solved"] == "3/4"


def test_extrinsics_empty_scatter_worst_is_none():
    names = ["exo_cam1", "exo_cam2", "exo_cam3"]
    T_root = {i: _pose([i, 0, 0]) for i in range(3)}
    out = build_extrinsics_output(
        calib=_calib(names), names=names, T_root=T_root, T_world_root=np.eye(4),
        rel_scatter=[], pair_scatter={}, residuals={n: 0.02 for n in names},
        shared=names, fit_rms_m=0.02)
    assert out["accuracy"]["relative"]["worst_m"] is None


def test_extrinsics_carries_intrinsics_block_through():
    names = ["exo_cam1", "exo_cam2", "exo_cam3"]
    calib = _calib(names)
    calib["accuracy"]["intrinsics"]["fx_spread_pct"] = 1.7
    T_root = {i: _pose([i, 0, 0]) for i in range(3)}
    out = build_extrinsics_output(
        calib=calib, names=names, T_root=T_root, T_world_root=np.eye(4),
        rel_scatter=[0.01], pair_scatter={}, residuals={n: 0.02 for n in names},
        shared=names, fit_rms_m=0.02)
    assert out["accuracy"]["intrinsics"]["fx_spread_pct"] == 1.7


def test_extrinsics_skips_cameras_whose_lens_failed():
    # the detection stage must skip a null-intrinsics camera (can't solvePnP) and
    # keep the rest -- this is the decision main() delegates here, so covering it
    # at L0 needs no video.
    names = ["exo_cam1", "exo_cam2", "exo_cam3"]
    calib = _calib(names, failed=["exo_cam2"])
    solvable, no_lens = cameras_to_solve(calib["cameras"], names)
    assert solvable == ["exo_cam1", "exo_cam3"]
    assert no_lens == ["exo_cam2"]


# --------------------------------------------------------------- intrinsics


def _entry(fx):
    return {"K": [[fx, 0, 960.0], [0, fx, 540.0], [0, 0, 1.0]],
            "dist": [0.0, 0.0, 0.0, 0.0, 0.0], "image_size": [1920, 1080],
            "rms_px": 0.4, "n_views": 50, "coverage": 0.9, "angle_spread": 0.6}


def test_lens_fails_to_calibrate():
    intr = {"exo_cam1": _entry(1350.0), "exo_cam2": {"failed": True}}
    out = build_intrinsics_output(intr, {"squares_x": 7})

    ok = out["cameras"]["exo_cam1"]
    assert ok["status"] == "ok"
    assert ok["intrinsics"]["K"][0][0] == 1350.0

    bad = out["cameras"]["exo_cam2"]
    assert bad["status"] == "failed_intrinsics"
    assert bad["intrinsics"] is None      # present-but-null, never a missing key
    assert bad["extrinsics"] is None

    # a failed camera has no fx, so it is left out of the quality table
    per = out["accuracy"]["intrinsics"]["per_camera"]
    assert "exo_cam2" not in per
    assert "exo_cam1" in per


def test_intrinsics_single_camera_zero_spread():
    out = build_intrinsics_output({"exo_cam1": _entry(1350.0)}, {"squares_x": 7})
    assert out["accuracy"]["intrinsics"]["fx_spread_pct"] == 0.0
    assert out["cameras"]["exo_cam1"]["extrinsics"] is None
    assert out["cameras"]["exo_cam1"]["intrinsics"]["K"][0][0] == 1350.0


def test_intrinsics_two_cameras_computes_spread():
    intr = {"exo_cam1": _entry(1000.0), "exo_cam2": _entry(1020.0)}
    out = build_intrinsics_output(intr, {"squares_x": 7})
    # (1020 - 1000) / 1000 * 100 = 2.0 %
    assert out["accuracy"]["intrinsics"]["fx_spread_pct"] == 2.0
    per = out["accuracy"]["intrinsics"]["per_camera"]
    assert set(per) == {"exo_cam1", "exo_cam2"}
    assert per["exo_cam2"]["fx"] == 1020.0


def test_calib_flags_pins_k3_by_default_frees_it_on_request():
    # both paths keep square pixels + no tangential; k3 is the only thing --free_k3
    # toggles.
    for f in (calib_flags(False), calib_flags(True)):
        assert f & cv2.CALIB_FIX_ASPECT_RATIO
        assert f & cv2.CALIB_ZERO_TANGENT_DIST
    assert calib_flags(False) & cv2.CALIB_FIX_K3          # default: k3 pinned to 0
    assert not (calib_flags(True) & cv2.CALIB_FIX_K3)     # --free_k3: k3 fitted


def test_intrinsics_schema_shape():
    out = build_intrinsics_output({"exo_cam1": _entry(1350.0)}, {"squares_x": 7})
    assert out["scene_details"] == {"board": {"squares_x": 7}}
    acc = out["accuracy"]["intrinsics"]
    assert {"describes", "fx_spread_pct", "per_camera"} <= set(acc)
    assert {"fx", "coverage", "rms_px", "angle_spread", "n_views"} == set(
        acc["per_camera"]["exo_cam1"])
