"""Unit tests for pipeline_calibration.

Run:  python -m pytest data-processing-scripts-v3/test_pipeline_calibration.py -q

Architecture mirrors the module's split:
  A. merge_calibration  -> PURE (dict in, dict out). Every gate decision
     (ok/broken/missing, null rules, drops, idempotency, no-mutation) is proven here
     with plain dicts — no files, no rosbags.
  B. annotate_calibration -> the thin FILE shell, tested with tmp_path: it only reads/
     writes and delegates logic to (A), so a couple of I/O cases suffice.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pipeline_calibration as pc  # noqa: E402


# --------------------------------------------------------------------------- #
# synthetic fixtures                                                          #
# --------------------------------------------------------------------------- #
def make_cam(status="ok", fx=963.9, x=2.2):
    """One calib camera block (K nested 3x3, dist, image_size + a 4x4 pose)."""
    return {
        "intrinsics": {"K": [[fx, 0.0, 622.0], [0.0, fx, 356.0], [0.0, 0.0, 1.0]],
                       "dist": [0.026, -0.044, 0.0, 0.0, 0.0],
                       "image_size": [1280, 720]},
        "status": status,
        "extrinsics": {"T_world_cam": [[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, 2.7],
                                       [0.0, 0.0, 1.0, 1.95], [0.0, 0.0, 0.0, 1.0]],
                       "position_m": [x, 2.7, 1.95]},
    }


def make_calib(cameras):
    """A calib dict with a board + accuracy (both must be dropped) and given cameras."""
    return {
        "scene_details": {"board": {"squares_x": 7, "dictionary": "DICT_5X5_100"}},
        "cameras": cameras,
        "accuracy": {"relative": {"worst_m": 0.02}, "cameras_solved": "4/4"},
    }


def base_meta():
    """A minimal post-extraction metadata.json: cam_ego intrinsics + the realsense
    extrinsic legs + a termination block, as color/depth/imu would have left it."""
    return {
        "metadata": {"dataset_name": "leo"},
        "camera_intrinsics": [{"camera": "cam_ego", "color": {"K": [900.0]},
                               "depth": {"K": [420.0]}}],
        "camera_extrinsics": [{"name": "depth_to_color", "source_topic": "/e/c",
                               "rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                               "translation": [0.015, 0, 0]}],
        "steps": {"streams": []},
        "termination": {"is_successful": True, "reason": []},
    }


def meta_with_exo(*cameras):
    """base_meta() plus recorded exo COLOR streams for the given cam names, as color
    extraction would have left steps.streams. Used to exercise the coverage gate, which keys
    off recorded streams (a cam only matters if it was actually recorded)."""
    m = base_meta()
    m["steps"]["streams"] = [{"kind": "color", "camera": c} for c in cameras]
    return m


def _intr(meta, camera):
    return next((c for c in meta["camera_intrinsics"] if c["camera"] == camera), None)


def _extr(meta, name):
    return next((e for e in meta["camera_extrinsics"] if e["name"] == name), None)


# --------------------------------------------------------------------------- #
# A. merge_calibration — PURE gate logic                                      #
# --------------------------------------------------------------------------- #
def test_ok_camera_writes_full_intrinsics_and_extrinsics():
    out = pc.merge_calibration(base_meta(), make_calib({"exo_cam1": make_cam("ok")}))

    intr = _intr(out, "exo_cam1")["color"]
    assert intr["status"] == "ok"
    assert intr["source"] == "charuco_calib"
    assert intr["width"] == 1280 and intr["height"] == 720
    assert intr["K"] == [963.9, 0.0, 622.0, 0.0, 963.9, 356.0, 0.0, 0.0, 1.0]  # flat-9
    assert intr["D"] == [0.026, -0.044, 0.0, 0.0, 0.0]

    ext = _extr(out, "exo_cam1_to_world")
    assert ext["status"] == "ok"
    assert ext["T_world_cam"][0][3] == 2.2 and ext["position_m"] == [2.2, 2.7, 1.95]
    assert ext["T_world_cam"] == make_cam("ok")["extrinsics"]["T_world_cam"]  # whole 4x4, verbatim
    assert ext["convention"] == pc.WORLD_POSE_CONVENTION


def test_broken_camera_writes_null_values_and_note():
    out = pc.merge_calibration(base_meta(),
                               make_calib({"exo_cam2": make_cam("low_coverage")}))

    intr = _intr(out, "exo_cam2")["color"]
    assert intr["status"] == "low_coverage"
    assert intr["K"] is None and intr["D"] is None          # solve values nulled
    assert intr["width"] == 1280 and intr["height"] == 720  # capture facts kept

    ext = _extr(out, "exo_cam2_to_world")
    assert ext["T_world_cam"] is None and ext["position_m"] is None

    # recorded as an informational note...
    assert out["steps"]["exo_calib_notes"] == \
        ["exo_cam2: status='low_coverage'; K/D and pose set null"]
    # ...but broken calibration does NOT fail termination (it limits use, not data validity)
    assert out["termination"]["reason"] == []
    assert out["termination"]["is_successful"] is True


def test_uncalibrated_camera_that_was_not_recorded_is_omitted():
    # calib has only exo_cam1; exo_cam3 is NOT in the calib AND was NOT recorded (streams=[]).
    # Coverage keys off RECORDED streams, so an unrecorded cam is simply never visited -> no
    # block, no error. (Contrast test_recorded_exo_cam_absent_from_calib_raises below.)
    out = pc.merge_calibration(base_meta(), make_calib({"exo_cam1": make_cam("ok")}))
    assert _intr(out, "exo_cam3") is None
    assert _extr(out, "exo_cam3_to_world") is None
    assert "exo_calib_notes" not in out["steps"]              # not recorded != broken (no note)
    assert out["termination"]["is_successful"] is True        # nothing recorded to cover -> silent


def test_recorded_exo_cam_absent_from_calib_raises():
    # exo_cam1 and exo_cam4 recorded, but calib only has exo_cam1 -> exo_cam4 is an
    # un-calibrated recorded cam -> hard episode failure (the 'missing' gate made loud).
    import pytest
    meta = meta_with_exo("exo_cam1", "exo_cam4")
    calib = make_calib({"exo_cam1": make_cam("ok")})
    with pytest.raises(pc.CalibrationCoverageError, match="exo_cam4"):
        pc.merge_calibration(meta, calib)


def test_recorded_exo_cam_present_but_broken_does_not_raise():
    # exo_cam4 recorded AND in calib but with a failed solve (status != "ok"): coverage is
    # satisfied (it has an entry), so NO raise — it is the ONE tolerated shortfall (null + note).
    meta = meta_with_exo("exo_cam4")
    out = pc.merge_calibration(meta, make_calib({"exo_cam4": make_cam("low_coverage")}))
    assert _intr(out, "exo_cam4")["color"]["K"] is None       # broken -> null, but no raise
    assert out["steps"]["exo_calib_notes"] == \
        ["exo_cam4: status='low_coverage'; K/D and pose set null"]
    assert out["termination"]["is_successful"] is True


def test_recorded_exo_cam_present_and_ok_does_not_raise():
    # the happy coverage case: every recorded exo cam has an ok solve -> full merge, no raise.
    meta = meta_with_exo("exo_cam1")
    out = pc.merge_calibration(meta, make_calib({"exo_cam1": make_cam("ok")}))
    assert _intr(out, "exo_cam1")["color"]["K"] is not None


def test_board_and_accuracy_are_dropped():
    out = pc.merge_calibration(base_meta(), make_calib({"exo_cam1": make_cam("ok")}))
    blob = json.dumps(out)
    assert "board" not in out and "scene_details" not in out
    assert "accuracy" not in out and "cameras_solved" not in blob


def test_existing_ego_blocks_untouched():
    out = pc.merge_calibration(base_meta(), make_calib({"exo_cam1": make_cam("ok")}))
    # cam_ego intrinsics (incl. its depth sub-block) and the realsense leg survive as-is
    assert _intr(out, "cam_ego") == base_meta()["camera_intrinsics"][0]
    assert _extr(out, "depth_to_color") == base_meta()["camera_extrinsics"][0]


def test_pure_does_not_mutate_inputs():
    meta, calib = base_meta(), make_calib({"exo_cam1": make_cam("ok")})
    meta_snapshot, calib_snapshot = copy.deepcopy(meta), copy.deepcopy(calib)
    pc.merge_calibration(meta, calib)
    assert meta == meta_snapshot and calib == calib_snapshot   # inputs unchanged


def test_rerun_is_idempotent_no_duplicates():
    calib = make_calib({"exo_cam1": make_cam("ok"), "exo_cam2": make_cam("bad")})
    once = pc.merge_calibration(base_meta(), calib)
    twice = pc.merge_calibration(once, calib)                  # merge on top of merged
    assert [c["camera"] for c in twice["camera_intrinsics"]] == \
        ["cam_ego", "exo_cam1", "exo_cam2"]                    # no dupes
    assert [e["name"] for e in twice["camera_extrinsics"]] == \
        ["depth_to_color", "exo_cam1_to_world", "exo_cam2_to_world"]
    assert twice["steps"]["exo_calib_notes"] == \
        ["exo_cam2: status='bad'; K/D and pose set null"]          # note not duplicated


def test_mixed_scenario_ok_broken_missing_together():
    # exo_cam1 ok, exo_cam2 broken, exo_cam4 missing (never in calib)
    calib = make_calib({"exo_cam1": make_cam("ok"),
                        "exo_cam2": make_cam("low_coverage")})
    out = pc.merge_calibration(base_meta(), calib)
    assert _intr(out, "exo_cam1")["color"]["K"] is not None    # ok -> values
    assert _intr(out, "exo_cam2")["color"]["K"] is None        # broken -> null
    assert _intr(out, "exo_cam4") is None                      # missing -> omitted
    assert out["termination"]["is_successful"] is True         # calibration never fails termination


# --------------------------------------------------------------------------- #
# B. annotate_calibration — the FILE shell                                    #
# --------------------------------------------------------------------------- #
def _write(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    return path


def test_annotate_writes_merged_metadata(tmp_path):
    out = tmp_path / "ep"
    _write(out / "metadata.json", base_meta())
    calib = _write(tmp_path / "calib-2008.json", make_calib({"exo_cam1": make_cam("ok")}))

    res = pc.annotate_calibration(out, calib)
    assert res["written"] is True and res["cameras"] == ["exo_cam1"]

    saved = json.loads((out / "metadata.json").read_text())
    assert _intr(saved, "exo_cam1")["color"]["K"][0] == 963.9
    assert _extr(saved, "exo_cam1_to_world")["position_m"] == [2.2, 2.7, 1.95]


def test_annotate_noop_when_metadata_absent(tmp_path):
    calib = _write(tmp_path / "calib.json", make_calib({"exo_cam1": make_cam("ok")}))
    res = pc.annotate_calibration(tmp_path / "nope", calib)      # no metadata.json
    assert res["written"] is False and res["reason"] == "no metadata.json"


def test_annotate_noop_when_calib_path_none(tmp_path):
    # a session that declares NO calibration (calib_path=None) -> clean no-op, file untouched.
    # This pins the contract that lets the wrapper pass None safely (the Path(None) bug class).
    out = tmp_path / "ep"
    _write(out / "metadata.json", base_meta())
    res = pc.annotate_calibration(out, None)
    assert res == {"written": False, "reason": "no calib path"}
    assert json.loads((out / "metadata.json").read_text()) == base_meta()   # untouched


def test_annotate_noop_when_calib_absent(tmp_path):
    out = tmp_path / "ep"
    _write(out / "metadata.json", base_meta())
    res = pc.annotate_calibration(out, tmp_path / "missing-calib.json")
    assert res["written"] is False and res["reason"] == "no calib file"
    # metadata.json is left exactly as it was
    assert json.loads((out / "metadata.json").read_text()) == base_meta()


def test_annotate_rejects_trailing_content(tmp_path):
    # the calib-2008 format is a single clean JSON object. Anything appended (a legend,
    # a second object, garbage) must fail LOUDLY, not be silently half-read.
    import pytest
    out = tmp_path / "ep"
    _write(out / "metadata.json", base_meta())
    calib = tmp_path / "calib.json"
    calib.write_text(json.dumps(make_calib({"exo_cam1": make_cam("ok")}), indent=2)
                     + "\n\n{ trailing garbage }\n")
    with pytest.raises(json.JSONDecodeError):
        pc.annotate_calibration(out, calib)


def test_annotate_raises_coverage_error_through_the_shell(tmp_path):
    # the file shell propagates CalibrationCoverageError from the pure merge: a recorded exo
    # cam with no calib entry -> the wrapper's calib step raises -> that bag crashes (loud).
    import pytest
    out = tmp_path / "ep"
    _write(out / "metadata.json", meta_with_exo("exo_cam1", "exo_cam4"))
    calib = _write(tmp_path / "calib.json", make_calib({"exo_cam1": make_cam("ok")}))
    with pytest.raises(pc.CalibrationCoverageError, match="exo_cam4"):
        pc.annotate_calibration(out, calib)


def test_annotate_is_idempotent_on_rerun(tmp_path):
    out = tmp_path / "ep"
    _write(out / "metadata.json", base_meta())
    calib = _write(tmp_path / "calib.json", make_calib({"exo_cam1": make_cam("ok")}))
    pc.annotate_calibration(out, calib)
    first = json.loads((out / "metadata.json").read_text())
    pc.annotate_calibration(out, calib)                         # run again
    second = json.loads((out / "metadata.json").read_text())
    assert first == second                                      # no drift, no duplicate blocks
