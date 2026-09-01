"""Unit tests for pipeline_metadata's shared metadata writers:
add_error (append-merge), upsert_extrinsic (extrinsics by name),
upsert_intrinsic (intrinsics by camera) and reorder_top_level (cosmetic order)."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pipeline_metadata import (add_error, upsert_extrinsic, upsert_intrinsic,
                               reorder_top_level, TOP_LEVEL_ORDER,
                               DEFAULT_EXTRINSIC_CONVENTION)


def test_creates_key_when_absent():
    c = {}
    add_error(c, "missing_stream_error", ["a"])
    assert c == {"missing_stream_error": ["a"]}


def test_appends_to_existing_without_overwriting():
    c = {"reason": ["missing_stream"]}
    add_error(c, "reason", ["extra_stream"])
    assert c["reason"] == ["missing_stream", "extra_stream"]   # both kept, order preserved


def test_skips_duplicates_idempotent():
    c = {"reason": ["depth"]}
    add_error(c, "reason", ["depth", "color"])
    add_error(c, "reason", ["color"])           # re-run
    assert c["reason"] == ["depth", "color"]     # no dupes


def test_returns_the_same_list_object():
    c = {"k": []}
    original = c["k"]
    returned = add_error(c, "k", ["x"])
    assert returned is original                  # appended in place, never reassigned


def test_empty_entries_leaves_key_present_and_unchanged():
    c = {}
    add_error(c, "missing_stream_error", [])
    assert c == {"missing_stream_error": []}      # key created empty, no error


# --- upsert_extrinsic -------------------------------------------------------
R9 = [1, 0, 0, 0, 1, 0, 0, 0, 1]
T3 = [0.015, 0.0, 0.0]


def test_extrinsic_creates_list_and_entry_when_absent():
    m = {}
    upsert_extrinsic(m, "depth_to_color", "/e/depth_to_color", R9, T3)
    assert [e["name"] for e in m["camera_extrinsics"]] == ["depth_to_color"]
    e = m["camera_extrinsics"][0]
    assert e["source_topic"] == "/e/depth_to_color"
    assert e["rotation"] == [1.0, 0, 0, 0, 1, 0, 0, 0, 1]
    assert e["translation"] == [0.015, 0.0, 0.0]
    assert e["convention"] == DEFAULT_EXTRINSIC_CONVENTION


def test_extrinsic_copies_numbers_verbatim_as_floats():
    # numpy inputs (as read_extrinsics returns) -> plain JSON-serializable floats.
    m = {}
    upsert_extrinsic(m, "depth_to_gyro", "/e/depth_to_gyro",
                     np.array(R9, np.float64), np.array([0.005, -0.002, 0.001]))
    e = m["camera_extrinsics"][0]
    assert all(isinstance(x, float) for x in e["rotation"] + e["translation"])
    assert e["translation"] == [0.005, -0.002, 0.001]


def test_extrinsic_distinct_names_compose_regardless_of_order():
    # depth (depth_to_color) and imu (gyro/accel) own disjoint names -> all coexist,
    # whichever writer ran first.
    m = {}
    upsert_extrinsic(m, "depth_to_gyro", "/e/g", R9, T3)      # imu first
    upsert_extrinsic(m, "depth_to_color", "/e/c", R9, T3)     # then depth
    upsert_extrinsic(m, "depth_to_accel", "/e/a", R9, T3)     # then imu again
    assert sorted(e["name"] for e in m["camera_extrinsics"]) == \
        ["depth_to_accel", "depth_to_color", "depth_to_gyro"]


def test_extrinsic_rerun_updates_in_place_no_duplicate():
    m = {}
    upsert_extrinsic(m, "depth_to_color", "/e/c", R9, [0.010, 0, 0])
    lst_before = m["camera_extrinsics"]
    upsert_extrinsic(m, "depth_to_color", "/e/c", R9, [0.015, 0, 0])   # re-run, new value
    assert len(m["camera_extrinsics"]) == 1                            # not duplicated
    assert m["camera_extrinsics"] is lst_before                       # list never reassigned
    assert m["camera_extrinsics"][0]["translation"] == [0.015, 0, 0]  # updated in place


def test_extrinsic_never_reassigns_shared_list():
    m = {"camera_extrinsics": [{"name": "depth_to_color", "source_topic": "/x"}]}
    original = m["camera_extrinsics"]
    upsert_extrinsic(m, "depth_to_gyro", "/e/g", R9, T3)   # a different writer's entry
    assert m["camera_extrinsics"] is original              # appended, not replaced
    assert len(original) == 2                              # existing entry survives


# --- upsert_intrinsic (camera_intrinsics by `camera`) -----------------------
COLOR = {"source": "charuco_calib", "width": 1280, "height": 720, "K": [1, 2, 3]}


def test_intrinsic_creates_list_and_block_when_absent():
    m = {}
    upsert_intrinsic(m, "exo_cam1", COLOR)
    assert m["camera_intrinsics"] == [{"camera": "exo_cam1", "color": COLOR}]


def test_intrinsic_rerun_updates_in_place_no_duplicate():
    m = {}
    upsert_intrinsic(m, "exo_cam1", {"K": [1]})
    lst_before = m["camera_intrinsics"]
    upsert_intrinsic(m, "exo_cam1", {"K": [2]})            # re-run, new value
    assert len(m["camera_intrinsics"]) == 1               # not duplicated
    assert m["camera_intrinsics"] is lst_before           # list never reassigned
    assert m["camera_intrinsics"][0]["color"] == {"K": [2]}


def test_intrinsic_distinct_cameras_coexist_and_preserve_siblings():
    # cam_ego (bag camera_info, with a depth sub-block) + exo (calib) own disjoint
    # cameras; writing exo must not clobber cam_ego's depth sub-block.
    m = {"camera_intrinsics": [{"camera": "cam_ego", "color": {"K": [0]},
                                "depth": {"K": [9]}}]}
    upsert_intrinsic(m, "exo_cam1", COLOR)                 # new camera
    upsert_intrinsic(m, "cam_ego", {"K": [1]})             # update existing color only
    ego = next(c for c in m["camera_intrinsics"] if c["camera"] == "cam_ego")
    assert ego["color"] == {"K": [1]}                      # color updated
    assert ego["depth"] == {"K": [9]}                      # sibling preserved
    assert [c["camera"] for c in m["camera_intrinsics"]] == ["cam_ego", "exo_cam1"]


# --- reorder_top_level (cosmetic key order) ---------------------------------
def _sample_meta():
    # deliberately out of order, plus an unknown key not in TOP_LEVEL_ORDER
    return {"camera_intrinsics": [1], "steps": {"a": 1}, "metadata": {"n": "leo"},
            "termination": {"ok": True}, "camera_extrinsics": [2],
            "episode_details": {"code": "7"}, "future_key": {"x": 1}}


def test_reorder_places_known_keys_in_order():
    out = reorder_top_level(_sample_meta())
    # known keys first, in TOP_LEVEL_ORDER; the unknown key trails, preserved
    assert list(out.keys()) == list(TOP_LEVEL_ORDER) + ["future_key"]


def test_reorder_is_value_preserving_and_never_drops_keys():
    src = _sample_meta()
    out = reorder_top_level(src)
    assert out == src                       # deep-equal: same keys, same values
    assert set(out) == set(src)             # nothing dropped or added
    assert out is not src                   # returns a new dict, input untouched
    assert list(src.keys())[0] == "camera_intrinsics"   # input order unchanged


def test_reorder_skips_absent_keys_no_null_fill():
    # a minimal meta missing most keys -> only the present ones appear, none invented
    out = reorder_top_level({"steps": {}, "metadata": {}})
    assert list(out.keys()) == ["metadata", "steps"]


def test_reorder_idempotent():
    once = reorder_top_level(_sample_meta())
    twice = reorder_top_level(once)
    assert list(twice.keys()) == list(once.keys()) and twice == once
