"""Unit tests for bag_integrity (step 0: structural check + spine init).

Run:  python -m pytest data-processing-scripts-v3/test_bag_integrity.py -q

Split mirrors the module:
  A. build_initial_spine -> PURE (dict in, dict out): spine shape, healthy vs corrupt.
  B. check_bag           -> reader shell: real bag ok; non-bag dirs -> corrupt.
  C. init_spine          -> file shell: writes metadata.json; corrupt verdict on disk.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bag_integrity as bi                       # noqa: E402
import test_pipeline_integration as T            # noqa: E402  (reuse its real-bag builder)

META = {"dataset_name": "leo", "dataset_version": "3.0", "robot_model": "human",
        "environment": "room", "setup": "v6", "subject": "s1", "bed_type": "single"}


# --------------------------------------------------------------------------- #
# A. build_initial_spine — PURE                                               #
# --------------------------------------------------------------------------- #
def test_healthy_spine_shape():
    spine = bi.build_initial_spine(META, corrupt=False, detail=[],
                                   date_recorded="2026-02-23", timestamp_range=[1.0, 5.0])
    assert list(spine.keys()) == ["metadata", "camera_intrinsics", "steps", "termination"]
    assert spine["metadata"]["dataset_name"] == "leo"
    assert spine["metadata"]["date_recorded"] == "2026-02-23"
    assert "fps" not in spine["metadata"]                 # top-level fps removed
    assert spine["camera_intrinsics"] == []               # color appends later
    assert spine["steps"] == {"streams": [], "timestamp_range": [1.0, 5.0]}
    assert spine["termination"] == {"is_successful": True, "reason": []}


def test_corrupt_spine_carries_token_and_detail():
    spine = bi.build_initial_spine(META, corrupt=True, detail=["bag failed to open"],
                                   date_recorded=None, timestamp_range=None)
    assert spine["termination"] == {"is_successful": False, "reason": ["rosbag_corruption"]}
    assert spine["steps"]["bag_corruption_error"] == ["bag failed to open"]
    assert spine["steps"]["streams"] == []
    assert spine["metadata"]["date_recorded"] is None


def test_metadata_block_order_matches_schema():
    spine = bi.build_initial_spine(META, corrupt=False, detail=[], date_recorded="2026-02-23")
    assert list(spine["metadata"].keys()) == [
        "dataset_name", "dataset_version", "robot_model", "environment",
        "date_recorded", "setup", "subject", "bed_type",
    ]


# --------------------------------------------------------------------------- #
# B. check_bag — reader shell                                                 #
# --------------------------------------------------------------------------- #
def test_check_real_bag_is_clean_with_timestamps(tmp_path):
    bag = T.build_pipeline_bag(tmp_path / "run1", n_color=8)
    corrupt, detail, date_recorded, ts_range = bi.check_bag(bag)
    assert corrupt is False and detail == []
    assert date_recorded is not None
    assert ts_range and ts_range[0] is not None and ts_range[1] is not None


def test_check_missing_path_is_corrupt(tmp_path):
    corrupt, detail, *_ = bi.check_bag(tmp_path / "nope")
    assert corrupt is True and "not found" in detail[0]


def test_check_dir_without_metadata_yaml_is_corrupt(tmp_path):
    d = tmp_path / "notabag"
    d.mkdir()
    (d / "readme.txt").write_text("x")
    corrupt, detail, *_ = bi.check_bag(d)
    assert corrupt is True and "metadata.yaml" in detail[0]


def test_check_file_path_is_corrupt(tmp_path):
    f = tmp_path / "afile"
    f.write_bytes(b"\x00")
    corrupt, detail, *_ = bi.check_bag(f)
    assert corrupt is True and "file" in detail[0]


# --------------------------------------------------------------------------- #
# C. init_spine — file shell                                                  #
# --------------------------------------------------------------------------- #
def test_init_spine_healthy_writes_clean_metadata(tmp_path):
    bag = T.build_pipeline_bag(tmp_path / "run1", n_color=8)
    out = tmp_path / "out"
    res = bi.init_spine(bag, out, META)
    assert res == {"corrupt": False, "detail": [], "written": True}
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["termination"]["is_successful"] is True
    assert meta["metadata"]["date_recorded"] is not None
    assert meta["steps"]["streams"] == []                 # nothing extracted yet


def test_init_spine_corrupt_writes_verdict_and_skips_nothing_else(tmp_path):
    d = tmp_path / "notabag"          # a dir with no metadata.yaml -> corrupt
    d.mkdir()
    out = tmp_path / "out"
    res = bi.init_spine(d, out, META)
    assert res["corrupt"] is True
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["termination"] == {"is_successful": False, "reason": ["rosbag_corruption"]}
    assert meta["steps"]["bag_corruption_error"]          # the reason is recorded
