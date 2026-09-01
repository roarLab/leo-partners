"""Unit suite for wrapper.py (the full color+depth pipeline).

Run:  conda activate leo-seg && pytest data-processing-scripts-v3/ -q

Scope: the WRAPPER's own responsibilities — the parts no subscript owns. The four
subscripts it orchestrates (rosbag_process_color_v3.main, rosbag_process_depth_v3.main,
validate_color_v3.validate_metadata, validate_depth_v3.validate_aligned_depth) are
MOCKED, so these are fast deterministic unit tests that need no real bags, color
topics, or the depth aligner. The scripts themselves are covered elsewhere
(test_rosbag_process_depth_v3.py + each script's own suite).

Four groups:
  A. Bag discovery & output routing   (is_bag / collect_bags / run_pipeline_dir)
  B. Step 3 sanity_check
  C. Step 6 extract_signals (the five error signals)
  D. Orchestration & isolation        (run_pipeline_for_bag / run_pipeline_dir)
"""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import wrapper as wrap                                  # noqa: E402

ALIGNED_H5 = "ego_aligned_depth_to_color.h5"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def make_bag(path: Path) -> Path:
    """A directory is_bag() accepts (has metadata.yaml). No topics needed —
    every step that would read it is mocked."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "metadata.yaml").write_text("version: 5\n")
    return path


def write_color_outputs(out_dir: Path, streams=None) -> None:
    """What a successful Step (color) leaves on disk: videos/ + timestamps/, and its
    slice APPENDED into metadata.json. Colour is an appender now: bag_integrity created
    the spine first, so we read-modify-write (preserving termination) rather than
    clobber. Falls back to creating a minimal spine when called standalone (no fixture)."""
    out_dir = Path(out_dir)
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    (out_dir / "videos" / "cam_ego.mp4").write_bytes(b"\x00")
    (out_dir / "timestamps").mkdir(parents=True, exist_ok=True)
    (out_dir / "timestamps" / "cam_ego.csv").write_text("ros_time_s\n0.0\n")
    p = out_dir / "metadata.json"
    meta = json.loads(p.read_text()) if p.is_file() else \
        {"steps": {"streams": []}, "termination": {"is_successful": True, "reason": []}}
    meta.setdefault("steps", {})["streams"] = streams or []
    p.write_text(json.dumps(meta))


def write_integrity_spine(out_dir: Path) -> None:
    """What bag_integrity (step 0) leaves for a clean bag: the metadata.json spine."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spine = {"metadata": {}, "camera_intrinsics": [],
             "steps": {"streams": [], "timestamp_range": None},
             "termination": {"is_successful": True, "reason": []}}
    (out_dir / "metadata.json").write_text(json.dumps(spine))


def write_depth_outputs(out_dir: Path) -> None:
    """What a successful Step-2 (depth) leaves on disk: the aligned h5 (+ CSV)."""
    out_dir = Path(out_dir)
    (out_dir / "depth_frames").mkdir(parents=True, exist_ok=True)
    (out_dir / "depth_frames" / ALIGNED_H5).write_bytes(b"\x00")
    (out_dir / "timestamps").mkdir(parents=True, exist_ok=True)
    (out_dir / "timestamps" / "ego_aligned_depth_to_color.csv").write_text("index\n0\n")


@pytest.fixture
def wired(monkeypatch):
    """Patch all subscripts with recording stubs that emit the on-disk
    artifacts a real run would. Returns a dict capturing call order + args."""
    rec = {"calls": [], "color_meta": None, "color_cameras": None, "depth_kwargs": None}

    def fake_integrity(bag=None, out_dir=None, meta=None):
        rec["calls"].append("bag_integrity")
        rec["integrity_meta"] = meta
        write_integrity_spine(out_dir)                 # a clean spine (never corrupt in the fixture)
        return {"corrupt": False, "detail": [], "written": True}

    def fake_color(bag=None, out_dir=None, meta=None, inspect_only=None, cameras=None):
        rec["calls"].append("color")
        rec["color_meta"] = meta
        rec["color_cameras"] = cameras
        write_color_outputs(out_dir)
        return {}

    def fake_depth(bag=None, out_dir=None, camera=None):
        rec["calls"].append("depth")
        rec["depth_kwargs"] = {"bag": bag, "out_dir": out_dir, "camera": camera}
        write_depth_outputs(out_dir)
        return {}

    def fake_imu(bag=None, out_dir=None, camera=None):
        rec["calls"].append("imu")
        rec["imu_kwargs"] = {"bag": bag, "out_dir": out_dir, "camera": camera}
        return {"extrinsics_found": ["depth_to_accel", "depth_to_gyro"],
                "imu_samples": 3, "imu_missing": False}

    def fake_episode_details(bag=None, out_dir=None):
        rec["calls"].append("episode_details")
        rec["episode_details_kwargs"] = {"bag": bag, "out_dir": out_dir}
        return {"bag": str(bag), "mistakes": [], "written": False,
                "code": None, "start_time_ns": None}

    def fake_val_color(out_dir):
        rec["calls"].append("val_color")

    def fake_val_depth(out_dir):
        rec["calls"].append("val_depth")

    def fake_val_imu(out_dir):
        rec["calls"].append("val_imu")

    monkeypatch.setattr(wrap.bag_integrity, "init_spine", fake_integrity)
    monkeypatch.setattr(wrap.rpc, "main", fake_color)
    monkeypatch.setattr(wrap.rpd, "main", fake_depth)
    monkeypatch.setattr(wrap.rpi, "main", fake_imu)
    monkeypatch.setattr(wrap.red, "extract_episode_details", fake_episode_details)
    monkeypatch.setattr(wrap.validate_color_v3, "validate_metadata", fake_val_color)
    monkeypatch.setattr(wrap.validate_depth_v3, "validate_aligned_depth", fake_val_depth)
    monkeypatch.setattr(wrap.validate_imu_v3, "validate_imu", fake_val_imu)
    return rec


# ===========================================================================
# A. Bag discovery & output routing
# ===========================================================================
def test_is_bag_accepts_metadata_yaml(tmp_path):
    assert wrap.is_bag(make_bag(tmp_path / "run1"))


def test_is_bag_accepts_flattened_mcap(tmp_path):
    d = tmp_path / "run1"
    d.mkdir()
    (d / "run1_0.mcap").write_bytes(b"\x00")
    assert wrap.is_bag(d)


def test_is_bag_rejects_plain_dir(tmp_path):
    d = tmp_path / "notabag"
    d.mkdir()
    (d / "readme.txt").write_text("x")
    assert not wrap.is_bag(d)


def test_collect_bags_single(tmp_path):
    bag = make_bag(tmp_path / "run1")
    assert wrap.collect_bags(bag) == [bag]


def test_collect_bags_parent_sorted_skips_junk(tmp_path):
    src = tmp_path / "leo_ws"
    (src / "build").mkdir(parents=True)      # not a bag
    (src / "install").mkdir()                # not a bag
    (src / ".hidden").mkdir()                # dot-prefixed
    make_bag(src / "run2")
    make_bag(src / "run1")
    assert [b.name for b in wrap.collect_bags(src)] == ["run1", "run2"]


def test_single_bag_output_not_nested(tmp_path, wired):
    bag = make_bag(tmp_path / "run1")
    rows = wrap.run_pipeline_dir(bag, tmp_path / "out", "ego", {})
    assert len(rows) == 1
    assert rows[0]["out_dir"] == str(tmp_path / "out")     # straight into destination
    assert not (tmp_path / "out" / "run1").exists()        # NOT nested


def test_parent_nests_each_bag(tmp_path, wired):
    src = tmp_path / "leo_ws"
    make_bag(src / "run1")
    make_bag(src / "run2")
    rows = wrap.run_pipeline_dir(src, tmp_path / "out", "ego", {})
    out_dirs = sorted(r["out_dir"] for r in rows)
    assert out_dirs == [str(tmp_path / "out" / "run1"), str(tmp_path / "out" / "run2")]


def test_no_bags_returns_empty(tmp_path):
    (tmp_path / "empty").mkdir()
    assert wrap.run_pipeline_dir(tmp_path / "empty", tmp_path / "out", "ego", {}) == []


# ===========================================================================
# B. Step 3 — sanity_check
# ===========================================================================
def _meta_streams(out_dir: Path, streams: list) -> None:
    """Write a metadata.json carrying just steps.streams (what sanity_check reads)."""
    (out_dir / "metadata.json").write_text(
        json.dumps({"steps": {"streams": streams}}), encoding="utf-8")


def test_sanity_all_declared_files_present_is_empty(tmp_path):
    (tmp_path / "videos").mkdir()
    (tmp_path / "depth_frames").mkdir()
    (tmp_path / "videos" / "cam_ego.mp4").write_bytes(b"\x00")
    (tmp_path / "depth_frames" / ALIGNED_H5).write_bytes(b"\x00")
    _meta_streams(tmp_path, [
        {"camera": "cam_ego", "kind": "color", "video": "videos/cam_ego.mp4"},
        {"camera": "ego", "kind": "aligned_depth_to_color",
         "frames_dir": f"depth_frames/{ALIGNED_H5}"},
    ])
    assert wrap.sanity_check(tmp_path) == []


def test_sanity_missing_metadata_flags_the_spine(tmp_path):
    missing = wrap.sanity_check(tmp_path)                 # no metadata.json at all
    assert [m["path"] for m in missing] == ["metadata.json"]


def test_sanity_flags_declared_but_absent_file(tmp_path):
    # metadata CLAIMS a depth h5 that is not on disk -> the vanished-file case
    _meta_streams(tmp_path, [
        {"camera": "ego", "kind": "aligned_depth_to_color",
         "frames_dir": f"depth_frames/{ALIGNED_H5}"},
    ])
    missing = wrap.sanity_check(tmp_path)
    assert missing == [{"camera": "ego", "kind": "aligned_depth_to_color",
                        "field": "frames_dir", "path": f"depth_frames/{ALIGNED_H5}"}]


def test_sanity_unrecorded_stream_declares_nothing(tmp_path):
    # a missing topic leaves a stream with no path fields -> nothing to check -> NOT
    # flagged (that absence is a data-quality flag, not an incomplete extraction)
    _meta_streams(tmp_path, [{"camera": "cam_ego", "kind": "color", "found": False}])
    assert wrap.sanity_check(tmp_path) == []


def test_sanity_null_path_field_is_skipped(tmp_path):
    # video is None (color found but no frames decoded); timestamps IS declared + present
    (tmp_path / "timestamps").mkdir()
    (tmp_path / "timestamps" / "cam_ego.csv").write_text("x")
    _meta_streams(tmp_path, [
        {"camera": "cam_ego", "kind": "color", "video": None,
         "timestamps": "timestamps/cam_ego.csv"},
    ])
    assert wrap.sanity_check(tmp_path) == []


def test_sanity_flags_each_missing_declared_field(tmp_path):
    # both declared files absent -> both reported
    _meta_streams(tmp_path, [
        {"camera": "cam_ego", "kind": "color",
         "video": "videos/cam_ego.mp4", "timestamps": "timestamps/cam_ego.csv"},
    ])
    missing = wrap.sanity_check(tmp_path)
    assert {m["field"] for m in missing} == {"video", "timestamps"}


# ===========================================================================
# C. Step 6 — extract_signals (the five error signals)
# ===========================================================================
def _write_meta(out_dir: Path, streams):
    write_color_outputs(out_dir, streams=streams)


def test_signals_clean_all_blank(tmp_path):
    _write_meta(tmp_path, [
        {"kind": "color", "camera": "cam_ego"},
        {"kind": "aligned_depth_to_color", "camera": "ego"},
    ])
    sig = wrap.extract_signals(tmp_path)
    assert sig.pop("exo_calib") == "null"   # no exo cams recorded here (enum, not an error)
    assert all(v == "" for v in sig.values())


def test_signals_color_errors_aggregated_per_camera(tmp_path):
    _write_meta(tmp_path, [
        {"kind": "color", "camera": "cam1", "data_error": ["82.0% frames vs cam_ego"]},
        {"kind": "color", "camera": "cam2", "timestamps_error": ["Index 5 period 3.0s"]},
    ])
    sig = wrap.extract_signals(tmp_path)
    assert sig["color_error"] == "cam1: 82.0% frames vs cam_ego"
    assert sig["color_timestamp_error"] == "cam2: Index 5 period 3.0s"


def test_signals_depth_data_error_split(tmp_path):
    # All depth data_error entries (h5 integrity AND two-way pairing) now flow into the
    # single depth_error column; the separate color_depth_mismatch column was removed.
    _write_meta(tmp_path, [{
        "kind": "aligned_depth_to_color", "camera": "ego",
        "timestamps_error": ["Frame 90 depth-stamp gap 2.0s"],
        "data_error": [
            "unpaired color→depth: 14.2% (142/1000)",
            "h5 frame count 998 != timestamps rows 1000",
        ],
    }])
    sig = wrap.extract_signals(tmp_path)
    assert sig["depth_error"] == ("unpaired color→depth: 14.2% (142/1000) | "
                                  "h5 frame count 998 != timestamps rows 1000")
    assert sig["depth_timestamp_error"] == "Frame 90 depth-stamp gap 2.0s"


def test_signals_missing_and_extra_from_steps_keys(tmp_path):
    # the two count-deviation signals come from the steps-level lists the extraction
    # scripts own, NOT from per-stream annotations.
    meta = {"steps": {
        "streams": [{"kind": "color", "camera": "cam_ego"}],
        "missing_stream_error": ["exo_cam: expected 4, found 3 (present: [...])"],
        "extra_stream_error": [],
    }}
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    sig = wrap.extract_signals(tmp_path)
    assert sig["missing_stream_error"] == "exo_cam: expected 4, found 3 (present: [...])"
    assert sig["extra_stream_error"] == ""


def test_signals_missing_metadata_all_blank(tmp_path):
    sig = wrap.extract_signals(tmp_path)    # no metadata.json
    assert set(sig) == {
        "color_error", "color_timestamp_error", "depth_error",
        "depth_timestamp_error", "imu_error", "imu_timestamp_error",
        "missing_stream_error", "extra_stream_error", "rosbag_corruption", "exo_calib",
    }
    assert all(v == "" for v in sig.values())   # missing metadata -> exo_calib/rosbag_corruption blank too


def test_signals_missing_imu_lands_in_missing_stream(tmp_path):
    # imu missing/empty is a PRESENCE failure recorded as a missing_stream entry (reason
    # imu_presence_err). The imu_error column is for DATA validation (validate_imu) — a mere
    # absence must NOT light it up; it stays blank while missing_stream_error carries it.
    meta = {
        "steps": {"streams": [{"kind": "color", "camera": "cam_ego"}],
                  "missing_stream_error": ["cam_ego imu: expected but not found (*/imu)"]},
        "termination": {"is_successful": False, "reason": ["imu_presence_err"]},
    }
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    sig = wrap.extract_signals(tmp_path)
    assert sig["missing_stream_error"] == "cam_ego imu: expected but not found (*/imu)"
    assert sig["imu_error"] == ""                            # validation column stays blank


# ===========================================================================
# D. Orchestration & isolation
# ===========================================================================
def test_happy_path_runs_all_in_order(tmp_path, wired):
    bag = make_bag(tmp_path / "run1")
    row = wrap.run_pipeline_for_bag(bag, tmp_path / "out", "ego", {})
    assert wired["calls"] == ["bag_integrity", "color", "depth", "imu", "episode_details", "val_color", "val_depth", "val_imu"]
    assert row["completed"] is True
    assert not (tmp_path / "out" / "pipeline_error.txt").exists()


def test_val_depth_runs_after_val_color(tmp_path, wired):
    bag = make_bag(tmp_path / "run1")
    wrap.run_pipeline_for_bag(bag, tmp_path / "out", "ego", {})
    calls = wired["calls"]
    assert calls.index("val_color") < calls.index("val_depth")


def test_colour_extract_and_validate_gated_off_when_no_colour_declared(tmp_path, wired):
    # With NO colour group declared (ego+exo present:False), BOTH the colour extract and
    # validate steps are gated off: there is no colour to extract, and validate_color's only
    # role is per-frame QUALITY of found colour streams (presence is extraction's
    # color_presence_err). bag_integrity is the ONLY always-run step; every extractor is
    # declaration-gated (colour on process_color, like depth/imu on their flags).
    bag = make_bag(tmp_path / "run1")
    cams = {"ego": {"present": False}, "exo": {"present": False},
            "depth": {"present": False}, "imu": {"present": True}}
    wrap.run_pipeline_for_bag(bag, tmp_path / "out", "ego", {}, cams)
    calls = wired["calls"]
    assert "color" not in calls          # extract gated off — no colour declared
    assert "val_color" not in calls      # validator gated off — no colour to quality-check
    assert "imu" in calls and "val_imu" in calls   # imu still processed (owns the verdict here)


def test_validate_color_runs_when_any_colour_group_present(tmp_path, wired):
    # exo off but ego on -> a colour group IS present -> validate colour still runs.
    bag = make_bag(tmp_path / "run1")
    cams = {"ego": {"present": True}, "exo": {"present": False}}
    wrap.run_pipeline_for_bag(bag, tmp_path / "out", "ego", {}, cams)
    assert "val_color" in wired["calls"]


def test_meta_passed_to_color_only(tmp_path, wired):
    bag = make_bag(tmp_path / "run1")
    meta = {"subject": "abc", "environment": "room 214"}
    wrap.run_pipeline_for_bag(bag, tmp_path / "out", "ego", meta)
    assert wired["color_meta"] == meta                       # color got it
    assert "meta" not in (wired["depth_kwargs"] or {})       # depth never sees it


def test_cameras_passed_to_color_only(tmp_path, wired):
    # a per-session camera layout reaches Step 1 (color); depth never sees it.
    bag = make_bag(tmp_path / "run1")
    cams = {"exo": {"suffix": "image_raw/compressed", "compressed": True, "label": "side_cam"}}
    wrap.run_pipeline_for_bag(bag, tmp_path / "out", "ego", {}, cams)
    assert wired["color_cameras"] == cams                    # color got the layout
    assert "cameras" not in (wired["depth_kwargs"] or {})    # depth never sees it


def test_cameras_defaults_to_none_when_omitted(tmp_path, wired):
    # standalone/legacy call sites that don't pass cameras -> color gets None
    # (rpc.main then falls back to its own DEFAULT_CAMERAS).
    bag = make_bag(tmp_path / "run1")
    wrap.run_pipeline_for_bag(bag, tmp_path / "out", "ego", {})
    assert wired["color_cameras"] is None


def test_merge_cameras_patches_group_field_keeps_others(tmp_path):
    base = {"ego": {"suffix": "d435i_ego/color/image_raw", "label": "cam_ego", "singleton": True},
            "exo": {"suffix": "image_raw/compressed", "label": "exo_cam", "compressed": True}}
    merged = wrap.merge_cameras(base, {"exo": {"label": "side_cam"}})
    assert merged["exo"]["label"] == "side_cam"              # patched field
    assert merged["exo"]["suffix"] == "image_raw/compressed" # untouched field kept
    assert merged["ego"] == base["ego"]                      # other group untouched
    assert base["exo"]["label"] == "exo_cam"                 # original not mutated


def test_merge_cameras_none_returns_base(tmp_path):
    base = {"exo": {"label": "exo_cam"}}
    assert wrap.merge_cameras(base, None) is base


# ---------------------------------------------------------------------------
# validate_cameras / validate_camera_configs — CONFIG-plane guard. Crashes LOUDLY
# before any extraction (operator mistake), the deliberate opposite of the per-bag
# DATA plane (missing/extra streams, which flag-and-continue).
# ---------------------------------------------------------------------------
def _camera_config(**patch):
    """A full wrapper-shape config (ego+exo color on, depth+imu off) for guard tests.
    Patch a group with _camera_config(ego={"present": False}, depth={"present": True})."""
    r = {
        "ego":   {"present": True, "suffix": "d435i_ego/color/image_raw",
                  "info_suffix": "d435i_ego/color/camera_info", "label": "cam_ego", "singleton": True},
        "exo":   {"present": True, "suffix": "image_raw/compressed", "label": "exo_cam", "count": 4},
        "depth": {"present": False, "suffix": "d435i_ego/depth/image_rect_raw"},
        "imu":   {"present": False, "suffix": "d435i_ego/imu"},
    }
    for grp, fields in patch.items():
        r[grp] = {**r[grp], **fields}
    return r


def test_validate_cameras_default_config_passes():
    assert wrap.validate_cameras(_camera_config()) is None           # no raise


def test_validate_cameras_depth_on_ego_off_raises():
    with pytest.raises(ValueError, match="depth requires ego color"):
        wrap.validate_cameras(_camera_config(ego={"present": False}, depth={"present": True}))


def test_validate_cameras_ego_off_depth_off_is_legal():
    # color stands alone: ego color off is fine as long as depth is off too
    assert wrap.validate_cameras(_camera_config(ego={"present": False})) is None


def test_validate_cameras_imu_not_constrained_against_color():
    # imu is standalone -> imu on + ego color off must NOT raise (guards over-constraining)
    assert wrap.validate_cameras(_camera_config(ego={"present": False}, imu={"present": True})) is None


def test_validate_cameras_missing_required_group_raises():
    r = _camera_config()
    del r["ego"]
    with pytest.raises(ValueError, match="group 'ego' is required"):
        wrap.validate_cameras(r)


def test_validate_cameras_all_absent_raises():
    # ego+exo off, depth+imu already off in the base -> nothing to extract
    with pytest.raises(ValueError, match="nothing to extract"):
        wrap.validate_cameras(_camera_config(ego={"present": False}, exo={"present": False}))


def test_validate_cameras_runs_on_merged_override_not_the_patch():
    # the override alone ({"depth": present True}) is harmless; only the MERGED config
    # (ego off from base) reveals the contradiction -> proves we validate POST-merge.
    base = _camera_config(ego={"present": False})
    with pytest.raises(ValueError, match="depth requires ego color"):
        wrap.validate_cameras(wrap.merge_cameras(base, {"depth": {"present": True}}))


def test_validate_camera_configs_fails_before_any_work_and_names_session():
    ls = [
        ("/data/good", "/out/good", None, None),
        ("/data/bad",  "/out/bad",  None, {"ego": {"present": False}, "depth": {"present": True}}),
    ]
    with pytest.raises(ValueError, match="/data/bad"):
        wrap.validate_camera_configs(ls, _camera_config())


def test_validate_camera_configs_all_valid_passes():
    ls = [("/data/a", "/out/a", None, None),
          ("/data/b", "/out/b", None, {"exo": {"present": False}})]
    assert wrap.validate_camera_configs(ls, _camera_config()) is None


# ---------------------------------------------------------------------------
# validate_calib_rows — CONFIG-plane guard: exo calibration is MANDATORY, so a
# missing/unset/malformed calib FILE fails the whole batch at startup (before any bag).
# (Per-episode COVERAGE — every recorded exo cam present — is the DATA plane, tested in
# test_pipeline_calibration.py; this guard only proves the file exists and parses.)
# ---------------------------------------------------------------------------
def _calib_json(tmp_path, name="calib-2008.json", obj=None):
    p = tmp_path / name
    p.write_text(json.dumps(obj if obj is not None else {"cameras": {}}))
    return str(p)


def test_validate_calib_rows_all_valid_passes(tmp_path):
    good = _calib_json(tmp_path)
    ls = [("/data/a", "/out/a", None, None, None),          # uses the default
          ("/data/b", "/out/b", None, None, good)]          # 5th-field override
    assert wrap.validate_calib_rows(ls, good) is None


def test_validate_calib_rows_missing_file_raises_and_names_session(tmp_path):
    good = _calib_json(tmp_path)
    ls = [("/data/good", "/out/good", None, None, good),
          ("/data/bad",  "/out/bad",  None, None, str(tmp_path / "nope.json"))]
    with pytest.raises(ValueError, match="/data/bad"):
        wrap.validate_calib_rows(ls, good)


def test_validate_calib_rows_unset_default_raises(tmp_path):
    # no override AND a falsy default -> calibration not configured -> hard fail (mandatory).
    ls = [("/data/x", "/out/x", None, None, None)]
    with pytest.raises(ValueError, match="no exo calibration configured"):
        wrap.validate_calib_rows(ls, None)


def test_validate_calib_rows_placeholder_default_raises(tmp_path):
    # the shipped placeholder ("filepath.json") does not exist -> loud fail, not a silent no-op.
    ls = [("/data/x", "/out/x", None, None, None)]
    with pytest.raises(ValueError, match="not found"):
        wrap.validate_calib_rows(ls, "filepath.json")


def test_validate_calib_rows_malformed_json_raises(tmp_path):
    bad = tmp_path / "calib.json"
    bad.write_text("{ not valid json ")
    ls = [("/data/x", "/out/x", None, None, str(bad))]
    with pytest.raises(ValueError, match="not valid JSON"):
        wrap.validate_calib_rows(ls, str(bad))


def test_validate_calib_rows_override_supplies_missing_default(tmp_path):
    # default is the broken placeholder, but a per-session 5th-field override points at a real
    # file -> that session passes (proves the resolve mirrors __main__: override or default).
    good = _calib_json(tmp_path)
    ls = [("/data/x", "/out/x", None, None, good)]
    assert wrap.validate_calib_rows(ls, "filepath.json") is None


@pytest.mark.parametrize("exc", [RuntimeError("boom"), SystemExit("fatal")])
def test_color_failure_isolated(tmp_path, monkeypatch, wired, exc):
    def boom(**kwargs):
        raise exc
    monkeypatch.setattr(wrap.rpc, "main", boom)
    bag = make_bag(tmp_path / "run1")
    row = wrap.run_pipeline_for_bag(bag, tmp_path / "out", "ego", {})
    assert row["completed"] is False
    assert (tmp_path / "out" / "pipeline_error.txt").exists()
    assert "depth" not in wired["calls"]                     # later steps skipped
    # bag_integrity wrote the spine before color crashed, so extract_signals runs on it;
    # exo_calib is a usability ENUM ("null" here), not a blank error cell -> excluded.
    assert all(row[c] == "" for c in wrap.REPORT_COLUMNS
               if c not in ("out_dir", "completed", "exo_calib"))


def test_depth_failure_skips_validators(tmp_path, monkeypatch, wired):
    monkeypatch.setattr(wrap.rpd, "main",
                        lambda **kw: (_ for _ in ()).throw(SystemExit("no depth topic")))
    bag = make_bag(tmp_path / "run1")
    row = wrap.run_pipeline_for_bag(bag, tmp_path / "out", "ego", {})
    assert row["completed"] is False
    assert wired["calls"] == ["bag_integrity", "color"]                       # nothing after depth failure
    assert "val_color" not in wired["calls"]


def test_missing_outputs_marks_incomplete_and_writes_marker(tmp_path, monkeypatch, wired):
    # Sanity is the FINAL VERDICT, not a mid-pipeline gate. Colour "succeeds" but DECLARES a
    # video file it never wrote (depth writes nothing), so every step (INCLUDING the
    # validators) still runs; the post-loop sanity then finds that declared file absent on
    # disk -> completed False. An INCOMPLETE is a graceful non-crash, but it DOES leave a
    # durable pipeline_error.txt marker. (bag_integrity always writes the spine now, so
    # "no metadata.json at all" no longer occurs — a declared-but-absent file is the case.)
    def color_declares_absent(**kw):
        wired["calls"].append("color")
        p = Path(kw["out_dir"]) / "metadata.json"
        meta = json.loads(p.read_text())          # the spine bag_integrity wrote
        meta["steps"]["streams"] = [
            {"camera": "cam_ego", "kind": "color", "video": "videos/cam_ego.mp4"}]
        p.write_text(json.dumps(meta))            # declares a file that is NOT on disk
        return {}
    monkeypatch.setattr(wrap.rpc, "main", color_declares_absent)
    monkeypatch.setattr(wrap.rpd, "main", lambda **kw: {})
    bag = make_bag(tmp_path / "run1")
    row = wrap.run_pipeline_for_bag(bag, tmp_path / "out", "ego", {})
    assert row["completed"] is False
    # validators PRECEDE the sanity verdict, so they DO run
    assert "val_color" in wired["calls"] and "val_depth" in wired["calls"]
    # INCOMPLETE writes the marker with the missing-file list, NOT a traceback
    err = tmp_path / "out" / "pipeline_error.txt"
    assert err.is_file()
    text = err.read_text()
    assert text.startswith("INCOMPLETE") and "Traceback" not in text


def test_incomplete_marker_lists_the_vanished_file(tmp_path, monkeypatch, wired):
    # A stream RECORDED a file in metadata but it isn't on disk -> INCOMPLETE, and the
    # marker names that exact declared file (camera/kind/field/path), not a traceback.
    def color_declares_absent_video(**kw):
        out = Path(kw["out_dir"])
        (out / "metadata.json").write_text(json.dumps(
            {"steps": {"streams": [
                {"camera": "cam_ego", "kind": "color", "video": "videos/cam_ego.mp4"}]}}))
        return {}
    monkeypatch.setattr(wrap.rpc, "main", color_declares_absent_video)
    bag = make_bag(tmp_path / "run1")
    cams = {"depth": {"present": False}, "imu": {"present": False}}   # color-only run
    row = wrap.run_pipeline_for_bag(bag, tmp_path / "out", "ego", {}, cams)
    assert row["completed"] is False
    text = (tmp_path / "out" / "pipeline_error.txt").read_text()
    assert "cam_ego (color) video: videos/cam_ego.mp4" in text
    assert "Traceback" not in text


def test_incomplete_flags_metadata_with_missing_file_token(tmp_path, monkeypatch, wired):
    # The machine-readable twin: a declared-but-absent file adds the missing_file_error
    # token to termination.reason and flips is_successful (per-file detail is in the txt,
    # NOT metadata — no steps key is written).
    def color_declares_absent_video(**kw):
        out = Path(kw["out_dir"])
        (out / "metadata.json").write_text(json.dumps(
            {"steps": {"streams": [
                {"camera": "cam_ego", "kind": "color", "video": "videos/cam_ego.mp4"}]},
             "termination": {"is_successful": True, "reason": []}}))
        return {}
    monkeypatch.setattr(wrap.rpc, "main", color_declares_absent_video)
    bag = make_bag(tmp_path / "run1")
    cams = {"depth": {"present": False}, "imu": {"present": False}}   # color-only run
    wrap.run_pipeline_for_bag(bag, tmp_path / "out", "ego", {}, cams)

    meta = json.loads((tmp_path / "out" / "metadata.json").read_text())
    assert wrap.MISSING_FILE_REASON_TOKEN in meta["termination"]["reason"]
    assert meta["termination"]["is_successful"] is False
    assert "sanity_error" not in meta.get("steps", {})               # no steps key written


def test_success_clears_stale_error_file(tmp_path, wired):
    out = tmp_path / "out"
    out.mkdir()
    (out / "pipeline_error.txt").write_text("previous failure\n")
    bag = make_bag(tmp_path / "run1")
    row = wrap.run_pipeline_for_bag(bag, out, "ego", {})
    assert row["completed"] is True
    assert not (out / "pipeline_error.txt").exists()         # cleared on success


def test_batch_continues_after_one_bag_fails(tmp_path, monkeypatch, wired):
    src = tmp_path / "root"
    make_bag(src / "good")
    make_bag(src / "bad")

    real_color = wrap.rpc.main
    def color_maybe_fail(bag=None, out_dir=None, meta=None, inspect_only=None, cameras=None):
        if Path(bag).name == "bad":
            raise SystemExit("bad bag")
        return real_color(bag=bag, out_dir=out_dir, meta=meta,
                          inspect_only=inspect_only, cameras=cameras)
    monkeypatch.setattr(wrap.rpc, "main", color_maybe_fail)

    rows = {Path(r["out_dir"]).name: r for r in
            wrap.run_pipeline_dir(src, tmp_path / "out", "ego", {})}
    assert rows["good"]["completed"] is True
    assert rows["bad"]["completed"] is False
    assert (tmp_path / "out" / "bad" / "pipeline_error.txt").exists()


def test_report_row_has_exactly_the_report_columns(tmp_path, wired):
    bag = make_bag(tmp_path / "run1")
    row = wrap.run_pipeline_for_bag(bag, tmp_path / "out", "ego", {})
    assert set(row) == set(wrap.REPORT_COLUMNS)


def test_imu_absent_present_flag_skips_imu_step(tmp_path, wired):
    # presence declared in the camera layout (like DEFAULT_CAMERAS): imu present=False.
    bag = make_bag(tmp_path / "run1")
    cams = {"imu": {"present": False}}
    row = wrap.run_pipeline_for_bag(bag, tmp_path / "out", "ego", {}, cams)
    assert wired["calls"] == ["bag_integrity", "color", "depth", "episode_details", "val_color", "val_depth"]   # no imu
    assert row["completed"] is True


def test_depth_absent_present_flag_skips_depth_and_its_validator(tmp_path, wired):
    # Depth present=False: depth align + validate depth drop out, and sanity no longer
    # needs the aligned h5 (no depth outputs here), so the bag still completes.
    bag = make_bag(tmp_path / "run1")
    cams = {"depth": {"present": False}}
    row = wrap.run_pipeline_for_bag(bag, tmp_path / "out", "ego", {}, cams)
    assert wired["calls"] == ["bag_integrity", "color", "imu", "episode_details", "val_color", "val_imu"]       # no depth / val_depth
    assert row["completed"] is True


def test_both_absent_runs_color_sanity_validatecolor_only(tmp_path, wired):
    bag = make_bag(tmp_path / "run1")
    cams = {"depth": {"present": False}, "imu": {"present": False}}
    row = wrap.run_pipeline_for_bag(bag, tmp_path / "out", "ego", {}, cams)
    assert wired["calls"] == ["bag_integrity", "color", "episode_details", "val_color"]
    assert row["completed"] is True


def test_default_cameras_none_processes_both(tmp_path, wired):
    # no camera layout -> both streams assumed present (default), so all steps run.
    bag = make_bag(tmp_path / "run1")
    wrap.run_pipeline_for_bag(bag, tmp_path / "out", "ego", {})
    assert wired["calls"] == ["bag_integrity", "color", "depth", "imu", "episode_details", "val_color", "val_depth", "val_imu"]


def test_corrupt_bag_skips_all_extraction_and_validation(tmp_path, monkeypatch, wired):
    # bag_integrity reports the bag corrupt -> the wrapper writes ONLY the spine (verdict)
    # and SKIPS all extraction + validation. completed:False; metadata carries
    # rosbag_corruption + is_successful False; the summary signal is set; a CORRUPT marker.
    def corrupt_integrity(bag=None, out_dir=None, meta=None):
        wired["calls"].append("bag_integrity")
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        (out / "metadata.json").write_text(json.dumps({
            "metadata": {}, "camera_intrinsics": [],
            "steps": {"streams": [], "timestamp_range": None,
                      "bag_corruption_error": ["bag failed to open or index: boom"]},
            "termination": {"is_successful": False, "reason": ["rosbag_corruption"]}}))
        return {"corrupt": True, "detail": ["bag failed to open or index: boom"], "written": True}
    monkeypatch.setattr(wrap.bag_integrity, "init_spine", corrupt_integrity)

    bag = make_bag(tmp_path / "run1")
    out = tmp_path / "out"
    row = wrap.run_pipeline_for_bag(bag, out, "ego", {})

    assert wired["calls"] == ["bag_integrity"]               # NOTHING else ran
    assert row["completed"] is False
    assert row["rosbag_corruption"]                          # summary signal populated
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["termination"]["reason"] == ["rosbag_corruption"]
    assert meta["termination"]["is_successful"] is False
    assert (out / "pipeline_error.txt").read_text().startswith("CORRUPT")


def test_stream_present_defaults_true_and_reads_flag():
    assert wrap.stream_present(None, "imu") is True           # None -> assume present
    assert wrap.stream_present({}, "depth") is True           # absent key -> present
    assert wrap.stream_present({"imu": {}}, "imu") is True    # key but no flag -> present
    assert wrap.stream_present({"imu": {"present": False}}, "imu") is False
    assert wrap.stream_present({"imu": {"present": False}}, "depth") is True   # per-stream


def test_present_flag_flows_through_cameras_override():
    # a per-session cameras_override opts a stream out through the SAME merge channel.
    merged = wrap.merge_cameras(
        {"imu": {"present": True, "suffix": "d435i_ego/imu"}},
        {"imu": {"present": False}})
    assert merged["imu"] == {"present": False, "suffix": "d435i_ego/imu"}   # patched, suffix kept
    assert wrap.stream_present(merged, "imu") is False


# ---------------------------------------------------------------------------
# E. Step 7 — session-summary.csv (ONE combined file; boolean error signals)
# ---------------------------------------------------------------------------
def _ep_out(dest: Path, name: str, details) -> Path:
    out = dest / name
    out.mkdir(parents=True, exist_ok=True)
    meta = {"steps": {"streams": []}}
    if details is not None:
        meta["episode_details"] = details
    (out / "metadata.json").write_text(json.dumps(meta))
    return out


def _row(out_dir: Path, *, completed=True, **errs) -> dict:
    return {**{c: "" for c in wrap.REPORT_COLUMNS}, "out_dir": str(out_dir),
            "completed": completed, **errs}


def test_session_summary_one_file_bool_errors_list_mistakes(tmp_path):
    import csv
    dest = tmp_path / "out"
    ep1 = _ep_out(dest, "2026-08-04_10_1", {
        "date": "2026-08-04", "episode_code": "10", "take": 1,
        "start_time_ns": 100, "duration_s": 1.0, "mistakes": ["m1", "m3"]})
    ep2 = _ep_out(dest, "2026-08-04_20_1", {
        "date": "2026-08-04", "episode_code": "20", "take": 1,
        "start_time_ns": 200, "duration_s": 2.0, "mistakes": []})
    rows = [_row(ep2),
            _row(ep1, color_error="cam_ego: 73% frames")]

    path = wrap.write_session_summary(rows, dest)

    # ONE file, in the session dir, named session-summary.csv; no stray index.csv
    assert path == dest / "session-summary.csv"
    assert not list(dest.glob("index.csv"))
    recs = list(csv.DictReader(path.read_text().splitlines()))
    assert list(recs[0]) == wrap.SUMMARY_COLUMNS               # exact column set + order
    assert [r["index"] for r in recs] == ["1", "2"]            # sorted by start_time_ns
    assert recs[0]["episode_code"] == "10" and recs[0]["take"] == "1"
    assert recs[0]["start_time_ns"] == "100" and recs[0]["duration_s"] == "1.0"
    # errors are booleans (True = tripped)
    assert recs[0]["color_error"] == "True" and recs[1]["color_error"] == "False"
    assert recs[0]["depth_error"] == "False" and recs[0]["completed"] == "True"
    # mistakes as a single-cell list literal
    assert recs[0]["mistakes"] == "['m1','m3']" and recs[1]["mistakes"] == "[]"
    assert recs[0]["out_dir"] == str(ep1)


def test_session_summary_crashed_episode_numbered_after_timed(tmp_path):
    # Two timed episodes + one that crashed before episode_details (no start_time_ns).
    # The crashed row must still get an index, numbered AFTER the timed ones (3).
    import csv
    dest = tmp_path / "out"
    ep1 = _ep_out(dest, "2026-08-04_10_1", {
        "date": "2026-08-04", "episode_code": "10", "take": 1,
        "start_time_ns": 100, "duration_s": 1.0, "mistakes": []})
    ep2 = _ep_out(dest, "2026-08-04_20_1", {
        "date": "2026-08-04", "episode_code": "20", "take": 1,
        "start_time_ns": 200, "duration_s": 2.0, "mistakes": []})
    crashed = _ep_out(dest, "2026-08-04_30_1", None)         # no episode_details -> no start_time_ns
    rows = [_row(crashed, completed=False), _row(ep2), _row(ep1)]

    path = wrap.write_session_summary(rows, dest)
    recs = list(csv.DictReader(path.read_text().splitlines()))
    assert [r["index"] for r in recs] == ["1", "2", "3"]     # crashed row appended as 3
    assert recs[2]["out_dir"] == str(crashed) and recs[2]["completed"] == "False"


def test_session_summary_index_follows_start_time_ns(tmp_path):
    # `index` is a report-time ordinal computed HERE from start_time_ns — NOT read from
    # metadata (these fixtures store no episode_index) and NOT the row arrival order.
    import csv
    dest = tmp_path / "out"
    a = _ep_out(dest, "2026-08-04_30_1", {"episode_code": "30", "take": 1,
                "date": "2026-08-04", "start_time_ns": 300, "duration_s": 1.0, "mistakes": []})
    b = _ep_out(dest, "2026-08-04_10_1", {"episode_code": "10", "take": 1,
                "date": "2026-08-04", "start_time_ns": 100, "duration_s": 1.0, "mistakes": []})
    c = _ep_out(dest, "2026-08-05_20_1", {"episode_code": "20", "take": 1,
                "date": "2026-08-05", "start_time_ns": 200, "duration_s": 1.0, "mistakes": []})
    rows = [_row(a), _row(b), _row(c)]                       # arrival order a,b,c = starts 300,100,200

    path = wrap.write_session_summary(rows, dest)
    recs = list(csv.DictReader(path.read_text().splitlines()))
    by_out = {r["out_dir"]: r["index"] for r in recs}
    assert by_out[str(b)] == "1"                             # start 100 -> earliest
    assert by_out[str(c)] == "2"                             # start 200
    assert by_out[str(a)] == "3"                             # start 300 -> latest


def test_session_summary_blank_identity_for_failed_episode(tmp_path):
    import csv
    dest = tmp_path / "out"
    out = _ep_out(dest, "2026-08-04_30_1", None)             # metadata.json, no episode_details
    rows = [_row(out, completed=False, depth_error="aligned h5 not found")]
    path = wrap.write_session_summary(rows, dest)
    rec = list(csv.DictReader(path.read_text().splitlines()))[0]
    # no episode_details (crashed before indexing) -> still gets a fallback index (the
    # only row, so 1) and its out_dir; the other identity cells stay blank.
    assert rec["index"] == "1" and rec["out_dir"] == str(out)
    assert rec["mistakes"] == "[]"
    assert rec["completed"] == "False" and rec["depth_error"] == "True"


def test_session_summary_has_missing_extra_columns(tmp_path):
    import csv
    dest = tmp_path / "out"
    ep = _ep_out(dest, "2026-08-06_test_1", {
        "episode_index": 1, "date": "2026-08-06", "episode_code": "test", "take": 1,
        "start_time_ns": 1, "duration_s": 1.0, "mistakes": []})
    rows = [_row(ep, missing_stream_error="exo: expected 4, found 3", extra_stream_error="")]
    path = wrap.write_session_summary(rows, dest)
    rec = list(csv.DictReader(path.read_text().splitlines()))[0]
    # both new columns exist and collapse to booleans (non-empty signal -> True)
    assert "missing_stream_error" in rec and "extra_stream_error" in rec
    assert rec["missing_stream_error"] == "True"
    assert rec["extra_stream_error"] == "False"


# ===========================================================================
# F. QUIET redirection + Step-6 read robustness
# ===========================================================================
def test_quiet_mode_writes_pipeline_log(tmp_path, wired, capsys):
    """quiet=True: the chatty step detail (the "===== step " markers + any
    subscript stdout) is redirected to <out_dir>/pipeline.log, while the
    wrapper's OWN per-step progress lines still reach the console."""
    bag = make_bag(tmp_path / "run1")
    out = tmp_path / "out"
    row = wrap.run_pipeline_for_bag(bag, out, "ego", {}, quiet=True)

    assert row["completed"] is True

    # the log exists, is non-empty, and carries the per-step header marker
    log_path = out / "pipeline.log"
    assert log_path.is_file()
    log_text = log_path.read_text()
    assert log_text.strip()                       # non-empty
    assert "===== step " in log_text              # step chatter landed in the log

    console = capsys.readouterr().out
    # the wrapper's own progress lines still show on the console …
    # (9 numbered steps: bag integrity, color, depth, imu, episode details, calib merge,
    #  validate color, validate depth, validate imu; sanity + the cosmetic reorder are
    #  post-loop, reported on the ⇒ line, not numbered steps)
    assert "step 1/9" in console
    assert "step 9/9" in console
    # … but the redirected step chatter (the "===== step " markers) did not
    assert "===== step " not in console


def test_corrupt_metadata_not_reported_completed(tmp_path, monkeypatch, wired):
    """All steps 'succeed' (artifacts on disk), but the final metadata.json is left
    corrupt at the moment the always-run report block (extract_signals) reads it.

    Design 2 guarantee: signal-capture sits in its own guarded block AFTER the step
    loop; a metadata.json that cannot be parsed there is a true pipeline error
    (corrupt final artifact) -> it flips `crashed`, writes error.txt, and forces
    completed False. A recorded error must never coexist with completed True.
    """
    def corrupt_val_depth(out_dir):
        wired["calls"].append("val_depth")
        # last writer of the shared metadata.json spine leaves it truncated
        (Path(out_dir) / "metadata.json").write_text("{ not valid json")
    monkeypatch.setattr(wrap.validate_depth_v3, "validate_aligned_depth",
                        corrupt_val_depth)

    bag = make_bag(tmp_path / "run1")
    out = tmp_path / "out"
    row = wrap.run_pipeline_for_bag(bag, out, "ego", {})

    # every step ran; the corruption only bites when the report step reads the json
    assert wired["calls"] == ["bag_integrity", "color", "depth", "imu", "episode_details", "val_color", "val_depth", "val_imu"]
    # extract_signals raised -> the wrapper recorded a pipeline error
    assert (out / "pipeline_error.txt").is_file()
    # DESIRED: a recorded error must not coexist with a completed=True row
    assert row["completed"] is False


# --- exo_calib usability enum (summary column) ------------------------------
# Pure function: recorded exo streams vs which carry a usable calib block -> enum.
def _color(cam):  # a color stream entry
    return {"camera": cam, "kind": "color"}


def _intr(cam, status="ok", K=(1, 2, 3)):  # a camera_intrinsics block
    return {"camera": cam, "color": {"status": status, "K": (list(K) if K is not None else None)}}


def test_exo_calib_full_all_usable():
    assert wrap._exo_calib_status(
        [_color("exo_cam1"), _color("exo_cam2")],
        [_intr("exo_cam1"), _intr("exo_cam2")]) == "full"


def test_exo_calib_partial_broken_or_missing():
    # one ok, one broken (null K)
    assert wrap._exo_calib_status(
        [_color("exo_cam1"), _color("exo_cam2")],
        [_intr("exo_cam1"), _intr("exo_cam2", "broken", None)]) == "partial"
    # one ok, one missing (no intrinsics block at all)
    assert wrap._exo_calib_status(
        [_color("exo_cam1"), _color("exo_cam2")],
        [_intr("exo_cam1")]) == "partial"


def test_exo_calib_none_usable_recorded_but_no_good_calib():
    assert wrap._exo_calib_status(
        [_color("exo_cam1"), _color("exo_cam2")],
        [_intr("exo_cam1", "broken", None)]) == "none_usable"


def test_exo_calib_null_no_exo_cams_recorded():
    # ego is recorded but excluded (realsense is always calibrated) -> null
    assert wrap._exo_calib_status([_color("cam_ego")], [_intr("cam_ego")]) == "null"


# --- calib wiring (e2e) + reorder on disk (I/O) -----------------------------
# These are NOT logic tests (merge/reorder logic is unit-tested in test_pipeline_
# calibration.py / test_pipeline_metadata.py). They prove the PLUMBING: that a real
# calib threads through the wrapper into metadata.json (happy path), that a bad calib
# fails the bag SAFELY without killing the batch, and that the on-disk reorder runs and
# its guard no-ops on a corrupt/absent file.
def _calib_file(path: Path, status="ok") -> Path:
    path.write_text(json.dumps({
        "scene_details": {"board": {"squares_x": 7}},
        "cameras": {"exo_cam1": {
            "intrinsics": {"K": [[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]],
                           "dist": [0, 0, 0, 0, 0], "image_size": [1280, 720]},
            "status": status,
            "extrinsics": {"T_world_cam": [[1, 0, 0, 2.0], [0, 1, 0, 2.0],
                                           [0, 0, 1, 1.9], [0, 0, 0, 1]],
                           "position_m": [2.0, 2.0, 1.9]}}},
        "accuracy": {"cameras_solved": "1/1"},
    }))
    return path


def test_wrapper_threads_real_calib_into_metadata(tmp_path, wired):
    # happy path: a real calib_path reaches the calib step and lands the exo block.
    bag = make_bag(tmp_path / "run1")
    out = tmp_path / "out"
    calib = _calib_file(tmp_path / "calib.json")
    row = wrap.run_pipeline_for_bag(bag, out, "ego", {}, calib_path=str(calib))
    assert row["completed"] is True
    meta = json.loads((out / "metadata.json").read_text())
    assert any(c["camera"] == "exo_cam1" for c in meta["camera_intrinsics"])   # merge ran
    assert any(e["name"] == "exo_cam1_to_world" for e in meta["camera_extrinsics"])


def test_wrapper_bad_calib_fails_bag_safely_not_batch(tmp_path, wired):
    # error path caught SAFELY: a malformed calib (trailing garbage) raises in the step,
    # the wrapper's try/except turns it into a failed bag — no exception escapes.
    bag = make_bag(tmp_path / "run1")
    out = tmp_path / "out"
    calib = tmp_path / "calib.json"
    calib.write_text('{"cameras": {}} trailing garbage')
    row = wrap.run_pipeline_for_bag(bag, out, "ego", {}, calib_path=str(calib))
    assert row["completed"] is False                       # bag failed
    assert (out / "pipeline_error.txt").is_file()          # recorded, batch can continue


def test_reorder_on_disk_happy_path(tmp_path):
    # out-of-order keys -> canonical order after the on-disk pass.
    (tmp_path / "metadata.json").write_text(json.dumps(
        {"steps": {}, "metadata": {"n": "leo"}, "termination": {}}))
    wrap._reorder_metadata_on_disk(tmp_path)
    keys = list(json.loads((tmp_path / "metadata.json").read_text()).keys())
    assert keys == ["metadata", "steps", "termination"]


def test_reorder_on_disk_guard_corrupt_json_noops(tmp_path):
    (tmp_path / "metadata.json").write_text("{ not valid json ")
    wrap._reorder_metadata_on_disk(tmp_path)               # must not raise
    assert (tmp_path / "metadata.json").read_text() == "{ not valid json "   # left as-is


def test_reorder_on_disk_guard_missing_file_noops(tmp_path):
    wrap._reorder_metadata_on_disk(tmp_path / "nope")      # no file -> must not raise
