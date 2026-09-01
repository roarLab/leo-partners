"""Unit tests for validate_depth_v3.py.

Run:  conda activate leo-seg && pytest data-processing-scripts-v3/ -q

Covers the two-way pairing check, the timestamp-gap check, h5 integrity
(count/shape/deep-scan), the idempotent stream annotation, and the merge-safe
"depth" termination token. Everything runs on synthetic h5 + CSV + metadata in
a tmpdir -- no real recordings or bags needed.
"""
import json
import sys
from pathlib import Path

import numpy as np
import h5py
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import validate_depth_v3 as vd                            # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic-output builders: an aligned h5, its timestamps CSV, and a metadata
# stub whose one aligned_depth_to_color stream points at them.
# ---------------------------------------------------------------------------
def write_h5(out_dir, has_depth, hc=6, wc=8, corrupt_frames=()):
    """Write depth_frames/ego_aligned_depth_to_color.h5 (N, hc, wc) uint16.

    Frame i is non-zero iff has_depth[i] (a blank/zero frame otherwise), UNLESS
    i is in corrupt_frames, which flips that frame's zero-ness to force a
    content-vs-flag disagreement (the deep-scan corruption case)."""
    out_dir = Path(out_dir)
    (out_dir / "depth_frames").mkdir(parents=True, exist_ok=True)
    n = len(has_depth)
    data = np.zeros((n, hc, wc), np.uint16)
    for i, flag in enumerate(has_depth):
        nonzero = bool(flag)
        if i in corrupt_frames:
            nonzero = not nonzero
        if nonzero:
            data[i] = 1000
    path = out_dir / "depth_frames" / "ego_aligned_depth_to_color.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data)
    return path


def write_csv(out_dir, has_depth, period=1.0 / 30, gap_at=None, gap=0.5):
    """Write timestamps/ego_aligned_depth_to_color.csv (color-indexed).

    Columns: index,color_stamp_s,depth_stamp_s,pair_dt_ms,has_depth. Paired rows
    (has_depth==1) carry a depth_stamp_s on a regular cadence; a large gap can be
    injected before the paired row whose index==gap_at."""
    out_dir = Path(out_dir)
    (out_dir / "timestamps").mkdir(parents=True, exist_ok=True)
    path = out_dir / "timestamps" / "ego_aligned_depth_to_color.csv"
    lines = ["index,color_stamp_s,depth_stamp_s,pair_dt_ms,has_depth"]
    t = 0.0
    for i, flag in enumerate(has_depth):
        cstamp = i * period
        if flag:
            step = gap if (gap_at is not None and i == gap_at) else period
            t += step
            lines.append(f"{i},{cstamp:.9f},{t:.9f},0.500,1")
        else:
            lines.append(f"{i},{cstamp:.9f},,,0")
    path.write_text("\n".join(lines) + "\n")
    return path


def write_metadata(out_dir, *, has_depth, hc=6, wc=8, n_depth_frames=None,
                   n_pair_missing_color=None, extra_reason=None, with_stream=True):
    """metadata.json whose aligned stream points at the h5/CSV above. n_paired /
    h5_index / n_pair_missing_depth are derived from has_depth."""
    out_dir = Path(out_dir)
    n = len(has_depth)
    n_paired = int(sum(has_depth))
    stream = {
        "camera": "cam_ego",
        "kind": "aligned_depth_to_color",
        "topic": "/ego/d435i_ego/depth/image_rect_raw",
        "frames_dir": "depth_frames/ego_aligned_depth_to_color.h5",
        "frame_dtype": "uint16",
        "depth_unit": "mm",
        "width": wc,
        "height": hc,
        "timestamps": "timestamps/ego_aligned_depth_to_color.csv",
        "fps_estimate": 30.0,
        "h5_index": n,
        "n_depth_frames": n if n_depth_frames is None else n_depth_frames,
        "n_paired": n_paired,
        "n_pair_missing_depth": n - n_paired,
        "n_pair_missing_color": (0 if n_pair_missing_color is None else n_pair_missing_color),
    }
    streams = [{"camera": "cam1", "kind": "color", "num_frames": n}]
    if with_stream:
        streams.append(stream)
    meta = {
        "metadata": {"dataset_name": "leo"},
        "camera_intrinsics": [{"camera": "cam_ego", "color": {}}],
        "steps": {"streams": streams, "timestamp_range": [0.0, 1.0]},
        "termination": {"is_successful": extra_reason is None,
                        "reason": list(extra_reason or [])},
    }
    path = out_dir / "metadata.json"
    path.write_text(json.dumps(meta, indent=2))
    return path


def load_meta(out_dir):
    return json.loads((Path(out_dir) / "metadata.json").read_text())


def aligned(meta):
    return next(s for s in meta["steps"]["streams"]
               if s["kind"] == "aligned_depth_to_color")


# ===========================================================================
# A. check_timestamps
# ===========================================================================
def build_ts_df(out_dir, has_depth, gap_at=None, gap=1.0):
    import pandas as pd
    write_csv(out_dir, has_depth, gap_at=gap_at, gap=gap)
    return pd.read_csv(out_dir / "timestamps" / "ego_aligned_depth_to_color.csv")


def test_timestamps_clean_has_no_errors(tmp_path):
    df = build_ts_df(tmp_path, [1] * 8)
    assert vd.check_timestamps(df, 5.0) == []


def test_timestamps_gap_flagged_at_frame_index(tmp_path):
    # a big depth-stamp gap before frame 5 -> one error naming that frame.
    df = build_ts_df(tmp_path, [1] * 8, gap_at=5)
    errors = vd.check_timestamps(df, 5.0)
    assert len(errors) == 1
    assert errors[0].startswith("Frame 5 ")


def test_timestamps_missing_columns_guarded():
    import pandas as pd
    df = pd.DataFrame({"foo": [1, 2, 3]})
    errors = vd.check_timestamps(df, 5.0)
    assert len(errors) == 1 and "missing expected columns" in errors[0]


def test_timestamps_too_few_paired_no_error(tmp_path):
    df = build_ts_df(tmp_path, [0, 1, 0, 0])   # only one paired row
    assert vd.check_timestamps(df, 5.0) == []


# ===========================================================================
# B. check_pairing (two-way)
# ===========================================================================
def test_pairing_color_side_over_threshold(tmp_path):
    import pandas as pd
    # 3/10 color frames blank -> 30% > 10%
    has = [1] * 7 + [0] * 3
    df = pd.DataFrame({"has_depth": has})
    stream = {"n_depth_frames": 10, "n_pair_missing_color": 0}
    errors = vd.check_pairing(stream, df, 0.10)
    assert any("color→depth" in e and "30.0%" in e for e in errors)
    assert not any("depth→color" in e for e in errors)


def test_pairing_depth_side_over_threshold(tmp_path):
    import pandas as pd
    df = pd.DataFrame({"has_depth": [1] * 10})           # color side clean
    stream = {"n_depth_frames": 100, "n_pair_missing_color": 20}  # 20% depth-side
    errors = vd.check_pairing(stream, df, 0.10)
    assert any("depth→color" in e and "20.0%" in e for e in errors)
    assert not any("color→depth" in e for e in errors)


def test_pairing_both_sides(tmp_path):
    import pandas as pd
    df = pd.DataFrame({"has_depth": [1] * 8 + [0] * 2})  # 20% color side
    stream = {"n_depth_frames": 100, "n_pair_missing_color": 15}  # 15% depth side
    errors = vd.check_pairing(stream, df, 0.10)
    assert any("color→depth" in e for e in errors)
    assert any("depth→color" in e for e in errors)


def test_pairing_symmetric_skew_still_flagged(tmp_path):
    # THE key case: equal totals (10 color, 10 depth) but 2 dropped each way.
    # Count subtraction would say 0; the true per-direction counts must flag both.
    import pandas as pd
    df = pd.DataFrame({"has_depth": [1] * 8 + [0] * 2})  # 2 color unpaired
    stream = {"n_depth_frames": 10, "n_pair_missing_color": 2}  # 2 depth unpaired
    errors = vd.check_pairing(stream, df, 0.10)
    assert any("color→depth" in e for e in errors)
    assert any("depth→color" in e for e in errors)


def test_pairing_clean_under_threshold(tmp_path):
    import pandas as pd
    df = pd.DataFrame({"has_depth": [1] * 99 + [0]})     # 1% color side
    stream = {"n_depth_frames": 100, "n_pair_missing_color": 1}  # 1% depth side
    assert vd.check_pairing(stream, df, 0.10) == []


def test_pairing_missing_persisted_counts_skips_depth_side(tmp_path, capsys):
    # No n_pair_missing_color -> depth side skipped with a note; color side still runs.
    import pandas as pd
    df = pd.DataFrame({"has_depth": [1] * 7 + [0] * 3})  # 30% color side
    stream = {}                                           # no persisted counts
    errors = vd.check_pairing(stream, df, 0.10)
    assert any("color→depth" in e for e in errors)
    assert not any("depth→color" in e for e in errors)
    assert "not checked" in capsys.readouterr().out


def test_pairing_messages_start_with_unpaired(tmp_path):
    # Routing contract: wrapper.extract_signals sends a depth data_error to the
    # report's color_depth_mismatch column IFF the message .lower().startswith
    # ("unpaired"), else to depth_error. Drive BOTH directions over threshold
    # (20% color→depth blanks AND 15% depth→color drops via the persisted
    # n_pair_missing_color path) and pin that EVERY pairing message honors it.
    import pandas as pd
    df = pd.DataFrame({"has_depth": [1] * 8 + [0] * 2})  # 2/10 = 20% color side
    stream = {"n_depth_frames": 100, "n_pair_missing_color": 15}  # 15% depth side
    errors = vd.check_pairing(stream, df, 0.10)
    assert any("color→depth" in e for e in errors)       # color side tripped
    assert any("depth→color" in e for e in errors)       # depth side tripped
    assert errors and all(e.lower().startswith("unpaired") for e in errors)


# ===========================================================================
# B2. check_episode_coverage (whole-device early stop) + color_frame_reference
# ===========================================================================
def test_episode_coverage_under_threshold_flags():
    # ego depth reached 482 frames; busiest color (cam1) got 658 -> 73.3% < 90%.
    errors = vd.check_episode_coverage({"h5_index": 482}, (658, "cam1"), 0.90)
    assert len(errors) == 1
    assert "73.3%" in errors[0] and "cam1" in errors[0]
    # must NOT start with "unpaired" -> wrapper routes it to depth_error, not mismatch
    assert not errors[0].lower().startswith("unpaired")


def test_episode_coverage_full_no_error():
    assert vd.check_episode_coverage({"h5_index": 658}, (658, "cam1"), 0.90) == []


def test_episode_coverage_just_under_threshold():
    # 89.9% < 90% trips; 90.0% does not (boundary is strict "<")
    assert vd.check_episode_coverage({"h5_index": 899}, (1000, "cam1"), 0.90)
    assert vd.check_episode_coverage({"h5_index": 900}, (1000, "cam1"), 0.90) == []


def test_episode_coverage_no_reference_skips():
    assert vd.check_episode_coverage({"h5_index": 482}, None, 0.90) == []
    assert vd.check_episode_coverage({"h5_index": 482}, (0, "N/A"), 0.90) == []


def test_episode_coverage_missing_h5_index_skips(capsys):
    assert vd.check_episode_coverage({}, (658, "cam1"), 0.90) == []
    assert "not checked" in capsys.readouterr().out


def test_color_frame_reference_picks_busiest():
    meta = {"steps": {"streams": [
        {"kind": "color", "camera": "cam1", "num_frames": 658, "found": True},
        {"kind": "color", "camera": "cam2", "num_frames": 657, "found": True},
        {"kind": "color", "camera": "cam_ego", "num_frames": 482, "found": True},
        {"kind": "aligned_depth_to_color", "camera": "cam_ego", "h5_index": 482}]}}
    assert vd.color_frame_reference(meta) == (658, "cam1")


def test_color_frame_reference_ignores_unfound_and_noncolor():
    meta = {"steps": {"streams": [
        {"kind": "color", "camera": "cam1", "num_frames": 999, "found": False},
        {"kind": "color", "camera": "cam2", "num_frames": 657, "found": True},
        {"kind": "aligned_depth_to_color", "camera": "cam_ego", "h5_index": 900}]}}
    assert vd.color_frame_reference(meta) == (657, "cam2")


# ===========================================================================
# C. check_data (h5 integrity)
# ===========================================================================
def test_data_clean(tmp_path):
    import pandas as pd
    has = [1] * 5 + [0]
    h5 = write_h5(tmp_path, has)
    df = pd.DataFrame({"has_depth": has})
    assert vd.check_data(h5, df, len(has), 6, 8, True) == []


def test_data_missing_h5(tmp_path):
    import pandas as pd
    errors = vd.check_data(tmp_path / "nope.h5", pd.DataFrame(), 6, 6, 8, True)
    assert len(errors) == 1 and "not found" in errors[0]


def test_data_frame_count_mismatch(tmp_path):
    import pandas as pd
    has = [1] * 6
    h5 = write_h5(tmp_path, has)
    df = pd.DataFrame({"has_depth": has})
    errors = vd.check_data(h5, df, 99, 6, 8, True)       # metadata claims 99
    assert any("!= metadata h5_index 99" in e for e in errors)


def test_data_shape_mismatch(tmp_path):
    import pandas as pd
    has = [1] * 6
    h5 = write_h5(tmp_path, has, hc=6, wc=8)
    df = pd.DataFrame({"has_depth": has})
    errors = vd.check_data(h5, df, 6, 10, 20, True)      # metadata claims 10x20
    assert any("frame shape" in e for e in errors)


def test_data_deep_scan_detects_corruption(tmp_path):
    import pandas as pd
    has = [1] * 5 + [0]
    # frame 2 flipped: CSV says has_depth=1 but the h5 frame is all-zero.
    h5 = write_h5(tmp_path, has, corrupt_frames=(2,))
    df = pd.DataFrame({"has_depth": has})
    errors = vd.check_data(h5, df, len(has), 6, 8, True)
    assert any("disagree between h5 content and" in e for e in errors)


def test_data_deep_scan_off_skips_corruption(tmp_path):
    import pandas as pd
    has = [1] * 5 + [0]
    h5 = write_h5(tmp_path, has, corrupt_frames=(2,))
    df = pd.DataFrame({"has_depth": has})
    assert vd.check_data(h5, df, len(has), 6, 8, False) == []   # deep_scan=False


# ===========================================================================
# D. validate_stream annotation
# ===========================================================================
def test_stream_annotated_on_error_and_returns_true(tmp_path):
    has = [1] * 7 + [0] * 3                               # 30% blank -> pairing error
    write_h5(tmp_path, has)
    write_csv(tmp_path, has)
    stream = aligned(json.loads(write_metadata(tmp_path, has_depth=has).read_text()))
    ts_bad, data_bad = vd.validate_stream(tmp_path, stream)
    assert data_bad is True
    assert "data_error" in stream and stream["data_error"]


def test_stream_clean_clears_stale_annotations(tmp_path):
    has = [1] * 8
    write_h5(tmp_path, has)
    write_csv(tmp_path, has)
    stream = aligned(json.loads(write_metadata(tmp_path, has_depth=has).read_text()))
    stream["data_error"] = ["stale"]                     # left over from a prior run
    stream["timestamps_error"] = ["stale"]
    ts_bad, data_bad = vd.validate_stream(tmp_path, stream)
    assert (ts_bad, data_bad) == (False, False)
    assert "data_error" not in stream and "timestamps_error" not in stream


# ===========================================================================
# E. validate_aligned_depth (entry point) + merge-safe termination
# ===========================================================================
def test_entry_clean_run_success(tmp_path):
    has = [1] * 8
    write_h5(tmp_path, has)
    write_csv(tmp_path, has)
    write_metadata(tmp_path, has_depth=has)
    vd.validate_aligned_depth(tmp_path)
    term = load_meta(tmp_path)["termination"]
    assert term["is_successful"] is True and term["reason"] == []


def test_entry_error_adds_depth_token(tmp_path):
    has = [1] * 6 + [0] * 4                               # 40% blank -> error
    write_h5(tmp_path, has)
    write_csv(tmp_path, has)
    write_metadata(tmp_path, has_depth=has)
    vd.validate_aligned_depth(tmp_path)
    term = load_meta(tmp_path)["termination"]
    assert "depth_data" in term["reason"] and term["is_successful"] is False


def test_entry_episode_coverage_trips_depth_when_internally_clean(tmp_path):
    # The 2026-08-03_rs-error case: depth is internally spotless (every frame paired,
    # no gaps, h5 intact) but the whole RealSense stopped early, so the busiest color
    # ran far longer. Pairing/timestamp/h5 all pass; only episode coverage catches it.
    has = [1] * 8                                        # 8 depth frames, all paired
    write_h5(tmp_path, has)
    write_csv(tmp_path, has)
    write_metadata(tmp_path, has_depth=has)
    meta = load_meta(tmp_path)
    for s in meta["steps"]["streams"]:                   # busiest color ran 20 frames
        if s.get("kind") == "color":
            s["num_frames"] = 20                         # 8/20 = 40% < 90%
    (tmp_path / "metadata.json").write_text(json.dumps(meta, indent=2))
    vd.validate_aligned_depth(tmp_path)
    m = load_meta(tmp_path)
    assert "depth_data" in m["termination"]["reason"]
    data_err = aligned(m).get("data_error", [])
    assert any("depth frames compared to cam1 color" in e for e in data_err)
    # coverage is the ONLY tripped check -> no unpaired/pairing message present
    assert not any(e.lower().startswith("unpaired") for e in data_err)


def test_entry_preserves_other_reasons(tmp_path):
    has = [1] * 6 + [0] * 4                               # depth error
    write_h5(tmp_path, has)
    write_csv(tmp_path, has)
    write_metadata(tmp_path, has_depth=has,
                   extra_reason=["color_data", "color_timestamps", "depth_presence_err"])
    vd.validate_aligned_depth(tmp_path)
    reason = load_meta(tmp_path)["termination"]["reason"]
    assert "color_data" in reason and "color_timestamps" in reason and "depth_data" in reason
    # extraction's presence token (data-plane miss/extra) is NOT owned by this quality
    # validator (owns only depth_data / depth_timestamps), so it is preserved, never stripped.
    assert "depth_presence_err" in reason


def test_entry_clears_stale_depth_on_clean_rerun(tmp_path):
    has = [1] * 8                                         # clean
    write_h5(tmp_path, has)
    write_csv(tmp_path, has)
    # a prior run had left "depth" (and an unrelated "color") in termination
    write_metadata(tmp_path, has_depth=has, extra_reason=["color", "depth"])
    vd.validate_aligned_depth(tmp_path)
    reason = load_meta(tmp_path)["termination"]["reason"]
    assert "depth" not in reason          # stale depth cleared
    assert "color" in reason              # unrelated reason preserved


def test_entry_no_aligned_stream_is_noop(tmp_path):
    write_metadata(tmp_path, has_depth=[1] * 4, with_stream=False)
    before = (tmp_path / "metadata.json").read_text()
    vd.validate_aligned_depth(tmp_path)
    assert (tmp_path / "metadata.json").read_text() == before   # untouched


def test_extra_topic_presence_token_survives_clean_validation(tmp_path):
    # The clobber scenario the presence token exists to prevent. A SURPLUS depth topic was
    # extracted (extraction flagged depth_presence_err) AND the extracted stream is otherwise
    # CLEAN. A stream EXISTS, so validate_depth does NOT early-return — it runs and re-derives
    # its own token (no depth_data: data is clean). Because depth_presence_err is not the
    # validator's to own, it must SURVIVE and keep is_successful False. Were presence folded
    # into depth_data, the clean-data re-derive would strip it and flip the verdict to true.
    has = [1] * 8                                        # clean: every color frame paired
    write_h5(tmp_path, has)
    write_csv(tmp_path, has)
    write_metadata(tmp_path, has_depth=has, extra_reason=["depth_presence_err"])
    vd.validate_aligned_depth(tmp_path)
    meta = load_meta(tmp_path)
    assert "depth_presence_err" in meta["termination"]["reason"]   # presence flag survives
    assert "depth_data" not in meta["termination"]["reason"]       # clean -> no quality token
    assert meta["termination"]["is_successful"] is False           # verdict not clobbered


def test_missing_topic_presence_token_survives_noop(tmp_path):
    # The missing arm: extraction aborted (no aligned stream) and flagged depth_presence_err.
    # validate_depth early-returns (nothing to validate), so the flag is preserved by the
    # no-op — doubly safe vs the extra arm, which relies on token disjointness instead.
    write_metadata(tmp_path, has_depth=[1] * 4, with_stream=False,
                   extra_reason=["depth_presence_err"])
    vd.validate_aligned_depth(tmp_path)
    meta = load_meta(tmp_path)
    assert "depth_presence_err" in meta["termination"]["reason"]
    assert meta["termination"]["is_successful"] is False


def test_entry_missing_metadata_no_crash(tmp_path):
    vd.validate_aligned_depth(tmp_path)                  # nothing written -> returns cleanly
    assert not (tmp_path / "metadata.json").exists()


def test_entry_idempotent(tmp_path):
    has = [1] * 6 + [0] * 4
    write_h5(tmp_path, has)
    write_csv(tmp_path, has)
    write_metadata(tmp_path, has_depth=has)
    vd.validate_aligned_depth(tmp_path)
    first = load_meta(tmp_path)["termination"]["reason"]
    vd.validate_aligned_depth(tmp_path)                  # re-run
    second = load_meta(tmp_path)["termination"]["reason"]
    assert first == second == ["depth_data"]


# ===========================================================================
# F. helpers
# ===========================================================================
def test_load_json_with_comments_strips_line_comments(tmp_path):
    p = tmp_path / "m.json"
    p.write_text('{\n  "a": 1, // trailing comment\n  // full line\n  "b": 2\n}\n')
    assert vd.load_json_with_comments(p) == {"a": 1, "b": 2}


def test_find_aligned_streams(tmp_path):
    meta = {"steps": {"streams": [
        {"kind": "color"}, {"kind": "aligned_depth_to_color", "camera": "cam_ego"}]}}
    found = vd.find_aligned_streams(meta)
    assert len(found) == 1 and found[0]["camera"] == "cam_ego"
