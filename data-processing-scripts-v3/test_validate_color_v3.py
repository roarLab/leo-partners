"""Unit tests for validate_color_v3.validate_metadata and load_json_with_comments."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import validate_color_v3 as v


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def make_color_stream(camera, num_frames, **extra):
    """Build a single color stream dict."""
    stream = {
        "camera": camera,
        "kind": "color",
        "num_frames": num_frames,
        "found": True,
    }
    stream.update(extra)
    return stream


def write_metadata(tmp_path, streams):
    """Write a metadata.json with the given streams and return its path."""
    meta = {"steps": {"streams": streams}}
    metadata_path = tmp_path / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return metadata_path


def load_metadata(tmp_path):
    """Reload the rewritten metadata.json."""
    with open(tmp_path / "metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)


def write_timestamps(tmp_path, name, periods, start=1000.0):
    """Write timestamps/<name>.csv from a list of inter-frame periods.

    Returns the relative path string to store in the stream["timestamps"] field.
    """
    ts_dir = tmp_path / "timestamps"
    ts_dir.mkdir(exist_ok=True)
    times = [start]
    for p in periods:
        times.append(times[-1] + p)
    lines = ["index,ros_time_s"]
    for idx, t in enumerate(times):
        lines.append(f"{idx},{t:.6f}")
    csv_path = ts_dir / f"{name}.csv"
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return f"timestamps/{name}.csv"


# ----------------------------------------------------------------------
# load_json_with_comments
# ----------------------------------------------------------------------
def test_load_json_with_comments_strips_line_comments(tmp_path):
    raw = (
        "{\n"
        "  // leading comment\n"
        "  \"a\": 1,\n"
        "  \"b\": 2  // trailing comment\n"
        "}\n"
    )
    p = tmp_path / "commented.json"
    p.write_text(raw, encoding="utf-8")
    result = v.load_json_with_comments(p)
    assert result == {"a": 1, "b": 2}


# ----------------------------------------------------------------------
# Missing metadata.json
# ----------------------------------------------------------------------
def test_missing_metadata_returns_without_raising(tmp_path, capsys):
    # No metadata.json in tmp_path.
    v.validate_metadata(tmp_path)
    captured = capsys.readouterr()
    assert "[ERROR]" in captured.out
    assert "metadata.json not found" in captured.out


# ----------------------------------------------------------------------
# Clean input
# ----------------------------------------------------------------------
def test_clean_input_is_successful(tmp_path):
    streams = [
        make_color_stream("cam_a", 100),
        make_color_stream("cam_b", 100),
    ]
    write_metadata(tmp_path, streams)
    v.validate_metadata(tmp_path)
    meta = load_metadata(tmp_path)
    assert meta["termination"]["is_successful"] is True
    assert meta["termination"]["reason"] == []
    for s in meta["steps"]["streams"]:
        assert "data_error" not in s
        assert "timestamps_error" not in s


# ----------------------------------------------------------------------
# Rule 1: color
# ----------------------------------------------------------------------
def test_color_error_below_threshold(tmp_path):
    # cam_b has 80/100 = 80% < 90% -> flagged.
    streams = [
        make_color_stream("cam_a", 100),
        make_color_stream("cam_b", 80),
    ]
    write_metadata(tmp_path, streams)
    v.validate_metadata(tmp_path)
    meta = load_metadata(tmp_path)
    by_cam = {s["camera"]: s for s in meta["steps"]["streams"]}
    assert "data_error" not in by_cam["cam_a"]
    assert "data_error" in by_cam["cam_b"]
    assert meta["termination"]["is_successful"] is False
    assert meta["termination"]["reason"] == ["color_data"]


def test_all_streams_empty_flagged_by_absolute_floor(tmp_path):
    # Every colour stream has 0 frames -> max_frames is 0, so the RELATIVE < 90% check is
    # blind. The absolute 0-floor must still flag each stream as empty (color_data).
    streams = [make_color_stream("cam_a", 0), make_color_stream("cam_b", 0)]
    write_metadata(tmp_path, streams)
    v.validate_metadata(tmp_path)
    meta = load_metadata(tmp_path)
    by_cam = {s["camera"]: s for s in meta["steps"]["streams"]}
    assert any("empty" in e for e in by_cam["cam_a"]["data_error"])
    assert any("empty" in e for e in by_cam["cam_b"]["data_error"])
    assert meta["termination"]["reason"] == ["color_data"]
    assert meta["termination"]["is_successful"] is False


def test_single_empty_stream_flagged_empty_not_percentage(tmp_path):
    # cam_b empty while cam_a has frames: the 0-floor fires with the "empty" message
    # (not a "0.0% frames" relative message).
    streams = [make_color_stream("cam_a", 100), make_color_stream("cam_b", 0)]
    write_metadata(tmp_path, streams)
    v.validate_metadata(tmp_path)
    by_cam = {s["camera"]: s for s in load_metadata(tmp_path)["steps"]["streams"]}
    assert "data_error" not in by_cam["cam_a"]
    assert any("empty" in e for e in by_cam["cam_b"]["data_error"])


def test_color_exactly_at_threshold_not_flagged(tmp_path):
    # 90/100 = 90% is NOT < 90% -> not flagged.
    streams = [
        make_color_stream("cam_a", 100),
        make_color_stream("cam_b", 90),
    ]
    write_metadata(tmp_path, streams)
    v.validate_metadata(tmp_path)
    meta = load_metadata(tmp_path)
    by_cam = {s["camera"]: s for s in meta["steps"]["streams"]}
    assert "data_error" not in by_cam["cam_b"]
    assert meta["termination"]["reason"] == []


def test_found_false_color_stream_excluded_from_max(tmp_path):
    # cam_big found==False -> still excluded from the max computation (so cam_a/cam_b
    # at 100 are the reference), but found==False is NO LONGER treated as corruption
    # here — a missing stream is the extraction script's missing_stream_error, not
    # validate_color's rosbag_corruption.
    streams = [
        make_color_stream("cam_big", 1000, found=False),
        make_color_stream("cam_a", 100),
        make_color_stream("cam_b", 100),
    ]
    write_metadata(tmp_path, streams)
    v.validate_metadata(tmp_path)
    meta = load_metadata(tmp_path)
    by_cam = {s["camera"]: s for s in meta["steps"]["streams"]}
    # cam_a and cam_b are 100/100, not flagged for color.
    assert "data_error" not in by_cam["cam_a"]
    assert "data_error" not in by_cam["cam_b"]
    # found==False no longer contributes rosbag_corruption.
    assert "rosbag_corruption" not in meta["termination"]["reason"]
    assert meta["termination"]["reason"] == []


# ----------------------------------------------------------------------
# Rule 3: undecodable colour frames -> color_data (colour QUALITY, NOT bag corruption)
# ----------------------------------------------------------------------
def test_decode_failures_flag_color_data(tmp_path):
    # Undecodable colour frames are a colour DATA-quality problem -> color_data, folded in
    # with frame-loss. Structural bag corruption is a SEPARATE token (rosbag_corruption)
    # owned by bag_integrity, which validate_color never emits.
    streams = [
        make_color_stream("cam_a", 100),
        make_color_stream("cam_b", 100, decode_failures=3),
    ]
    write_metadata(tmp_path, streams)
    v.validate_metadata(tmp_path)
    meta = load_metadata(tmp_path)
    by_cam = {s["camera"]: s for s in meta["steps"]["streams"]}
    assert by_cam["cam_b"]["data_error"] == ["3 frame(s) failed to decode"]
    assert meta["termination"]["reason"] == ["color_data"]
    assert "rosbag_corruption" not in meta["termination"]["reason"]  # not validate_color's token
    assert meta["termination"]["is_successful"] is False


def test_found_false_no_longer_triggers_rosbag_corruption(tmp_path):
    # A missing stream (found==False) is owned by the extraction script's
    # missing_stream_error, NOT re-classified as corruption by validate_color.
    streams = [
        make_color_stream("cam_a", 100),
        make_color_stream("cam_b", 100, found=False),
    ]
    write_metadata(tmp_path, streams)
    v.validate_metadata(tmp_path)
    meta = load_metadata(tmp_path)
    assert meta["termination"]["reason"] == []


# ----------------------------------------------------------------------
# Rule 2: timestamps
# ----------------------------------------------------------------------
def test_timestamp_gap_flags_color_stream(tmp_path):
    # Many ~0.033s periods and one large 0.5s gap that exceeds 5x mean.
    periods = [0.033] * 20 + [0.5] + [0.033] * 20
    rel = write_timestamps(tmp_path, "cam_a", periods)
    streams = [make_color_stream("cam_a", 100, timestamps=rel)]
    write_metadata(tmp_path, streams)
    v.validate_metadata(tmp_path)
    meta = load_metadata(tmp_path)
    s = meta["steps"]["streams"][0]
    assert "timestamps_error" in s
    assert meta["termination"]["reason"] == ["color_timestamps"]
    assert meta["termination"]["is_successful"] is False


def test_non_color_stream_timestamps_are_ignored(tmp_path):
    # validate_color now owns COLOUR timestamps only: a non-colour stream (depth here)
    # with a timestamp gap is left entirely untouched — no annotation, no token. Depth
    # timestamps are validate_depth_v3's job (depth_timestamps), imu is validate_imu's.
    periods = [0.033] * 20 + [0.5] + [0.033] * 20
    rel = write_timestamps(tmp_path, "depth_cam", periods)
    depth_stream = {
        "camera": "depth_cam",
        "kind": "depth",
        "num_frames": 100,
        "found": True,
        "timestamps": rel,
    }
    streams = [make_color_stream("cam_a", 100), depth_stream]
    write_metadata(tmp_path, streams)
    v.validate_metadata(tmp_path)
    meta = load_metadata(tmp_path)
    assert meta["termination"]["reason"] == []            # colour clean, depth ignored
    ds = next(s for s in meta["steps"]["streams"] if s["kind"] == "depth")
    assert "timestamps_error" not in ds and "timestamp_error" not in ds


def test_uniform_timestamps_no_error(tmp_path):
    periods = [0.033] * 30
    rel = write_timestamps(tmp_path, "cam_a", periods)
    streams = [make_color_stream("cam_a", 100, timestamps=rel)]
    write_metadata(tmp_path, streams)
    v.validate_metadata(tmp_path)
    meta = load_metadata(tmp_path)
    s = meta["steps"]["streams"][0]
    assert "timestamps_error" not in s
    assert meta["termination"]["reason"] == []


def test_missing_timestamp_file_ignored(tmp_path):
    # Stream references a timestamps file that does not exist.
    streams = [make_color_stream("cam_a", 100, timestamps="timestamps/nope.csv")]
    write_metadata(tmp_path, streams)
    v.validate_metadata(tmp_path)
    meta = load_metadata(tmp_path)
    s = meta["steps"]["streams"][0]
    assert "timestamps_error" not in s
    assert meta["termination"]["reason"] == []


# ----------------------------------------------------------------------
# Reason ordering / rebuild-from-scratch
# ----------------------------------------------------------------------
def test_reason_order_owned_categories(tmp_path):
    # Trigger validate_color's TWO owned colour tokens together (color_data, color_timestamps).
    # Frame-loss AND decode failures both fold into the single color_data token; the timestamp
    # gap adds color_timestamps. rosbag_corruption is bag_integrity's, never emitted here.
    color_periods = [0.033] * 10 + [0.5] + [0.033] * 10
    color_rel = write_timestamps(tmp_path, "cam_b", color_periods)
    streams = [
        make_color_stream("cam_a", 100),
        # cam_b: color loss (70%) + timestamp gap + decode failures -> color_data + color_timestamps.
        make_color_stream(
            "cam_b", 70, timestamps=color_rel, decode_failures=1
        ),
    ]
    write_metadata(tmp_path, streams)
    v.validate_metadata(tmp_path)
    meta = load_metadata(tmp_path)
    assert meta["termination"]["reason"] == [
        "color_data",
        "color_timestamps",
    ]
    assert meta["termination"]["is_successful"] is False


def test_termination_owned_tokens_recomputed_foreign_preserved(tmp_path):
    # validate_color OWNS only {color_data, color_timestamps}: it clears its own stale tokens
    # on a clean re-run, but PRESERVES any reason another writer set (here depth, owned by
    # validate_depth). No wholesale rewrite.
    streams = [
        make_color_stream("cam_a", 100),
        make_color_stream("cam_b", 100),
    ]
    meta = {
        "steps": {"streams": streams},
        "termination": {
            "is_successful": False,
            "reason": ["color_data", "depth"],   # stale own token + a foreign one
        },
    }
    metadata_path = tmp_path / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    v.validate_metadata(tmp_path)
    result = load_metadata(tmp_path)
    # clean color data -> the OWN "color_data" token is dropped; the foreign
    # "depth" survives; is_successful reflects the surviving reason.
    assert result["termination"]["reason"] == ["depth"]
    assert result["termination"]["is_successful"] is False


def test_extraction_presence_token_survives_validate(tmp_path):
    # REGRESSION — missing-stream presence-erasure bug. A declared-but-unrecorded color cam
    # has NO stream entry, so extraction records the miss as `color_presence_err` in
    # termination. validate_color must NOT erase it: presence (color_presence_err, owned by
    # extraction) and per-frame quality (color_data, owned here) are DISJOINT tokens. The
    # recorded streams are healthy, so validate adds no quality token — but the presence flag
    # must survive and is_successful must stay False. (Before the fix, extraction wrote
    # `color_data`, which validate owned and stripped -> is_successful wrongly flipped True.)
    streams = [
        make_color_stream("exo_cam1", 100),
        make_color_stream("exo_cam2", 100),
        make_color_stream("exo_cam3", 100),
        # exo_cam4 declared-but-missing -> no stream entry (extraction's domain, not here)
    ]
    meta = {
        "steps": {
            "streams": streams,
            "missing_stream_error": ["exo_cam: declared [1,2,3,4], missing [4]"],
        },
        "termination": {"is_successful": False, "reason": ["color_presence_err"]},
    }
    metadata_path = tmp_path / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    v.validate_metadata(tmp_path)
    result = load_metadata(tmp_path)
    assert result["termination"]["reason"] == ["color_presence_err"]   # preserved, not erased
    assert result["termination"]["is_successful"] is False


# ----------------------------------------------------------------------
# Stale stream-level annotations must be cleared on a clean re-run
# ----------------------------------------------------------------------
def test_stale_stream_annotations_cleared_on_clean_rerun(tmp_path):
    # A prior dirty run left data_error + timestamps_error on cam_b, but the data
    # is now CLEAN: cam_b sits at the 100% reference, has no decode failures, and
    # its timestamps are uniform. termination.reason is rebuilt to [] (already
    # works), but wrapper.extract_signals reads the stream-level data_error /
    # timestamps_error fields DIRECTLY -- so a clean re-run must CLEAR the stale
    # annotations, mirroring validate_depth_v3.validate_stream's .pop(). This
    # asserts that desired idempotent behavior.
    periods = [0.033] * 30
    rel = write_timestamps(tmp_path, "cam_b", periods)
    streams = [
        make_color_stream("cam_a", 100),
        make_color_stream(
            "cam_b",
            100,
            timestamps=rel,
            data_error=["stale: 42.0% frames compared to cam_a as 100%"],
            timestamps_error=["stale: Index 7 period 0.90s > mean period 0.03s"],
        ),
    ]
    write_metadata(tmp_path, streams)
    v.validate_metadata(tmp_path)
    meta = load_metadata(tmp_path)

    by_cam = {s["camera"]: s for s in meta["steps"]["streams"]}
    # Clean data -> empty reason (this part already works today).
    assert meta["termination"]["reason"] == []
    assert meta["termination"]["is_successful"] is True
    # Desired idempotent behavior: the stale annotations are gone on the clean run.
    assert not by_cam["cam_b"].get("data_error")
    assert not by_cam["cam_b"].get("timestamps_error")
