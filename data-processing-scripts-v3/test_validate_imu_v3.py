"""Unit tests for validate_imu_v3.validate_imu.

Synthetic metadata.json + imu/*.csv in tmp_path — no real bags. Mirrors the colour
and depth validator suites: a timestamp gap raises imu_timestamps, a bad mean rate
raises imu_data, and the merge is owner-scoped (only imu_data / imu_timestamps are
this validator's to add or clear)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import validate_imu_v3 as vi


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
IMU_CSV_REL = "imu/cam_ego_imu.csv"


def write_imu_csv(tmp_path: Path, times) -> None:
    """Write imu/cam_ego_imu.csv with a ros_time_s column (the only column the
    validator reads); the motion columns are filler."""
    (tmp_path / "imu").mkdir(parents=True, exist_ok=True)
    n = len(times)
    pd.DataFrame({
        "index": np.arange(n),
        "ros_time_s": np.asarray(times, dtype=np.float64),
        "color_frame_index": np.zeros(n, dtype=np.int64),
        "wx": np.zeros(n), "wy": np.zeros(n), "wz": np.zeros(n),
        "ax": np.zeros(n), "ay": np.zeros(n), "az": np.zeros(n),
    }).to_csv(tmp_path / IMU_CSV_REL, index=False)


def write_meta(tmp_path: Path, *, file_rel=IMU_CSV_REL, extra_reason=None,
               with_stream=True) -> None:
    streams = []
    if with_stream:
        streams.append({"camera": "cam_ego", "kind": "imu", "file": file_rel,
                        "num_samples": 100, "found": True})
    meta = {
        "steps": {"streams": streams},
        "termination": {"is_successful": extra_reason is None,
                        "reason": list(extra_reason or [])},
    }
    (tmp_path / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_meta(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "metadata.json").read_text())


def imu_stream(meta: dict) -> dict:
    return next(s for s in meta["steps"]["streams"] if s["kind"] == "imu")


# Fixtures are built RELATIVE to the operational constant vi.EXPECTED_IMU_HZ, never a
# hardcoded rate, so the suite is invariant to whatever value the rig/config sets: a
# "clean" stream is sampled at exactly that rate, a "wrong" stream at half it (-50%,
# well outside IMU_RATE_TOLERANCE). Changing EXPECTED_IMU_HZ must NOT break these tests —
# only a logic change should. (A test runs in a clean env; it should not encode an
# operational default.)
PERIOD = 1.0 / vi.EXPECTED_IMU_HZ          # seconds/sample at the expected rate -> clean
HALF_RATE_PERIOD = 2.0 * PERIOD            # half the expected rate -> 50% below -> imu_data

# clean: 100 samples at exactly EXPECTED_IMU_HZ, no gaps
CLEAN_TIMES = (np.arange(100) * PERIOD).tolist()


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------
def test_clean_stream_no_tokens(tmp_path):
    write_imu_csv(tmp_path, CLEAN_TIMES)
    write_meta(tmp_path)
    vi.validate_imu(tmp_path)
    meta = load_meta(tmp_path)
    assert meta["termination"]["reason"] == []
    assert meta["termination"]["is_successful"] is True
    s = imu_stream(meta)
    assert "data_error" not in s and "timestamps_error" not in s


# ---------------------------------------------------------------------------
# timestamp gap -> imu_timestamps
# ---------------------------------------------------------------------------
def test_timestamp_gap_flags_imu_timestamps(tmp_path):
    times = list(np.arange(50) * PERIOD)            # expected cadence …
    times += [times[-1] + 10 * PERIOD]              # … then one 10x-mean gap (ts flag)
    times += list(times[-1] + np.arange(1, 50) * PERIOD)
    write_imu_csv(tmp_path, times)
    write_meta(tmp_path)
    vi.validate_imu(tmp_path)
    meta = load_meta(tmp_path)
    assert "imu_timestamps" in meta["termination"]["reason"]
    assert "imu_data" not in meta["termination"]["reason"]   # rate stays within band
    assert imu_stream(meta).get("timestamps_error")


# ---------------------------------------------------------------------------
# bad mean rate -> imu_data
# ---------------------------------------------------------------------------
def test_wrong_mean_rate_flags_imu_data(tmp_path):
    write_imu_csv(tmp_path, (np.arange(100) * HALF_RATE_PERIOD).tolist())   # half rate -> 50% below
    write_meta(tmp_path)
    vi.validate_imu(tmp_path)
    meta = load_meta(tmp_path)
    assert "imu_data" in meta["termination"]["reason"]
    assert "imu_timestamps" not in meta["termination"]["reason"]  # uniform -> no gap
    assert imu_stream(meta).get("data_error")


def test_missing_csv_flags_imu_data(tmp_path):
    write_meta(tmp_path, file_rel="imu/nope.csv")   # stream claims a CSV that is absent
    vi.validate_imu(tmp_path)
    meta = load_meta(tmp_path)
    assert "imu_data" in meta["termination"]["reason"]
    assert any("missing or unreadable" in e for e in imu_stream(meta)["data_error"])


# ---------------------------------------------------------------------------
# entry-point / merge behaviour
# ---------------------------------------------------------------------------
def test_no_imu_stream_is_noop(tmp_path):
    write_meta(tmp_path, with_stream=False)
    before = (tmp_path / "metadata.json").read_text()
    vi.validate_imu(tmp_path)
    assert (tmp_path / "metadata.json").read_text() == before   # untouched


def test_extra_topic_presence_token_survives_clean_validation(tmp_path):
    # The clobber scenario the presence token exists to prevent. A SURPLUS /imu topic was
    # extracted (extraction flagged imu_presence_err) AND the extracted samples are CLEAN.
    # A stream EXISTS, so validate_imu does NOT no-op — it runs and re-derives its own token
    # (no imu_data: rate is on-band). Because imu_presence_err is not the validator's to own,
    # it must SURVIVE and keep is_successful False. A shared token would be stripped on clean
    # data, flipping the verdict to true.
    write_imu_csv(tmp_path, CLEAN_TIMES)
    write_meta(tmp_path, extra_reason=["imu_presence_err"])
    vi.validate_imu(tmp_path)
    meta = load_meta(tmp_path)
    assert "imu_presence_err" in meta["termination"]["reason"]   # presence flag survives
    assert "imu_data" not in meta["termination"]["reason"]       # clean -> no quality token
    assert meta["termination"]["is_successful"] is False         # verdict not clobbered


def test_missing_topic_presence_token_survives_noop(tmp_path):
    # The missing arm: a missing/empty /imu was flagged imu_presence_err and no imu stream
    # was written, so validate_imu no-ops (nothing to validate) and the flag is preserved —
    # doubly safe vs the extra arm, which relies on token disjointness instead.
    write_meta(tmp_path, with_stream=False, extra_reason=["imu_presence_err"])
    vi.validate_imu(tmp_path)
    meta = load_meta(tmp_path)
    assert "imu_presence_err" in meta["termination"]["reason"]
    assert meta["termination"]["is_successful"] is False


def test_missing_metadata_no_crash(tmp_path):
    vi.validate_imu(tmp_path)                        # nothing written -> returns cleanly
    assert not (tmp_path / "metadata.json").exists()


def test_preserves_other_reasons(tmp_path):
    write_imu_csv(tmp_path, (np.arange(100) * HALF_RATE_PERIOD).tolist())   # rate error -> imu_data
    write_meta(tmp_path, extra_reason=["color_data", "depth_data", "imu_presence_err"])
    vi.validate_imu(tmp_path)
    reason = load_meta(tmp_path)["termination"]["reason"]
    assert "color_data" in reason and "depth_data" in reason and "imu_data" in reason
    # extraction's presence token (data-plane miss/extra) is NOT owned by this quality
    # validator (owns only imu_data / imu_timestamps), so it is preserved, never stripped.
    assert "imu_presence_err" in reason


def test_clears_stale_own_token_on_clean_rerun(tmp_path):
    write_imu_csv(tmp_path, CLEAN_TIMES)
    # a prior dirty run left imu_data (own) + color_data (foreign) in termination
    write_meta(tmp_path, extra_reason=["imu_data", "color_data"])
    vi.validate_imu(tmp_path)
    reason = load_meta(tmp_path)["termination"]["reason"]
    assert "imu_data" not in reason        # stale own token cleared (clean data)
    assert "color_data" in reason          # foreign reason preserved


def test_idempotent(tmp_path):
    write_imu_csv(tmp_path, (np.arange(100) * HALF_RATE_PERIOD).tolist())   # rate error
    write_meta(tmp_path)
    vi.validate_imu(tmp_path)
    first = load_meta(tmp_path)["termination"]["reason"]
    vi.validate_imu(tmp_path)
    second = load_meta(tmp_path)["termination"]["reason"]
    assert first == second == ["imu_data"]
