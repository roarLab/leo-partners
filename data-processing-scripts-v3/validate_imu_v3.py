#!/usr/bin/env python3
"""
IMU Validation Script
---------------------
Companion to validate_color_v3.py / validate_depth_v3.py, for the ego /imu sample
stream that rosbag_process_imu_v3.py appended to metadata.json (kind == "imu"). It
validates that stream and, on any problem, annotates it + flags the split imu tokens
in termination.reason.

Two error classes (mirrors the colour/depth data/timestamp split):

  TIMESTAMP ERROR  inter-sample gaps: a period > PERIOD_MULTIPLIER x the mean period
                   marks an IMU dropout. Written to stream["timestamps_error"];
                   raises the 'imu_timestamps' termination token.

  DATA ERROR       sample-rate sanity: the mean sample frequency deviates from the
                   expected rate (EXPECTED_IMU_HZ) by more than IMU_RATE_TOLERANCE.
                   A wrong rate means the driver mis-configured or samples were
                   dropped wholesale. Written to stream["data_error"]; raises the
                   'imu_data' termination token.

This script OWNS only {imu_data, imu_timestamps} — it PRESERVES every other reason
another writer set (color_* from validate_color, depth_* from validate_depth,
missing/extra from the extraction scripts) and clears a stale own-token when a re-run
is clean. Same owner-scoped / append-only discipline as the other validators, so the
reasons compose regardless of the order the writers run in.

NOTE ON EXPECTED_IMU_HZ: the design slide specifies 100 Hz. The Intel D435i united
/imu stream can actually run ~200 Hz depending on the driver's gyro/accel config — if
this flags every good episode, set EXPECTED_IMU_HZ to the rig's real rate. It is a
single config constant precisely so this is a one-line change.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUT_DIR = r"/Volumes/TRANSCEND4/output/2026-07-31_run2"  # root: metadata.json + imu/*.csv
PERIOD_MULTIPLIER = 5.0       # timestamp error if an inter-sample interval > 5x the mean
EXPECTED_IMU_HZ = 200.0       # expected mean sample rate (see NOTE in the module docstring)
IMU_RATE_TOLERANCE = 0.20     # data error if the mean rate deviates > 20% from EXPECTED_IMU_HZ

IMU_KIND = "imu"
IMU_DATA_TOKEN = "imu_data"
IMU_TS_TOKEN = "imu_timestamps"
_OWNED_TOKENS = {IMU_DATA_TOKEN, IMU_TS_TOKEN}


def load_json_with_comments(filepath: Path) -> dict:
    """Reads JSON, tolerating JS-style // comments (the hand-edited template carries
    them; files written by the pipeline do not)."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    content = re.sub(r"^\s*//.*$", "", content, flags=re.MULTILINE)
    content = re.sub(r"(?<!:)\s*//.*$", "", content, flags=re.MULTILINE)
    return json.loads(content)


def find_imu_streams(meta: dict) -> List[dict]:
    """Every steps.streams entry produced by rosbag_process_imu_v3 (usually one)."""
    streams = meta.get("steps", {}).get("streams", [])
    return [s for s in streams if s.get("kind") == IMU_KIND]


def _sample_times(out_dir: Path, stream: dict) -> Optional[np.ndarray]:
    """The IMU sample times (seconds) from the stream's CSV, or None if unreadable.
    The IMU stream carries its CSV under 'file' (columns index, ros_time_s, ...),
    NOT the 'timestamps' key the colour/depth streams use."""
    rel = stream.get("file")
    if not rel:
        return None
    csv = out_dir / rel
    if not csv.exists():
        return None
    try:
        df = pd.read_csv(csv)
    except Exception:  # noqa: BLE001
        return None
    if "ros_time_s" not in df.columns:
        return None
    return pd.to_numeric(df["ros_time_s"], errors="coerce").dropna().to_numpy()


def check_timestamps(times: np.ndarray, multiplier: float) -> List[str]:
    """Inter-sample gaps: a period > multiplier x the mean period is an IMU dropout."""
    if times.size < 2:
        return []
    periods = np.diff(np.sort(times))
    mean_period = float(periods.mean())
    if mean_period <= 0:
        return []
    errors: List[str] = []
    for k, p in enumerate(periods):
        if p > multiplier * mean_period:
            errors.append(f"Sample {k + 1} gap {p:.4f}s > {multiplier:.0f}x mean "
                          f"period {mean_period:.4f}s")
    return errors


def check_rate(times: np.ndarray, expected_hz: float, tolerance: float) -> List[str]:
    """Mean sample rate sanity: flag when the mean frequency deviates from
    expected_hz by more than `tolerance` (a fraction). A wrong rate = driver
    mis-config or wholesale sample loss."""
    if times.size < 2:
        return []
    periods = np.diff(np.sort(times))
    mean_period = float(periods.mean())
    if mean_period <= 0:
        return []
    mean_hz = 1.0 / mean_period
    if abs(mean_hz - expected_hz) / expected_hz > tolerance:
        return [f"mean rate {mean_hz:.1f} Hz deviates > {tolerance * 100:.0f}% from "
                f"expected {expected_hz:.0f} Hz"]
    return []


def validate_stream(out_dir: Path, stream: dict) -> tuple:
    """Validate one IMU stream, annotating it in place. Returns
    (has_timestamp_error, has_data_error) so the caller can raise the split tokens
    (imu_timestamps / imu_data)."""
    times = _sample_times(out_dir, stream)
    if times is None:
        # No readable CSV -> a data error (the samples the stream claims are unusable).
        ts_errors: List[str] = []
        data_errors = [f"IMU samples CSV missing or unreadable: {stream.get('file')}"]
    else:
        ts_errors = check_timestamps(times, PERIOD_MULTIPLIER)
        data_errors = check_rate(times, EXPECTED_IMU_HZ, IMU_RATE_TOLERANCE)

    if ts_errors:
        stream["timestamps_error"] = ts_errors
    else:
        stream.pop("timestamps_error", None)
    if data_errors:
        stream["data_error"] = data_errors
    else:
        stream.pop("data_error", None)

    cam = stream.get("camera", "unknown")
    if ts_errors or data_errors:
        print(f"[FAIL] {cam} {IMU_KIND}:")
        for e in ts_errors:
            print(f"    [timestamp] {e}")
        for e in data_errors:
            print(f"    [data]      {e}")
    else:
        print(f"[OK] {cam} {IMU_KIND}: timestamp + rate checks passed.")
    return bool(ts_errors), bool(data_errors)


def validate_imu(out_dir: Path) -> None:
    meta_path = out_dir / "metadata.json"
    if not meta_path.exists():
        print(f"[ERROR] metadata.json not found at {meta_path}")
        return
    try:
        meta = load_json_with_comments(meta_path)
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] failed to parse {meta_path.name}: {e}")
        return

    streams = find_imu_streams(meta)
    if not streams:
        # No imu stream = a missing/empty /imu (already flagged as imu_data by the
        # extraction script's presence check), or imu not recorded. Nothing to
        # validate; leave metadata untouched.
        print(f"[WARN] no '{IMU_KIND}' stream in metadata.json; nothing to validate.")
        return

    results = [validate_stream(out_dir, s) for s in streams]
    has_ts_error = any(ts for ts, _ in results)
    has_data_error = any(data for _, data in results)

    # --- merge-safe termination update: own ONLY imu_data / imu_timestamps ---
    term = meta.get("termination") or {}
    reasons = [r for r in (term.get("reason") or []) if r not in _OWNED_TOKENS]
    if has_data_error:
        reasons.append(IMU_DATA_TOKEN)
    if has_ts_error:
        reasons.append(IMU_TS_TOKEN)
    meta["termination"] = {"is_successful": len(reasons) == 0, "reason": reasons}

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[SUCCESS] updated {meta_path} "
          f"(termination.reason={meta['termination']['reason']})")


if __name__ == "__main__":
    try:
        validate_imu(Path(OUTPUT_DIR))
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(130)
