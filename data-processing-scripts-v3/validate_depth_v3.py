#!/usr/bin/env python3
"""
Aligned-Depth Validation Script
-------------------------------
Companion to validate_color_v3.py, but for the offline depth->color
alignment output (rosbag_process_depth_v3.py). It validates the aligned-depth
stream(s) that rosbag_process_depth_v3 appended to metadata.json and, on any
problem, annotates the stream + flags "depth" in termination.reason.

Two error classes (mirrors the color validator's color/timestamp split):

  TIMESTAMP ERROR  inter-frame gaps in the PAIRED depth stamps: a period
                   > PERIOD_MULTIPLIER x the mean period marks a depth dropout.
                   Written to stream["timestamp_error"].

  DATA ERROR       the aligned h5 + pairing + episode coverage. Covers: h5
                   missing/unreadable, frame-count or shape mismatch vs metadata,
                   (deep scan) h5 content disagreeing with the has_depth flag
                   (silent corruption), the TWO-WAY pairing check -- > MAX_UNPAIRED_RATIO
                   of frames unpaired in EITHER direction (color->depth blanks
                   from the CSV, depth->color drops from the persisted counts) --
                   and the EPISODE COVERAGE check: the aligned depth reaches
                   < MIN_EPISODE_FRAME_RATIO of the busiest color stream's frames,
                   i.e. the whole RealSense stopped early / started late relative to
                   the other cameras. That case is invisible to the pairing/timestamp
                   checks (they measure depth against its OWN ego color partner, which
                   co-dies on a device crash), so depth is compared to the busiest
                   color instead -- in frames, mirroring validate_color_v3.
                   All written to stream["data_error"].

Either class is a "depth error": "depth" is added to termination.reason and
is_successful is recomputed. This script OWNS the "depth" token only -- it
PRESERVES every other reason (color / timestamps / rosbag_corruption set by
validate_color; missing_stream / extra_stream set by the extraction scripts) and
clears a stale "depth" when a re-run is clean. Every writer follows this same
owner-scoped / append-only discipline (see pipeline_metadata.add_error), so the
reasons compose regardless of the order the writers run in.

The aligned-depth timestamps CSV schema differs from the color CSVs
(index,color_stamp_s,depth_stamp_s,pair_dt_ms,has_depth -- NOT ros_time_s),
which is why this is a separate script.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import h5py
except Exception as e:  # noqa: BLE001
    raise SystemExit(f"validate_depth_v3 requires h5py: {e}")

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUT_DIR = r"/Volumes/TRANSCEND4/output/2026-07-31_run2"  # root: metadata.json + depth_frames/ + timestamps/
MAX_UNPAIRED_RATIO = 0.10     # data error if > 10% of frames are unpaired (EITHER direction)
PERIOD_MULTIPLIER = 5.0       # timestamp error if a paired-depth interval > 5x the mean
DEEP_SCAN = True              # read the h5 to cross-check content vs the has_depth flag
MIN_EPISODE_FRAME_RATIO = 0.90  # data error if the aligned depth reaches < 90% of the
                                # busiest color stream's frame count (whole-device early
                                # stop/late start); mirrors validate_color_v3.COLOR_THRESHOLD_RATIO

ALIGNED_KIND = "aligned_depth_to_color"
# Stream-first tokens (mirrors colour's color_data/color_timestamps): the data/h5/pairing
# class -> depth_data ; the paired-stamp gap class -> depth_timestamps. The legacy single
# "depth" token is stripped on write so old metadata migrates cleanly.
DEPTH_DATA_TOKEN = "depth_data"
DEPTH_TS_TOKEN = "depth_timestamps"
_LEGACY_DEPTH_TOKENS = {"depth", DEPTH_DATA_TOKEN, DEPTH_TS_TOKEN}


def load_json_with_comments(filepath: Path) -> dict:
    """Reads JSON, tolerating JS-style // comments (the hand-edited template
    carries them; files written by the pipeline do not)."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    content = re.sub(r"^\s*//.*$", "", content, flags=re.MULTILINE)
    content = re.sub(r"(?<!:)\s*//.*$", "", content, flags=re.MULTILINE)
    return json.loads(content)


def find_aligned_streams(meta: dict) -> List[dict]:
    """Every steps.streams entry produced by rosbag_process_depth_v3 (usually one)."""
    streams = meta.get("steps", {}).get("streams", [])
    return [s for s in streams if s.get("kind") == ALIGNED_KIND]


def color_frame_reference(meta: dict) -> tuple:
    """(max_frames, camera_name) of the busiest FOUND color stream -- the same
    reference validate_color_v3 grades color streams against. This is the yardstick
    for check_episode_coverage: the aligned depth's OWN ego color partner co-fails
    on a device crash, so depth is measured against the busiest color instead."""
    streams = meta.get("steps", {}).get("streams", [])
    max_frames, max_cam = 0, "N/A"
    for s in streams:
        if s.get("kind") == "color" and s.get("found", True):
            nf = int(s.get("num_frames", 0) or 0)
            if nf > max_frames:
                max_frames, max_cam = nf, s.get("camera", "unknown")
    return max_frames, max_cam


def check_timestamps(df: pd.DataFrame, multiplier: float) -> List[str]:
    """Gaps in the PAIRED depth stamps -> depth dropouts. Reports the color frame
    index where each gap lands (the CSV 'index' column)."""
    if "has_depth" not in df.columns or "depth_stamp_s" not in df.columns:
        return [f"timestamps CSV missing expected columns "
                f"(need has_depth + depth_stamp_s; got {list(df.columns)})"]
    paired = df[df["has_depth"].astype(int) == 1].copy()
    paired["depth_stamp_s"] = pd.to_numeric(paired["depth_stamp_s"], errors="coerce")
    paired = paired.dropna(subset=["depth_stamp_s"]).sort_values("depth_stamp_s")
    stamps = paired["depth_stamp_s"].to_numpy()
    if stamps.size < 2:
        return []
    frame_idx = (paired["index"].to_numpy() if "index" in paired.columns
                 else np.arange(stamps.size))
    periods = np.diff(stamps)
    mean_period = float(periods.mean())
    if mean_period <= 0:
        return []
    errors = []
    for k, p in enumerate(periods):
        if p > multiplier * mean_period:
            errors.append(f"Frame {int(frame_idx[k + 1])} depth-stamp gap {p:.2f}s "
                          f"> {multiplier:.0f}x mean period {mean_period:.2f}s")
    return errors


def check_pairing(stream: dict, df: pd.DataFrame, max_unpaired_ratio: float) -> List[str]:
    """Two-way pairing check -- BOTH a pairing miss is a data error:

      color->depth  color frames with no depth partner (blank h5 frames).
                    Read from the CSV has_depth flag (color-indexed, so every
                    color frame is a row; unpaired ones have has_depth==0).
      depth->color  depth msgs with no color partner. NOT reconstructable from
                    the CSV/h5 (both color-indexed) -- read from the counts
                    rosbag_process_depth_v3 persisted into the stream entry
                    (n_depth_no_partner / n_depth_msgs).

    Count subtraction (n_color vs n_depth_msgs) is deliberately NOT used: equal
    totals can still hide symmetric skew drops, so the true per-direction counts
    are the only correct source."""
    errors: List[str] = []

    # --- color->depth : color frames without a depth partner (CSV truth) ---
    n_color = n_paired = None
    if "has_depth" in df.columns and len(df):
        flags = df["has_depth"].astype(int).to_numpy()
        n_color, n_paired = len(flags), int(flags.sum())
    elif stream.get("h5_index") is not None and stream.get("n_paired") is not None:
        n_color, n_paired = int(stream["h5_index"]), int(stream["n_paired"])
    if n_color and n_paired is not None:
        n_blank = n_color - n_paired
        ratio = n_blank / n_color
        if ratio > max_unpaired_ratio:
            errors.append(f"unpaired color→depth: {ratio * 100:.1f}% "
                          f"({n_blank}/{n_color} color frames without depth) "
                          f"exceeds {max_unpaired_ratio * 100:.0f}%")

    # --- depth->color : depth msgs without a color partner (persisted counts) ---
    n_depth = stream.get("n_depth_frames")
    n_missing_color = stream.get("n_pair_missing_color")
    if n_depth and n_missing_color is not None:
        ratio = n_missing_color / n_depth
        if ratio > max_unpaired_ratio:
            errors.append(f"unpaired depth→color: {ratio * 100:.1f}% "
                          f"({n_missing_color}/{n_depth} depth frames without color) "
                          f"exceeds {max_unpaired_ratio * 100:.0f}%")
    elif n_missing_color is None:
        print("    [note] depth→color pairing not checked: metadata lacks "
              "n_pair_missing_color/n_depth_frames (re-run rosbag_process_depth_v3 to persist them).")

    return errors


def check_episode_coverage(stream: dict, color_ref: Optional[tuple],
                           min_ratio: float) -> List[str]:
    """Whole-device coverage: did the aligned depth reach as far as the episode?

    The pairing/timestamp/h5 checks all measure depth against its OWN ego color
    partner -- which co-dies when the RealSense crashes, so a whole-device early
    stop (or late start) stays invisible to them. This compares the aligned depth's
    frame reach (h5_index) against the busiest color stream (color_ref), exactly as
    validate_color_v3 grades a color stream: reach < min_ratio of the reference is a
    data error. Measured in frames -- depth is 1:1 color-indexed -- to stay
    consistent with the color validator's yardstick and message.

    Skip-safe: returns [] when there is no color reference (color_ref falsy or
    max_frames 0) or the stream carries no h5_index."""
    if not color_ref:
        return []
    max_frames, max_cam = color_ref
    reach = stream.get("h5_index")
    if not max_frames or reach is None:
        if reach is None:
            print("    [note] episode coverage not checked: stream lacks h5_index.")
        return []
    ratio = int(reach) / max_frames
    if ratio < min_ratio:
        return [f"{ratio * 100:.1f}% depth frames compared to {max_cam} color as 100%"]
    return []


def check_data(h5_path: Path, df: pd.DataFrame, h5_index: Optional[int],
               height: Optional[int], width: Optional[int],
               deep_scan: bool) -> List[str]:
    """Integrity of the aligned h5 against metadata + the CSV (pairing coverage is
    handled separately by check_pairing)."""
    if not h5_path.exists():
        return [f"aligned depth h5 not found: {h5_path}"]

    errors: List[str] = []
    try:
        with h5py.File(h5_path, "r") as f:
            if "data" not in f:
                return [f"h5 has no 'data' dataset: {h5_path.name}"]
            dset = f["data"]
            n_h5 = int(dset.shape[0])

            # --- shape / count integrity vs metadata + CSV ---
            if h5_index is not None and n_h5 != int(h5_index):
                errors.append(f"h5 frame count {n_h5} != metadata h5_index {h5_index}")
            if len(df) and len(df) != n_h5:
                errors.append(f"h5 frame count {n_h5} != timestamps rows {len(df)}")
            if height is not None and width is not None and \
                    tuple(dset.shape[1:]) != (int(height), int(width)):
                errors.append(f"h5 frame shape {tuple(dset.shape[1:])} "
                              f"!= metadata (h,w)=({int(height)}, {int(width)})")

            # --- has_depth flag (needed only for the deep content cross-check;
            #     pairing coverage itself is check_pairing's job) ---
            has_flag = (df["has_depth"].astype(int).to_numpy()
                        if "has_depth" in df.columns and len(df) else None)

            # --- deep content cross-check: h5 non-zero-ness vs has_depth flag ---
            if deep_scan and has_flag is not None and len(has_flag) == n_h5:
                mismatches = sum(1 for i in range(n_h5)
                                 if bool(np.any(dset[i])) != bool(has_flag[i]))
                if mismatches:
                    errors.append(f"{mismatches} frame(s) disagree between h5 content and "
                                  "has_depth flag (silent corruption)")
    except Exception as e:  # noqa: BLE001
        return [f"aligned depth h5 unreadable ({h5_path.name}): {e}"]
    return errors


def validate_stream(out_dir: Path, stream: dict,
                    color_ref: Optional[tuple] = None) -> tuple:
    """Validate one aligned-depth stream, annotating it in place. Returns
    (has_timestamp_error, has_data_error) so the caller can raise the split tokens
    (depth_timestamps / depth_data). color_ref = (max_frames, cam) of the busiest
    color stream (from color_frame_reference); None skips the episode-coverage check."""
    # --- timestamps ---
    ts_errors: List[str] = []
    ts_rel = stream.get("timestamps")
    df = pd.DataFrame()
    if ts_rel and (out_dir / ts_rel).exists():
        try:
            df = pd.read_csv(out_dir / ts_rel)
            ts_errors = check_timestamps(df, PERIOD_MULTIPLIER)
        except Exception as e:  # noqa: BLE001
            ts_errors = [f"timestamps CSV unreadable ({ts_rel}): {e}"]
    else:
        ts_errors = [f"timestamps CSV missing: {ts_rel}"]

    # --- data / h5 : integrity + two-way pairing (both are data errors) ---
    frames_rel = stream.get("frames_dir")
    if frames_rel:
        data_errors = check_data(out_dir / frames_rel, df, stream.get("h5_index"),
                                 stream.get("height"), stream.get("width"), DEEP_SCAN)
    else:
        data_errors = ["stream has no frames_dir (h5 path) in metadata"]
    data_errors += check_pairing(stream, df, MAX_UNPAIRED_RATIO)
    data_errors += check_episode_coverage(stream, color_ref, MIN_EPISODE_FRAME_RATIO)

    # --- annotate the stream idempotently (set or clear) ---
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
        print(f"[FAIL] {cam} {ALIGNED_KIND}:")
        for e in ts_errors:
            print(f"    [timestamp] {e}")
        for e in data_errors:
            print(f"    [data]      {e}")
    else:
        print(f"[OK] {cam} {ALIGNED_KIND}: timestamp + data checks passed.")
    return bool(ts_errors), bool(data_errors)


def validate_aligned_depth(out_dir: Path) -> None:
    meta_path = out_dir / "metadata.json"
    if not meta_path.exists():
        print(f"[ERROR] metadata.json not found at {meta_path}")
        return
    try:
        meta = load_json_with_comments(meta_path)
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] failed to parse {meta_path.name}: {e}")
        return

    streams = find_aligned_streams(meta)
    if not streams:
        print(f"[WARN] no '{ALIGNED_KIND}' stream in metadata.json; nothing to validate "
              "(run rosbag_process_depth_v3 first). Leaving metadata untouched.")
        return

    color_ref = color_frame_reference(meta)
    results = [validate_stream(out_dir, s, color_ref) for s in streams]
    has_ts_error = any(ts for ts, _ in results)
    has_data_error = any(data for _, data in results)

    # --- merge-safe termination update: own ONLY the depth tokens (depth_data /
    # depth_timestamps), stripping the legacy "depth" too so old metadata migrates. ---
    term = meta.get("termination") or {}
    reasons = [r for r in (term.get("reason") or []) if r not in _LEGACY_DEPTH_TOKENS]
    if has_data_error:
        reasons.append(DEPTH_DATA_TOKEN)
    if has_ts_error:
        reasons.append(DEPTH_TS_TOKEN)
    meta["termination"] = {"is_successful": len(reasons) == 0, "reason": reasons}

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[SUCCESS] updated {meta_path} "
          f"(termination.reason={meta['termination']['reason']})")


if __name__ == "__main__":
    try:
        validate_aligned_depth(Path(OUTPUT_DIR))
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(130)
