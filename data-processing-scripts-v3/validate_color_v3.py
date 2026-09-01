#!/usr/bin/env python3
"""
Standalone Metadata Validation Script
-------------------------------------
1. Reads metadata.json and timestamps/*.csv files in OUTPUT_DIR.
2. Strips JS-style comments (// ...) from metadata.json safely before parsing.
3. Checks color streams against the highest frame count camera (Color Error if < 90%).
4. Checks inter-frame intervals against stream mean period (Timestamp Error if > 5x mean).
5. Appends errors to metadata.json and updates termination status.
"""

import json
import re
import pandas as pd
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUT_DIR = r"/Volumes/TRANSCEND4/2026-07-31_run2"  # Root directory containing metadata.json and timestamps/*.csv
COLOR_THRESHOLD_RATIO = 0.90  # Flag if frames < 90% of max frames (>= 10% loss)
PERIOD_MULTIPLIER = 5.0      # Flag if inter-frame interval > 5x mean period


def load_json_with_comments(filepath: Path) -> dict:
    """Reads a JSON file and strips JS-style comments (// ...) before parsing."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    # Strip line comments (// ...) while preserving URLs/strings
    content_clean = re.sub(r'^\s*//.*$', '', content, flags=re.MULTILINE)
    content_clean = re.sub(r'(?<!:)\s*//.*$', '', content_clean, flags=re.MULTILINE)
    return json.loads(content_clean)


def validate_metadata(out_dir: Path):
    metadata_path = out_dir / "metadata.json"

    if not metadata_path.exists():
        print(f"[ERROR] metadata.json not found at {metadata_path}")
        return

    try:
        meta = load_json_with_comments(metadata_path)
    except Exception as e:
        print(f"[ERROR] Failed to parse {metadata_path.name}: {e}")
        return

    streams = meta.get("steps", {}).get("streams", [])

    # ------------------------------------------------------------------
    # RULE 1: Color Validation (Compare against max frames among color streams)
    # ------------------------------------------------------------------
    color_streams = [s for s in streams if s.get("kind") == "color" and s.get("found", True)]
    max_frames = 0
    max_cam_name = "N/A"

    for s in color_streams:
        nf = s.get("num_frames", 0)
        if nf > max_frames:
            max_frames = nf
            max_cam_name = s.get("camera", "unknown")

    has_color_error = False
    has_color_timestamp_error = False  # colour-stream ts gaps (new 'color_timestamps' token)

    for s in streams:
        camera = s.get("camera", "unknown")
        kind = s.get("kind", "unknown")
        num_frames = s.get("num_frames", 0)

        # Rule 1: colour DATA quality on FOUND colour streams -> color_data. TWO sources,
        # both per-frame quality: frame-count loss, and undecodable frames (decode_failures,
        # counted by extraction). Undecodable colour frames are a colour DATA problem, NOT
        # bag corruption — structural bag corruption is the SEPARATE `rosbag_corruption`
        # token owned by bag_integrity, never touched here. A missing stream (found:False)
        # is extraction's color_presence_err, skipped here. The ABSOLUTE 0-floor runs first
        # (catches the all-empty episode the relative "< 90% of busiest" test is blind to
        # when max_frames is 0).
        if kind == "color" and s.get("found", True):
            col_errors = []
            if num_frames == 0:                       # 0 frames is unambiguously broken
                col_errors.append("stream is empty (0 frames)")
            elif max_frames > 0:                      # relative loss vs the busiest stream
                ratio = (num_frames / max_frames) * 100
                if ratio < (COLOR_THRESHOLD_RATIO * 100):
                    col_errors.append(f"{ratio:.1f}% frames compared to {max_cam_name} as 100%")
            n_decode = s.get("decode_failures", 0)
            if n_decode > 0:                          # undecodable frames = colour DATA quality
                col_errors.append(f"{n_decode} frame(s) failed to decode")
            if col_errors:
                has_color_error = True
                s["data_error"] = col_errors
            else:
                s.pop("data_error", None)   # clean color stream -> clear stale annotation

        # ------------------------------------------------------------------
        # RULE 2: Timestamp Validation (Inter-frame interval check)
        # ------------------------------------------------------------------
        # COLOUR streams only: depth-stream timestamps are validate_depth's job
        # (depth_timestamps), and the imu stream is validate_imu's — this validator
        # never touches a non-colour stream's timestamps.
        ts_rel_path = s.get("timestamps")
        if ts_rel_path and kind == "color":
            ts_file = out_dir / ts_rel_path
            if ts_file.exists():
                df_ts = pd.read_csv(ts_file)
                if "ros_time_s" in df_ts.columns and len(df_ts) > 1:
                    ts_vals = df_ts["ros_time_s"].values
                    periods = ts_vals[1:] - ts_vals[:-1]
                    mean_period = periods.mean()

                    ts_errors = []
                    if mean_period > 0:
                        for idx, p in enumerate(periods):
                            if p > (PERIOD_MULTIPLIER * mean_period):
                                frame_idx = idx + 1
                                ts_errors.append(
                                    f"Index {frame_idx} period {p:.2f}s > mean period {mean_period:.2f}s"
                                )

                    if ts_errors:
                        s["timestamps_error"] = ts_errors
                        has_color_timestamp_error = True
                    else:
                        s.pop("timestamps_error", None)   # clean stream -> clear stale annotation

    # ------------------------------------------------------------------
    # Termination — merge-safe: this validator OWNS only its per-frame QUALITY colour tokens
    # (color_data [frame loss + undecodable frames] / color_timestamps). It recomputes those
    # from scratch each run (so a re-run that is now clean clears them) but PRESERVES every
    # other reason another writer set — color_presence_err / color_info from EXTRACTION
    # (stream / intrinsics presence), rosbag_corruption from bag_integrity (structural bag
    # corruption), depth_* from validate_depth, imu_* from validate_imu.
    # Disjoint ownership: presence (color_presence_err, extraction), structural corruption
    # (rosbag_corruption, bag_integrity), and per-frame quality (color_data, here) are SEPARATE
    # tokens, so recomputing quality can never strip a presence or corruption flag. Order
    # does not matter.
    # ------------------------------------------------------------------
    OWNED = {"color_data", "color_timestamps"}
    term = meta.get("termination") or {}
    reasons = [r for r in (term.get("reason") or []) if r not in OWNED]
    if has_color_error:
        reasons.append("color_data")
    if has_color_timestamp_error:
        reasons.append("color_timestamps")
    # has_depth_error is set from a depth stream's timestamp gap; the "depth" token
    # itself is owned by validate_depth_v3, so we do NOT add it here.

    meta["termination"] = {
        "is_successful": len(reasons) == 0,
        "reason": reasons,
    }

    # Save clean, strictly valid metadata.json
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f"[SUCCESS] Validation complete. Updated {metadata_path.name, metadata_path}")


if __name__ == "__main__":
    validate_metadata(Path(OUTPUT_DIR))