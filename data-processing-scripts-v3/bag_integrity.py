#!/usr/bin/env python3
"""
Bag integrity + spine initialisation  (bag_integrity)
-----------------------------------------------------
STEP 0 of the v3 pipeline. It runs FIRST — before any extractor — and does two
things no other step can:

  1. STRUCTURAL bag check. Open the rosbag2 and read its index/summary (topics +
     start/end time) WITHOUT decoding any payload. If the bag cannot be opened or
     its index cannot be read, the bag is CORRUPT.
  2. CREATE metadata.json (the spine). Every later step (color, depth, imu,
     episode_details, calib, validators) APPENDS to this file; none of them create
     it any more. The spine carries the `metadata` block, index-derived
     `date_recorded` / `steps.timestamp_range`, empty `camera_intrinsics`, an empty
     `steps.streams`, and the initial `termination`.

OWNERSHIP: this step OWNS the `rosbag_corruption` termination token. It means exactly
one thing — the bag is not structurally openable/indexable. It does NOT mean "a color
frame failed to decode" (that is per-frame colour QUALITY -> color_data, owned by
validate_color) nor "a message deep in a stream won't deserialize" (the extractor
reading that stream hits it -> its own token, or a crash). There is no separate
message-scan pass: the extractors already read every message they need.

FAIL-FAST: when the bag is corrupt, the wrapper writes only this spine (with
`rosbag_corruption` + is_successful False) and SKIPS all extraction and validation —
the verdict is already known. A corrupt bag therefore still gets a real metadata.json
documenting why, instead of only a crash marker.

TESTABILITY: all shape logic is in the pure function build_initial_spine(...) (dict
in, dict out, no I/O). check_bag() is the thin reader shell; init_spine() is the thin
file shell around both.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rosbags.highlevel import AnyReader

METADATA_FILENAME = "metadata.json"
CORRUPTION_TOKEN = "rosbag_corruption"      # the ONLY token this step owns
# The descriptive metadata block fields (order matches the on-disk schema). Values come
# from the resolved `meta` dict the wrapper passes; date_recorded is filled from the index.
METADATA_FIELDS = ("dataset_name", "dataset_version", "robot_model", "environment",
                   "setup", "subject", "bed_type")


def build_initial_spine(meta: Dict[str, Any], *, corrupt: bool, detail: List[str],
                        date_recorded: Optional[str] = None,
                        timestamp_range: Optional[list] = None) -> Dict[str, Any]:
    """PURE: return the initial metadata.json spine every later step appends to.

    metadata block: the descriptive fields from `meta` (already resolved by the caller)
    plus the index-derived `date_recorded`. `camera_intrinsics` starts empty (color
    appends via upsert_intrinsic); `steps.streams` starts empty (each extractor appends);
    `steps.timestamp_range` is the index range (or None). termination carries
    rosbag_corruption when the bag is structurally corrupt. No `fps` field — fps is a
    per-stream stat (steps.streams[].fps), never a single top-level value."""
    block = {f: meta.get(f) for f in METADATA_FIELDS}
    block["date_recorded"] = date_recorded
    # restore the canonical block order (date_recorded sits after environment)
    ordered_block = {
        "dataset_name": block.get("dataset_name"),
        "dataset_version": block.get("dataset_version"),
        "robot_model": block.get("robot_model"),
        "environment": block.get("environment"),
        "date_recorded": date_recorded,
        "setup": block.get("setup"),
        "subject": block.get("subject"),
        "bed_type": block.get("bed_type"),
    }
    steps: Dict[str, Any] = {"streams": [], "timestamp_range": timestamp_range}
    if corrupt:
        steps["bag_corruption_error"] = list(detail)   # informational: WHY it is corrupt
    return {
        "metadata": ordered_block,
        "camera_intrinsics": [],
        "steps": steps,
        "termination": {
            "is_successful": not corrupt,
            "reason": [CORRUPTION_TOKEN] if corrupt else [],
        },
    }


def check_bag(bag) -> Tuple[bool, List[str], Optional[str], Optional[list]]:
    """READER shell: structural integrity of the rosbag2 at `bag`. Opens the bag and
    reads its index (topics + start/end time) WITHOUT decoding payloads. Returns
    (corrupt, detail, date_recorded, timestamp_range).

    Cheap structural preconditions first (path exists, is a directory, has metadata.yaml),
    then AnyReader open + index read. ANY failure -> corrupt=True with the reason; the
    timestamps come back None. A clean open yields date_recorded (from the index start
    time) and the [start, end] second range."""
    bag = Path(bag)
    if not bag.exists():
        return True, [f"bag path not found: {bag}"], None, None
    if bag.is_file():
        return True, [f"bag path is a file, not a rosbag2 directory: {bag}"], None, None
    if not (bag / "metadata.yaml").exists():
        return True, [f"no metadata.yaml in {bag} — not a rosbag2 folder"], None, None
    try:
        with AnyReader([bag]) as reader:
            _ = reader.topics                 # touch the connection index
            start_ns = reader.start_time      # index-level; no payload decode
            end_ns = reader.end_time
    except Exception as e:                     # noqa: BLE001 — any open/index failure = corrupt
        return True, [f"bag failed to open or index: {e}"], None, None

    date_recorded: Optional[str] = None
    ts_range: Optional[list] = None
    if start_ns is not None:
        start_s = start_ns / 1e9
        end_s = end_ns / 1e9 if end_ns is not None else None
        date_recorded = time.strftime("%Y-%m-%d", time.gmtime(start_s))
        ts_range = [start_s, end_s]
    return False, [], date_recorded, ts_range


def init_spine(bag, out_dir, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """FILE shell: check the bag, build the spine, write out_dir/metadata.json. Returns a
    small summary {corrupt, detail, written} for the wrapper's per-step line and its
    fail-fast decision. `meta` is the RESOLVED descriptive metadata dict."""
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    corrupt, detail, date_recorded, ts_range = check_bag(bag)
    spine = build_initial_spine(meta or {}, corrupt=corrupt, detail=detail,
                                date_recorded=date_recorded, timestamp_range=ts_range)
    (out_root / METADATA_FILENAME).write_text(json.dumps(spine, indent=2))
    where = out_root / METADATA_FILENAME
    if corrupt:
        print(f"[bag_integrity] {out_root.name}: CORRUPT — {'; '.join(detail)} -> {where}")
    else:
        print(f"[bag_integrity] {out_root.name}: ok (date={date_recorded}) -> spine written {where}")
    return {"corrupt": corrupt, "detail": detail, "written": True}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Structural bag check + metadata.json spine init")
    ap.add_argument("--bag", required=True, help="rosbag2 directory")
    ap.add_argument("--out-dir", required=True, help="episode out_dir to write metadata.json into")
    args = ap.parse_args()
    res = init_spine(args.bag, args.out_dir, {})
    raise SystemExit(1 if res["corrupt"] else 0)
