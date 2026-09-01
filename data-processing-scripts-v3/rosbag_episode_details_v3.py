#!/usr/bin/env python3
"""
Episode-details extractor  (rosbag_episode_details_v3)
------------------------------------------------------
A PER-BAG EXTRACTION step of the v3 pipeline, exactly like color/depth/imu: the
wrapper hands it one bag + its out_dir, and it writes that episode's identity,
timing and mistakes into the episode's `metadata.json` under `episode_details`.
Everything is read WHILE the raw bag is in hand, so there is no post-loop pass, no
source_bag, and no reaching back into the raw bag later.

One rosbag folder IS one episode, named `<date>_<code>_<take>`. Mistakes are
annotated after the fact as empty marker files under `<bag>/oops/` (the filename IS
the mistake code). None of that is in the bag's message stream; this step derives
identity from the folder name, timing from the bag's own `metadata.yaml`, and
mistakes from `oops/`, then injects:

    "episode_details": {
      "episode_code": "23",        # <code> from the folder name (opaque string)
      "take": 2,                   # <take> from the folder name (per-code counter)
      "date": "2026-08-04",        # <date> from the folder name
      "start_time_ns": 1690000000000000000,   # metadata.yaml starting_time
      "duration_s": 42.7,          # metadata.yaml duration
      "objects": [],               # owned by the code->codebook step (left as-is here)
      "actions": [],               # owned by the code->codebook step (left as-is here)
      "mistakes": ["m1", "m3"]     # sorted oops/ marker filenames (opaque strings)
    }

NO episode_index: recording order is fully determined by `start_time_ns`, so storing
a precomputed rank per-episode would only invite staleness. The 1..N ordinal is a
DATASET-VIEW number, computed at report time in wrapper.py's session-summary.csv by
sorting rows on start_time_ns — not persisted here.

OWNERSHIP: this step owns episode identity + timing + mistakes. It does NOT own
`objects`/`actions` (a separate code->{objects,actions} codebook step fills those),
so any existing objects/actions are PRESERVED, not clobbered. Idempotent / rerunnable.

INPUT  : one bag folder (holds metadata.yaml + optional oops/) + its out_dir (holds
         the processed metadata.json). OUTPUT: episode_details written into it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

METADATA_FILENAME = "metadata.json"
OOPS_DIRNAME = "oops"
# Folder-name convention: <date>_<code>_<take>, where the ONLY two "_" separate the
# three parts (date uses "-", code is alphanumeric with no "_"/"-"). See spec.
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# --------------------------------------------------------------------------- #
# parsing helpers                                                             #
# --------------------------------------------------------------------------- #
def parse_episode_folder(name: str) -> Optional[dict]:
    """`<date>_<code>_<take>` -> {date, code, take:int}, or None if the name does
    not conform (the dataset is mid-migration, so non-conforming names are skipped,
    not errors). take must be a positive integer; date must look like YYYY-MM-DD."""
    parts = name.split("_")
    if len(parts) != 3:
        return None
    date, code, take = parts
    if not DATE_RE.match(date) or not code:
        return None
    try:
        take_i = int(take)
    except ValueError:
        return None
    if take_i < 1:
        return None
    return {"date": date, "code": code, "take": take_i}


def read_bag_timing(bag_dir: Path) -> Optional[dict]:
    """Read start time + duration straight from the rosbag2 `metadata.yaml`
    (no yaml dependency, no need to open the .mcap). Returns
    {start_time_ns:int, duration_s:float|None} or None if start time is unreadable.

    Paths (rosbag2 v5, confirmed against a real bag):
      rosbag2_bagfile_information.starting_time.nanoseconds_since_epoch
      rosbag2_bagfile_information.duration.nanoseconds
    """
    yaml_path = bag_dir / "metadata.yaml"
    if not yaml_path.is_file():
        return None
    text = yaml_path.read_text(encoding="utf-8", errors="replace")
    m_start = re.search(r"nanoseconds_since_epoch:\s*(\d+)", text)
    if not m_start:
        return None
    start_ns = int(m_start.group(1))
    # scope the duration match to the `duration:` block so it can't grab a stray number
    m_dur = re.search(r"duration:\s*\n\s*nanoseconds:\s*(\d+)", text)
    duration_s = round(int(m_dur.group(1)) / 1e9, 3) if m_dur else None
    return {"start_time_ns": start_ns, "duration_s": duration_s}


def read_mistakes(bag_dir: Path) -> List[str]:
    """Sorted mistake codes = the filenames in `<bag>/oops/`. Each file is empty;
    the name is the whole datum, treated as an opaque string (no m<digit> parsing).
    No oops/ dir, or an empty one, means no mistakes -> []."""
    oops = bag_dir / OOPS_DIRNAME
    if not oops.is_dir():
        return []
    return sorted(p.name for p in oops.iterdir() if p.is_file())


def extract_episode_details(bag=None, out_dir=None) -> dict:
    """Per-bag EXTRACTION entry (a wrapper step, run while the raw bag is in hand):
    from ONE episode bag, derive its identity (folder name), timing (metadata.yaml)
    and mistakes (oops/), and write them into that episode's metadata.json under
    `episode_details`. Read-modify-write: creates the block if absent, PRESERVES any
    existing objects/actions (owned by the codebook step), touches no other top-level
    key. No-op if metadata.json is absent (run color extraction first). Idempotent.

    Deliberately NO episode_index: recording order is fully determined by
    start_time_ns, so the 1..N ordinal (a dataset-wide view) is computed at report
    time in session-summary.csv, never stored per-episode. A bag whose folder name is
    non-conforming, or whose metadata.yaml is unreadable, still gets its mistakes; the
    missing fields are simply left out (warned).

    Returns a summary dict for the wrapper's per-step line."""
    bag = Path(bag)
    if not bag.exists():
        raise SystemExit(f"Bag not found: {bag}")
    out_root = Path(out_dir) if out_dir else bag.parent
    meta_path = out_root / METADATA_FILENAME

    parsed = parse_episode_folder(bag.name)          # {date, code, take} or None
    timing = read_bag_timing(bag)                    # {start_time_ns, duration_s} or None
    mistakes = read_mistakes(bag)

    if not meta_path.is_file():
        print(f"[episode_details] {meta_path} not found; skipping "
              "(run color extraction first).")
        return {"bag": str(bag), "mistakes": mistakes, "written": False}

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    ed = meta.setdefault("episode_details", {})
    if parsed is not None:
        ed["episode_code"], ed["take"], ed["date"] = parsed["code"], parsed["take"], parsed["date"]
    else:
        print(f"[episode_details] WARN bag folder '{bag.name}' is not "
              "<date>_<code>_<take>; identity left blank.")
    if timing is not None:
        ed["start_time_ns"], ed["duration_s"] = timing["start_time_ns"], timing["duration_s"]
    else:
        print(f"[episode_details] WARN no start time in {bag / 'metadata.yaml'}; "
              "timing left blank.")
    ed["mistakes"] = mistakes
    ed.setdefault("objects", [])                     # owned by the codebook step; preserve
    ed.setdefault("actions", [])

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[episode_details] {bag.name}: code={ed.get('episode_code')} "
          f"take={ed.get('take')} start={ed.get('start_time_ns')} "
          f"mistakes={mistakes or '[]'} -> {meta_path}")
    return {"bag": str(bag), "mistakes": mistakes, "written": True,
            "code": ed.get("episode_code"), "start_time_ns": ed.get("start_time_ns")}


