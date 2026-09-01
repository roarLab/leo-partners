"""Unit tests for rosbag_episode_details_v3.py.

Run:  python -m pytest data-processing-scripts-v3/test_episode_details_v3.py -q

Everything is synthetic: fake bag folders (each named <date>_<code>_<take>, holding a
metadata.yaml and optionally an oops/ marker dir) + a processed out/<name>/metadata.json.
extract_episode_details reads the bag IN HAND (a per-bag extraction step); no real bags,
no rosbag deps. Recording order (episode_index) is a report-time ordinal owned by
wrapper.py's session-summary, so it is tested there, not here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rosbag_episode_details_v3 as ed  # noqa: E402


# --------------------------------------------------------------------------- #
# synthetic fixtures                                                          #
# --------------------------------------------------------------------------- #
def write_metadata_yaml(bag_dir: Path, start_ns: int, dur_ns: int | None = 1_000_000_000):
    bag_dir.mkdir(parents=True, exist_ok=True)
    lines = ["rosbag2_bagfile_information:"]
    if dur_ns is not None:
        lines += ["  duration:", f"    nanoseconds: {dur_ns}"]
    lines += ["  starting_time:", f"    nanoseconds_since_epoch: {start_ns}",
              "  message_count: 10"]
    (bag_dir / "metadata.yaml").write_text("\n".join(lines) + "\n")


def write_oops(bag_dir: Path, codes):
    oops = bag_dir / "oops"
    oops.mkdir(parents=True, exist_ok=True)
    for c in codes:
        (oops / c).write_text("")


def make_episode(root: Path, name: str, start_ns: int, *, dur_ns=1_000_000_000,
                 mistakes=(), existing_details=None, extra_meta=None):
    """Create the raw bag (root/bags/name, with metadata.yaml + optional oops/) and the
    processed output (root/out/name/metadata.json). Returns (bag_dir, out_dir)."""
    bag_dir = root / "bags" / name
    write_metadata_yaml(bag_dir, start_ns, dur_ns)
    if mistakes:
        write_oops(bag_dir, mistakes)
    out_dir = root / "out" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"metadata": {"dataset_name": "leo"}, "steps": {"streams": []}}
    if extra_meta:
        meta.update(extra_meta)
    if existing_details is not None:
        meta["episode_details"] = existing_details
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return bag_dir, out_dir


def load(out_dir: Path) -> dict:
    return json.loads((out_dir / "metadata.json").read_text())


# --------------------------------------------------------------------------- #
# A. pure parsers                                                             #
# --------------------------------------------------------------------------- #
def test_parse_episode_folder_valid():
    assert ed.parse_episode_folder("2026-08-04_23_1") == {"date": "2026-08-04", "code": "23", "take": 1}
    assert ed.parse_episode_folder("2026-08-04_pickA_2") == {"date": "2026-08-04", "code": "pickA", "take": 2}


def test_parse_episode_folder_rejects_nonconforming():
    # mid-migration names that don't split into exactly [date, code, take]
    assert ed.parse_episode_folder("2026-08-03_rs-error") is None   # 2 parts
    assert ed.parse_episode_folder("2026-08-04_23") is None         # no take
    assert ed.parse_episode_folder("2026-08-04_23_1_extra") is None # 4 parts
    assert ed.parse_episode_folder("2026-08-04_23_x") is None       # take not int
    assert ed.parse_episode_folder("2026-08-04_23_0") is None       # take < 1
    assert ed.parse_episode_folder("notadate_23_1") is None         # bad date


def test_read_bag_timing(tmp_path):
    bag = tmp_path / "b"
    write_metadata_yaml(bag, start_ns=1785834513211243216, dur_ns=12163103628)
    t = ed.read_bag_timing(bag)
    assert t["start_time_ns"] == 1785834513211243216
    assert t["duration_s"] == round(12163103628 / 1e9, 3)


def test_read_bag_timing_missing_or_partial(tmp_path):
    assert ed.read_bag_timing(tmp_path / "nope") is None            # no metadata.yaml
    bag = tmp_path / "b"
    write_metadata_yaml(bag, start_ns=100, dur_ns=None)             # start only, no duration
    t = ed.read_bag_timing(bag)
    assert t["start_time_ns"] == 100 and t["duration_s"] is None


def test_read_mistakes(tmp_path):
    bag = tmp_path / "b"
    bag.mkdir()
    assert ed.read_mistakes(bag) == []                             # no oops/
    write_oops(bag, ["m3", "m1"])                                   # unordered on disk
    assert ed.read_mistakes(bag) == ["m1", "m3"]                    # returned sorted
    empty = tmp_path / "c"
    (empty / "oops").mkdir(parents=True)
    assert ed.read_mistakes(empty) == []                           # empty oops/


# --------------------------------------------------------------------------- #
# B. extract_episode_details — the per-bag EXTRACTION step                    #
# --------------------------------------------------------------------------- #
def test_extract_episode_details_writes_all_fields(tmp_path):
    bag, out = make_episode(tmp_path, "2026-08-04_23_2", start_ns=500,
                            dur_ns=42_700_000_000, mistakes=["m3", "m1"])
    summary = ed.extract_episode_details(bag=bag, out_dir=out)
    assert summary["written"] is True

    d = load(out)["episode_details"]
    assert d["episode_code"] == "23"
    assert d["take"] == 2
    assert d["date"] == "2026-08-04"
    assert d["start_time_ns"] == 500
    assert d["duration_s"] == 42.7
    assert d["mistakes"] == ["m1", "m3"]           # sorted
    assert d["objects"] == [] and d["actions"] == []
    assert "episode_index" not in d                # index is a report-time ordinal, not stored


def test_extract_episode_details_no_oops_empty_mistakes(tmp_path):
    bag, out = make_episode(tmp_path, "2026-08-04_23_1", start_ns=1)
    ed.extract_episode_details(bag=bag, out_dir=out)
    assert load(out)["episode_details"]["mistakes"] == []


def test_extract_episode_details_noop_without_metadata(tmp_path):
    bag = tmp_path / "bags" / "2026-08-04_23_1"
    write_metadata_yaml(bag, start_ns=1)
    write_oops(bag, ["m1"])
    out = tmp_path / "out" / "2026-08-04_23_1"
    out.mkdir(parents=True)                        # deliberately NO metadata.json
    summary = ed.extract_episode_details(bag=bag, out_dir=out)
    assert summary["written"] is False
    assert not (out / "metadata.json").exists()


def test_extract_episode_details_nonconforming_folder_keeps_mistakes(tmp_path):
    # a mid-migration name that doesn't parse: identity is left OUT, but this step never
    # drops an episode — timing + mistakes are still written.
    bag, out = make_episode(tmp_path, "2026-08-03_rs-error", start_ns=100, mistakes=["m1"])
    ed.extract_episode_details(bag=bag, out_dir=out)
    d = load(out)["episode_details"]
    assert "episode_code" not in d and "take" not in d and "date" not in d
    assert d["start_time_ns"] == 100               # timing still read
    assert d["mistakes"] == ["m1"]


def test_extract_episode_details_missing_yaml_leaves_timing_blank(tmp_path):
    # bag folder with no metadata.yaml: timing left OUT, identity + mistakes still written
    bag = tmp_path / "bags" / "2026-08-04_23_1"
    write_oops(bag, ["m1"])                         # creates the bag dir + oops, NO metadata.yaml
    out = tmp_path / "out" / "2026-08-04_23_1"
    out.mkdir(parents=True)
    (out / "metadata.json").write_text(json.dumps({"metadata": {}}))
    ed.extract_episode_details(bag=bag, out_dir=out)
    d = load(out)["episode_details"]
    assert "start_time_ns" not in d and "duration_s" not in d   # timing blank
    assert d["episode_code"] == "23"               # identity still parsed
    assert d["mistakes"] == ["m1"]


def test_extract_episode_details_preserves_objects_actions(tmp_path):
    # a codebook step already filled objects/actions -> we must not clobber them
    bag, out = make_episode(tmp_path, "2026-08-04_23_1", start_ns=1,
                            existing_details={"objects": ["duvet"], "actions": ["fold"]})
    ed.extract_episode_details(bag=bag, out_dir=out)
    d = load(out)["episode_details"]
    assert d["objects"] == ["duvet"] and d["actions"] == ["fold"]
    assert d["episode_code"] == "23"               # our fields still written


def test_extract_episode_details_preserves_other_metadata_keys(tmp_path):
    bag, out = make_episode(tmp_path, "2026-08-04_23_1", start_ns=1,
                            extra_meta={"termination": {"is_successful": True, "reason": []}})
    before = load(out)
    ed.extract_episode_details(bag=bag, out_dir=out)
    after = load(out)
    assert after["metadata"] == before["metadata"]
    assert after["termination"] == before["termination"]
    assert after["steps"] == before["steps"]


def test_extract_episode_details_idempotent(tmp_path):
    bag, out = make_episode(tmp_path, "2026-08-04_23_1", start_ns=1, mistakes=["m1"])
    ed.extract_episode_details(bag=bag, out_dir=out)
    first = load(out)["episode_details"]
    ed.extract_episode_details(bag=bag, out_dir=out)
    assert load(out)["episode_details"] == first   # stable across re-runs
