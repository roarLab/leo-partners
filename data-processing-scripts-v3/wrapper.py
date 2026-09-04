"""
Full color+depth+imu extraction/validation pipeline wrapper
-----------------------------------------------------------
Runs the whole per-bag pipeline offline (no shell) for a batch of bags and
writes one Excel report row per bag. For EACH bag it runs, in order:

  STEP 1  rosbag_process_color_v3.main(bag, out_dir, meta)   -> CREATES metadata.json
                                                                 + videos/ + timestamps/
  STEP 2  rosbag_process_depth_v3.main(bag, out_dir, camera) -> APPENDS aligned-depth
                                                                 stream + depth_frames/*.h5
                                                                 + depth_to_color extrinsic
  STEP 3  rosbag_process_imu_v3.main(bag, out_dir, camera)   -> APPENDS depth_to_gyro/accel
                                                                 extrinsics + imu/*.csv stream;
                                                                 records missing_stream if /imu
                                                                 is missing or empty
  STEP 4  rosbag_episode_details_v3.extract_episode_details(bag, out_dir) -> WRITES the
                                                                 episode_details block
                                                                 (identity + timing + mistakes),
                                                                 read from the bag in hand
  STEP 5  validate_color_v3.validate_metadata(out_dir)       -> REWRITES termination
                                                                 (color_data/color_timestamps)
  STEP 6  validate_depth_v3.validate_aligned_depth(out_dir)  -> MERGES depth_data/depth_timestamps
  STEP 7  validate_imu_v3.validate_imu(out_dir)              -> MERGES imu_data/imu_timestamps
  STEP 8  sanity_check(out_dir)   (THIS wrapper, FINAL VERDICT) -> declared outputs on disk?

Steps 2, 3 and 7 are gated by each stream's declared presence (DEFAULT_CAMERAS["depth"]
/ ["imu"], present=True by default; opt out per session with a cameras_override like
{"imu": {"present": False}}): a stream a rig did NOT record is skipped entirely — its
extract step, its sanity requirement, and (depth) its validator all drop out, and its
absence is NOT flagged. The live "step n/total" numbering is assigned after the enabled
set is known, so it stays correct when depth/imu are off.

episode_details (STEP 4) is a PER-BAG extraction step: while the raw bag is in hand it
writes identity (folder name) + timing (metadata.yaml) + mistakes (oops/) into that
episode's metadata.json. There is NO post-loop pass. The 1..N recording-order index is
NOT stored per-bag — it is a dataset-view ordinal computed at report time in
write_session_summary (sort on start_time_ns).

metadata.json is the shared spine: Step 1 creates it, Steps 2 & 3 append to it,
Steps 5 & 6 annotate it. Ordering is forced by that file — depth/imu append to the
json color created (each appends no-op if it is absent), and depth-validate MUST run
after color-validate (which rebuilds termination wholesale; depth-validate then
merges only its own "depth" token). The imu step is owner-scoped the same way: it
merges only depth_to_gyro/accel + (on a missing/extra /imu or extrinsic) an
"imu_data"/"imu_info" token, clobbering nothing.

OWNERSHIP
  Steps 1-7 are owned by their existing scripts — this wrapper only calls them and
  passes paths. Sanity (Step 8) and the report step are the wrapper's own.
  episode_details (Step 4) is owned by rosbag_episode_details_v3, run per-bag in the
  loop like the other extractors — not a post-loop pass.

PROGRESS OUTPUT
  Each bag prints a banner (`=== [i/N] <bag> ===`) and one live line per step
  (`→ running …` then `✓ done (Ns) — <summary>` / `✗ FAILED`), then a listing of
  every artifact written with its size, and a one-line per-bag verdict. The batch
  ends with a `<done>/<failed>` count above the report table.
  The step scripts are chatty (topic tables, calibration, a live frame
  counter). By default (QUIET=False, a single config for the whole run) that
  chatter is interleaved inline with the wrapper progress; set QUIET=True to
  redirect it to <out_dir>/pipeline.log so the console shows only the wrapper's
  progress.

INPUT / OUTPUT LAYOUT
  Each `ls` entry is ONE recording SESSION: `source` is the session dir holding the
  episode bag subfolders (one bag = one episode). collect_bags() enumerates them and
  each episode nests under `destination/<episode>`, mirroring the input tree. (A single
  bag folder as `source` still works — it writes straight into `destination`.)

METADATA SEAM
  The 7 descriptive fields (dataset_name, subject, environment, ...) are owned by
  the wrapper as a `meta` dict, passed to color (Step 1) ONLY. Depth never sees
  it. Each `ls` entry may carry its own override; None -> DEFAULT_META.

REPORT  (session-summary.csv, ONE per session in its output dir; one row per episode)
  index,date,episode_code,take,start_time_ns,duration_s,mistakes | all but `index` come
                           from the episode_details block (written per-bag by
                           rosbag_episode_details_v3); `index` is the 1..N recording order
                           computed HERE by sorting rows on start_time_ns (NOT stored in
                           metadata.json). mistakes is a list literal, e.g. ['m1','m3']
  completed              | EXECUTION INTEGRITY: every ENABLED step ran without crashing AND
                           every file a stream DECLARED in metadata.json is on disk (the
                           final metadata-driven sanity verdict). False has two causes, both
                           leaving a pipeline_error.txt: a crash (traceback) or an INCOMPLETE
                           (a declared file absent -> the missing-file list). Distinct from
                           termination.is_successful, which is DATA QUALITY. (True/False)
  color_error            | True = a color stream lost >10% frames
  color_timestamp_error  | True = a color inter-frame gap > 5x mean
  depth_error            | True = aligned h5 broken (missing/shape/count/corruption) OR
                           >10% frames unpaired either direction (validate_depth: depth_data)
  depth_timestamp_error  | True = paired-depth stamp gap > 5x mean (depth_timestamps)
  imu_error              | True = IMU sample-rate off vs expected OR CSV unreadable
                           (validate_imu: imu_data). A mere missing/empty /imu is a PRESENCE
                           failure -> missing_stream_error, NOT this validation column.
  imu_timestamp_error    | True = an IMU inter-sample gap > 5x mean (imu_timestamps)
  missing_stream_error   | True = fewer streams than declared (a camera/depth/imu topic or
                           extrinsic expected but absent, or /imu present but empty) — a
                           PRESENCE check owned by the extraction scripts
  extra_stream_error     | True = more streams than the declared count (an undeclared
                           camera/topic/extrinsic was found; still extracted)
  out_dir                | this episode's output directory
  Error cells are booleans; the full reason message stays in the episode's metadata.json.

CONFIG
  Edit CAMERA, DEFAULT_META and the `ls` list at the bottom. DEFAULT_CAMERAS declares
  the rig's streams: the color groups (ego, exo) AND the ego depth / imu streams, each
  with present=True by default. Set present=False (globally or per session via a
  cameras_override) to skip a stream's step + validation and NOT flag its absence.
  Advanced depth knobs are set once by assigning rpd.<CONST>, e.g.
  rpd.PAIR_TOLERANCE_MS = 20; imu topic overrides likewise via rpi.<CONST>.
"""

import contextlib
import csv
import json
import time
import traceback
from pathlib import Path

import rosbag_process_color_v3 as rpc
import rosbag_process_depth_v3 as rpd
import rosbag_process_imu_v3 as rpi
import validate_color_v3
import validate_depth_v3
import validate_imu_v3
import rosbag_episode_details_v3 as red
import pipeline_calibration as pc
import bag_integrity                              # STEP 0: structural bag check + metadata.json spine creation
from pipeline_metadata import add_error, reorder_top_level

COLOR_KIND = "color"
ALIGNED_KIND = "aligned_depth_to_color"
IMU_KIND = "imu"
# Termination reason token the FINAL sanity verdict owns: a stream declared a file in
# metadata.json that is absent on disk (an INCOMPLETE extraction). It flips
# termination.is_successful; the per-file detail lives only in pipeline_error.txt.
MISSING_FILE_REASON_TOKEN = "missing_file_error"

REPORT_COLUMNS = [
    "out_dir",
    "completed",
    "color_error",
    "color_timestamp_error",
    "depth_error",
    "depth_timestamp_error",
    "imu_error",
    "imu_timestamp_error",
    "missing_stream_error",
    "extra_stream_error",
    "exo_calib",          # ENUM (full/partial/none_usable/null), NOT a boolean error signal
    "rosbag_corruption",  # bag_integrity gate (step 1): bag not structurally openable/indexable
]

# ----------------------------------------------------------------------------
# session-summary.csv: ONE combined file per session (wrapper-owned), written into
# the session's output dir. One row per episode = the episode_details identity/timing
# columns (read from each metadata.json) + the pipeline pass/fail signals as booleans
# + completed + out_dir. The `index` column is the 1..N recording order, computed here
# by sorting rows on start_time_ns (a dataset-view ordinal, not stored per-episode).
# ----------------------------------------------------------------------------
SUMMARY_FILENAME = "session-summary.csv"
ERROR_COLUMNS = ["color_error", "color_timestamp_error", "depth_error",
                 "depth_timestamp_error", "imu_error", "imu_timestamp_error",
                 "missing_stream_error", "extra_stream_error", "rosbag_corruption"]
# exo_calib is a usability ENUM (not booleanised like ERROR_COLUMNS): full = every recorded
# exo cam usable, partial = some, none_usable = cams recorded but none usable, null = no exo
# cams recorded. Reports downstream 3D usability; it does NOT reflect a data-quality failure.
# rosbag_corruption is booleanised (in ERROR_COLUMNS) but placed LAST in the CSV, after the
# exo_calib enum — the bag-level structural gate sits at the far end of the row.
SUMMARY_COLUMNS = ["index", "out_dir", "date", "episode_code", "take", "start_time_ns",
                   "duration_s", "mistakes", "completed",
                   *[c for c in ERROR_COLUMNS if c != "rosbag_corruption"],
                   "exo_calib", "rosbag_corruption"]


def _fmt_mistakes(mistakes) -> str:
    """Mistakes as a single-cell list literal, e.g. ['m1','m3']."""
    return "[" + ",".join(f"'{m}'" for m in (mistakes or [])) + "]"


def build_summary_row(row: dict) -> dict:
    """One session-summary row: the episode_details identity (read from the episode's
    metadata.json) + the pipeline signals from `row`, with the five error columns
    collapsed to booleans (True = tripped). A skipped/failed episode with no
    episode_details keeps blank identity cells but still reports its signals."""
    ed = {}
    meta_path = Path(row.get("out_dir", "")) / "metadata.json"
    if meta_path.is_file():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                ed = json.load(f).get("episode_details") or {}
        except Exception:  # noqa: BLE001
            ed = {}
    rec = {
        # `index` is filled by write_session_summary (report-time ordinal from
        # start_time_ns); it is NOT stored in metadata.json.
        "index": "",
        "date": ed.get("date", ""),
        "episode_code": ed.get("episode_code", ""),
        "take": ed.get("take", ""),
        "start_time_ns": ed.get("start_time_ns", ""),
        "duration_s": ed.get("duration_s", ""),
        "mistakes": _fmt_mistakes(ed.get("mistakes")),
        "completed": bool(row.get("completed")),
        "out_dir": row.get("out_dir", ""),
    }
    for c in ERROR_COLUMNS:
        rec[c] = bool(row.get(c))
    rec["exo_calib"] = row.get("exo_calib", "")   # enum string, carried verbatim (not booleanised)
    return rec


def write_session_summary(rows: list[dict], destination) -> Path:
    """Write ONE session-summary.csv into `destination` (the session output dir): one
    row per episode, ordered by recording `start_time_ns` (the source of truth for
    sequence). The 1..N `index` is a DATASET-VIEW ordinal computed HERE by that sort —
    it is NOT stored in metadata.json. A bag that crashed before episode_details (or a
    non-conforming folder) has no start_time_ns, so those rows sort last by out_dir and
    are numbered after the timed ones — every row still carries an index."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    recs = [build_summary_row(r) for r in rows]

    def _order(r):
        ts = r.get("start_time_ns")
        timed = isinstance(ts, int)          # blank ("") -> untimed -> sorts last
        return (not timed, ts if timed else 0, r.get("out_dir", ""))

    recs.sort(key=_order)
    for i, r in enumerate(recs, start=1):    # ordinal from the start-time sort
        r["index"] = i
    out_path = destination / SUMMARY_FILENAME
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        w.writerows(recs)
    return out_path


# ----------------------------------------------------------------------------
# Bag discovery (single rosbag folder OR a parent of bag folders)
# ----------------------------------------------------------------------------
def is_bag(folder: Path) -> bool:
    """A rosbag folder has a metadata.yaml (or, if flattened, an .mcap/.db3)."""
    if not folder.is_dir():
        return False
    if (folder / "metadata.yaml").exists():
        return True
    return any(f.is_file() and f.suffix in (".mcap", ".db3")
               and not f.name.startswith("._") for f in folder.iterdir())


def collect_bags(source: Path) -> list[Path]:
    """Accept a single rosbag folder OR a parent that contains bag folders."""
    if is_bag(source):
        return [source]
    return [d for d in sorted(source.iterdir())
            if d.is_dir() and not d.name.startswith(".") and is_bag(d)]


def merge_cameras(base: dict, override: dict | None) -> dict:
    """One-level-deep merge of a per-session camera-layout override over `base`
    (rpc.DEFAULT_CAMERAS). Each override key patches that group's fields; groups it
    omits are kept as-is. None -> `base` unchanged. This is what lets a shared setup
    tweak e.g. the exo label for one session, or opt a session out of a stream
    ({"imu": {"present": False}}), without restating the whole layout."""
    if not override:
        return base
    merged = {grp: dict(vals) for grp, vals in base.items()}
    for grp, vals in override.items():
        merged[grp] = {**merged.get(grp, {}), **(vals or {})}
    return merged


def stream_present(cameras: dict | None, name: str) -> bool:
    """Whether the ego device stream `name` ("depth" / "imu") is declared present in
    the camera layout. Declared alongside the color groups in DEFAULT_CAMERAS, so a
    per-session cameras_override can opt out ({"imu": {"present": False}}) through the
    SAME channel as a camera tweak. Default = present: an absent key (or None cameras)
    means "assume the rig recorded it", so nothing changes for rigs that have both.
    present=False skips that stream's extract step, its sanity requirement, and (depth)
    its validator, and does NOT flag its absence."""
    return bool((cameras or {}).get(name, {}).get("present", True))


def color_group_present(cameras: dict | None) -> bool:
    """Whether ANY colour group (ego or exo) is declared present. Gates the VALIDATE COLOUR
    step ONLY. validate_color's sole role is per-frame QUALITY of FOUND colour streams —
    presence (a missing/extra stream) is extraction's `color_presence_err`, not its job — so
    with no colour declared it has nothing to validate, and its termination recompute is
    redundant: no colour forces `imu` on (depth requires ego, so ego off => depth off), and
    validate_imu then recomputes the verdict. The colour EXTRACT step is NOT gated by this —
    it writes the metadata.json spine and owns presence detection, so it always runs."""
    c = cameras or {}
    return bool(c.get("ego", {}).get("present", True) or c.get("exo", {}).get("present", True))


def validate_cameras(cameras: dict) -> None:
    """CONFIG-plane guard: reject a self-contradictory camera config LOUDLY, before any
    extraction runs. This is the deliberate opposite of the per-bag DATA plane (missing /
    extra streams, which flag-and-continue): a contradiction here is an operator mistake in
    DEFAULT_CAMERAS or a cameras_override, so we crash rather than silently produce a wrong
    dataset. Run on the MERGED config (merge_cameras output), so a default contradiction and
    an override-induced one hit the exact same check.

    Raises ValueError on:
      - a missing required color group ('ego' / 'exo' — rpc.main subscripts them directly,
        so an override that drops one would otherwise KeyError mid-bag);
      - depth declared present while ego color is absent — depth aligns ONTO the ego color
        stream (rosbag_process_depth_v3 raises 'No color frames to index against'), so this
        combination cannot produce depth. Caught here as an early, readable crash instead;
      - nothing declared present at all (ego color, exo color, depth, imu all off) — an empty
        session with nothing to extract.
    imu is intentionally NOT constrained against color: it is a standalone stream."""
    for grp in ("ego", "exo"):
        if grp not in cameras:
            raise ValueError(f"camera group '{grp}' is required but missing from the config")

    ego_present = bool(cameras["ego"].get("present", True))
    exo_present = bool(cameras["exo"].get("present", True))
    depth_present = stream_present(cameras, "depth")
    imu_present = stream_present(cameras, "imu")

    if depth_present and not ego_present:
        raise ValueError(
            "depth requires ego color: depth aligns onto the ego color stream, but "
            "ego.present=False while depth.present=True. "
            "Set ego present:True, or depth present:False.")

    if not (ego_present or exo_present or depth_present or imu_present):
        raise ValueError(
            "no streams declared present (ego color, exo color, depth, imu all off) — "
            "nothing to extract; check the config.")


def validate_camera_configs(ls, base: dict) -> None:
    """Pre-flight EVERY session's resolved config before a single bag is processed, so a
    contradictory cameras_override fails the whole batch at startup rather than mid-run.
    Mirrors the per-session merge in __main__ (merge_cameras(base, override)) and runs the
    same validate_cameras gate on each merged result; on failure re-raises naming the
    offending session path, since with many `ls` rows you need to know which one is broken.
    `ls` rows are (session_dir, output_root, meta_override, cameras_override)."""
    for row in ls:
        source = row[0]
        cameras_override = row[3] if len(row) > 3 else None
        try:
            validate_cameras(merge_cameras(base, cameras_override))
        except ValueError as e:
            raise ValueError(f"invalid camera config for session '{source}': {e}") from e


def validate_calib_rows(ls, default_calib) -> None:
    """Pre-flight EVERY session's resolved exo-calibration FILE before a single bag runs.
    Exo calibration is MANDATORY for this dataset — a session without a usable calib file
    cannot be delivered as a multi-view capture — so an unset / missing / malformed calib
    file is an operator config error that fails the WHOLE batch at startup, the same
    CONFIG-plane fail-fast as validate_camera_configs (not a mid-run per-bag flag). Mirrors the
    per-session resolve in __main__: `calib_override or DEFAULT_CALIB_JSON`.

    Raises ValueError (naming the offending session) when a row's resolved calib path is
    empty/unset, does not point at an existing file, or does not parse as JSON.

    NOT checked here (DATA-plane, needs the recorded streams -> lives in merge_calibration):
    per-episode COVERAGE (every recorded exo cam present). Also not here: per-camera solve
    QUALITY (status != 'ok'), which is tolerated (null + note)."""
    for row in ls:
        source = row[0]
        calib_override = row[4] if len(row) > 4 else None
        calib_path = calib_override or default_calib
        if not calib_path:
            raise ValueError(
                f"no exo calibration configured for session '{source}': set DEFAULT_CALIB_JSON "
                f"or the row's 5th field to a calib-<date>.json (calibration is mandatory).")
        calib_file = Path(calib_path)
        if not calib_file.is_file():
            raise ValueError(
                f"exo calibration file not found for session '{source}': {calib_file} "
                f"(calibration is mandatory — fix the path or supply the file).")
        try:
            pc.load_calib(calib_file)                      # strict parse; JSONDecodeError is a ValueError
        except ValueError as e:
            raise ValueError(
                f"exo calibration file for session '{source}' is not valid JSON: "
                f"{calib_file} ({e}).") from e


# ----------------------------------------------------------------------------
# FINAL VERDICT — sanity check (wrapper-owned), run LAST and METADATA-DRIVEN: every
# file a stream DECLARED in metadata.json must exist on disk. A miss => the bag is
# INCOMPLETE (completed False) and a pipeline_error.txt marker is written (with the
# missing-file list, not a traceback). This checks EXECUTION INTEGRITY (did we produce
# what we said we did), NOT data quality (that is termination.reason). A stream that was
# never recorded (a missing topic left only a missing_stream flag, no stream entry, or a
# null path field) declares nothing here, so it does NOT fail this check.
# ----------------------------------------------------------------------------
def sanity_check(out_dir: Path) -> list[dict]:
    """Declared-output presence check. For every path a stream recorded in metadata.json
    (steps.streams[] -> video / timestamps / frames_dir / file), that file must exist on
    disk. Returns the declared-but-absent files as dicts {camera, kind, field, path}
    (empty list = every declared output present).

    metadata.json itself is the spine: if it is absent the bag produced nothing, reported
    as the single missing output. A stream that declared no path (a missing topic ->
    found:False, or a null field) is skipped — its absence is a data-quality flag
    (termination), not an incomplete extraction (completed)."""
    meta_path = out_dir / "metadata.json"
    if not meta_path.is_file():
        return [{"camera": "-", "kind": "-", "field": "metadata.json", "path": "metadata.json"}]
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:  # noqa: BLE001 — unreadable metadata is caught upstream (report block)
        return []
    missing: list[dict] = []
    for s in meta.get("steps", {}).get("streams", []):
        for field in ("video", "timestamps", "frames_dir", "file"):
            rel = s.get(field)
            if rel and not (out_dir / rel).exists():
                missing.append({"camera": s.get("camera", "?"), "kind": s.get("kind", "?"),
                                "field": field, "path": rel})
    return missing


# ----------------------------------------------------------------------------
# STEP 6 — report signals (wrapper-owned): read the final annotated metadata.json
# ----------------------------------------------------------------------------
def _join(messages: list[str]) -> str:
    return " | ".join(m for m in messages if m)


def _exo_calib_status(streams: list[dict], cam_intrinsics: list[dict]) -> str:
    """Per-episode exo-calibration USABILITY enum for the summary (not an error signal):
      full        - every recorded exo cam has a usable calib block (status ok, non-null K)
      partial     - some recorded exo cams usable, some not (broken or missing)
      none_usable - exo cams recorded, but none usable
      null        - no exo cams recorded at all
    Ego is excluded on purpose: a realsense D435i ships factory-calibrated, so ego is always
    available and is not an exo concern. Exo cams are identified by the 'exo' label prefix
    (the DEFAULT_CAMERAS exo label + the calib file's own exo_camN keys)."""
    recorded = {s.get("camera") for s in streams
                if s.get("kind") == COLOR_KIND and str(s.get("camera", "")).startswith("exo")}
    if not recorded:
        return "null"
    usable = {c.get("camera") for c in cam_intrinsics
              if str(c.get("camera", "")).startswith("exo")
              and (c.get("color") or {}).get("status") == "ok"
              and (c.get("color") or {}).get("K") is not None}
    n_ok = len(recorded & usable)
    if n_ok == len(recorded):
        return "full"
    if n_ok == 0:
        return "none_usable"
    return "partial"


def extract_signals(out_dir: Path) -> dict:
    """Pull the error signals out of the fully-annotated metadata.json (written by
    Steps 1,2,4,5). Each value is the joined reason message(s) when the check
    tripped, else an empty string. The two count-deviation signals come from the
    steps-level lists the EXTRACTION scripts own (missing_stream_error /
    extra_stream_error), not from per-stream annotations."""
    signals = {c: "" for c in REPORT_COLUMNS if c not in ("out_dir", "completed")}
    meta_path = out_dir / "metadata.json"
    if not meta_path.is_file():
        return signals
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    steps = meta.get("steps", {})
    streams = steps.get("streams", [])

    color_err, color_ts = [], []
    for s in (s for s in streams if s.get("kind") == COLOR_KIND):
        cam = s.get("camera", "unknown")
        color_err += [f"{cam}: {m}" for m in (s.get("data_error") or [])]
        color_ts += [f"{cam}: {m}" for m in (s.get("timestamps_error") or [])]

    depth_err, depth_ts = [], []
    for s in (s for s in streams if s.get("kind") == ALIGNED_KIND):
        depth_ts += list(s.get("timestamps_error") or [])
        depth_err += list(s.get("data_error") or [])   # h5 integrity + two-way pairing

    imu_err, imu_ts = [], []
    for s in (s for s in streams if s.get("kind") == IMU_KIND):
        imu_ts += list(s.get("timestamps_error") or [])
        imu_err += list(s.get("data_error") or [])      # sample-rate + CSV integrity

    signals["color_error"] = _join(color_err)
    signals["color_timestamp_error"] = _join(color_ts)
    signals["depth_error"] = _join(depth_err)
    signals["depth_timestamp_error"] = _join(depth_ts)
    signals["imu_error"] = _join(imu_err)
    signals["imu_timestamp_error"] = _join(imu_ts)
    # missing_stream_error / extra_stream_error carry the extraction-owned PRESENCE
    # failures across all streams (missing/extra colour, depth, imu topics/extrinsics).
    signals["missing_stream_error"] = _join(steps.get("missing_stream_error") or [])
    signals["extra_stream_error"] = _join(steps.get("extra_stream_error") or [])
    # rosbag_corruption: bag_integrity's structural verdict, read from termination.reason
    # (the detail, if any, lives in steps.bag_corruption_error). A corrupt bag skips all
    # extraction, so this is the one signal that can be set with an otherwise-empty spine.
    if "rosbag_corruption" in ((meta.get("termination") or {}).get("reason") or []):
        signals["rosbag_corruption"] = _join(steps.get("bag_corruption_error")
                                             or ["structural bag corruption"])
    # exo-calibration usability enum (not an error): computed from recorded exo streams vs
    # which of them carry a usable calib block. Detail is in steps.exo_calib_notes.
    signals["exo_calib"] = _exo_calib_status(streams, meta.get("camera_intrinsics", []))
    return signals


# ----------------------------------------------------------------------------
# Progress reporting (wrapper-owned): step banners, per-step summaries, file list
# ----------------------------------------------------------------------------
def _fmt_size(nbytes: int) -> str:
    mb = nbytes / (1024 * 1024)
    return f"{mb / 1024:.2f} GB" if mb >= 1024 else f"{mb:.1f} MB"


def _run_step(n: int, total: int, label: str, fn, log, summarize=None):
    """Announce step `n`, time it, capture its stdout to `log` (a file handle, or
    None to leave it inline), and print a one-line ✓/✗ verdict. Returns fn()'s
    value; on the ✓ line, appends `summarize(result)` when a summarizer is given.
    Re-raises on failure (after the ✗ line) so the caller's handler still runs."""
    print(f"  → step {n}/{total}: {label} …", flush=True)
    t0 = time.time()
    try:
        if log is not None:
            log.write(f"\n===== step {n}/{total}: {label} =====\n")
            log.flush()
            with contextlib.redirect_stdout(log):
                result = fn()
        else:
            result = fn()
    # KeyboardInterrupt (BaseException) is intentionally not caught here.
    except (Exception, SystemExit) as e:  # noqa: BLE001
        print(f"  ✗ step {n}/{total}: {label} FAILED ({time.time() - t0:.1f}s) — {e}",
              flush=True)
        raise
    extra = ""
    if summarize is not None:
        try:
            extra = f" — {summarize(result)}"
        except Exception:  # a summary bug must never break the pipeline
            extra = ""
    print(f"  ✓ step {n}/{total}: {label} ({time.time() - t0:.1f}s){extra}", flush=True)
    return result


def _summary_color(md: dict) -> str:
    steps = (md or {}).get("steps", {})
    streams = steps.get("streams", []) or []
    found = [s for s in streams if s.get("found", True)]
    frames = sum(int(s.get("num_frames", 0) or 0) for s in streams)
    ok = (md or {}).get("termination", {}).get("is_successful")
    tail = "" if ok in (None, True) else "  [termination NOT successful]"
    return f"{len(found)}/{len(streams)} streams, {frames} frames{tail}"


def _summary_depth(s: dict) -> str:
    if not s:
        return "no summary returned"
    calib = "calib ok" if s.get("calib_ok") else "CALIB CHECK FAILED"
    return (f"{s.get('n_color', '?')} color-indexed / {s.get('n_paired', '?')} paired, "
            f"max dt {s.get('max_pair_dt_ms', '?')} ms, {calib}")


def _summary_imu(s: dict) -> str:
    if not s:
        return "no summary returned"
    exts = s.get("extrinsics_found") or []
    ext_txt = ", ".join(exts) if exts else "no extrinsics"
    tail = "  [IMU MISSING/EMPTY]" if s.get("imu_missing") else ""
    return f"{s.get('imu_samples', 0)} samples, {len(exts)} extrinsic(s) ({ext_txt}){tail}"


def _summary_episode_details(s: dict) -> str:
    if not s:
        return "no summary returned"
    ms = s.get("mistakes") or []
    return (f"code={s.get('code')} start={s.get('start_time_ns')} "
            f"{len(ms)} mistake(s): {ms or '[]'}")


def _list_outputs(out_dir: Path) -> None:
    """Print every artifact written for this bag, grouped, with file sizes."""
    subdirs = ["videos", "timestamps", "depth_frames", "overlays"]
    files = [f for f in [out_dir / "metadata.json"] if f.is_file()]
    for d in subdirs:
        if (out_dir / d).is_dir():
            files += sorted(f for f in (out_dir / d).iterdir() if f.is_file())
    print("  files:")
    if not files:
        print("      (none)")
        return
    for f in files:
        print(f"      {str(f.relative_to(out_dir)):<46s} {_fmt_size(f.stat().st_size):>10s}")


# ----------------------------------------------------------------------------
# Per-bag orchestration (owns sequencing + isolation; the extract/validate steps
# delegate). Sanity is NOT a step — it is the post-loop FINAL VERDICT (below).
# ----------------------------------------------------------------------------
def _record_crash(err_path: Path, bag: Path, exc) -> None:
    """Record a TRUE crash: print the failure, dump the traceback, and persist it to
    `err_path`. Used both for a step that raised and for an unreadable final
    metadata.json (a corrupt final artifact is itself a pipeline error). Distinct from
    a graceful "incomplete" verdict (a declared output missing from disk), which writes
    NO error file — the reason there already lives in metadata.json + the summary row.
    Must be called from inside an `except` block (relies on the live exception for the
    traceback)."""
    print(f"[pipeline] {bag.name}: FAILED — {exc}")
    traceback.print_exc()
    with open(err_path, "w", encoding="utf-8") as f:
        f.write(f"{exc}\n")
        f.write(f"bag: {bag}\n")
        f.write(traceback.format_exc())


def _summary_integrity(s: dict) -> str:
    """One-line verdict for the bag-integrity step."""
    if s.get("corrupt"):
        return f"CORRUPT — {'; '.join(s.get('detail', []))}"
    return "spine written (metadata.json)"


def _record_corrupt(err_path: Path, bag: Path, detail: list[str]) -> None:
    """Record a CORRUPT bag: bag_integrity could not open/index it, so extraction and
    validation were skipped. Durable per-bag marker; the verdict itself already lives in
    metadata.json (rosbag_corruption + is_successful False)."""
    print(f"[pipeline] {bag.name}: CORRUPT — {'; '.join(detail)}")
    with open(err_path, "w", encoding="utf-8") as f:
        f.write("CORRUPT — bag failed the structural integrity check (rosbag_corruption)\n")
        f.write(f"bag: {bag}\n")
        for d in detail:
            f.write(f"  - {d}\n")


def _record_incomplete(err_path: Path, bag: Path, missing: list[dict]) -> None:
    """Record an INCOMPLETE bag: it ran to the end without crashing, but a stream
    declared a file in metadata.json that is absent on disk (silent extraction loss).
    Mirrors _record_crash as a durable per-bag marker, but lists the missing declared
    files instead of a traceback. Line 1 (`INCOMPLETE — ...`) is what distinguishes this
    file from a crash file (an exception message + traceback) at a glance."""
    print(f"[pipeline] {bag.name}: INCOMPLETE — {len(missing)} declared output(s) missing")
    with open(err_path, "w", encoding="utf-8") as f:
        f.write("INCOMPLETE — declared outputs missing from disk\n")
        f.write(f"bag: {bag}\n")
        for m in missing:
            f.write(f"  - {m['camera']} ({m['kind']}) {m['field']}: {m['path']}\n")


def _flag_incomplete_in_metadata(out_dir: Path) -> None:
    """The machine-readable twin of the pipeline_error.txt marker: add the
    'missing_file_error' token to termination.reason in metadata.json (owner-scoped, run
    last) and recompute is_successful, so the json alone flags the extraction incomplete.
    The per-file detail lives only in pipeline_error.txt — no steps key is written here.
    add_error dedups and never clobbers another writer's reasons. No-op if metadata.json
    is absent — completed:false + the marker already carry it."""
    meta_path = out_dir / "metadata.json"
    if not meta_path.is_file():
        return
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:  # noqa: BLE001 — an unreadable json is already a crash upstream
        return
    term = meta.setdefault("termination", {"is_successful": True, "reason": []})
    add_error(term, "reason", [MISSING_FILE_REASON_TOKEN])
    term["is_successful"] = not term.get("reason")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _reorder_metadata_on_disk(out_dir: Path) -> None:
    """Rewrite metadata.json with its top-level keys in the canonical order
    (reorder_top_level). COSMETIC and value-preserving; GUARDED so a missing or corrupt
    metadata simply no-ops rather than failing the bag — a key-order pass must never
    change a verdict. Runs last of all, after every content writer."""
    meta_path = Path(out_dir) / "metadata.json"
    try:
        meta = json.loads(meta_path.read_text())
        meta_path.write_text(json.dumps(reorder_top_level(meta), indent=2))
    except (OSError, ValueError):  # absent / unreadable / invalid json -> leave as-is
        pass


def run_pipeline_for_bag(bag: Path, out_dir: Path, camera: str, meta: dict,
                         cameras: dict | None = None, calib_path=None,
                         *, idx: int = 1, total: int = 1, quiet: bool = False) -> dict:
    bag = Path(bag)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    err_path = out_dir / "pipeline_error.txt"
    row = {c: "" for c in REPORT_COLUMNS}
    row["out_dir"] = str(out_dir)
    row["completed"] = False

    # Which optional ego streams to process — declared in the camera layout (default
    # present); a session opts out via cameras_override, e.g. {"imu": {"present": False}}.
    process_depth = stream_present(cameras, "depth")
    process_imu = stream_present(cameras, "imu")
    # Gates BOTH colour steps (extract + validate). bag_integrity owns the spine now, so colour
    # extract is a plain declaration-gated appender like depth/imu — with no colour declared there
    # is nothing to extract, and validate_color (colour QUALITY of found streams) has nothing to
    # check (presence is extraction's color_presence_err).
    process_color = color_group_present(cameras)

    print(f"\n=== [{idx}/{total}] {bag.name} ===", flush=True)
    print(f"  src: {bag}")
    print(f"  out: {out_dir}")
    log = None
    if quiet:
        log = open(out_dir / "pipeline.log", "w", encoding="utf-8")
        print("  log: pipeline.log  (full step detail; set QUIET=False for inline)")

    t_bag = time.time()
    crashed = False
    bag_corrupt = False
    corrupt_detail: list[str] = []
    try:
        # STEP 1 (always first): bag_integrity — structural bag check + metadata.json spine
        # creation. It OWNS the spine and the rosbag_corruption token. A step that RAISES is a
        # true crash (breaks the loop -> later steps skipped -> completed False).
        #   Extraction + validation are STEPS 2..N; they only run if the bag opens cleanly —
        # a CORRUPT bag writes just the spine (verdict) and skips them all. Each of those
        # steps flags-and-continues internally; the validators annotate the spine. A skipped
        # depth/imu stream drops its extract AND its validator; colour is the same — its
        # extract is an appender gated on `process_color` (no colour declared -> nothing to
        # extract or validate). bag_integrity is the ONLY unconditionally-always step; every
        # extractor now runs only when its stream is declared. (The config guard forbids
        # depth-without-ego-colour, so process_color is always true when depth is on; gating
        # here only skips the imu-only config, where colour extract would be a no-op.)
        rest = []
        if process_color:
            rest.append(("color extract",
                          lambda: rpc.main(bag=bag, out_dir=out_dir, meta=meta, cameras=cameras),
                          _summary_color))
        if process_depth:
            rest.append(("depth align",
                          lambda: rpd.main(bag=bag, out_dir=out_dir, camera=camera),
                          _summary_depth))
        if process_imu:
            rest.append(("imu extract",
                          lambda: rpi.main(bag=bag, out_dir=out_dir, camera=camera),
                          _summary_imu))
        # episode_details is not stream-gated: identity/timing/mistakes apply to every episode.
        rest.append(("episode details",
                      lambda: red.extract_episode_details(bag=bag, out_dir=out_dir),
                      _summary_episode_details))
        # calib merge: stamp the session's exo calibration into the spine (after it
        # exists). No-ops cleanly if calib_path is unset/absent; the exo intrinsics/
        # extrinsics + ok/broken/missing gate live in pipeline_calibration.
        rest.append(("calib merge",
                      lambda: pc.annotate_calibration(out_dir, calib_path),
                      lambda s: (f"merged {len(s.get('cameras', []))} exo cam(s)"
                                 f"{', broken=' + str(s['broken']) if s.get('broken') else ''}"
                                 if s.get("written") else f"skipped ({s.get('reason')})")))
        if process_color:
            rest.append(("validate color",
                          lambda: validate_color_v3.validate_metadata(out_dir),
                          lambda _: "metadata termination rebuilt"))
        if process_depth:
            rest.append(("validate depth",
                          lambda: validate_depth_v3.validate_aligned_depth(out_dir),
                          lambda _: "depth token merged"))
        if process_imu:
            rest.append(("validate imu",
                          lambda: validate_imu_v3.validate_imu(out_dir),
                          lambda _: "imu tokens merged"))

        total_steps = 1 + len(rest)
        # STEP 1: bag integrity + spine. Resolve the descriptive meta once (bag_integrity
        # writes the metadata block; colour and the rest only append to the spine it creates).
        integ = _run_step(1, total_steps, "bag integrity",
                          lambda: bag_integrity.init_spine(bag, out_dir, rpc._resolve_meta(meta)),
                          log, _summary_integrity)
        bag_corrupt = bool(integ.get("corrupt"))
        corrupt_detail = list(integ.get("detail", []))
        if bag_corrupt:
            print(f"  ⚠ bag corrupt — skipping steps 2..{total_steps} (extraction + validation)",
                  flush=True)
        else:
            for i, (label, fn, summ) in enumerate(rest, 2):
                _run_step(i, total_steps, label, fn, log, summ)
    # SystemExit is how the extraction scripts surface fatal input errors (missing
    # topic, unreadable bag); catch it too so one bad bag can't kill the batch.
    # KeyboardInterrupt (BaseException, not Exception) is intentionally NOT caught.
    except (Exception, SystemExit) as e:  # noqa: BLE001
        crashed = True
        _record_crash(err_path, bag, e)
    finally:
        if log is not None:
            log.close()

    # --- report signals: ALWAYS read the (fully-annotated) metadata, even for a bag
    # that crashed or is incomplete — otherwise the very bags that failed would carry
    # the LEAST useful summary rows. extract_signals no-ops cleanly on absent metadata;
    # a metadata.json that cannot be PARSED here is itself a pipeline error (corrupt
    # final artifact), so it flips `crashed`.
    try:
        row.update(extract_signals(out_dir))
        _list_outputs(out_dir)
    except (Exception, SystemExit) as e:  # noqa: BLE001
        # First-writer-wins: if the step loop already crashed and wrote err_path, keep
        # that ORIGINAL traceback rather than clobbering it with this follow-on failure.
        if not crashed:
            _record_crash(err_path, bag, e)
        crashed = True

    # --- FINAL VERDICT (sanity, last): complete only if every step ran AND every file a
    # stream DECLARED in metadata.json is on disk. A declared-but-absent file is a graceful
    # INCOMPLETE (completed False) — it did NOT crash, but it DOES leave a pipeline_error.txt
    # marker (the missing-file list) so the failure is durable, not just a console line.
    # Skip the disk check entirely if we crashed OR the bag was corrupt (nothing was
    # extracted, so nothing is declared). A CORRUPT bag is completed:False by definition —
    # its verdict already lives in metadata.json (rosbag_corruption + is_successful False).
    missing = [] if (crashed or bag_corrupt) else sanity_check(out_dir)
    row["completed"] = (not crashed) and (not bag_corrupt) and (not missing)
    if row["completed"]:
        if err_path.is_file():
            err_path.unlink()                                  # clean run clears a stale marker
    elif crashed:
        pass                                                   # _record_crash already wrote err_path
    elif bag_corrupt:
        _record_corrupt(err_path, bag, corrupt_detail)         # CORRUPT -> durable marker
    else:
        _record_incomplete(err_path, bag, missing)             # INCOMPLETE -> durable marker
        _flag_incomplete_in_metadata(out_dir)                  # ...and the token in metadata.json

    # Cosmetic FINAL pass: canonical top-level key order. Runs LAST and ALWAYS — after
    # every writer, including the incomplete-flag write just above — and is guarded so a
    # missing/corrupt metadata never turns a cosmetic reorder into a bag failure.
    _reorder_metadata_on_disk(out_dir)

    dt = time.time() - t_bag
    if row["completed"]:
        tripped = [c for c in REPORT_COLUMNS
                   if c not in ("out_dir", "completed", "exo_calib") and row.get(c)]
        verdict = "no errors" if not tripped else \
            f"{len(tripped)} error(s): {', '.join(tripped)}"
        print(f"  ⇒ [{idx}/{total}] {bag.name}: completed in {dt:.1f}s — {verdict}")
    elif crashed:
        print(f"  ⇒ [{idx}/{total}] {bag.name}: FAILED in {dt:.1f}s — see {err_path.name}")
    elif bag_corrupt:
        print(f"  ⇒ [{idx}/{total}] {bag.name}: CORRUPT in {dt:.1f}s — see {err_path.name}")
    else:
        print(f"  ⇒ [{idx}/{total}] {bag.name}: INCOMPLETE in {dt:.1f}s — see {err_path.name}")
    return row


def run_pipeline_dir(source, destination, camera: str, meta: dict,
                     cameras: dict | None = None, calib_path=None,
                     *, quiet: bool = False) -> list[dict]:
    source = Path(source)
    destination = Path(destination)
    bags = collect_bags(source)
    if not bags:
        print(f"[warn] no rosbag found at/under {source}")
        return []
    single = is_bag(source)   # source IS the bag -> output straight into destination
    total = len(bags)
    print(f"[batch] {total} bag(s) to process under {source}")
    rows = []
    for i, bag in enumerate(bags, 1):
        out_dir = destination if single else destination / bag.name
        rows.append(run_pipeline_for_bag(bag, out_dir, camera, meta, cameras, calib_path,
                                         idx=i, total=total, quiet=quiet))
    return rows


if __name__ == "__main__":
    CAMERA = "ego"                                   # substring identifying the ego D435i
    QUIET = False    # False (default): everything inline — each bag's step chatter is
                     # interleaved with the wrapper progress on the console. True: step
                     # chatter -> <out_dir>/pipeline.log, console shows only wrapper progress.

    # Descriptive metadata written into metadata.json["metadata"] by Step 1 (color).
    # Applied to every bag unless an `ls` entry overrides it.
    DEFAULT_META = {
        "dataset_name": "leo",
        "dataset_version": "4.0",
        "robot_model": "human",
        "environment": "sutd parcel d",
        "setup": "v7",
        "subject": "",
        "bed_type": "single",
    }
    
    # Default exo calibration for every session, unless an ls row overrides it (5th field).
    # Exo calibration is MANDATORY: this path MUST resolve to an existing, valid calib-<date>.json.
    # A missing/unset/malformed file fails the WHOLE batch at startup (validate_calib_rows below).
    # The value below is a placeholder — set it to your real calib file (or override per session)
    # or the run refuses to start. (This is deliberate: a broken calib path fails LOUD, not silent.)
    DEFAULT_CALIB_JSON = "filepath.json"

    # Rig stream layout (Step 1+) — CANONICAL, edit here. The color GROUPS (ego, exo) are
    # DISCOVERED by topic suffix and written as cam_ego / exo_cam<id>. singleton=True -> one
    # topic, label used verbatim (cam_ego); otherwise topics are labelled label+<trailing
    # device number>. exo.ids is the SET of webcam device numbers to expect: a declared id
    # not found -> missing_stream, a found id not declared -> extra_stream (add an id to
    # include a new webcam cleanly; omit one, e.g. [1, 2, 4], to skip cam3). info_suffix ->
    # the camera_info carrying that group's intrinsics.
    #   The ego non-color STREAMS (depth, imu) are declared the SAME way with present=True
    # (assume the rig recorded them). Set present=False — globally here or per session via a
    # cameras_override, e.g. {"imu": {"present": False}} — to SKIP that stream's extract step,
    # its sanity requirement and (depth) its validator, and NOT flag its absence. Left
    # present, a missing/empty topic trips missing_stream (both depth and imu — a presence
    # check, not data validation). Their `suffix` is documentation; discovery lives in rpd / rpi.
    #   Applied to every session unless an `ls` row overrides it (a per-session patch, e.g.
    # {"exo": {"label": "side_cam"}}). (rpc.DEFAULT_CAMERAS is the color script's STANDALONE
    # fallback for its own run / unit tests; it stays RAW ego ON PURPOSE — this OPERATOR
    # default records ego color COMPRESSED. For an old raw-ego bag, set the ego group back to
    # suffix "d435i_ego/color/image_raw" + compressed:False; depth auto-handles either.)
    #   The COLOR groups (ego, exo) take the SAME present flag: present:False skips that
    # group's extract + flagging (main still writes the spine). depth requires ego color,
    # so validate_cameras rejects depth present + ego present:False before any bag runs.
    DEFAULT_CAMERAS = {
        "ego": {"present": True,
                "suffix": "d435i_ego/color/image_raw/compressed",
                "info_suffix": "d435i_ego/color/camera_info",
                "compressed": True,
                "label": "cam_ego",
                "singleton": True},
        "exo": {"present": True,
                "suffix": "image_raw/compressed",
                "compressed": True,  "label": "exo_cam", "ids": [1, 2, 3, 4]},
        "depth": {"present": True, "suffix": "d435i_ego/depth/image_rect_raw"},
        "imu":   {"present": True, "suffix": "d435i_ego/imu"},
    }

    # Advanced depth knobs (uncomment to change from defaults):
    # rpd.PAIR_TOLERANCE_MS = 16.0
    # rpd.HOLE_FILL = True

    # (session_dir, output_root, meta_override, cameras_override) — ONE row per session.
    #   session_dir = a folder of episode bag subfolders -> each episode nests under
    #                 output_root/<episode>; the whole session is processed in one go.
    #   (a single bag folder as session_dir still works -> writes straight into output_root.)
    #   meta_override    = None -> use DEFAULT_META;    or a partial dict to override per session.
    #   cameras_override = None -> use DEFAULT_CAMERAS; or a partial group patch per session.
    #   calib_json_filepath_override = None -> USE_DEFAULT_JSON
    ls = [
        ("/Volumes/TRANSCEND/test-sessions",  
         "./test/test-sessions",  None, None, None),
    ]

    # Config-plane pre-flight: reject a contradictory config (e.g. depth on + ego color
    # off, or all streams off) for ANY session BEFORE opening a single bag — a loud crash
    # at startup, not a mid-batch surprise. Extraction below is reached only if this passes.
    validate_camera_configs(ls, DEFAULT_CAMERAS)
    validate_calib_rows(ls, DEFAULT_CALIB_JSON)     # exo calibration mandatory: fail fast on a bad/absent file

    start = time.time()
    print(f"start time is {start}")
    all_rows = []
    summaries = []
    for source, destination, meta_override, cameras_override, calib_override in ls:
        meta = {**DEFAULT_META, **(meta_override or {})}
        cameras = merge_cameras(DEFAULT_CAMERAS, cameras_override)
        calib_path = calib_override or DEFAULT_CALIB_JSON     # None -> session-default calib
        rows = run_pipeline_dir(source, destination, CAMERA, meta, cameras, calib_path, quiet=QUIET)
        all_rows.extend(rows)
        # episode_details is written PER-BAG in the loop above (identity/timing/mistakes);
        # there is no post-loop pass. The 1..N recording-order index is a dataset-view
        # ordinal computed at report time below (sort on start_time_ns), not stored per-bag.
        # ONE combined session-summary.csv in the session output dir.
        summary_path = write_session_summary(rows, destination)
        summaries.append(summary_path)
        print(f"[summary] wrote {summary_path}")

    end = time.time()
    n_done = sum(1 for r in all_rows if r.get("completed"))
    n_fail = len(all_rows) - n_done
    print(f"\n===== batch complete: {n_done} completed / {n_fail} failed "
          f"out of {len(all_rows)} bag(s) =====")
    for p in summaries:
        print(f"  summary: {p}")
    print(f"total time is {end - start:.1f}s")
    print("safely exit")
