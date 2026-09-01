#!/usr/bin/env python3
"""
Exo-calibration merge  (pipeline_calibration)
---------------------------------------------
A SESSION-scoped annotation step of the v3 pipeline. Where color/depth/imu read one
raw bag, this reads one OFFLINE calibration file (`calib-<date>.json`, produced by the
ChArUco solve) and stamps its per-camera intrinsics + extrinsics into an already-
extracted episode `metadata.json`. One calib = one rig setup = applies to every
episode in the session, so the wrapper calls this for each episode's out_dir with the
same calib file.

WHAT IT MERGES (see CALIB_MERGE_PLAN / PIPELINE.md):
  calib.cameras.exo_camN.intrinsics  -> camera_intrinsics[]  (K nested 3x3 -> flat-9,
                                        dist -> D, image_size -> width/height)
  calib.cameras.exo_camN.extrinsics  -> camera_extrinsics[]  (T_world_cam kept WHOLE
                                        4x4; NOT decomposed into rotation/translation,
                                        which mean something else — see convention)
  calib.cameras.exo_camN.status      -> the gate (below) + carried on both blocks
  calib.scene_details.board          -> DROPPED (calibration scaffolding)
  calib.accuracy                     -> DROPPED

THE GATE:
  ok       (status == "ok")   -> write full values
  broken   (status != "ok")   -> write the block, but solve values are NULL (K, D,
                                 T_world_cam, position_m); width/height kept (a capture
                                 fact, not a solve output); status carried so a reader
                                 knows WHY it is null. Recorded as an informational note in
                                 steps.exo_calib_notes. TOLERATED: a failed solve limits
                                 downstream 3D use but does not fail the episode.
  missing  (a RECORDED exo cam absent from calib.cameras) -> CalibrationCoverageError.
                                 Exo calibration is MANDATORY for this dataset, so a recorded
                                 exo camera with no calib entry FAILS the episode (loud). A
                                 cam that was NOT recorded and is absent from calib is simply
                                 never visited -> omitted, no error.

TWO ENFORCEMENT PLANES (do not confuse them):
  - the calib FILE existing / parsing is a CONFIG-plane check, enforced UPSTREAM in the
    wrapper pre-flight (validate_calib_rows): a missing/unset/malformed file fails the WHOLE
    batch at startup, before a bag opens. Not this function's job.
  - per-episode COVERAGE (every recorded exo cam present) is a DATA-plane check — it needs
    the recorded streams, known only post-extraction — so it lives HERE and raises.

Broken calibration does NOT fail termination.is_successful — a bad exo solve LIMITS
downstream application (multi-view 3D for that camera) but does not corrupt the recorded
data; it is reported per-episode by the wrapper's `exo_calib` summary enum, not the verdict.
The color STREAM in steps.streams is never gated here — video frames are independent of
whether the offline solve succeeded.

OWNERSHIP: this step owns the exo_cam* camera blocks in camera_intrinsics and the
exo_cam*_to_world entries in camera_extrinsics (find-or-replace by camera / name, so it
is idempotent and composes with the realsense-owned depth_to_* legs). It touches no
other key. It deliberately does NOT reorder top-level keys — that is a separate,
always-run wrapper pass (reorder_top_level), so sessions with no calib still get ordered.

TESTABILITY: all decision logic lives in the pure function merge_calibration(meta,
calib) -> new meta (no I/O, no mutation of inputs). annotate_calibration() is the thin
file shell around it.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))

from pipeline_metadata import add_error, upsert_intrinsic  # noqa: E402

METADATA_FILENAME = "metadata.json"

CALIB_SOURCE = "charuco_calib"          # provenance marker on every calib-sourced block
STATUS_OK = "ok"                        # the ONLY status treated as a good solve
CALIB_NOTES_KEY = "exo_calib_notes"     # steps[] informational record; does NOT gate termination
COLOR_KIND = "color"                    # stream kind of a colour video stream (== wrapper.COLOR_KIND)
EXO_PREFIX = "exo"                       # exo cams are named exo_cam<id>; ego is excluded (factory-calibrated)


class CalibrationCoverageError(Exception):
    """A RECORDED exo camera has no entry in the calib file. Exo calibration is mandatory for
    this dataset, so an un-calibrated recorded exo cam FAILS the episode: raised from the pure
    merge, surfaced by the wrapper's calib-merge step as a per-bag crash + durable
    pipeline_error.txt (the batch continues). Distinct from a present-but-broken solve
    (status != 'ok'), which is tolerated (null values + note)."""
# The exo pose is the camera's placement in the world frame the board defines. This is a
# DIFFERENT object and layout from the realsense depth_to_* legs (column-major, intra-
# device), so it carries its own convention string and is stored as a whole 4x4.
WORLD_POSE_CONVENTION = (
    "row-major; T_world_cam maps cam->world, last col = camera position in world (m)"
)


def _flatten_K(K: Any) -> List[float]:
    """Nested 3x3 camera matrix (calib's form) -> flat-9 row-major (camera_intrinsics'
    form, matching cam_ego's `color.K`)."""
    return [float(x) for row in K for x in row]


def _upsert_world_pose(meta: Dict[str, Any], entry: Dict[str, Any]) -> Dict[str, Any]:
    """Insert-or-replace ONE exo world-pose entry (matched by `name`) in
    meta['camera_extrinsics']. Owner-scoped to exo_cam*_to_world names, so the
    realsense depth_to_* legs are never touched; find-or-replace keeps it idempotent
    without duplicating on a re-run. (A sibling of upsert_extrinsic, but for the whole-
    4x4 world-pose shape rather than the decomposed rotation/translation shape.)"""
    exts = meta.setdefault("camera_extrinsics", [])
    idx = next((i for i, e in enumerate(exts) if e.get("name") == entry["name"]), None)
    if idx is None:
        exts.append(entry)
    else:
        exts[idx] = entry
    return entry


def merge_calibration(meta: Dict[str, Any], calib: Dict[str, Any]) -> Dict[str, Any]:
    """PURE: return a NEW metadata dict with the calib merged in per the gate above.
    Does not mutate `meta` or `calib`. Missing cameras (absent from calib.cameras) are
    simply never visited -> omitted. Board and accuracy are ignored (never read)."""
    meta = copy.deepcopy(meta)
    notes: List[str] = []

    # COVERAGE GATE (mandatory calibration): every RECORDED exo camera must have a calib
    # entry. A recorded exo cam absent from calib.cameras is a hard episode failure — the
    # 'missing' state made loud. Keyed on RECORDED streams (steps.streams color kind, exo
    # prefix), the same source as the wrapper's exo_calib enum: a cam that was EXPECTED but
    # never recorded is a separate missing_stream flag, not our concern. A present-but-broken
    # solve (status != "ok") is NOT a coverage gap — it is handled (null + note) in the loop.
    calib_cams = calib.get("cameras", {})
    recorded_exo = {s.get("camera") for s in meta.get("steps", {}).get("streams", [])
                    if s.get("kind") == COLOR_KIND
                    and str(s.get("camera", "")).startswith(EXO_PREFIX)}
    uncalibrated = sorted(c for c in recorded_exo if c not in calib_cams)
    if uncalibrated:
        raise CalibrationCoverageError(
            f"recorded exo camera(s) have no calibration entry: {uncalibrated}; "
            f"calib provides {sorted(calib_cams)}. Exo calibration is mandatory — add these "
            f"cameras to the calib file (as a real solve, or a block with status != 'ok').")

    for name, cam in calib.get("cameras", {}).items():
        status = cam.get("status")
        ok = status == STATUS_OK
        intr = cam.get("intrinsics", {}) or {}
        w, h = (intr.get("image_size") or [None, None])[:2]

        # intrinsics: solve values (K/D) null when broken; width/height always kept.
        upsert_intrinsic(meta, name, {
            "source": CALIB_SOURCE,
            "status": status,
            "width": w,
            "height": h,
            "K": _flatten_K(intr["K"]) if ok else None,
            "D": [float(x) for x in intr["dist"]] if ok else None,
        })

        # extrinsics: whole 4x4 kept when ok, null when broken.
        ext = cam.get("extrinsics", {}) or {}
        _upsert_world_pose(meta, {
            "name": f"{name}_to_world",
            "source": CALIB_SOURCE,
            "status": status,
            "T_world_cam": ext.get("T_world_cam") if ok else None,
            "position_m": ext.get("position_m") if ok else None,
            "convention": WORLD_POSE_CONVENTION,
        })

        if not ok:
            notes.append(f"{name}: status='{status}'; K/D and pose set null")

    # Informational only: broken calibration LIMITS downstream 3D use but does not corrupt
    # the recorded data, so it does NOT touch termination.is_successful. The wrapper's
    # exo_calib summary enum is where a broken/missing solve is reported per episode.
    if notes:
        add_error(meta.setdefault("steps", {}), CALIB_NOTES_KEY, notes)

    return meta


def load_calib(calib_path: Path) -> Dict[str, Any]:
    """Read a calib file STRICTLY (the calib-2008 format is the contract: one clean
    JSON object, nothing after it). Trailing content — an appended legend, a second
    object, garbage — raises json.JSONDecodeError rather than being silently ignored,
    so a malformed calib fails loudly instead of half-merging."""
    return json.loads(Path(calib_path).read_text())


def annotate_calibration(out_dir, calib_path) -> Dict[str, Any]:
    """File shell: merge `calib_path` into out_dir/metadata.json in place. Returns a small
    summary for the wrapper's per-step line, and is idempotent. Does NOT reorder keys (a
    separate wrapper pass).

    RAISES CalibrationCoverageError when a recorded exo cam has no calib entry (propagated
    from merge_calibration) — the wrapper turns that into a per-bag crash. Also propagates a
    json.JSONDecodeError from a malformed calib file. The file's EXISTENCE is enforced upstream
    (wrapper validate_calib_rows pre-flight), so the absent-file branches below are a defensive
    no-op for direct/standalone callers, not the mandatory-calibration gate."""
    out_root = Path(out_dir)
    meta_path = out_root / METADATA_FILENAME
    if not meta_path.is_file():
        print(f"[calibration] {meta_path} not found; skipping (run extraction first).")
        return {"written": False, "reason": "no metadata.json"}
    if not calib_path:                       # None / "" -> session declared no calibration
        print("[calibration] no calib path configured; skipping.")
        return {"written": False, "reason": "no calib path"}
    calib_file = Path(calib_path)
    if not calib_file.is_file():
        print(f"[calibration] calib file {calib_file} not found; skipping.")
        return {"written": False, "reason": "no calib file"}

    meta = json.loads(meta_path.read_text())
    calib = load_calib(calib_file)
    merged = merge_calibration(meta, calib)
    meta_path.write_text(json.dumps(merged, indent=2))

    cams = list(calib.get("cameras", {}))
    broken = [c for c, v in calib.get("cameras", {}).items()
              if v.get("status") != STATUS_OK]
    print(f"[calibration] {meta_path.parent.name}: merged {len(cams)} camera(s) "
          f"{cams}; broken={broken or '[]'} -> {meta_path}")
    return {"written": True, "cameras": cams, "broken": broken}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Merge an exo calibration into an episode's metadata.json")
    ap.add_argument("--out-dir", required=True, help="episode out_dir holding metadata.json")
    ap.add_argument("--calib", required=True, help="path to calib-<date>.json")
    args = ap.parse_args()
    annotate_calibration(args.out_dir, args.calib)
