"""Shared metadata-error helpers for the v3 pipeline.

The color/depth extraction scripts and the two validators all write error
signals into the SAME metadata.json, in an order that is not guaranteed. To keep
them from clobbering one another, error lists are only ever APPENDED to, never
reassigned — `add_error` is the single choke point for that.

Ownership rule (who writes which token/key):
  - rosbag_process_color_v3 (color extraction)  -> steps.missing_stream_error,
                                                    steps.extra_stream_error, and the
                                                    "missing_stream" / "extra_stream"
                                                    termination tokens.
  - rosbag_process_depth_v3  (depth extraction)  -> a "missing_stream" entry/token when
                                                    the depth topic is absent, AND the
                                                    "depth_to_color" camera_extrinsics entry.
  - rosbag_process_imu_v3    (imu extraction)    -> the "depth_to_gyro" / "depth_to_accel"
                                                    camera_extrinsics entries, AND a
                                                    "missing_stream" entry/token when /imu is
                                                    absent or empty (a presence check — imu
                                                    is not data-validated yet).
  - validate_color_v3 / validate_depth_v3        -> their OWN validation tokens only
                                                    (color / timestamps / rosbag_corruption
                                                    / depth), via an owner-scoped merge.
Nobody rewrites another writer's signals. The extraction-owned lists/tokens are
never cleared (color creates metadata.json fresh each run, so they start empty).
"""
from __future__ import annotations
from typing import Any, Dict, Iterable, List


def add_error(container: Dict[str, Any], key: str, entries: Iterable[str]) -> List[str]:
    """Append `entries` into the list at container[key] (created if absent),
    skipping any already present. NEVER reassigns the list, so writers to the same
    key compose regardless of order, and re-runs don't duplicate. (These writers run
    sequentially, not in parallel — 'compose regardless of order' means the pipeline
    doesn't depend on which step ran first, not thread-safety.)"""
    lst = container.setdefault(key, [])
    for e in entries:
        if e not in lst:
            lst.append(e)
    return lst


# Column-major rotation is librealsense's own storage order (rsutil.h); depth_to_X
# is the transform FROM depth TO X, i.e. p_X = R.p_depth + t (see
# rosbag_process_depth_v3's header). Stored verbatim so the VIO consumer composes
# any chain (e.g. imu->color) itself, with full knowledge of the convention.
DEFAULT_EXTRINSIC_CONVENTION = (
    "librealsense column-major rotation; p_target = R.p_source + t "
    "(source/target named by the entry, e.g. depth_to_color: source=depth, target=color)"
)


def upsert_extrinsic(meta: Dict[str, Any], name: str, source_topic: str,
                     rotation: Iterable[float], translation: Iterable[float],
                     convention: str = DEFAULT_EXTRINSIC_CONVENTION) -> Dict[str, Any]:
    """Insert-or-update ONE entry (matched by `name`) in meta['camera_extrinsics'],
    the verbatim way: `rotation` (9) and `translation` (3) are copied as-is from the
    bag's realsense2_camera_msgs/Extrinsics message — no composition, no derived
    geometry. Owner-scoped like add_error: each writer touches only the name(s) it
    owns (depth -> depth_to_color; imu -> depth_to_gyro/accel), so the entries
    compose regardless of run order. The shared list is never reassigned (find-or-
    append), and a re-run updates the matching entry in place instead of
    duplicating. Returns the entry written."""
    exts = meta.setdefault("camera_extrinsics", [])
    payload = {
        "name": name,
        "source_topic": source_topic,
        "rotation": [float(x) for x in rotation],
        "translation": [float(x) for x in translation],
        "convention": convention,
    }
    entry = next((e for e in exts if e.get("name") == name), None)
    if entry is None:
        exts.append(payload)
    else:
        entry.update(payload)
    return payload


def upsert_intrinsic(meta: Dict[str, Any], camera: str,
                     color: Dict[str, Any]) -> Dict[str, Any]:
    """Insert-or-update ONE camera block (matched by `camera`) in
    meta['camera_intrinsics'], setting its `color` sub-block. Owner-scoped and
    find-or-replace like upsert_extrinsic: a re-run updates the matching camera in
    place instead of appending a duplicate, and the shared list is never reassigned.
    Only the `color` sub-block is touched — any sibling sub-block (e.g. a `depth`
    block written by the depth extractor for cam_ego) is preserved. Returns the
    block written.

    (camera_extrinsics is keyed by transform `name`; camera_intrinsics is keyed by
    `camera` — hence a separate helper. The exo calibration owns the exo_cam* camera
    blocks; the bag's camera_info owns cam_ego — disjoint keys, so they compose.)"""
    cams = meta.setdefault("camera_intrinsics", [])
    entry = next((c for c in cams if c.get("camera") == camera), None)
    if entry is None:
        entry = {"camera": camera, "color": color}
        cams.append(entry)
    else:
        entry["color"] = color
    return entry


# Fixed, human-friendly top-level key order for metadata.json. The pipeline writes
# keys in run order (color, then depth/imu append, then episode_details, then calib),
# so the on-disk order is otherwise haphazard. reorder_top_level applies this order
# as a final, cosmetic pass.
TOP_LEVEL_ORDER = (
    "metadata", "episode_details", "camera_intrinsics",
    "camera_extrinsics", "steps", "termination",
)


def reorder_top_level(meta: Dict[str, Any],
                      order: Iterable[str] = TOP_LEVEL_ORDER) -> Dict[str, Any]:
    """Return a NEW dict with `meta`'s top-level keys placed in `order`, then any
    remaining keys appended in their original order. COSMETIC ONLY: it never reads,
    writes, drops, or mutates a value — the returned dict compares deep-equal to the
    input (same keys, same values), differing only in key order. Unknown/future keys
    are preserved (append-remaining), so this can never silently drop a key it did
    not know about. Safe to run last and always, on any metadata dict, regardless of
    which step produced which key."""
    ordered = {k: meta[k] for k in order if k in meta}
    ordered.update({k: v for k, v in meta.items() if k not in ordered})
    return ordered
