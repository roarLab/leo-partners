"""Validate CLI path arguments up front, so a typo'd --flag or a missing file
fails immediately with a message that names the flag and the path -- not a raw
FileNotFoundError traceback, and not a misleading 'no camera calibrated' three
stages later once video_streams() has silently returned [].

Board-logic-free on purpose: the camera-naming check takes the already-computed
stream list (require_exo_cams) rather than importing board, so this module stays
a plain CLI-input validator that any tool in the repo can reuse.
"""

import json
import os
import re

# New-setup camera naming: a stream must be exo_cam<N> (exo_cam1.mp4, ...).
# Old names (cam1/cam2/cam3) and the ego camera do not match, so they are
# ignored rather than mistaken for an exo camera.
_EXO = re.compile(r"^exo_cam\d+$")


def require_file(path, flag, script):
    """Exit with a clear message unless `path` is an existing file."""
    if not os.path.isfile(path):
        raise SystemExit(f"[{script}] {flag} file not found: {path} -- "
                         "check the path / typo")


def require_dir(path, flag, script):
    """Exit with a clear message unless `path` is an existing directory."""
    if not os.path.isdir(path):
        raise SystemExit(f"[{script}] {flag} dir not found: {path} -- "
                         "check the path / typo")


def load_json(path, flag, script):
    """require_file, then parse -- turning a missing file or malformed JSON into
    a named SystemExit instead of a raw traceback."""
    require_file(path, flag, script)
    try:
        with open(path) as fh:
            return json.load(fh)
    except json.JSONDecodeError as e:
        raise SystemExit(f"[{script}] {flag} is not valid JSON: {path} -- {e}")


def require_exo_cams(streams, path, flag, script):
    """Keep only exo_cam<N> clips (the new-setup naming) from `streams` (the
    result of video_streams(path)); ignore anything else, printing what it
    skipped so a misnamed file does not vanish silently. Exit if none remain --
    catches a wrong subfolder AND an old-named/typo'd capture, naming what it did
    find so the cause is obvious. Returns the filtered [(name, path), ...]."""
    good = [(n, p) for n, p in streams if _EXO.match(n)]
    skipped = [n for n, _ in streams if not _EXO.match(n)]
    if skipped:
        print(f"[skip] {flag}: ignoring non-exo clips: {', '.join(skipped)}")
    if not good:
        found = ", ".join(n for n, _ in streams) or "none"
        raise SystemExit(f"[{script}] {flag} has no exo_cam*.mp4 clips: {path} "
                         f"-- expected exo_cam1.mp4, exo_cam2.mp4 ...; "
                         f"found: {found}")
    return good
