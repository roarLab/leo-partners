"""Guard-logic tests for inputs.py (the CLI-input validator).

Pure: inputs.py imports only json/os/re, so these run without OpenCV or a
synthetic capture -- unlike the rest of the suite. They assert the guard raises
SystemExit (a named one-line message, NOT a raw FileNotFoundError/JSONDecodeError
traceback) and that the message names the flag, the path, and -- for the naming
guard -- what was actually found.
"""

import json

import pytest

from inputs import load_json, require_dir, require_exo_cams, require_file


# ----------------------------------------------------------- require_file / dir


def test_require_file_missing_names_flag_and_path():
    with pytest.raises(SystemExit) as e:
        require_file("nope/typo.json", "--calib", "exo_extrinsics")
    msg = str(e.value)
    assert "--calib" in msg and "nope/typo.json" in msg
    # the whole point: a clean exit, not the raw filesystem error
    assert not isinstance(e.value, FileNotFoundError)


def test_require_file_existing_does_not_raise(tmp_path):
    f = tmp_path / "ok.json"
    f.write_text("{}")
    require_file(str(f), "--calib", "exo_extrinsics")   # no raise


def test_require_dir_missing_names_flag_and_path():
    with pytest.raises(SystemExit) as e:
        require_dir("capture/typo_dir", "--walk", "exo_extrinsics")
    msg = str(e.value)
    assert "--walk" in msg and "capture/typo_dir" in msg


def test_require_dir_existing_does_not_raise(tmp_path):
    require_dir(str(tmp_path), "--walk", "exo_extrinsics")   # no raise


# ------------------------------------------------------------------- load_json


def test_load_json_missing_file_exits(tmp_path):
    with pytest.raises(SystemExit) as e:
        load_json(str(tmp_path / "gone.json"), "--calib", "exo_extrinsics")
    assert "--calib" in str(e.value) and "file not found" in str(e.value)


def test_load_json_malformed_exits(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid json")
    with pytest.raises(SystemExit) as e:
        load_json(str(bad), "--calib", "exo_extrinsics")
    # our prefix only -- the decoder's own tail (char N) is not ours to pin
    assert "--calib" in str(e.value) and "is not valid JSON" in str(e.value)
    assert not isinstance(e.value, json.JSONDecodeError)


def test_load_json_valid_returns_parsed(tmp_path):
    f = tmp_path / "cams.json"
    f.write_text('{"exo_cam1": [1, 2, 3]}')
    assert load_json(str(f), "--camera_positions", "exo_extrinsics") == {
        "exo_cam1": [1, 2, 3]}


# -------------------------------------------------------------- require_exo_cams


def _streams(*names):
    return [(n, f"/cap/{n}.mp4") for n in names]


def test_exo_cams_all_valid_returns_all_no_skip(capsys):
    out = require_exo_cams(_streams("exo_cam1", "exo_cam2"),
                           "/cap", "--intrinsics", "exo_intrinsics")
    assert [n for n, _ in out] == ["exo_cam1", "exo_cam2"]
    assert "[skip]" not in capsys.readouterr().out


def test_exo_cams_mixed_keeps_valid_and_reports_skip(capsys):
    out = require_exo_cams(_streams("exo_cam1", "cam2"),
                           "/cap", "--intrinsics", "exo_intrinsics")
    assert [n for n, _ in out] == ["exo_cam1"]        # old name ignored
    skip = capsys.readouterr().out
    assert "[skip]" in skip and "cam2" in skip         # but not silently


def test_exo_cams_all_old_named_exits_listing_found():
    with pytest.raises(SystemExit) as e:
        require_exo_cams(_streams("cam1", "cam2"),
                         "/cap", "--walk", "exo_extrinsics")
    msg = str(e.value)
    assert "no exo_cam*.mp4 clips" in msg
    assert "cam1" in msg and "cam2" in msg             # names what it found


def test_exo_cams_typo_exits_listing_the_typo():
    with pytest.raises(SystemExit) as e:
        require_exo_cams(_streams("exo_cma1"),
                         "/cap", "--intrinsics", "exo_intrinsics")
    assert "exo_cma1" in str(e.value)                  # the misspelling is shown


def test_exo_cams_empty_exits_found_none():
    with pytest.raises(SystemExit) as e:
        require_exo_cams([], "/cap", "--intrinsics", "exo_intrinsics")
    assert "found: none" in str(e.value)
