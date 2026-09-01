"""L2 -- the happy-path end-to-end. Build a full synthetic session, run the real
importable pipeline over it (exo_intrinsics.main then exo_extrinsics.main), and
check the recovered lenses and camera positions against the truth we rendered
from. One slow test (it renders and encodes video); marked `e2e` so quick L0/L1
iteration can skip it with `pytest -m "not e2e"`."""

import argparse
import json
import os

import numpy as np
import pytest

import exo_extrinsics
import exo_intrinsics


def _ns(**kw):
    return argparse.Namespace(**kw)


@pytest.mark.e2e
def test_pipeline_recovers_lenses_and_positions(synth_capture):
    cap = synth_capture
    root = os.path.dirname(cap["intrinsics"])
    intr_json = os.path.join(root, "calib", "intrinsics.json")
    calib_json = os.path.join(root, "calib", "calib.json")

    exo_intrinsics.main(_ns(
        intrinsics=cap["intrinsics"], board=cap["board"], out=intr_json,
        stride=3, min_corners=8, max_views=120, free_k3=False))

    exo_extrinsics.main(_ns(
        walk=cap["walk"], calib=intr_json, camera_positions=cap["camera_positions"],
        board=cap["board"], out=calib_json, stride=2, min_corners=10,
        max_reproj=2.0, sync_ms=33.0, min_covis=8))

    assert os.path.exists(calib_json)
    cams = json.load(open(calib_json))["cameras"]

    for name, T_true in cap["cams"].items():
        entry = cams.get(name)
        assert entry is not None and entry["extrinsics"] is not None, \
            f"{name} missing from result"
        fx = entry["intrinsics"]["K"][0][0]
        ferr = abs(fx - cap["K_true"][0, 0]) / cap["K_true"][0, 0] * 100
        perr = np.linalg.norm(np.array(entry["extrinsics"]["position_m"])
                              - T_true[:3, 3])
        assert ferr < 2.0, f"{name} focal error {ferr:.2f}% too high"
        assert perr < 0.05, f"{name} position error {perr*100:.1f} cm too high"


@pytest.mark.e2e
def test_starved_camera_fails_intrinsics(starved_intrinsics_capture):
    """A camera with too few close-up views must surface in the real intrinsics
    file as status failed_intrinsics with a null lens -- not vanish, and not drag
    the others down."""
    cap = starved_intrinsics_capture
    root = os.path.dirname(cap["intrinsics"])
    intr_json = os.path.join(root, "calib", "intrinsics.json")

    exo_intrinsics.main(_ns(
        intrinsics=cap["intrinsics"], board=cap["board"], out=intr_json,
        stride=3, min_corners=8, max_views=120, free_k3=False))

    cams = json.load(open(intr_json))["cameras"]

    bad = cams[cap["failed_cam"]]
    assert bad["status"] == "failed_intrinsics"
    assert bad["intrinsics"] is None       # present-but-null, key not missing
    assert bad["extrinsics"] is None

    for good in cap["ok_cams"]:
        assert cams[good]["status"] == "ok"
        assert cams[good]["intrinsics"] is not None
