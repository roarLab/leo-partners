"""Edge-case regression suite for rosbag_process_depth_v3.py.

Run:  conda activate leo-seg && pytest data-processing-scripts-v3/ -q

Covers: aligner geometry (vs a scalar librealsense transcription), frame
pairing / drops / skew, calibration + topic discovery, and the metadata.json
merge. Everything runs on tiny synthetic ROS2 bags built in a tmpdir, so no
real recordings are needed. The full color+depth pipeline wrapper is tested
separately in test_wrapper.py.
"""
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import h5py
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rosbag_process_depth_v3 as ead                  # noqa: E402

from rosbags.rosbag2 import Writer                      # noqa: E402
from rosbags.typesys import get_typestore, Stores, get_types_from_msg  # noqa: E402

ZEROS5 = [0.0, 0.0, 0.0, 0.0, 0.0]
IDENTITY9 = [1, 0, 0, 0, 1, 0, 0, 0, 1]


# ---------------------------------------------------------------------------
# Scalar reference: literal float32 transcription of librealsense align.cpp +
# rsutil.h. The aligner must match this bit-for-bit. (pinhole path; zero coeffs)
# ---------------------------------------------------------------------------
f32 = np.float32


def _deproj(intr, u, v, d):
    x = (f32(u) - f32(intr.cx)) / f32(intr.fx)
    y = (f32(v) - f32(intr.cy)) / f32(intr.fy)
    return [f32(d) * x, f32(d) * y, f32(d)]


def _xform(rot, trans, p):
    return [rot[0]*p[0] + rot[3]*p[1] + rot[6]*p[2] + trans[0],
            rot[1]*p[0] + rot[4]*p[1] + rot[7]*p[2] + trans[1],
            rot[2]*p[0] + rot[5]*p[1] + rot[8]*p[2] + trans[2]]


def _proj(intr, p):
    x = p[0] / p[2]
    y = p[1] / p[2]
    return [x * f32(intr.fx) + f32(intr.cx), y * f32(intr.fy) + f32(intr.cy)]


def scalar_align(raw, depth, color, rot, trans, z_scale=0.001):
    Wc, Hc = color.width, color.height
    out = np.zeros(Hc * Wc, np.uint16)
    rf = [f32(v) for v in rot]
    tf = [f32(v) for v in trans]
    for dy in range(depth.height):
        for dx in range(depth.width):
            z = int(raw[dy, dx])
            if z == 0:
                continue
            d = f32(z) * f32(z_scale)
            pA = _proj(color, _xform(rf, tf, _deproj(depth, f32(dx) - f32(0.5), f32(dy) - f32(0.5), d)))
            x0 = int(f32(pA[0]) + f32(0.5)); y0 = int(f32(pA[1]) + f32(0.5))
            pB = _proj(color, _xform(rf, tf, _deproj(depth, f32(dx) + f32(0.5), f32(dy) + f32(0.5), d)))
            x1 = int(f32(pB[0]) + f32(0.5)); y1 = int(f32(pB[1]) + f32(0.5))
            if x0 < 0 or y0 < 0 or x1 >= Wc or y1 >= Hc:
                continue
            for yy in range(y0, y1 + 1):
                for xx in range(x0, x1 + 1):
                    idx = yy * Wc + xx
                    out[idx] = min(int(out[idx]), z) if out[idx] else z
    return out.reshape(Hc, Wc)


def make_intr(w, h, fx, fy, cx, cy, model=ead.RS2_DISTORTION_BROWN_CONRADY, coeffs=ZEROS5):
    return ead.Intrinsics(width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy,
                          model=model, coeffs=coeffs)


# ---------------------------------------------------------------------------
# Synthetic bag builder
# ---------------------------------------------------------------------------
def build_bag(path, *, n_color=8, drop_depth=(), drop_color=(), depth_skew_s=0.0,
              depth_fn=None, rotation=IDENTITY9, translation=(0.015, 0.0, 0.0),
              depth_coeffs=ZEROS5, color_coeffs=ZEROS5, distortion_model="plumb_bob",
              with_extrinsics=True, second_color_camera=False, with_depth=True,
              depth_dims=(16, 12), color_dims=(24, 18),
              depth_k=(12.0, 12.0, 8.0, 6.0), color_k=(18.0, 18.0, 12.0, 9.0),
              prefix="/ego/camera"):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    ts = get_typestore(Stores.ROS2_HUMBLE)
    ts.register(get_types_from_msg("float64[9] rotation\nfloat64[3] translation\n",
                                   "realsense2_camera_msgs/msg/Extrinsics"))
    Time = ts.types["builtin_interfaces/msg/Time"]
    Header = ts.types["std_msgs/msg/Header"]
    Image = ts.types["sensor_msgs/msg/Image"]
    CI = ts.types["sensor_msgs/msg/CameraInfo"]
    ROI = ts.types["sensor_msgs/msg/RegionOfInterest"]
    Extr = ts.types["realsense2_camera_msgs/msg/Extrinsics"]
    DW, DH = depth_dims
    CW, CH = color_dims

    def hdr(t, frame):
        return Header(stamp=Time(sec=t // 1_000_000_000, nanosec=t % 1_000_000_000), frame_id=frame)

    def cinfo(t, w, h, k, coeffs, frame):
        fx, fy, cx, cy = k
        return CI(header=hdr(t, frame), height=h, width=w, distortion_model=distortion_model,
                  d=np.array(coeffs, np.float64), k=np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1], np.float64),
                  r=np.eye(3).ravel().astype(np.float64),
                  p=np.array([fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0], np.float64),
                  binning_x=0, binning_y=0,
                  roi=ROI(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False))

    if depth_fn is None:
        yy, xx = np.mgrid[0:DH, 0:DW]
        base = (900 + 8 * xx + 5 * yy).astype(np.uint16)
        depth_fn = lambda i: base.copy()  # noqa: E731

    with Writer(path, version=9) as w:
        c_color = w.add_connection(f"{prefix}/color/image_raw", Image.__msgtype__, typestore=ts)
        c_cinfo = w.add_connection(f"{prefix}/color/camera_info", CI.__msgtype__, typestore=ts)
        if with_depth:  # with_depth=False omits the depth image topic (a truly missing
            c_depth = w.add_connection(f"{prefix}/depth/image_rect_raw", Image.__msgtype__, typestore=ts)
        c_dinfo = w.add_connection(f"{prefix}/depth/camera_info", CI.__msgtype__, typestore=ts)  # stream, not empty)
        if with_extrinsics:
            c_extr = w.add_connection(f"{prefix}/extrinsics/depth_to_color", Extr.__msgtype__, typestore=ts)
            ex = Extr(rotation=np.array(rotation, np.float64), translation=np.array(translation, np.float64))
            w.write(c_extr, 0, ts.serialize_cdr(ex, Extr.__msgtype__))
        if second_color_camera:
            c_color2 = w.add_connection("/wrist/camera/color/image_raw", Image.__msgtype__, typestore=ts)
            c_cinfo2 = w.add_connection("/wrist/camera/color/camera_info", CI.__msgtype__, typestore=ts)

        for i in range(n_color):
            t = int(i * 33_333_333)
            w.write(c_cinfo, t, ts.serialize_cdr(cinfo(t, CW, CH, color_k, color_coeffs, "color"), CI.__msgtype__))
            w.write(c_dinfo, t, ts.serialize_cdr(cinfo(t, DW, DH, depth_k, depth_coeffs, "depth"), CI.__msgtype__))
            if i not in drop_color:
                color = np.dstack([np.full((CH, CW), 50 + i, np.uint8),
                                   np.full((CH, CW), 100, np.uint8),
                                   np.full((CH, CW), 150, np.uint8)])
                w.write(c_color, t, ts.serialize_cdr(
                    Image(header=hdr(t, "color"), height=CH, width=CW, encoding="rgb8",
                          is_bigendian=0, step=CW * 3, data=color.reshape(-1)), Image.__msgtype__))
                if second_color_camera:
                    w.write(c_color2, t, ts.serialize_cdr(
                        Image(header=hdr(t, "color2"), height=CH, width=CW, encoding="rgb8",
                              is_bigendian=0, step=CW * 3, data=color.reshape(-1)), Image.__msgtype__))
            if with_depth and i not in drop_depth:
                td = int(t + depth_skew_s * 1e9)
                depth = depth_fn(i)
                w.write(c_depth, td, ts.serialize_cdr(
                    Image(header=hdr(td, "depth"), height=DH, width=DW, encoding="16UC1",
                          is_bigendian=0, step=DW * 2, data=depth.view(np.uint8).reshape(-1)), Image.__msgtype__))
    return path


def run_main(bag, out, camera="ego"):
    return ead.main(bag=str(bag), out_dir=str(out), camera=camera)


def read_h5(out, camera="ego"):
    with h5py.File(Path(out) / "depth_frames" / f"{camera}_aligned_depth_to_color.h5", "r") as f:
        return f["data"][:], dict(f["data"].attrs)


# ---------------------------------------------------------------------------
# Fixtures: restore mutated module globals between tests
# ---------------------------------------------------------------------------
_SAVE = ["STRIDE", "LIMIT", "PAIR_TOLERANCE_MS", "HOLE_FILL", "Z_SCALE", "OVERLAY_PNG",
         "ROTATION", "TRANSLATION", "COLOR_MODEL", "DEPTH_MODEL",
         "CAMERA", "DEPTH_TOPIC", "COLOR_TOPIC", "DEPTH_INFO_TOPIC", "COLOR_INFO_TOPIC",
         "EXTRINSICS_TOPIC"]


@pytest.fixture(autouse=True)
def reset_config():
    saved = {k: getattr(ead, k) for k in _SAVE}
    yield
    for k, v in saved.items():
        setattr(ead, k, v)


# ===========================================================================
# A. Aligner geometry
# ===========================================================================
@pytest.mark.parametrize("dd,cd,ang", [
    ((16, 12), (24, 18), 0.0),      # upscale, no rotation
    ((24, 18), (12, 9), 0.0),       # downscale -> z-buffer collisions
    ((16, 12), (24, 18), 0.4),      # tilted rotation
])
def test_aligner_bit_identical_to_scalar(dd, cd, ang):
    depth = make_intr(dd[0], dd[1], dd[0] * 0.9, dd[0] * 0.9, dd[0] / 2, dd[1] / 2)
    color = make_intr(cd[0], cd[1], cd[0] * 0.9, cd[0] * 0.9, cd[0] / 2, cd[1] / 2)
    a = math.radians(ang)
    Rmat = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    rot = Rmat.T.reshape(-1).tolist()
    trans = [0.0151, -0.0003, -0.0003]
    rng = np.random.RandomState(1)
    yy, xx = np.mgrid[0:dd[1], 0:dd[0]]
    raw = (700 + 3 * xx + 2 * yy).astype(np.uint16)
    raw[rng.rand(dd[1], dd[0]) < 0.1] = 0

    ref = scalar_align(raw, depth, color, rot, trans)
    got = ead.DepthToColorAligner(depth, color, rot, trans, hole_fill=True).align(raw)
    assert np.array_equal(ref, got)


def test_all_zero_depth_gives_zero_output():
    depth = make_intr(16, 12, 12, 12, 8, 6)
    color = make_intr(24, 18, 18, 18, 12, 9)
    out = ead.DepthToColorAligner(depth, color, IDENTITY9, (0.015, 0, 0)).align(np.zeros((12, 16), np.uint16))
    assert out.shape == (18, 24)
    assert int(out.max()) == 0


def test_holes_are_skipped():
    depth = make_intr(16, 12, 12, 12, 8, 6)
    color = make_intr(24, 18, 18, 18, 12, 9)
    raw = np.full((12, 16), 1000, np.uint16)
    raw[6, 8] = 0
    out = ead.DepthToColorAligner(depth, color, IDENTITY9, (0.015, 0, 0)).align(raw)
    assert out.max() == 1000                     # the plane got aligned
    assert (out == 0).any()                      # some output pixels remain empty


def test_out_of_bounds_depth_dropped_no_crash():
    depth = make_intr(16, 12, 12, 12, 8, 6)
    color = make_intr(24, 18, 18, 18, 12, 9)
    raw = np.full((12, 16), 5, np.uint16)        # 5 mm -> huge parallax -> projects off-frame
    out = ead.DepthToColorAligner(depth, color, IDENTITY9, (0.5, 0, 0)).align(raw)
    assert out.shape == (18, 24)                  # returns cleanly (mostly/all zero)


def test_center_only_is_subset_of_hole_fill():
    depth = make_intr(16, 12, 12, 12, 8, 6)
    color = make_intr(24, 18, 18, 18, 12, 9)
    raw = (900 + np.mgrid[0:12, 0:16][1] * 8).astype(np.uint16)
    filled = ead.DepthToColorAligner(depth, color, IDENTITY9, (0.015, 0, 0), hole_fill=True).align(raw)
    sparse = ead.DepthToColorAligner(depth, color, IDENTITY9, (0.015, 0, 0), hole_fill=False).align(raw)
    assert (sparse > 0).sum() <= (filled > 0).sum()
    # hole-fill uses a stricter both-corners bounds check, so at the very image
    # border center-only can fill a pixel hole-fill skips; compare the interior.
    interior = np.zeros_like(sparse, dtype=bool)
    interior[1:-1, 1:-1] = True
    assert np.all(filled[(sparse > 0) & interior] > 0)   # every interior sparse pixel is covered


def test_zbuffer_keeps_nearest():
    # downscale so multiple depth pixels land on the same color pixel
    depth = make_intr(24, 18, 24, 24, 12, 9)
    color = make_intr(12, 9, 12, 12, 6, 4.5)
    raw = np.full((18, 24), 2000, np.uint16)
    raw[0:9, :] = 800                              # a nearer band
    out = ead.DepthToColorAligner(depth, color, IDENTITY9, (0.0, 0, 0)).align(raw)
    nz = out[out > 0]
    assert nz.min() == 800                         # nearest wins somewhere
    assert nz.max() <= 2000


# ===========================================================================
# B. Pairing / sync
# ===========================================================================
def test_perfect_sync_all_paired(tmp_path):
    bag = build_bag(tmp_path / "bag", n_color=8)
    s = run_main(bag, tmp_path / "out")
    assert s["n_color"] == 8 and s["n_paired"] == 8 and s["n_blank_color"] == 0
    data, _ = read_h5(tmp_path / "out")
    assert data.shape == (8, 18, 24)
    assert all(data[i].max() > 0 for i in range(8))


def test_dropped_depth_makes_blank_frame(tmp_path):
    bag = build_bag(tmp_path / "bag", n_color=8, drop_depth=(3,))
    s = run_main(bag, tmp_path / "out")
    assert s["n_paired"] == 7 and s["n_blank_color"] == 1
    data, _ = read_h5(tmp_path / "out")
    assert int(data[3].max()) == 0                 # the dropped index is a zero frame
    assert int(data[2].max()) > 0 and int(data[4].max()) > 0


def test_dropped_color_makes_depth_no_partner(tmp_path):
    bag = build_bag(tmp_path / "bag", n_color=8, drop_color=(4,))
    s = run_main(bag, tmp_path / "out")
    assert s["n_color"] == 7                        # one fewer color frame
    assert s["n_depth_no_partner"] >= 1             # the depth at t=4 has no color within tol


def test_skew_within_tolerance_pairs(tmp_path):
    bag = build_bag(tmp_path / "bag", n_color=6, depth_skew_s=0.005)   # 5 ms < 16 ms
    s = run_main(bag, tmp_path / "out")
    assert s["n_paired"] == 6
    assert 4.0 < s["max_pair_dt_ms"] < 6.0


def test_skew_beyond_tolerance_blanks(tmp_path, capsys):
    # 16.3 ms skew: at 30 fps (33.3 ms period) the nearest color on BOTH sides is
    # >16 ms away, so every depth frame is beyond tolerance and blanks.
    bag = build_bag(tmp_path / "bag", n_color=6, depth_skew_s=0.0163)
    s = run_main(bag, tmp_path / "out")
    assert s["n_paired"] == 0 and s["n_blank_color"] == 6
    assert "skew" in capsys.readouterr().out.lower()   # the "not a true drop" note fired


def test_wider_tolerance_recovers_skew(tmp_path):
    ead.PAIR_TOLERANCE_MS = 40.0
    bag = build_bag(tmp_path / "bag", n_color=6, depth_skew_s=0.0163)
    s = run_main(bag, tmp_path / "out")
    assert s["n_paired"] == 6   # 16.3 ms < 40 ms and each depth's own color is nearest


def test_stride_subsamples(tmp_path):
    ead.STRIDE = 2
    bag = build_bag(tmp_path / "bag", n_color=8)
    s = run_main(bag, tmp_path / "out")
    assert s["n_color"] == 4                         # every 2nd color frame


def test_limit_caps_frames(tmp_path):
    ead.LIMIT = 3
    bag = build_bag(tmp_path / "bag", n_color=8)
    s = run_main(bag, tmp_path / "out")
    assert s["n_color"] == 3


def test_zero_color_frames_raises(tmp_path):
    bag = build_bag(tmp_path / "bag", n_color=0)
    with pytest.raises(SystemExit):
        run_main(bag, tmp_path / "out")


# ===========================================================================
# C. Calibration / topics
# ===========================================================================
def test_custom_extrinsics_type_is_read(tmp_path):
    bag = build_bag(tmp_path / "bag", n_color=4, translation=(0.015, 0, 0))
    run_main(bag, tmp_path / "out")
    _, attrs = read_h5(tmp_path / "out")
    t = np.array(attrs["depth_to_color_translation_m"])
    assert abs(t[0] - 0.015) < 1e-6


def test_missing_extrinsics_uses_override(tmp_path):
    ead.ROTATION = IDENTITY9
    ead.TRANSLATION = [0.02, 0.0, 0.0]
    bag = build_bag(tmp_path / "bag", n_color=4, with_extrinsics=False)
    s = run_main(bag, tmp_path / "out")
    assert s["n_paired"] == 4
    _, attrs = read_h5(tmp_path / "out")
    assert abs(np.array(attrs["depth_to_color_translation_m"])[0] - 0.02) < 1e-6


def test_missing_extrinsics_without_override_flags_depth_info(tmp_path):
    # No extrinsics topic and no ROTATION/TRANSLATION override -> FLAG AND CONTINUE
    # (slide 7 row 2): record depth_info in missing_stream_error, return early (no h5),
    # do NOT crash. Colour must already exist so the write target metadata.json is present.
    import json
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metadata.json").write_text(
        json.dumps({"steps": {"streams": [], "missing_stream_error": []},
                    "termination": {"is_successful": True, "reason": []}}), encoding="utf-8")
    bag = build_bag(tmp_path / "bag", n_color=4, with_extrinsics=False)
    s = run_main(bag, out)                                    # must NOT raise
    assert s["aligned"] is False and s["reason"] == "missing_depth_extrinsics"
    meta = json.loads((out / "metadata.json").read_text())
    assert any("extrinsics" in e for e in meta["steps"]["missing_stream_error"])
    assert meta["termination"]["reason"] == ["depth_info"]


def test_ambiguous_topics_raise_without_camera(tmp_path):
    bag = build_bag(tmp_path / "bag", n_color=4, second_color_camera=True)
    ead.CAMERA = None                                  # main() falls back to CAMERA when camera arg is None
    with pytest.raises(SystemExit):
        run_main(bag, tmp_path / "out", camera=None)   # two color/image_raw, no filter -> ambiguous


def test_camera_substring_disambiguates(tmp_path):
    bag = build_bag(tmp_path / "bag", n_color=4, second_color_camera=True)
    s = run_main(bag, tmp_path / "out", camera="ego")
    assert s["n_paired"] == 4


def test_nonzero_coeffs_warns(tmp_path, capsys):
    bag = build_bag(tmp_path / "bag", n_color=4, color_coeffs=[0.1, -0.2, 0.0, 0.0, 0.05])
    run_main(bag, tmp_path / "out")
    assert "NON-ZERO distortion" in capsys.readouterr().out


def test_far_from_identity_warns(tmp_path, capsys):
    a = math.radians(20)
    Rmat = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    bag = build_bag(tmp_path / "bag", n_color=4, rotation=Rmat.T.reshape(-1).tolist(),
                    translation=(0.2, 0.0, 0.0))
    run_main(bag, tmp_path / "out")
    out = capsys.readouterr().out
    assert "far from identity" in out and "baseline" in out


def test_h5_attrs_and_csv(tmp_path):
    bag = build_bag(tmp_path / "bag", n_color=5, drop_depth=(2,))
    run_main(bag, tmp_path / "out")
    data, attrs = read_h5(tmp_path / "out")
    assert data.dtype == np.uint16 and attrs["align_direction"] == "depth_to_color"
    assert attrs["hole_fill"] == "corner_rect" and "librealsense_ref" in attrs
    csv = (Path(tmp_path / "out") / "timestamps" / "ego_aligned_depth_to_color.csv").read_text().splitlines()
    assert csv[0] == "index,color_stamp_s,depth_stamp_s,pair_dt_ms,has_depth"
    assert len(csv) == 6                              # header + 5 rows
    assert csv[3].split(",")[-1] == "0"              # index 2 row -> has_depth=0 (dropped)


# ===========================================================================
# D. metadata.json merge (append_aligned_depth_to_metadata + main() wiring)
#
# main() appends the aligned-depth stream into the metadata.json that
# rosbag_process_color_v3 already wrote in the SHARED out_dir, touching no other
# keys. These tests pre-write a minimal stub of that file (the fields the merge
# reads/updates) and assert the merge behaviour end-to-end via main().
# ===========================================================================
def write_stub_metadata(out_root, camera_label="cam_ego"):
    """Minimal stand-in for the metadata.json rosbag_process_color_v3 writes:
    just the keys the merge reads (camera_intrinsics + steps.streams) plus a few
    unrelated keys used to prove the merge leaves everything else untouched."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    meta = {
        "metadata": {"dataset_name": "leo", "subject": "abc-123"},
        "camera_intrinsics": [
            {"camera": camera_label, "color": {"width": 24, "height": 18}},
            {"camera": "cam1", "color": {"width": 640, "height": 480}},
        ],
        "steps": {
            "streams": [{"camera": camera_label, "kind": "color", "num_frames": 8}],
            "timestamp_range": [1.0, 2.0],
        },
        "termination": {"is_successful": True, "reason": []},
    }
    path = out_root / "metadata.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return path


def _load_meta(out):
    with open(Path(out) / "metadata.json", encoding="utf-8") as f:
        return json.load(f)


def _aligned_entries(meta):
    return [s for s in meta["steps"]["streams"] if s["kind"] == "aligned_depth_to_color"]


def test_metadata_absent_is_noop_not_crash(tmp_path):
    # No metadata.json pre-written: merge must no-op, not fabricate a file or crash.
    bag = build_bag(tmp_path / "bag", n_color=4)
    s = run_main(bag, tmp_path / "out")
    assert s["n_paired"] == 4                               # alignment still succeeded
    assert not (Path(tmp_path / "out") / "metadata.json").exists()


def test_metadata_stream_appended_with_expected_fields(tmp_path):
    bag = build_bag(tmp_path / "bag", n_color=8)
    out = tmp_path / "out"
    write_stub_metadata(out)
    s = run_main(bag, out)
    aligned = _aligned_entries(_load_meta(out))
    assert len(aligned) == 1
    e = aligned[0]
    assert e["camera"] == "cam_ego"
    assert e["frame_dtype"] == "uint16"
    assert e["depth_unit"] == "mm"
    assert e["frames_dir"] == "depth_frames/ego_aligned_depth_to_color.h5"
    assert e["timestamps"] == "timestamps/ego_aligned_depth_to_color.csv"
    assert e["width"] == 24 and e["height"] == 18          # COLOR resolution (aligned frame)
    assert e["topic"].endswith("/depth/image_rect_raw")
    assert e["h5_index"] == s["n_color"] == 8
    assert e["fps_estimate"] and e["fps_estimate"] > 0
    # ts_min/ts_max: span of the color-indexed timeline, present and ordered
    assert e["ts_min"] is not None and e["ts_max"] is not None
    assert e["ts_max"] >= e["ts_min"]


def test_metadata_persists_pairing_counts(tmp_path):
    # The new extraction: the pairing counts must land in the metadata stream entry
    # (n_pair_missing_color enables the depth->color check -- it exists ONLY here,
    # not in the CSV/h5).
    bag = build_bag(tmp_path / "bag", n_color=8, drop_depth=(3,), drop_color=(5,))
    out = tmp_path / "out"
    write_stub_metadata(out)
    s = run_main(bag, out)
    e = _aligned_entries(_load_meta(out))[0]
    assert e["n_paired"] == s["n_paired"]
    assert e["n_pair_missing_depth"] == s["n_blank_color"]
    assert e["n_depth_frames"] == s["n_depth_msgs"]
    assert e["n_pair_missing_color"] == s["n_depth_no_partner"]
    assert e["n_pair_missing_color"] >= 1      # a dropped color -> a depth with no partner


def test_metadata_h5_index_is_full_h5_length_not_paired(tmp_path):
    # h5_index must equal the h5 length (blanks included), NOT n_paired.
    bag = build_bag(tmp_path / "bag", n_color=8, drop_depth=(3,))
    out = tmp_path / "out"
    write_stub_metadata(out)
    s = run_main(bag, out)
    e = _aligned_entries(_load_meta(out))[0]
    assert s["n_paired"] == 7 and s["n_blank_color"] == 1
    assert e["h5_index"] == 8                               # includes the blank slot
    data, _ = read_h5(out)
    assert data.shape[0] == e["h5_index"]                  # matches the dataset shape


def test_metadata_depth_intrinsics_block_added(tmp_path):
    bag = build_bag(tmp_path / "bag", n_color=4)
    out = tmp_path / "out"
    write_stub_metadata(out)
    run_main(bag, out)
    ego = next(c for c in _load_meta(out)["camera_intrinsics"] if c["camera"] == "cam_ego")
    assert "color" in ego                                   # pre-existing block untouched
    d = ego["depth"]                                        # new block
    assert d["width"] == 16 and d["height"] == 12          # depth (native) resolution
    assert d["camera_info_topic"].endswith("/depth/camera_info")
    assert len(d["K"]) == 9 and len(d["P"]) == 12 and len(d["R"]) == 9


def test_metadata_preserves_all_other_keys(tmp_path):
    bag = build_bag(tmp_path / "bag", n_color=4)
    out = tmp_path / "out"
    write_stub_metadata(out)
    before = _load_meta(out)
    run_main(bag, out)
    after = _load_meta(out)
    assert after["metadata"] == before["metadata"]
    assert after["termination"] == before["termination"]
    assert after["steps"]["timestamp_range"] == before["steps"]["timestamp_range"]
    assert any(s["kind"] == "color" for s in after["steps"]["streams"])   # color stream kept
    cam1_before = next(c for c in before["camera_intrinsics"] if c["camera"] == "cam1")
    cam1_after = next(c for c in after["camera_intrinsics"] if c["camera"] == "cam1")
    assert cam1_after == cam1_before                       # unrelated camera untouched


def test_metadata_append_is_idempotent(tmp_path):
    bag = build_bag(tmp_path / "bag", n_color=4)
    out = tmp_path / "out"
    write_stub_metadata(out)
    run_main(bag, out)
    run_main(bag, out)                                     # re-run: must replace, not duplicate
    meta = _load_meta(out)
    assert len(_aligned_entries(meta)) == 1
    ego = next(c for c in meta["camera_intrinsics"] if c["camera"] == "cam_ego")
    assert "depth" in ego


def _extrinsic(meta, name):
    return next((e for e in meta.get("camera_extrinsics", []) if e["name"] == name), None)


def test_metadata_depth_to_color_extrinsic_written_verbatim(tmp_path):
    # The depth step records the depth->color extrinsic it aligned with, VERBATIM,
    # into camera_extrinsics (numbers copied as-is, no composition).
    rot = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    trans = (0.0151, -0.0002, 0.0003)
    bag = build_bag(tmp_path / "bag", n_color=4, rotation=rot, translation=trans)
    out = tmp_path / "out"
    write_stub_metadata(out)
    run_main(bag, out)
    e = _extrinsic(_load_meta(out), "depth_to_color")
    assert e is not None
    assert e["rotation"] == [float(x) for x in rot]
    assert e["translation"] == [float(x) for x in trans]
    assert e["source_topic"].endswith("extrinsics/depth_to_color")
    assert "column-major" in e["convention"]


def test_metadata_depth_to_color_extrinsic_idempotent(tmp_path):
    bag = build_bag(tmp_path / "bag", n_color=4, translation=(0.015, 0, 0))
    out = tmp_path / "out"
    write_stub_metadata(out)
    run_main(bag, out)
    run_main(bag, out)                                     # re-run must not duplicate
    exts = _load_meta(out).get("camera_extrinsics", [])
    assert [e["name"] for e in exts].count("depth_to_color") == 1


# --- pure-helper unit tests -------------------------------------------------
def test_relpath_relativizes_and_falls_back():
    assert ead._relpath("/a/b/c/depth_frames/x.h5", "/a/b/c") == "depth_frames/x.h5"
    assert ead._relpath("/x/y.h5", "/a/b/c") == "/x/y.h5"   # unrelated -> abs string


def test_fps_from_stamps_uses_robust_median():
    assert abs(ead._fps_from_stamps(np.arange(10) / 30.0) - 30.0) < 1e-6
    assert ead._fps_from_stamps(np.array([0.0, 0.1])) is None          # too few
    # one big gap (a dropped frame): median ignores it, mean would not.
    gappy = np.array([0.0, 1 / 30, 2 / 30, 10 / 30])
    assert abs(ead._fps_from_stamps(gappy) - 30.0) < 1e-6


def test_append_returns_false_when_no_metadata(tmp_path):
    color = make_intr(24, 18, 18, 18, 12, 9)
    summary = {"h5": str(tmp_path / "depth_frames" / "ego_aligned_depth_to_color.h5"),
               "csv": str(tmp_path / "timestamps" / "ego_aligned_depth_to_color.csv"),
               "n_color": 5}
    ok = ead.append_aligned_depth_to_metadata(
        tmp_path, summary, None, np.array([0.0, 1 / 30, 2 / 30]), color,
        "/ego/camera/depth/image_rect_raw")
    assert ok is False


def test_append_without_matching_camera_appends_stream_but_skips_depth(tmp_path):
    # metadata has no cam_ego entry -> stream still appended, depth block skipped.
    write_stub_metadata(tmp_path, camera_label="cam_other")
    color = make_intr(24, 18, 18, 18, 12, 9)
    summary = {"h5": str(tmp_path / "depth_frames" / "ego_aligned_depth_to_color.h5"),
               "csv": str(tmp_path / "timestamps" / "ego_aligned_depth_to_color.csv"),
               "n_color": 5}
    depth_block = {"camera_info_topic": "x/depth/camera_info", "width": 16, "height": 12,
                   "K": [1, 0, 8, 0, 1, 6, 0, 0, 1]}
    ok = ead.append_aligned_depth_to_metadata(
        tmp_path, summary, depth_block, np.array([0.0, 1 / 30, 2 / 30, 3 / 30]), color,
        "/ego/camera/depth/image_rect_raw")
    assert ok is True
    meta = _load_meta(tmp_path)
    assert len(_aligned_entries(meta)) == 1
    assert all("depth" not in c for c in meta["camera_intrinsics"])   # nothing to attach to


# ===========================================================================
# E. Returned summary calib guard + pairing-index edge cases
#
# These target the exact booleans wrapper.py:296 gates on (the RETURNED summary
# dict, not stdout) and the two index-remap branches in the pairing loop that
# build_bag's one-depth-per-color-shared-stamp shape can't express, so those two
# use a small custom bag built with the same rosbags Writer pattern build_bag
# uses (explicit per-frame HEADER stamps).
# ===========================================================================
def build_bag_stamps(path, color_stamps_ns, depth_specs, *,
                     rotation=IDENTITY9, translation=(0.015, 0.0, 0.0),
                     depth_dims=(16, 12), color_dims=(24, 18),
                     depth_k=(12.0, 12.0, 8.0, 6.0), color_k=(18.0, 18.0, 12.0, 9.0),
                     prefix="/ego/camera"):
    """Custom bag with explicit per-frame HEADER stamps (build_bag ties one depth
    to one color at a shared stamp, which can't express two-depth-to-one-color or
    shuffled color stamps). Bag RECORD time is a separate monotonic counter, so
    arrival order == write order while header stamps are whatever we pass.
      color_stamps_ns: header stamp (ns) per color frame, in write/arrival order.
      depth_specs:     (header_stamp_ns, fill_value) per depth frame, write order.
    One color_info + depth_info + extrinsics message is written up front."""
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    ts = get_typestore(Stores.ROS2_HUMBLE)
    ts.register(get_types_from_msg("float64[9] rotation\nfloat64[3] translation\n",
                                   "realsense2_camera_msgs/msg/Extrinsics"))
    Time = ts.types["builtin_interfaces/msg/Time"]
    Header = ts.types["std_msgs/msg/Header"]
    Image = ts.types["sensor_msgs/msg/Image"]
    CI = ts.types["sensor_msgs/msg/CameraInfo"]
    ROI = ts.types["sensor_msgs/msg/RegionOfInterest"]
    Extr = ts.types["realsense2_camera_msgs/msg/Extrinsics"]
    DW, DH = depth_dims
    CW, CH = color_dims

    def hdr(t, frame):
        return Header(stamp=Time(sec=t // 1_000_000_000, nanosec=t % 1_000_000_000), frame_id=frame)

    def cinfo(w, h, k, frame):
        fx, fy, cx, cy = k
        return CI(header=hdr(0, frame), height=h, width=w, distortion_model="plumb_bob",
                  d=np.array(ZEROS5, np.float64), k=np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1], np.float64),
                  r=np.eye(3).ravel().astype(np.float64),
                  p=np.array([fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0], np.float64),
                  binning_x=0, binning_y=0,
                  roi=ROI(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False))

    rec = 0  # monotonic bag record time (ns); header stamps set independently above
    with Writer(path, version=9) as w:
        c_color = w.add_connection(f"{prefix}/color/image_raw", Image.__msgtype__, typestore=ts)
        c_cinfo = w.add_connection(f"{prefix}/color/camera_info", CI.__msgtype__, typestore=ts)
        c_depth = w.add_connection(f"{prefix}/depth/image_rect_raw", Image.__msgtype__, typestore=ts)
        c_dinfo = w.add_connection(f"{prefix}/depth/camera_info", CI.__msgtype__, typestore=ts)
        c_extr = w.add_connection(f"{prefix}/extrinsics/depth_to_color", Extr.__msgtype__, typestore=ts)
        w.write(c_extr, rec, ts.serialize_cdr(
            Extr(rotation=np.array(rotation, np.float64), translation=np.array(translation, np.float64)),
            Extr.__msgtype__))
        w.write(c_cinfo, rec, ts.serialize_cdr(cinfo(CW, CH, color_k, "color"), CI.__msgtype__))
        w.write(c_dinfo, rec, ts.serialize_cdr(cinfo(DW, DH, depth_k, "depth"), CI.__msgtype__))
        rec += 1
        for i, cs in enumerate(color_stamps_ns):
            color = np.dstack([np.full((CH, CW), 50 + i, np.uint8),
                               np.full((CH, CW), 100, np.uint8),
                               np.full((CH, CW), 150, np.uint8)])
            w.write(c_color, rec, ts.serialize_cdr(
                Image(header=hdr(cs, "color"), height=CH, width=CW, encoding="rgb8",
                      is_bigendian=0, step=CW * 3, data=color.reshape(-1)), Image.__msgtype__))
            rec += 1
        for ds, val in depth_specs:
            depth = np.full((DH, DW), val, np.uint16)
            w.write(c_depth, rec, ts.serialize_cdr(
                Image(header=hdr(ds, "depth"), height=DH, width=DW, encoding="16UC1",
                      is_bigendian=0, step=DW * 2, data=depth.view(np.uint8).reshape(-1)), Image.__msgtype__))
            rec += 1
    return path


def test_summary_calib_flags_clean(tmp_path):
    # Clean bag: every guard passes and calib_ok is the AND of them. Asserts the
    # RETURNED dict (the exact booleans wrapper.py:296 gates on), not stdout.
    bag = build_bag(tmp_path / "bag", n_color=4)
    s = run_main(bag, tmp_path / "out")
    assert s["calib_ok"] is True
    assert s["depth_distortion_ok"] is True
    assert s["color_distortion_ok"] is True
    assert s["rotation_ok"] is True
    assert s["translation_ok"] is True
    assert s["rect_clamped"] == 0


def test_summary_calib_flags_color_distortion_fails(tmp_path):
    # Same injection test_nonzero_coeffs_warns uses: non-zero COLOR distortion
    # coeffs. The single failing guard must drag calib_ok False.
    bag = build_bag(tmp_path / "bag", n_color=4, color_coeffs=[0.1, -0.2, 0.0, 0.0, 0.05])
    s = run_main(bag, tmp_path / "out")
    assert s["color_distortion_ok"] is False
    assert s["calib_ok"] is False


def test_summary_calib_flags_rotation_fails(tmp_path):
    # Same injection test_far_from_identity_warns uses: R rotated 20 deg off
    # identity. rotation_ok must be False and calib_ok False with it.
    a = math.radians(20)
    Rmat = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    bag = build_bag(tmp_path / "bag", n_color=4, rotation=Rmat.T.reshape(-1).tolist(),
                    translation=(0.2, 0.0, 0.0))
    s = run_main(bag, tmp_path / "out")
    assert s["rotation_ok"] is False
    assert s["calib_ok"] is False


def test_closer_depth_reclaims_slot(tmp_path):
    # Two depth frames both within PAIR_TOLERANCE_MS (16 ms) of ONE color frame's
    # stamp. The NEARER depth must win the single color slot (production guard
    # `if best >= best_dt[cidx]: continue`). The two carry distinguishable constant
    # depth values so the aligned h5 slot reveals which frame landed. The farther
    # frame is written FIRST, so the nearer one reclaims (overwrites) the slot.
    color_at = 100_000_000                     # 0.100 s
    far = (color_at - 4_000_000, 700)          # 0.096 s, dt 4 ms  -> loser
    near = (color_at + 2_000_000, 1500)        # 0.102 s, dt 2 ms  -> winner
    bag = build_bag_stamps(tmp_path / "bag", [color_at], [far, near])
    s = run_main(bag, tmp_path / "out")
    assert s["n_color"] == 1
    assert s["n_depth_msgs"] == 2
    assert s["n_paired"] == 1                   # the one color slot, paired once
    assert s["n_blank_color"] == 0
    assert s["n_depth_no_partner"] == 0         # a reclaim is not a no-partner drop
    data, _ = read_h5(tmp_path / "out")
    nz = data[0][data[0] > 0]
    assert nz.size > 0
    assert nz.min() == 1500 and nz.max() == 1500   # only the nearer frame landed
    assert not (data[0] == 700).any()              # the farther frame never wrote


def test_nonmonotonic_color_stamps_remap(tmp_path, capsys):
    # Color HEADER stamps are shuffled vs write/arrival order: write order carries
    # stamps [0.2, 0.1, 0.3]. main() sorts them for searchsorted (order=[1,0,2])
    # and must map each pairing back through order[best_k] to the color frame's
    # ORIGINAL write-index. Each depth pairs unambiguously to one color stamp and
    # carries a distinct value, so the h5 slot it lands in proves the remap: it
    # must land at the color's write-index, NOT its sorted position.
    c0, c1, c2 = 200_000_000, 100_000_000, 300_000_000   # write order (shuffled)
    color_stamps = [c0, c1, c2]
    depth_specs = [(c0, 1000), (c1, 2000), (c2, 3000)]   # value tied to a stamp
    bag = build_bag_stamps(tmp_path / "bag", color_stamps, depth_specs)
    s = run_main(bag, tmp_path / "out")
    assert "not monotonic" in capsys.readouterr().out    # the remap path was taken
    assert s["n_color"] == 3 and s["n_paired"] == 3 and s["n_depth_no_partner"] == 0
    data, _ = read_h5(tmp_path / "out")
    # depth@0.2 -> write-index 0 (sorted pos 1); depth@0.1 -> write-index 1
    # (sorted pos 0); depth@0.3 -> write-index 2. Landing at the write-index and
    # not the sorted position is the whole point of order[best_k].
    assert data[0].max() == 1000
    assert data[1].max() == 2000
    assert data[2].max() == 3000


# ---------------------------------------------------------------------------
# record_missing_depth: an absent depth topic -> extraction-owned missing_stream
# entry + token, APPENDED into the color-written metadata.json (never clobbering
# the color streams / other reasons).
# ---------------------------------------------------------------------------
def test_record_missing_depth_appends_without_clobbering(tmp_path):
    meta = {
        "steps": {
            "streams": [{"camera": "cam_ego", "kind": "color"}],
            "missing_stream_error": [],
            "extra_stream_error": [],
        },
        "termination": {"is_successful": False, "reason": ["extra_stream"]},
    }
    (tmp_path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    assert ead.record_depth_presence(
        tmp_path, [f"cam_ego depth: expected but not found (*{ead.DEPTH_SUFFIX})"],
        [], [], []) is True
    out = json.loads((tmp_path / "metadata.json").read_text())

    # the color stream is untouched
    assert out["steps"]["streams"] == [{"camera": "cam_ego", "kind": "color"}]
    # a missing_stream entry naming the depth camera was appended
    assert len(out["steps"]["missing_stream_error"]) == 1
    assert "depth" in out["steps"]["missing_stream_error"][0]
    # the data-plane token is added, and the pre-existing foreign token is preserved
    assert out["termination"]["reason"] == ["extra_stream", "depth_presence_err"]
    assert out["termination"]["is_successful"] is False


def test_missing_depth_topic_flags_and_continues(tmp_path):
    # Worklist #4: a bag with NO depth image topic must NOT raise. main() records the
    # extraction-owned missing_stream error and RETURNS (flag-and-continue), writing no
    # h5. Marking the bag incomplete is the wrapper's final sanity verdict, not a crash.
    bag = build_bag(tmp_path / "bag", n_color=4, with_depth=False)
    out = tmp_path / "out"
    out.mkdir()
    # emulate the color step having written metadata.json first (so the flag can land)
    (out / "metadata.json").write_text(json.dumps(
        {"steps": {"streams": [], "missing_stream_error": []},
         "termination": {"is_successful": True, "reason": []}}), encoding="utf-8")

    s = ead.main(bag=str(bag), out_dir=str(out), camera="ego")   # must NOT raise
    assert s.get("aligned") is False

    meta = json.loads((out / "metadata.json").read_text())
    assert len(meta["steps"]["missing_stream_error"]) == 1        # depth miss flagged
    assert not any((out / "depth_frames").glob("*.h5"))           # no aligned h5 written


def test_record_missing_depth_noop_without_metadata(tmp_path):
    # standalone run with no color metadata.json yet -> no-op, returns False
    assert ead.record_depth_presence(tmp_path, ["cam_ego depth: not found"], [], [], []) is False
    assert not (tmp_path / "metadata.json").exists()


def test_record_missing_depth_idempotent(tmp_path):
    (tmp_path / "metadata.json").write_text(
        json.dumps({"steps": {}, "termination": {"is_successful": True, "reason": []}}),
        encoding="utf-8")
    entry = [f"cam_ego depth: expected but not found (*{ead.DEPTH_SUFFIX})"]
    ead.record_depth_presence(tmp_path, entry, [], [], [])
    ead.record_depth_presence(tmp_path, entry, [], [], [])   # re-run
    out = json.loads((tmp_path / "metadata.json").read_text())
    assert len(out["steps"]["missing_stream_error"]) == 1   # not duplicated
    assert out["termination"]["reason"] == ["depth_presence_err"]
