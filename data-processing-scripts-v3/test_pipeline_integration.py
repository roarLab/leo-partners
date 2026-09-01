"""Integration suite for the full color+depth pipeline (wrapper.py).

Unlike test_wrapper.py -- which MOCKS the four subscripts and tests the wrapper's
own logic in isolation -- this file runs the REAL rosbag_process_color_v3,
rosbag_process_depth_v3, validate_color_v3 and validate_depth_v3 end-to-end on a
tiny synthetic ROS2 bag built in a tmpdir. Nothing is mocked: every function is
actually called.

Its job is the one thing the unit suite structurally cannot do -- prove the four
scripts actually compose on disk (metadata.json created -> appended -> validated
-> read back), and that each of the five report signals is produced by the REAL
validators, not by a hand-written fixture. A reworded validator message or a
changed annotation key breaks a test here while the mocked unit tests stay green.

The synthetic bag emits one "ego" camera under /ego/d435i_ego/, whose color
suffixes are exactly the color script's hardcoded ego topics AND match the depth
aligner's suffix discovery -- so a single camera drives both scripts. Optional
knobs add a second (c922) color camera for the frame-loss case and an uneven
timestamp for the timing-gap case.

Run:  conda activate leo-seg && pytest data-processing-scripts-v3/ -q
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import wrapper as wrap                          # noqa: E402
import rosbag_process_color_v3 as rpc           # noqa: E402

# --- bag geometry (kept tiny; content is irrelevant, only counts/stamps matter) ---
EGO_PREFIX = "/ego/d435i_ego"
FRAME_NS = 33_333_333          # ~30 fps
GAP_NS = 1_000_000_000         # 1 s jump -- far above 5x the frame period
DEPTH_DIMS = (16, 12)          # (width, height)
COLOR_DIMS = (24, 18)
DEPTH_K = (12.0, 12.0, 8.0, 6.0)
COLOR_K = (18.0, 18.0, 12.0, 9.0)
ZEROS5 = [0.0, 0.0, 0.0, 0.0, 0.0]
IDENTITY9 = [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]
TRANSLATION = (0.015, 0.0, 0.0)


def build_pipeline_bag(path, *, n_color=8, drop_depth=(),
                       c922_counts=None, gap_after=None):
    """Write a tiny synthetic ROS2 bag both scripts can read.

    n_color      color+depth frame count for the ego camera.
    drop_depth   ego color-frame indices whose DEPTH partner is omitted
                 (-> blank h5 frames -> color->depth mismatch when > 10%).
    c922_counts  frame count per c922 color camera, e.g. [8, 8, 8] emits
                 /c922_1../c922_3 each with 8 frames. validate_color compares
                 every color camera against the busiest, so a CLEAN run needs all
                 four cameras present at full count (a missing camera reads as
                 0 frames -> color error); a short entry here is genuine frame
                 loss. None emits no c922 cameras.
    gap_after    if set, shift every frame after this index by GAP_NS, creating a
                 single interval > 5x the mean (-> color + depth timestamp error).
    """
    path = Path(path)
    ts = get_typestore(Stores.ROS2_HUMBLE)
    ts.register(get_types_from_msg(
        "float64[9] rotation\nfloat64[3] translation\n",
        "realsense2_camera_msgs/msg/Extrinsics"))
    Time = ts.types["builtin_interfaces/msg/Time"]
    Header = ts.types["std_msgs/msg/Header"]
    Image = ts.types["sensor_msgs/msg/Image"]
    CompressedImage = ts.types["sensor_msgs/msg/CompressedImage"]
    CI = ts.types["sensor_msgs/msg/CameraInfo"]
    ROI = ts.types["sensor_msgs/msg/RegionOfInterest"]
    Extr = ts.types["realsense2_camera_msgs/msg/Extrinsics"]
    DW, DH = DEPTH_DIMS
    CW, CH = COLOR_DIMS

    def stamp(i):
        return i * FRAME_NS + (GAP_NS if (gap_after is not None and i > gap_after) else 0)

    def hdr(t, frame):
        return Header(stamp=Time(sec=t // 1_000_000_000, nanosec=t % 1_000_000_000),
                      frame_id=frame)

    def cinfo(t, w, h, k, frame):
        fx, fy, cx, cy = k
        return CI(header=hdr(t, frame), height=h, width=w, distortion_model="plumb_bob",
                  d=np.array(ZEROS5, np.float64),
                  k=np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1], np.float64),
                  r=np.eye(3).ravel().astype(np.float64),
                  p=np.array([fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0], np.float64),
                  binning_x=0, binning_y=0,
                  roi=ROI(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False))

    yy, xx = np.mgrid[0:DH, 0:DW]
    depth_base = (900 + 8 * xx + 5 * yy).astype(np.uint16)
    color_bgr = np.dstack([np.full((CH, CW), 50, np.uint8),
                           np.full((CH, CW), 100, np.uint8),
                           np.full((CH, CW), 150, np.uint8)])
    _, jpeg = cv2.imencode(".jpg", color_bgr)
    jpeg_bytes = np.frombuffer(jpeg.tobytes(), dtype=np.uint8)

    with Writer(path, version=9) as w:
        c_color = w.add_connection(f"{EGO_PREFIX}/color/image_raw", Image.__msgtype__, typestore=ts)
        c_cinfo = w.add_connection(f"{EGO_PREFIX}/color/camera_info", CI.__msgtype__, typestore=ts)
        c_depth = w.add_connection(f"{EGO_PREFIX}/depth/image_rect_raw", Image.__msgtype__, typestore=ts)
        c_dinfo = w.add_connection(f"{EGO_PREFIX}/depth/camera_info", CI.__msgtype__, typestore=ts)
        c_extr = w.add_connection(f"{EGO_PREFIX}/extrinsics/depth_to_color", Extr.__msgtype__, typestore=ts)
        w.write(c_extr, 0, ts.serialize_cdr(
            Extr(rotation=np.array(IDENTITY9, np.float64),
                 translation=np.array(TRANSLATION, np.float64)), Extr.__msgtype__))

        c922_conns = []
        for j, count in enumerate(c922_counts or []):
            cam = f"/c922_{j + 1}"
            c_img = w.add_connection(f"{cam}/image_raw/compressed",
                                     CompressedImage.__msgtype__, typestore=ts)
            c_ci = w.add_connection(f"{cam}/camera_info", CI.__msgtype__, typestore=ts)
            c922_conns.append((c_img, c_ci, count))

        for i in range(n_color):
            t = stamp(i)
            w.write(c_cinfo, t, ts.serialize_cdr(cinfo(t, CW, CH, COLOR_K, "color"), CI.__msgtype__))
            w.write(c_dinfo, t, ts.serialize_cdr(cinfo(t, DW, DH, DEPTH_K, "depth"), CI.__msgtype__))
            color_rgb = color_bgr[..., ::-1].copy()
            w.write(c_color, t, ts.serialize_cdr(
                Image(header=hdr(t, "color"), height=CH, width=CW, encoding="rgb8",
                      is_bigendian=0, step=CW * 3, data=color_rgb.reshape(-1)), Image.__msgtype__))
            if i not in drop_depth:
                w.write(c_depth, t, ts.serialize_cdr(
                    Image(header=hdr(t, "depth"), height=DH, width=DW, encoding="16UC1",
                          is_bigendian=0, step=DW * 2,
                          data=depth_base.view(np.uint8).reshape(-1)), Image.__msgtype__))
            for c_img, c_ci, count in c922_conns:
                if i < count:
                    w.write(c_ci, t, ts.serialize_cdr(cinfo(t, CW, CH, COLOR_K, "c922"), CI.__msgtype__))
                    w.write(c_img, t, ts.serialize_cdr(
                        CompressedImage(header=hdr(t, "c922"), format="jpeg", data=jpeg_bytes),
                        CompressedImage.__msgtype__))
    return path


@pytest.fixture(autouse=True)
def portable_codec(monkeypatch):
    """Force the OpenCV mp4v writer so the suite stays hermetic and needs no ffmpeg
    binary. The default 'h264' path pipes frames to ffmpeg+libx264 (imageio-ffmpeg or a
    system ffmpeg) — exercised in the real pipeline, not required for these tests."""
    monkeypatch.setattr(rpc, "VIDEO_CODEC", "mp4v")


def _run(bag, out):
    """Run the whole real pipeline for one bag and return the report row."""
    return wrap.run_pipeline_for_bag(bag, out, "ego", {"subject": "integration-test"})


# ===========================================================================
# 1. No injected faults -> pipeline completes and reports no errors
# ===========================================================================
def test_pipeline_with_no_faults_reports_no_errors(tmp_path):
    # all four color cameras at full count -> validate_color has nothing to flag
    bag = build_pipeline_bag(tmp_path / "run1", n_color=8, c922_counts=[8, 8, 8])
    out = tmp_path / "out"
    row = _run(bag, out)

    assert row["completed"] is True
    assert (out / "depth_frames" / "ego_aligned_depth_to_color.h5").exists()
    # every error signal is silent on a clean bag
    for col in ("color_error", "color_timestamp_error", "depth_error",
                "depth_timestamp_error", "imu_error", "imu_timestamp_error"):
        assert row[col] == "", f"{col} should be blank, got {row[col]!r}"


# ===========================================================================
# 2. Dropped depth frames -> real validator reports a color/depth mismatch
# ===========================================================================
def test_depth_dropout_is_reported_as_depth_error(tmp_path):
    # 4 of 12 color frames (33%) get no depth partner -- well past the 10% gate.
    # The two-way pairing miss now folds into the depth_error column (color_depth_mismatch
    # was removed); the "unpaired ..." message is preserved.
    bag = build_pipeline_bag(tmp_path / "run1", n_color=12, drop_depth=(2, 3, 4, 5))
    row = _run(bag, tmp_path / "out")

    assert row["completed"] is True
    assert "unpaired" in row["depth_error"], row["depth_error"]


# ===========================================================================
# 3. The report row from a real run has exactly the seven columns
# ===========================================================================
def test_report_row_has_the_seven_columns(tmp_path):
    bag = build_pipeline_bag(tmp_path / "run1", n_color=8)
    out = tmp_path / "out"
    row = _run(bag, out)

    assert set(row) == set(wrap.REPORT_COLUMNS)
    assert row["out_dir"] == str(out)


# ===========================================================================
# 4. One camera loses >10% of frames -> real validator reports a color error
# ===========================================================================
def test_color_frame_loss_is_reported_as_color_error(tmp_path):
    # every camera full except c922_1 at 6/10 (60%) -> only that loss is flagged
    bag = build_pipeline_bag(tmp_path / "run1", n_color=10, c922_counts=[6, 10, 10])
    row = _run(bag, tmp_path / "out")

    assert row["completed"] is True
    assert row["color_error"] != "", "expected the c922 frame loss to be flagged"


# ===========================================================================
# 5. An oversized timing gap -> real validators report timestamp errors
# ===========================================================================
def test_timing_gap_is_reported_as_timestamp_error(tmp_path):
    # one ~1 s jump after frame 3, everything else at ~30 fps
    bag = build_pipeline_bag(tmp_path / "run1", n_color=8, gap_after=3)
    row = _run(bag, tmp_path / "out")

    assert row["completed"] is True
    assert row["color_timestamp_error"] != "", "color stamps carry the gap"
    assert row["depth_timestamp_error"] != "", "paired-depth stamps carry the gap"
