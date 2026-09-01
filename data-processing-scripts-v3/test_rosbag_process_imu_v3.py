"""Unit tests for rosbag_process_imu_v3.py (IMU extrinsics + sample extraction).

Run:  conda activate leo-seg && pytest data-processing-scripts-v3/ -q

Builds tiny synthetic ROS2 bags carrying the depth_to_gyro / depth_to_accel extrinsic
topics (realsense2_camera_msgs/Extrinsics) and — for the sample tests — a united
sensor_msgs/Imu stream. Asserts the extrinsics land VERBATIM in camera_extrinsics
(owner-scoped: the depth-owned depth_to_color entry is never touched) and the /imu
samples land in imu/cam_ego_imu.csv + a steps.streams entry, each tagged with the ego
color frame it falls under by the PRECEDING rule.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rosbag_process_imu_v3 as imu                          # noqa: E402

from rosbags.rosbag2 import Writer                            # noqa: E402
from rosbags.typesys import get_typestore, Stores, get_types_from_msg  # noqa: E402

IDENTITY9 = [1, 0, 0, 0, 1, 0, 0, 0, 1]
GYRO_T = (0.005, -0.002, 0.001)
ACCEL_T = (0.021, -0.005, 0.011)


# ---------------------------------------------------------------------------
# Synthetic bag builder — the IMU extrinsic topics (+ one color frame so the bag
# is a well-formed recording), mirroring test_rosbag_process_depth_v3's approach.
# ---------------------------------------------------------------------------
def build_bag(path, *, gyro=(IDENTITY9, GYRO_T), accel=(IDENTITY9, ACCEL_T),
              with_gyro=True, with_accel=True, prefix="/ego/d435i_ego",
              imu_samples=None, imu_frame="camera_imu_optical_frame"):
    """imu_samples: optional list of (bag_time_ns, (gx,gy,gz), (ax,ay,az)) written to
    {prefix}/imu as a united sensor_msgs/Imu (orientation left unpopulated, covariances
    the fixed 0.01 default — exactly what the real driver emits, so the extractor's drop
    of those fields is exercised)."""
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    ts = get_typestore(Stores.ROS2_HUMBLE)
    ts.register(get_types_from_msg("float64[9] rotation\nfloat64[3] translation\n",
                                   "realsense2_camera_msgs/msg/Extrinsics"))
    Time = ts.types["builtin_interfaces/msg/Time"]
    Header = ts.types["std_msgs/msg/Header"]
    Image = ts.types["sensor_msgs/msg/Image"]
    Extr = ts.types["realsense2_camera_msgs/msg/Extrinsics"]
    Imu = ts.types["sensor_msgs/msg/Imu"]
    Vec3 = ts.types["geometry_msgs/msg/Vector3"]
    Quat = ts.types["geometry_msgs/msg/Quaternion"]

    def hdr(t, frame):
        return Header(stamp=Time(sec=t // 1_000_000_000, nanosec=t % 1_000_000_000),
                      frame_id=frame)

    def write_extr(w, conn, r, t):
        w.write(conn, 0, ts.serialize_cdr(
            Extr(rotation=np.array(r, np.float64), translation=np.array(t, np.float64)),
            Extr.__msgtype__))

    def make_imu(t_ns, g, a):
        return Imu(
            header=hdr(t_ns, imu_frame),
            orientation=Quat(x=0.0, y=0.0, z=0.0, w=0.0),               # driver leaves unpopulated
            orientation_covariance=np.array([-1.0] + [0.0] * 8, np.float64),
            angular_velocity=Vec3(x=g[0], y=g[1], z=g[2]),
            angular_velocity_covariance=np.full(9, 0.01, np.float64),   # fixed config default
            linear_acceleration=Vec3(x=a[0], y=a[1], z=a[2]),
            linear_acceleration_covariance=np.full(9, 0.01, np.float64),
        )

    with Writer(path, version=9) as w:
        c_color = w.add_connection(f"{prefix}/color/image_raw", Image.__msgtype__, typestore=ts)
        if with_gyro:
            r, t = gyro
            write_extr(w, w.add_connection(f"{prefix}/extrinsics/depth_to_gyro",
                                           Extr.__msgtype__, typestore=ts), r, t)
        if with_accel:
            r, t = accel
            write_extr(w, w.add_connection(f"{prefix}/extrinsics/depth_to_accel",
                                           Extr.__msgtype__, typestore=ts), r, t)
        img = np.zeros((4, 6, 3), np.uint8)
        w.write(c_color, 1, ts.serialize_cdr(
            Image(header=hdr(1, "color"), height=4, width=6, encoding="rgb8",
                  is_bigendian=0, step=6 * 3, data=img.reshape(-1)), Image.__msgtype__))
        # IMU samples last so bag-time is non-decreasing across writes (extr@0, color@1, imu@~10s).
        # `is not None` (not truthiness) so imu_samples=[] creates the /imu topic with ZERO
        # messages — the present-but-empty case (librealsense <2.57) the missing tests exercise.
        if imu_samples is not None:
            imu_conn = w.add_connection(f"{prefix}/imu", Imu.__msgtype__, typestore=ts)
            for t_ns, g, a in imu_samples:
                w.write(imu_conn, t_ns, ts.serialize_cdr(make_imu(t_ns, g, a), Imu.__msgtype__))
    return path


def write_stub_metadata(out_root, *, seed_extrinsics=True, color_times=None):
    """Minimal stand-in for the color-written metadata.json. seed_extrinsics mirrors
    the hand-seeded template (name-only stubs) so we can prove they fill in place.
    color_times: optional list of cam_ego color-frame times (seconds); when given, a
    cam_ego color stream + its timestamps/cam_ego.csv are written so the sample
    extractor has real frames to match color_frame_index against."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    streams = []
    if color_times is not None:
        (out_root / "timestamps").mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"index": np.arange(len(color_times)),
                      "ros_time_s": np.asarray(color_times, float)}
                     ).to_csv(out_root / "timestamps" / "cam_ego.csv", index=False)
        streams.append({"camera": "cam_ego", "kind": "color",
                        "topic": "/ego/d435i_ego/color/image_raw",
                        "timestamps": "timestamps/cam_ego.csv",
                        "num_frames": len(color_times), "found": True})
    meta = {
        "metadata": {"dataset_name": "leo"},
        "camera_intrinsics": [{"camera": "cam_ego", "color": {"width": 6, "height": 4}}],
        "steps": {"streams": streams, "timestamp_range": [1.0, 2.0]},
        "termination": {"is_successful": True, "reason": []},
    }
    if seed_extrinsics:
        meta["camera_extrinsics"] = [{"name": "depth_to_color"},
                                     {"name": "depth_to_gyro"},
                                     {"name": "depth_to_accel"}]
    path = out_root / "metadata.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def _imu_stream(meta):
    return next((s for s in meta.get("steps", {}).get("streams", [])
                 if s.get("kind") == "imu"), None)


def _read_imu_csv(out):
    return pd.read_csv(Path(out) / "imu" / "cam_ego_imu.csv")


def _load(out):
    return json.loads((Path(out) / "metadata.json").read_text(encoding="utf-8"))


def _extrinsic(meta, name):
    return next((e for e in meta.get("camera_extrinsics", []) if e["name"] == name), None)


# ---------------------------------------------------------------------------
# Restore any config globals a test may set (only _cli mutates them, but be safe).
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def reset_config():
    saved = {k: getattr(imu, k)
             for k in ("CAMERA", "EXTRINSICS_TOPIC_OVERRIDES", "IMU_TOPIC_OVERRIDE")}
    yield
    for k, v in saved.items():
        setattr(imu, k, v)


# ===========================================================================
# Tests
# ===========================================================================
def test_both_extrinsics_written_verbatim(tmp_path):
    bag = build_bag(tmp_path / "bag")
    out = tmp_path / "out"
    write_stub_metadata(out)
    summary = imu.main(bag=bag, out_dir=out, camera="ego")

    meta = _load(out)
    g = _extrinsic(meta, "depth_to_gyro")
    a = _extrinsic(meta, "depth_to_accel")
    assert g["rotation"] == [float(x) for x in IDENTITY9]
    assert g["translation"] == [float(x) for x in GYRO_T]
    assert g["source_topic"].endswith("extrinsics/depth_to_gyro")
    assert a["translation"] == [float(x) for x in ACCEL_T]
    assert "column-major" in g["convention"]
    assert summary["extrinsics_written"] == 2
    assert summary["extrinsics_found"] == ["depth_to_accel", "depth_to_gyro"]


def test_fills_seeded_stub_in_place_no_duplicate(tmp_path):
    bag = build_bag(tmp_path / "bag")
    out = tmp_path / "out"
    write_stub_metadata(out, seed_extrinsics=True)     # name-only stubs present
    imu.main(bag=bag, out_dir=out, camera="ego")
    names = [e["name"] for e in _load(out)["camera_extrinsics"]]
    assert names.count("depth_to_gyro") == 1           # stub filled, not duplicated
    assert names.count("depth_to_accel") == 1


def test_does_not_touch_depth_to_color_entry(tmp_path):
    # imu owns only gyro/accel; the depth_to_color stub (depth's to own) stays as-is.
    bag = build_bag(tmp_path / "bag")
    out = tmp_path / "out"
    write_stub_metadata(out, seed_extrinsics=True)
    imu.main(bag=bag, out_dir=out, camera="ego")
    assert _extrinsic(_load(out), "depth_to_color") == {"name": "depth_to_color"}


def test_works_without_seeded_stubs(tmp_path):
    # No pre-seeded camera_extrinsics: upsert creates the list + entries.
    bag = build_bag(tmp_path / "bag")
    out = tmp_path / "out"
    write_stub_metadata(out, seed_extrinsics=False)
    imu.main(bag=bag, out_dir=out, camera="ego")
    assert sorted(e["name"] for e in _load(out)["camera_extrinsics"]) == \
        ["depth_to_accel", "depth_to_gyro"]


def test_missing_accel_flagged_imu_info_not_fatal(tmp_path):
    # A missing extrinsic leg is non-fatal (the other leg still lands) but is now a
    # presence failure on the INFO plane -> reason 'imu_info', not silently skipped.
    bag = build_bag(tmp_path / "bag", with_accel=False, imu_samples=IMU_SAMPLES)
    out = tmp_path / "out"
    write_stub_metadata(out, seed_extrinsics=False, color_times=COLOR_TIMES)
    summary = imu.main(bag=bag, out_dir=out, camera="ego")
    meta = _load(out)
    assert _extrinsic(meta, "depth_to_gyro") is not None
    assert _extrinsic(meta, "depth_to_accel") is None      # absent leg simply not written
    assert summary["extrinsics_found"] == ["depth_to_gyro"]
    assert summary["extrinsics_written"] == 1
    # the absent leg is flagged as an info-plane presence failure
    assert "imu_info" in meta["termination"]["reason"]
    assert "imu_presence_err" not in meta["termination"]["reason"]  # /imu samples ARE present
    assert any("depth_to_accel" in e for e in meta["steps"]["missing_stream_error"])
    assert meta["termination"]["is_successful"] is False


def test_metadata_absent_is_noop_not_crash(tmp_path):
    # No metadata.json: main reads the extrinsics but writes nothing (no fabricated file).
    bag = build_bag(tmp_path / "bag")
    out = tmp_path / "out"
    summary = imu.main(bag=bag, out_dir=out, camera="ego")
    assert summary["extrinsics_found"] == ["depth_to_accel", "depth_to_gyro"]
    assert summary["extrinsics_written"] == 0
    assert not (out / "metadata.json").exists()


def test_camera_substring_filters_out_other_devices(tmp_path):
    # Extrinsics under a non-ego prefix are ignored when camera="ego".
    bag = build_bag(tmp_path / "bag", prefix="/other_cam")
    out = tmp_path / "out"
    write_stub_metadata(out, seed_extrinsics=False)
    summary = imu.main(bag=bag, out_dir=out, camera="ego")
    assert summary["extrinsics_found"] == []
    assert summary["extrinsics_written"] == 0


def test_missing_bag_raises(tmp_path):
    with pytest.raises(SystemExit):
        imu.main(bag=tmp_path / "nope", out_dir=tmp_path / "out", camera="ego")


# ===========================================================================
# IMU SAMPLE extraction (united /imu -> CSV + color-frame match)
# ===========================================================================
# color frames at 10.0/10.1/10.2 s (index 0/1/2); samples placed to land -1,0,1,2
COLOR_TIMES = [10.0, 10.1, 10.2]
IMU_SAMPLES = [
    (9_950_000_000,  (0.1, 0.2, 0.3), (1.0, 2.0, 3.0)),      # before frame 0 -> -1
    (10_050_000_000, (0.4, 0.5, 0.6), (4.0, 5.0, 6.0)),      # [10.0,10.1) -> 0
    (10_150_000_000, (0.7, 0.8, 0.9), (7.0, 8.0, 9.0)),      # [10.1,10.2) -> 1
    (10_250_000_000, (1.0, 1.1, 1.2), (10.0, 11.0, 12.0)),   # >= 10.2 -> 2 (preceding = last)
]
IMU_COLS = ["index", "ros_time_s", "color_frame_index", "wx", "wy", "wz", "ax", "ay", "az"]


def test_imu_samples_written_with_preceding_match(tmp_path):
    bag = build_bag(tmp_path / "bag", imu_samples=IMU_SAMPLES)
    out = tmp_path / "out"
    write_stub_metadata(out, color_times=COLOR_TIMES)
    summary = imu.main(bag=bag, out_dir=out, camera="ego")

    assert summary["imu_samples"] == 4
    df = _read_imu_csv(out)
    # exact column set/order — proves orientation + covariances are dropped
    assert list(df.columns) == IMU_COLS
    assert df["color_frame_index"].tolist() == [-1, 0, 1, 2]          # preceding rule
    assert df["index"].tolist() == [0, 1, 2, 3]
    assert df["ros_time_s"].tolist() == pytest.approx([9.95, 10.05, 10.15, 10.25])
    assert df["wx"].tolist() == pytest.approx([0.1, 0.4, 0.7, 1.0])   # gyro kept verbatim
    assert df["az"].tolist() == pytest.approx([3.0, 6.0, 9.0, 12.0])  # accel kept verbatim


def test_imu_stream_entry_metadata(tmp_path):
    bag = build_bag(tmp_path / "bag", imu_samples=IMU_SAMPLES)
    out = tmp_path / "out"
    write_stub_metadata(out, color_times=COLOR_TIMES)
    imu.main(bag=bag, out_dir=out, camera="ego")

    s = _imu_stream(_load(out))
    assert s is not None
    assert s["camera"] == "cam_ego" and s["kind"] == "imu"
    assert s["file"] == "imu/cam_ego_imu.csv"
    assert s["frame_id"] == "camera_imu_optical_frame"
    assert s["compose_leg"] == "extrinsics/depth_to_gyro"
    assert s["matched_color_stream"] == "cam_ego"
    assert s["color_frame_match_rule"] == "preceding"
    assert s["num_samples"] == 4
    assert s["n_unmatched_leading"] == 1                              # the one pre-frame-0 sample
    assert s["units"] == {"angular_velocity": "rad/s", "linear_acceleration": "m/s^2"}


def test_imu_stream_idempotent_on_rerun(tmp_path):
    bag = build_bag(tmp_path / "bag", imu_samples=IMU_SAMPLES)
    out = tmp_path / "out"
    write_stub_metadata(out, color_times=COLOR_TIMES)
    imu.main(bag=bag, out_dir=out, camera="ego")
    imu.main(bag=bag, out_dir=out, camera="ego")                      # re-run

    streams = _load(out)["steps"]["streams"]
    assert sum(1 for s in streams if s.get("kind") == "imu") == 1     # replaced, not duplicated
    assert len(_read_imu_csv(out)) == 4


def test_imu_without_color_stream_indexes_minus_one(tmp_path):
    # metadata present but no cam_ego color stream -> samples still written, all unmatched.
    bag = build_bag(tmp_path / "bag", imu_samples=IMU_SAMPLES)
    out = tmp_path / "out"
    write_stub_metadata(out, seed_extrinsics=False)                   # no color_times
    imu.main(bag=bag, out_dir=out, camera="ego")

    df = _read_imu_csv(out)
    assert df["color_frame_index"].tolist() == [-1, -1, -1, -1]
    assert _imu_stream(_load(out))["n_unmatched_leading"] == 4


def test_no_imu_topic_leaves_samples_untouched(tmp_path):
    # a bag with extrinsics but no /imu topic: extrinsics still land, no imu CSV/stream.
    bag = build_bag(tmp_path / "bag")                                 # imu_samples=None
    out = tmp_path / "out"
    write_stub_metadata(out, color_times=COLOR_TIMES)
    summary = imu.main(bag=bag, out_dir=out, camera="ego")

    assert summary["imu_samples"] == 0
    assert summary["extrinsics_written"] == 2
    assert not (out / "imu" / "cam_ego_imu.csv").exists()
    assert _imu_stream(_load(out)) is None


def test_imu_samples_noop_when_metadata_absent(tmp_path):
    # no metadata.json: sample extraction is skipped (needs color frames to match against).
    bag = build_bag(tmp_path / "bag", imu_samples=IMU_SAMPLES)
    out = tmp_path / "out"
    summary = imu.main(bag=bag, out_dir=out, camera="ego")

    assert summary["imu_samples"] == 0
    assert not (out / "imu" / "cam_ego_imu.csv").exists()
    assert not (out / "metadata.json").exists()


# ===========================================================================
# MISSING / EMPTY /imu  ->  imu_presence_err (a data-plane presence failure, like missing
# colour frames; a distinct token from validate_imu's imu_data QUALITY token so validation
# can never strip it). Missing extrinsics -> imu_info (the info plane).
# ===========================================================================
def _imu_missing_entries(meta):
    return [m for m in meta.get("steps", {}).get("missing_stream_error", []) if " imu:" in m]


def test_missing_imu_topic_records_missing_stream(tmp_path):
    bag = build_bag(tmp_path / "bag")                       # imu_samples=None -> no /imu topic
    out = tmp_path / "out"
    write_stub_metadata(out, seed_extrinsics=False)
    summary = imu.main(bag=bag, out_dir=out, camera="ego")

    meta = _load(out)
    assert summary["imu_missing"] is True
    assert "imu_presence_err" in meta["termination"]["reason"]
    assert "imu_error" not in meta["termination"]["reason"]   # no validation-style token
    assert meta["termination"]["is_successful"] is False
    assert _imu_missing_entries(meta) and "not found" in _imu_missing_entries(meta)[0]


def test_empty_imu_topic_records_missing_stream(tmp_path):
    bag = build_bag(tmp_path / "bag", imu_samples=[])       # /imu present but 0 messages
    out = tmp_path / "out"
    write_stub_metadata(out, color_times=COLOR_TIMES)
    summary = imu.main(bag=bag, out_dir=out, camera="ego")

    meta = _load(out)
    assert summary["imu_missing"] is True and summary["imu_samples"] == 0
    assert "imu_presence_err" in meta["termination"]["reason"]
    assert "streamed no messages" in _imu_missing_entries(meta)[0]
    assert not (out / "imu" / "cam_ego_imu.csv").exists()   # no CSV for an empty stream


def test_present_imu_records_no_missing_stream(tmp_path):
    bag = build_bag(tmp_path / "bag", imu_samples=IMU_SAMPLES)
    out = tmp_path / "out"
    write_stub_metadata(out, color_times=COLOR_TIMES)
    summary = imu.main(bag=bag, out_dir=out, camera="ego")

    meta = _load(out)
    assert summary["imu_missing"] is False
    assert "imu_presence_err" not in (meta["termination"].get("reason") or [])
    assert _imu_missing_entries(meta) == []


def test_missing_imu_is_idempotent(tmp_path):
    bag = build_bag(tmp_path / "bag")                       # no /imu
    out = tmp_path / "out"
    write_stub_metadata(out, seed_extrinsics=False)
    imu.main(bag=bag, out_dir=out, camera="ego")
    imu.main(bag=bag, out_dir=out, camera="ego")            # re-run must not duplicate

    meta = _load(out)
    assert meta["termination"]["reason"].count("imu_presence_err") == 1
    assert len(_imu_missing_entries(meta)) == 1


def test_missing_imu_preserves_other_termination_reasons(tmp_path):
    # imu_presence_err is append-only: it must compose with a pre-existing reason, not clobber it.
    bag = build_bag(tmp_path / "bag")                       # no /imu
    out = tmp_path / "out"
    write_stub_metadata(out, seed_extrinsics=False)
    m = _load(out)
    m["termination"] = {"is_successful": False, "reason": ["timestamps"]}
    (out / "metadata.json").write_text(json.dumps(m), encoding="utf-8")

    imu.main(bag=bag, out_dir=out, camera="ego")
    reasons = _load(out)["termination"]["reason"]
    assert "timestamps" in reasons and "imu_presence_err" in reasons


def test_missing_imu_noop_when_metadata_absent(tmp_path):
    bag = build_bag(tmp_path / "bag")                       # no /imu, no metadata.json
    out = tmp_path / "out"
    summary = imu.main(bag=bag, out_dir=out, camera="ego")
    assert summary["imu_missing"] is True                   # flagged in the summary
    assert not (out / "metadata.json").exists()             # but nothing recorded (no file to touch)
