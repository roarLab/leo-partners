"""Unit + integration test suite for rosbag_process_color_v3.py.

Run:  conda activate leo-data-collection
      python -m pytest data-processing-scripts-v3/test_rosbag_process_color_v3.py -q

Two layers:
  A) Pure helpers (no bag): sanitize_topic, compute_fps, convert_ros_to_iso,
     _to_jsonable.
  B) End-to-end via main(): a tiny synthetic ROS2 bag is written into a tmpdir
     with the exact BAG_TOPICS the script expects, main() is driven through the
     module-level constants, and the resulting metadata.json / videos / CSVs are
     asserted. Everything is self-contained; no real recordings or network.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rosbag_process_color_v3 as rp                     # noqa: E402

from rosbags.rosbag2 import Writer                        # noqa: E402
from rosbags.typesys import get_typestore, Stores         # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic bag builder
# ---------------------------------------------------------------------------
# A slightly larger frame (64x48) so cv2.VideoWriter("mp4v") reliably opens in a
# headless environment; tiny frames sometimes fail to initialise the encoder.
CW, CH = 64, 48


def _cinfo(ts, CI, ROI, hdr, w, h, k):
    fx, fy, cx, cy = k
    return CI(
        header=hdr,
        height=h,
        width=w,
        distortion_model="plumb_bob",
        d=np.zeros(5, np.float64),
        k=np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1], np.float64),
        r=np.eye(3).ravel().astype(np.float64),
        p=np.array([fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0], np.float64),
        binning_x=0,
        binning_y=0,
        roi=ROI(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False),
    )


def build_bag(path, *, n_frames=6, n_webcams=4, drop_c922=(), corrupt=None, ego_prefix="ego",
              extra_ego_prefixes=(), ego_ci="present"):
    """Write a rosbag2 folder holding the color-pipeline topics.

    n_webcams: how many c922 webcams to write (c922_1..N). Discovery is by suffix,
    so this is the ONLY knob that changes how many exo streams appear — no per-camera
    edit anywhere else. Each c922 also gets a (zeros) /c922_i/camera_info connection
    that must NOT be discovered as a color stream.
    drop_c922: tuple of c922 indices to omit entirely (no connection), to exercise
    the "extract whatever's present" path (a dropped webcam is simply absent).
    corrupt: optional (c922_index, frame_index) tuple. That single c922 frame is
    written with non-jpeg garbage bytes instead of a valid jpeg, so cv2.imdecode
    returns None and the decode-failure path is exercised.
    ego_prefix: namespace segment(s) in front of the ego device, e.g. "robotA/ego".
    The ego suffix must still match regardless of how deep it is nested.
    ego_ci: "present" (default) writes the ego camera_info topic + a message per frame;
    "empty" adds the topic connection but writes NO camera_info message (topic present,
    no decodable message); "absent" omits the connection entirely. The latter two
    exercise the missing-intrinsics presence failure.
    """
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)

    ts = get_typestore(Stores.ROS2_HUMBLE)
    Time = ts.types["builtin_interfaces/msg/Time"]
    Header = ts.types["std_msgs/msg/Header"]
    Image = ts.types["sensor_msgs/msg/Image"]
    CImage = ts.types["sensor_msgs/msg/CompressedImage"]
    CI = ts.types["sensor_msgs/msg/CameraInfo"]
    ROI = ts.types["sensor_msgs/msg/RegionOfInterest"]

    def hdr(t, frame):
        return Header(
            stamp=Time(sec=t // 1_000_000_000, nanosec=t % 1_000_000_000),
            frame_id=frame,
        )

    ego_k = (60.0, 60.0, 32.0, 24.0)
    c922_k = (55.0, 55.0, 32.0, 24.0)

    with Writer(path, version=9) as w:
        c_ego = w.add_connection(f"/{ego_prefix}/d435i_ego/color/image_raw", Image.__msgtype__, typestore=ts)
        c_ego_ci = (None if ego_ci == "absent" else
                    w.add_connection(f"/{ego_prefix}/d435i_ego/color/camera_info", CI.__msgtype__, typestore=ts))

        # extra topics that ALSO end with the ego suffix -> exercise the >1-match guard
        extra_ego = {p: w.add_connection(f"/{p}/d435i_ego/color/image_raw", Image.__msgtype__, typestore=ts)
                     for p in extra_ego_prefixes}

        c922_conns = {}
        for i in range(1, n_webcams + 1):
            if i in drop_c922:
                continue
            c_img = w.add_connection(f"/c922_{i}/image_raw/compressed", CImage.__msgtype__, typestore=ts)
            c_ci = w.add_connection(f"/c922_{i}/camera_info", CI.__msgtype__, typestore=ts)
            c922_conns[i] = (c_img, c_ci)

        for f in range(n_frames):
            t = int(f * 33_333_333)

            # ego RealSense raw color (rgb8)
            rgb = np.dstack([
                np.full((CH, CW), 40 + f, np.uint8),
                np.full((CH, CW), 90, np.uint8),
                np.full((CH, CW), 160, np.uint8),
            ])
            w.write(c_ego, t, ts.serialize_cdr(
                Image(header=hdr(t, "cam_ego"), height=CH, width=CW, encoding="rgb8",
                      is_bigendian=0, step=CW * 3, data=rgb.reshape(-1)), Image.__msgtype__))
            if ego_ci == "present":
                w.write(c_ego_ci, t, ts.serialize_cdr(
                    _cinfo(ts, CI, ROI, hdr(t, "cam_ego"), CW, CH, ego_k), CI.__msgtype__))
            for pref, c in extra_ego.items():
                w.write(c, t, ts.serialize_cdr(
                    Image(header=hdr(t, pref), height=CH, width=CW, encoding="rgb8",
                          is_bigendian=0, step=CW * 3, data=rgb.reshape(-1)), Image.__msgtype__))

            # c922 compressed color (jpeg)
            bgr = np.dstack([
                np.full((CH, CW), 160, np.uint8),
                np.full((CH, CW), 90, np.uint8),
                np.full((CH, CW), 40 + f, np.uint8),
            ])
            jpeg = cv2.imencode(".jpg", bgr)[1].tobytes()
            for i, (c_img, c_ci) in c922_conns.items():
                if corrupt is not None and corrupt == (i, f):
                    # deliberately NOT a valid jpeg -> cv2.imdecode returns None
                    payload = np.frombuffer(b"\xde\xad\xbe\xef not a jpeg", np.uint8)
                else:
                    payload = np.frombuffer(jpeg, np.uint8)
                w.write(c_img, t, ts.serialize_cdr(
                    CImage(header=hdr(t, f"c922_{i}"), format="jpeg",
                           data=payload), CImage.__msgtype__))
                w.write(c_ci, t, ts.serialize_cdr(
                    _cinfo(ts, CI, ROI, hdr(t, f"c922_{i}"), CW, CH, c922_k), CI.__msgtype__))

    return path


# ---------------------------------------------------------------------------
# Fixtures: restore every module global we mutate between tests
# ---------------------------------------------------------------------------
_SAVE = ["BAG_PATH", "OUT_DIR", "VIDEO_CODEC", "INSPECT_ONLY", "STRIDE", "FORCE_FPS"]


@pytest.fixture(autouse=True)
def reset_config():
    saved = {k: getattr(rp, k) for k in _SAVE}
    yield
    for k, v in saved.items():
        setattr(rp, k, v)


def drive_main(bag_dir, out_dir, cameras=None):
    rp.BAG_PATH = str(bag_dir)
    rp.OUT_DIR = str(out_dir)
    rp.VIDEO_CODEC = "mp4v"   # keep tests hermetic: force the OpenCV writer, no ffmpeg dependency
    rp.INSPECT_ONLY = False
    rp.STRIDE = 1
    rp.FORCE_FPS = 0.0
    return rp.main(cameras=cameras)


def cams_with_exo_ids(ids):
    """DEFAULT_CAMERAS with the exo `ids` set overridden, so a test can make the webcams
    present deliberately match (clean) or mismatch (missing/extra) the declared ids."""
    import copy
    c = copy.deepcopy(rp.DEFAULT_CAMERAS)
    c["exo"]["ids"] = list(ids)
    return c


def _load_meta(out_dir):
    with open(Path(out_dir) / "metadata.json", encoding="utf-8") as f:
        return json.load(f)


# ===========================================================================
# A. Pure helpers
# ===========================================================================
def test_sanitize_topic_replaces_non_alnum():
    assert rp.sanitize_topic("/c922_1/image_raw/compressed") == "c922_1_image_raw_compressed"
    assert rp.sanitize_topic("/ego/d435i_ego/color/image_raw") == "ego_d435i_ego_color_image_raw"
    # leading/trailing separators stripped, run of symbols collapsed
    assert rp.sanitize_topic("///a..b--c///") == "a_b_c"


def test_sanitize_topic_truncates_to_120():
    long = "/" + "a" * 200
    assert len(rp.sanitize_topic(long)) == 120


def test_compute_fps_few_samples_returns_default():
    assert rp.compute_fps(np.array([], dtype=float)) == 20.0
    assert rp.compute_fps(np.array([1.0])) == 20.0
    assert rp.compute_fps(np.array([1.0, 2.0])) == 20.0


def test_compute_fps_regular_spacing():
    assert abs(rp.compute_fps(np.arange(10) / 30.0) - 30.0) < 1e-6


def test_compute_fps_all_equal_timestamps_returns_default():
    # dt filtered to > 0 leaves nothing -> fall back to 20.0
    assert rp.compute_fps(np.array([5.0, 5.0, 5.0, 5.0])) == 20.0


def test_compute_fps_clips_to_bounds():
    # 1 s spacing -> 1 fps, clipped to the lower bound of 1.0
    assert rp.compute_fps(np.array([0.0, 1.0, 2.0, 3.0])) == 1.0


def test_convert_ros_to_iso_epoch_zero():
    out = rp.convert_ros_to_iso(0.0)
    assert out == "1970-01-01T00:00:00.000Z"
    assert out.endswith("Z")


def test_convert_ros_to_iso_millisecond_precision():
    out = rp.convert_ros_to_iso(1_600_000_000.123456)
    # ISO8601 with millisecond precision and a trailing Z
    assert out.endswith("Z")
    assert out.startswith("2020-09-13T")
    # exactly three fractional digits before the Z
    frac = out.split(".")[1][:-1]
    assert len(frac) == 3


def test_to_jsonable_numpy_and_bytes():
    assert rp._to_jsonable(np.array([1, 2, 3])) == [1, 2, 3]
    assert rp._to_jsonable(np.float32(3.5)) == 3.5
    assert isinstance(rp._to_jsonable(np.float32(3.5)), float)
    assert rp._to_jsonable(np.int64(7)) == 7
    assert isinstance(rp._to_jsonable(np.int64(7)), int)
    assert rp._to_jsonable(b"abcd") == {"__bytes_len__": 4}
    assert rp._to_jsonable((1, 2)) == [1, 2]
    assert rp._to_jsonable("plain") == "plain"


def test_to_jsonable_is_json_serializable():
    val = {"a": np.array([1.0, 2.0]), "b": np.float64(1.0), "c": (np.int32(3),)}
    # every leaf must survive json.dumps after conversion
    converted = {k: rp._to_jsonable(v) for k, v in val.items()}
    json.dumps(converted)


# ===========================================================================
# B. Integration via main()
# ===========================================================================
def test_main_happy_path_structure(tmp_path):
    bag = build_bag(tmp_path / "bag", n_frames=6)
    out = tmp_path / "out"
    meta = drive_main(bag, out)

    # metadata.json written and re-loadable
    assert (out / "metadata.json").exists()
    loaded = _load_meta(out)
    assert loaded == meta

    # top-level structure
    assert "metadata" in loaded
    assert loaded["metadata"]["dataset_name"] == rp.DATASET_NAME
    # top-level `fps` was removed (fps is per-stream; see steps.streams[].fps below):
    assert "fps" not in loaded["metadata"]
    assert all(s["fps"] > 0 for s in loaded["steps"]["streams"] if s.get("kind") == "color")

    # camera_intrinsics: ONLY cam_ego (RealSense). The c922 webcams have no
    # meaningful intrinsics, so they are excluded (their camera_info is ignored).
    intr = loaded["camera_intrinsics"]
    assert [c["camera"] for c in intr] == ["cam_ego"]
    color = intr[0]["color"]
    assert len(color["K"]) == 9
    assert color["width"] == CW and color["height"] == CH
    # the webcams are still present as color streams, just without intrinsics,
    # and labelled exo_cam<N> (discovered by suffix, not a hardcoded map)
    stream_cams = {s["camera"] for s in loaded["steps"]["streams"]}
    assert {"exo_cam1", "exo_cam2", "exo_cam3", "exo_cam4"} <= stream_cams

    # steps.streams: 5 (ego + 4 c922)
    streams = loaded["steps"]["streams"]
    assert len(streams) >= 5
    assert all(s["found"] for s in streams)
    assert all(s["num_frames"] == 6 for s in streams)

    # count matches (4 webcams == default count) -> no deviation, clean success
    assert loaded["steps"]["missing_stream_error"] == []
    assert loaded["steps"]["extra_stream_error"] == []
    assert loaded["termination"]["is_successful"] is True
    assert loaded["termination"]["reason"] == []


def test_main_writes_videos_and_timestamps(tmp_path):
    bag = build_bag(tmp_path / "bag", n_frames=6)
    out = tmp_path / "out"
    drive_main(bag, out)

    for name in ["cam_ego", "exo_cam1", "exo_cam2", "exo_cam3", "exo_cam4"]:
        vid = out / "videos" / f"{name}.mp4"
        csv = out / "timestamps" / f"{name}.csv"
        assert vid.exists() and vid.stat().st_size > 0, f"missing video {name}"
        assert csv.exists(), f"missing timestamps {name}"
        lines = csv.read_text().splitlines()
        assert lines[0] == "index,ros_time_s"
        assert len(lines) == 1 + 6            # header + 6 frames


def test_missing_webcam_flagged_but_others_extracted(tmp_path):
    # Default ids [1,2,3,4], but c922_3 dropped -> devices 1,2,4 present. Discovery extracts
    # exactly what's there (no phantom exo_cam3), and the missing id 3 is flagged as a
    # missing_stream (owned by the extraction script) via the declared-vs-found set diff.
    bag = build_bag(tmp_path / "bag", n_frames=6, n_webcams=4, drop_c922=(3,))
    out = tmp_path / "out"
    meta = drive_main(bag, out)

    cams = {s["camera"] for s in meta["steps"]["streams"]}
    assert cams == {"cam_ego", "exo_cam1", "exo_cam2", "exo_cam4"}   # exactly what's present
    assert "exo_cam3" not in cams                                    # no placeholder record

    for name in ("exo_cam1", "exo_cam2", "exo_cam4"):
        s = next(x for x in meta["steps"]["streams"] if x["camera"] == name)
        assert s["found"] is True and s["num_frames"] == 6

    # id deviation: declared 3 not found -> data-plane miss -> color_presence_err, fails termination
    assert len(meta["steps"]["missing_stream_error"]) == 1
    assert "missing [3]" in meta["steps"]["missing_stream_error"][0]
    assert meta["steps"]["extra_stream_error"] == []
    assert meta["termination"]["reason"] == ["color_presence_err"]
    assert meta["termination"]["is_successful"] is False


def test_extra_webcam_extracted_and_flagged(tmp_path):
    # Default ids [1,2,3,4] but 5 c922s present -> ALL extracted (never capped), and the
    # undeclared id 5 is flagged as a data-plane extra -> color_presence_err (non-crash, trust bag).
    bag = build_bag(tmp_path / "bag", n_frames=6, n_webcams=5)
    out = tmp_path / "out"
    meta = drive_main(bag, out)   # default ids = [1,2,3,4]

    cams = {s["camera"] for s in meta["steps"]["streams"]}
    assert {"exo_cam1", "exo_cam2", "exo_cam3", "exo_cam4", "exo_cam5"} <= cams
    assert (out / "videos" / "exo_cam5.mp4").exists()               # surplus still extracted

    assert len(meta["steps"]["extra_stream_error"]) == 1
    assert "undeclared [5]" in meta["steps"]["extra_stream_error"][0]
    assert meta["steps"]["missing_stream_error"] == []
    assert meta["termination"]["reason"] == ["color_presence_err"]
    assert meta["termination"]["is_successful"] is False


def test_skip_specific_webcam_via_ids(tmp_path):
    # Intentionally run WITHOUT cam3: declare ids [1,2,4] and record only those. The set
    # diff is empty both ways -> clean, no missing/extra, labels keep their physical ids.
    bag = build_bag(tmp_path / "bag", n_frames=6, n_webcams=4, drop_c922=(3,))
    out = tmp_path / "out"
    meta = drive_main(bag, out, cameras=cams_with_exo_ids([1, 2, 4]))

    cams = {s["camera"] for s in meta["steps"]["streams"]}
    assert cams == {"cam_ego", "exo_cam1", "exo_cam2", "exo_cam4"}   # gap preserved, no rename
    assert meta["steps"]["missing_stream_error"] == []              # 3 was never expected
    assert meta["steps"]["extra_stream_error"] == []
    assert meta["termination"]["is_successful"] is True


def test_ids_detect_missing_and_extra_simultaneously(tmp_path):
    # The set diff catches BOTH at once — what a bare count would miss (found==declared==4
    # here would look clean). Declare [1,2,3,4]; record 1,2,4,5 (drop 3, add 5).
    bag = build_bag(tmp_path / "bag", n_frames=6, n_webcams=5, drop_c922=(3,))
    out = tmp_path / "out"
    meta = drive_main(bag, out)   # default ids [1,2,3,4]; found {1,2,4,5}

    assert len(meta["steps"]["missing_stream_error"]) == 1
    assert "missing [3]" in meta["steps"]["missing_stream_error"][0]
    assert len(meta["steps"]["extra_stream_error"]) == 1
    assert "undeclared [5]" in meta["steps"]["extra_stream_error"][0]
    assert meta["termination"]["reason"] == ["color_presence_err"]
    assert meta["termination"]["is_successful"] is False


def test_ids_none_skips_deviation_check(tmp_path):
    # ids absent -> no expectation, so any webcam count is accepted (extract-all, no flag).
    bag = build_bag(tmp_path / "bag", n_frames=6, n_webcams=4)
    out = tmp_path / "out"
    cams = cams_with_exo_ids([1, 2, 3, 4])
    del cams["exo"]["ids"]
    meta = drive_main(bag, out, cameras=cams)

    assert {"exo_cam1", "exo_cam2", "exo_cam3", "exo_cam4"} <= {s["camera"] for s in meta["steps"]["streams"]}
    assert meta["steps"]["missing_stream_error"] == []
    assert meta["steps"]["extra_stream_error"] == []
    assert meta["termination"]["is_successful"] is True


def _cams(**patch):
    """DEFAULT_CAMERAS deep-copied with per-group field patches, e.g.
    _cams(ego={"present": False}). Used to drive the color-group opt-out."""
    import copy
    c = copy.deepcopy(rp.DEFAULT_CAMERAS)
    for grp, fields in patch.items():
        c[grp].update(fields)
    return c


def test_ego_present_false_skips_ego_but_writes_spine(tmp_path):
    # present:False on ego -> no discovery/video/intrinsics/flag, but metadata.json (the
    # spine depth/imu append to) is still written. Absent-by-declaration is NOT a miss.
    bag = build_bag(tmp_path / "bag", n_frames=6, n_webcams=4)
    out = tmp_path / "out"
    meta = drive_main(bag, out, cameras=_cams(ego={"present": False}))

    cams = {s["camera"] for s in meta["steps"]["streams"]}
    assert "cam_ego" not in cams                              # ego skipped
    assert cams == {"exo_cam1", "exo_cam2", "exo_cam3", "exo_cam4"}
    assert not (out / "videos" / "cam_ego.mp4").exists()      # no ego video
    assert meta["camera_intrinsics"] == []                    # no ego intrinsics
    assert meta["steps"]["missing_stream_error"] == []        # NOT a missing_stream
    assert meta["termination"]["is_successful"] is True
    assert (out / "metadata.json").exists()                   # spine written


def test_exo_present_false_skips_exo_keeps_ego(tmp_path):
    # present:False on exo -> no exo videos and the count check is skipped, so 4 c922s in
    # the bag are silently ignored (no missing/extra). ego still extracts normally.
    bag = build_bag(tmp_path / "bag", n_frames=6, n_webcams=4)
    out = tmp_path / "out"
    meta = drive_main(bag, out, cameras=_cams(exo={"present": False}))

    cams = {s["camera"] for s in meta["steps"]["streams"]}
    assert cams == {"cam_ego"}                                # only ego
    assert not (out / "videos" / "exo_cam1.mp4").exists()
    assert meta["steps"]["missing_stream_error"] == []        # count check skipped
    assert meta["steps"]["extra_stream_error"] == []
    assert meta["termination"]["is_successful"] is True


def test_both_color_groups_off_writes_empty_spine(tmp_path):
    bag = build_bag(tmp_path / "bag", n_frames=6, n_webcams=4)
    out = tmp_path / "out"
    meta = drive_main(bag, out, cameras=_cams(ego={"present": False}, exo={"present": False}))

    assert meta["steps"]["streams"] == []                     # zero color streams
    assert meta["camera_intrinsics"] == []
    assert meta["steps"]["missing_stream_error"] == []
    assert meta["steps"]["extra_stream_error"] == []
    assert (out / "metadata.json").exists()                   # spine for depth/imu/sanity


def test_present_absent_key_defaults_to_present(tmp_path):
    # back-compat: a group with NO `present` key behaves exactly as before (present).
    bag = build_bag(tmp_path / "bag", n_frames=6, n_webcams=4)
    out = tmp_path / "out"
    cams = _cams()
    del cams["ego"]["present"]
    del cams["exo"]["present"]
    meta = drive_main(bag, out, cameras=cams)

    cams_out = {s["camera"] for s in meta["steps"]["streams"]}
    assert "cam_ego" in cams_out
    assert {"exo_cam1", "exo_cam2", "exo_cam3", "exo_cam4"} <= cams_out
    assert meta["steps"]["missing_stream_error"] == []        # clean, exactly as before


def test_corrupt_compressed_frame_counted_and_flagged(tmp_path):
    # c922_2 (exo_cam2) gets ONE non-jpeg garbage frame (index 3) among 6 valid ones:
    # cv2.imdecode returns None -> _decode_compressed_frame raises RuntimeError ->
    # export_video_stream catches it, bumps n_failed, and skips that frame.
    bag = build_bag(tmp_path / "bag", n_frames=6, corrupt=(2, 3))
    out = tmp_path / "out"
    meta = drive_main(bag, out)

    cam2 = next(s for s in meta["steps"]["streams"] if s["camera"] == "exo_cam2")
    assert cam2["found"] is True
    assert cam2["decode_failures"] == 1        # the garbage frame was caught + counted
    assert cam2["num_frames"] == 5             # 6 written, corrupt one skipped not counted

    # the other c922 streams are untouched: all 6 frames, no failures
    for name in ("exo_cam1", "exo_cam3"):
        s = next(x for x in meta["steps"]["streams"] if x["camera"] == name)
        assert s["decode_failures"] == 0
        assert s["num_frames"] == 6

    # decode failures are NOT a termination concern of the extraction script — that's
    # validate_color's rosbag_corruption token. Extraction only owns missing/extra,
    # and the count is fine here (4 webcams == count), so its termination is clean.
    assert meta["steps"]["missing_stream_error"] == []
    assert meta["steps"]["extra_stream_error"] == []
    assert meta["termination"]["is_successful"] is True


# ===========================================================================
# C. Suffix discovery (the behaviour this refactor is about)
# ===========================================================================
def _open(bag):
    return rp.AnyReader([Path(bag)], default_typestore=rp.typestore)


def test_label_for_singleton_and_derived():
    ego = {"label": "cam_ego", "singleton": True}
    exo = {"label": "exo_cam"}
    # singleton uses the label verbatim, regardless of topic
    assert rp.label_for("/ego/d435i_ego/color/image_raw", ego) == "cam_ego"
    assert rp.label_for("/anything/at/all", ego) == "cam_ego"
    # exo derives the trailing number of the device segment
    assert rp.label_for("/c922_4/image_raw/compressed", exo) == "exo_cam4"
    assert rp.label_for("/c922_12/image_raw/compressed", exo) == "exo_cam12"
    # a device segment with no trailing number -> sanitized fallback
    assert rp.label_for("/leftcam/image_raw/compressed", exo) == "exo_cam_leftcam"


def test_discover_topics_matches_by_suffix_ignores_camera_info(tmp_path):
    bag = build_bag(tmp_path / "bag", n_frames=3, n_webcams=4)
    with _open(bag) as reader:
        exo = rp.discover_topics(reader, "image_raw/compressed")
        assert exo == [f"/c922_{i}/image_raw/compressed" for i in (1, 2, 3, 4)]
        ego = rp.discover_topics(reader, "d435i_ego/color/image_raw")
        assert ego == ["/ego/d435i_ego/color/image_raw"]
        # the c922 camera_info connections are NOT color streams
        assert all("camera_info" not in t for t in exo + ego)


def test_discovers_arbitrary_number_of_webcams(tmp_path):
    # FIVE webcams, zero code change: c922_1..5 -> exo_cam1..5, each with video+csv.
    # This is the original bug (cam4 never extracted) proven fixed, and then some.
    # Declare ids 1..5 so this stays a pure discovery test (no missing/extra noise).
    bag = build_bag(tmp_path / "bag", n_frames=6, n_webcams=5)
    out = tmp_path / "out"
    meta = drive_main(bag, out, cameras=cams_with_exo_ids([1, 2, 3, 4, 5]))

    cams = {s["camera"] for s in meta["steps"]["streams"]}
    assert {"exo_cam1", "exo_cam2", "exo_cam3", "exo_cam4", "exo_cam5"} <= cams
    for name in ("exo_cam4", "exo_cam5"):
        assert (out / "videos" / f"{name}.mp4").exists()
        assert (out / "timestamps" / f"{name}.csv").exists()
        s = next(x for x in meta["steps"]["streams"] if x["camera"] == name)
        assert s["found"] and s["num_frames"] == 6
    assert meta["termination"]["is_successful"] is True   # count matches -> clean


def test_suffix_match_namespaced_ego(tmp_path):
    # ego device nested under an extra namespace -> still matched by suffix, still
    # labelled cam_ego, still the only camera carrying intrinsics.
    bag = build_bag(tmp_path / "bag", n_frames=6, n_webcams=2, ego_prefix="robotA/ego")
    out = tmp_path / "out"
    meta = drive_main(bag, out, cameras=cams_with_exo_ids([1, 2]))   # ids 1,2 -> clean

    assert [c["camera"] for c in meta["camera_intrinsics"]] == ["cam_ego"]
    ego = next(s for s in meta["steps"]["streams"] if s["camera"] == "cam_ego")
    assert ego["found"] and ego["num_frames"] == 6
    assert ego["topic"] == "/robotA/ego/d435i_ego/color/image_raw"


def test_duplicate_ego_extracted_as_extra(tmp_path, capsys):
    # Two topics end with the ego suffix -> BOTH extracted (no crash, no data loss):
    # the first (sorted) stays cam_ego (intrinsics + depth anchor); the second gets a
    # distinct cam_ego_2 label so it neither overwrites cam_ego.mp4 nor is dropped, and
    # it is flagged as a data-plane extra -> color_presence_err. 4 webcams keeps the exo count
    # clean, isolating the ego duplicate as the only deviation.
    bag = build_bag(tmp_path / "bag", n_frames=6, n_webcams=4, extra_ego_prefixes=("ego2",))
    out = tmp_path / "out"
    meta = drive_main(bag, out)                       # must not raise

    cams = [s["camera"] for s in meta["steps"]["streams"]]
    assert cams.count("cam_ego") == 1                 # exactly one canonical cam_ego
    assert "cam_ego_2" in cams                        # the duplicate, extracted distinctly
    assert (out / "videos" / "cam_ego_2.mp4").exists()
    primary = next(s for s in meta["steps"]["streams"] if s["camera"] == "cam_ego")
    assert primary["topic"] == "/ego/d435i_ego/color/image_raw"          # first sorted = primary
    # only cam_ego carries intrinsics (the duplicate does not)
    assert [c["camera"] for c in meta["camera_intrinsics"]] == ["cam_ego"]

    assert len(meta["steps"]["extra_stream_error"]) == 1
    assert meta["steps"]["missing_stream_error"] == []
    assert meta["termination"]["reason"] == ["color_presence_err"]
    assert "UNEXPECTED EXTRA EGO" in capsys.readouterr().out             # loud warning


def test_missing_ego_camera_info_topic_flags_color_info(tmp_path):
    # Ego color frames extract fine, but the camera_info TOPIC is absent -> no intrinsics.
    # This is an info-plane presence failure: it lands in missing_stream_error (like any
    # miss) but under the color_info reason, NOT color_presence_err (the frames are present).
    bag = build_bag(tmp_path / "bag", n_frames=6, n_webcams=4, ego_ci="absent")
    out = tmp_path / "out"
    meta = drive_main(bag, out)                       # must not raise

    # frames still extracted; the ego is just missing its intrinsics block
    assert (out / "videos" / "cam_ego.mp4").exists()
    assert meta["camera_intrinsics"] == [{"camera": "cam_ego"}]   # no 'color' K attached

    errs = meta["steps"]["missing_stream_error"]
    assert any("camera_info" in e for e in errs)
    assert meta["steps"]["extra_stream_error"] == []
    assert meta["termination"]["reason"] == ["color_info"]
    assert meta["termination"]["is_successful"] is False


def test_empty_ego_camera_info_flags_color_info(tmp_path):
    # camera_info TOPIC exists but streams no decodable message -> same info-plane failure
    # as an absent topic. The sub-branch the slide calls out separately.
    bag = build_bag(tmp_path / "bag", n_frames=6, n_webcams=4, ego_ci="empty")
    out = tmp_path / "out"
    meta = drive_main(bag, out)

    errs = meta["steps"]["missing_stream_error"]
    assert any("camera_info" in e for e in errs)
    assert meta["termination"]["reason"] == ["color_info"]
    assert meta["termination"]["is_successful"] is False


# ===========================================================================
# D. Video encoding backend — VIDEO_CODEC selection (h264 pipe vs OpenCV fourcc)
# ===========================================================================
# The frame-reading / timestamp / CSV / metadata machinery in export_video_stream is
# codec-agnostic and already covered by sections B/C. What is NEW is the writer
# boundary: _open_video_writer choosing a backend, and _FfmpegWriter piping frames to
# ffmpeg+libx264. These cover exactly that surface — the routing hermetically (stub the
# writers, no process/file), plus one real end-to-end h264 encode.

def _make_fake_writer(kind, sink):
    """Stand-in for _FfmpegWriter/_Cv2Writer that opens nothing and records
    (kind, constructor-args), so a test can assert which backend was chosen and how."""
    class _Fake:
        def __init__(self, *args):
            sink.append((kind, args))

        def write(self, *_):
            pass

        def close(self):
            pass
    return _Fake


def _stub_writers(monkeypatch):
    """Replace both real writers with recorders; return the shared call log."""
    calls = []
    monkeypatch.setattr(rp, "_FfmpegWriter", _make_fake_writer("ffmpeg", calls))
    monkeypatch.setattr(rp, "_Cv2Writer", _make_fake_writer("cv2", calls))
    return calls


def test_open_writer_mp4v_uses_cv2(monkeypatch, tmp_path):
    # mp4v -> OpenCV writer with the mp4v fourcc, even when an ffmpeg IS available
    # (a non-h264 codec must never be routed through the ffmpeg pipe).
    calls = _stub_writers(monkeypatch)
    monkeypatch.setattr(rp, "_resolve_ffmpeg", lambda: "/fake/ffmpeg")
    monkeypatch.setattr(rp, "VIDEO_CODEC", "mp4v")
    rp._open_video_writer(tmp_path / "x.mp4", 30.0, 64, 48)
    assert len(calls) == 1
    kind, args = calls[0]
    assert kind == "cv2"
    assert args[-1] == "mp4v"                 # fourcc = _Cv2Writer's last constructor arg


def test_open_writer_arbitrary_fourcc_passthrough(monkeypatch, tmp_path):
    # An unrecognised value is used verbatim as the OpenCV fourcc — NOT silently coerced
    # to mp4v. Regression guard for the single-knob (collapsed CODEC) fix.
    calls = _stub_writers(monkeypatch)
    monkeypatch.setattr(rp, "_resolve_ffmpeg", lambda: "/fake/ffmpeg")
    monkeypatch.setattr(rp, "VIDEO_CODEC", "XVID")
    rp._open_video_writer(tmp_path / "x.mp4", 30.0, 64, 48)
    assert calls[0][0] == "cv2"
    assert calls[0][1][-1] == "XVID"


@pytest.mark.parametrize("alias", ["h264", "avc1", "libx264", "x264", "H264", "AVC1"])
def test_open_writer_h264_aliases_use_ffmpeg(monkeypatch, tmp_path, alias):
    # every H.264 alias (case-insensitive) routes to the ffmpeg pipe, threading the
    # resolved binary through as _FfmpegWriter's first arg.
    calls = _stub_writers(monkeypatch)
    monkeypatch.setattr(rp, "_resolve_ffmpeg", lambda: "/fake/ffmpeg")
    monkeypatch.setattr(rp, "VIDEO_CODEC", alias)
    rp._open_video_writer(tmp_path / "x.mp4", 30.0, 64, 48)
    assert calls[0][0] == "ffmpeg"
    assert calls[0][1][0] == "/fake/ffmpeg"


def test_open_writer_h264_falls_back_to_mp4v_without_ffmpeg(monkeypatch, tmp_path, capsys):
    # h264 requested but no ffmpeg anywhere -> degrade to OpenCV mp4v (never hard-fail),
    # with a visible warning. The portability safety net.
    calls = _stub_writers(monkeypatch)
    monkeypatch.setattr(rp, "_resolve_ffmpeg", lambda: None)
    monkeypatch.setattr(rp, "VIDEO_CODEC", "h264")
    rp._open_video_writer(tmp_path / "x.mp4", 30.0, 64, 48)
    assert calls[0][0] == "cv2"
    assert calls[0][1][-1] == "mp4v"
    assert "No ffmpeg found" in capsys.readouterr().out


def _count_readable_frames(path):
    """Frames OpenCV can decode back out of `path` (-1 if it won't even open)."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return -1
        n = 0
        while cap.read()[0]:
            n += 1
        return n
    finally:
        cap.release()


def test_h264_encode_produces_readable_video(tmp_path):
    # The one end-to-end encode: run main() through the REAL ffmpeg+libx264 pipe and
    # prove the files it emits are valid H.264 that decodes back to the right frame
    # count — for both the raw ego stream and a compressed exo stream. Skips cleanly
    # where imageio-ffmpeg (and thus a bundled ffmpeg) isn't installed.
    pytest.importorskip("imageio_ffmpeg")

    bag = build_bag(tmp_path / "bag", n_frames=6)
    out = tmp_path / "out"
    rp.BAG_PATH = str(bag)
    rp.OUT_DIR = str(out)
    rp.VIDEO_CODEC = "h264"          # exercise the ffmpeg pipe, not the mp4v stub
    rp.INSPECT_ONLY = False
    rp.STRIDE = 1
    rp.FORCE_FPS = 0.0
    rp.main(cameras=None)

    for name in ("cam_ego", "exo_cam1"):     # raw ego + compressed exo, both via the pipe
        vid = out / "videos" / f"{name}.mp4"
        assert vid.exists() and vid.stat().st_size > 0, f"missing/empty {name}"
        assert _count_readable_frames(vid) == 6, f"{name} did not decode back to 6 frames"
