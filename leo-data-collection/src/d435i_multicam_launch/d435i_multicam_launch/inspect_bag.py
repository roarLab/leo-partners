"""Post-hoc sanity check for a recorded bag: does every stream match the rig spec?

Why this exists
---------------
Some faults are invisible after the fact. The worst was C922 resolution: the
launch asked for 1280x720, gscam silently recorded 1920x1080, and nothing
surfaced it -- the JPEG is an opaque blob and the camera_info for these cameras
is published as 0x0. Weeks of sessions were wrong before anyone decoded a frame.

So this reads what actually landed in the bag and checks it against what the rig
is configured to record:

  * resolution -- the colour streams (the four C922 + the ego colour) against the
    colour size, and the ego depth stream against the depth size;
  * framerate  -- every colour and depth stream against the expected fps,
    measured as (message count / bag duration);
  * IMU        -- present, non-empty, and averaging its expected rate.

The expected values below MIRROR the launch config (c922.launch.py and
four_d435i.launch.py). They are the rig's contract: keep them in sync with the
launch files, or override any of them per run with the flags. A no-argument run
checks against the shipped defaults.

Usage
-----
    ros2 run d435i_multicam_launch inspect_bag <bag-dir>
    ros2 run d435i_multicam_launch inspect_bag <bag-dir> --depth-width 640 --depth-height 480
    ros2 run d435i_multicam_launch inspect_bag <bag-dir> --fps 15 --imu-fps 100

A non-zero exit means at least one stream does not match.
"""
import argparse
import sys

from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, Info, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message

# --- The rig contract. A mirror of the launch config; override with flags. ---
# Colour: the four C922 (c922.launch.py DEFAULTS) and the ego colour
# (four_d435i.launch.py rgb_camera.color_profile) both run at this size.
COLOR_W, COLOR_H = 1280, 720
# Depth: the ego depth stream (four_d435i.launch.py depth_module.depth_profile).
DEPTH_W, DEPTH_H = 848, 480
# Framerate every colour and depth stream is asked to hold.
IMAGE_FPS = 30
# The ego IMU rate (gyro/accel united at 200 Hz).
IMU_FPS = 200
# A stream passes the framerate check if its average is at least this fraction
# of the expected rate -- slack for the start/stop edges of a short bag, but a
# dropped-frame session recorded at half rate still fails loudly.
FPS_FLOOR_FRAC = 0.90

COMPRESSED = "sensor_msgs/msg/CompressedImage"
RAW = "sensor_msgs/msg/Image"


def jpeg_dims(data):
    """Width, height from a JPEG's SOF marker -- the only record of the size.

    Walks the marker segments to the Start-Of-Frame (0xFFC0..0xCF, minus the
    non-SOF markers DHT/JPG/DAC) and reads the two big-endian shorts it carries.
    """
    i, n = 2, len(data)
    while i < n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height = (data[i + 5] << 8) | data[i + 6]
            width = (data[i + 7] << 8) | data[i + 8]
            return width, height
        i += 2 + ((data[i + 2] << 8) | data[i + 3])
    return None


def read_bag(bag):
    """Everything the checks need, in one pass plus the metadata.

    Returns (types, frames, counts, duration):
      * types    -- {topic: type_name} for EVERY recorded topic, even empty ones,
                    so a camera that produced zero frames is still visible.
      * frames   -- {topic: raw_bytes} of the first message on each topic that
                    produced one (absent for empty topics).
      * counts   -- {topic: message_count} from the metadata (no scan).
      * duration -- bag length in seconds, from the metadata.
    """
    reader = SequentialReader()
    reader.open(StorageOptions(uri=bag, storage_id="mcap"),
                ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}

    frames = {}
    while reader.has_next() and len(frames) < len(types):
        topic, data, _ = reader.read_next()
        if topic not in frames:
            frames[topic] = data

    meta = Info().read_metadata(bag, "mcap")
    counts = {t.topic_metadata.name: t.message_count
              for t in meta.topics_with_message_count}
    return types, frames, counts, meta.duration.total_seconds()


def image_dims(type_name, raw):
    """(width, height) for one image message, or None if it carries no size."""
    if type_name == COMPRESSED:
        msg = deserialize_message(raw, get_message(COMPRESSED))
        return jpeg_dims(bytes(msg.data))
    if type_name == RAW:
        msg = deserialize_message(raw, get_message(RAW))
        return msg.width, msg.height
    return None


def classify(topic, type_name):
    """Which check group a topic belongs to: 'color', 'depth', 'imu', or None.

    None means the topic is not checked (camera_info, extrinsics).
    """
    if topic.endswith("/imu"):
        return "imu"
    if type_name == COMPRESSED:
        return "color"                 # the C922 exo cameras publish JPEG
    if type_name == RAW and "depth" in topic:
        return "depth"
    if type_name == RAW and "color" in topic:
        return "color"                 # the ego colour
    return None


def parse_args(argv):
    p = argparse.ArgumentParser(prog="inspect_bag")
    p.add_argument("bag", help="bag directory (contains metadata.yaml)")
    p.add_argument("--color-width", type=int, default=COLOR_W)
    p.add_argument("--color-height", type=int, default=COLOR_H)
    p.add_argument("--depth-width", type=int, default=DEPTH_W)
    p.add_argument("--depth-height", type=int, default=DEPTH_H)
    p.add_argument("--fps", type=int, default=IMAGE_FPS,
                   help="expected fps for every colour and depth stream")
    p.add_argument("--imu-fps", type=int, default=IMU_FPS,
                   help="expected IMU rate in Hz")
    p.add_argument("--fps-floor-frac", type=float, default=FPS_FLOOR_FRAC,
                   help="a stream passes if its average rate is at least this "
                        "fraction of expected (default 0.90)")
    return p.parse_args(argv)


def main(args=None):
    o = parse_args(sys.argv[1:] if args is None else args)

    want_res = {"color": (o.color_width, o.color_height),
                "depth": (o.depth_width, o.depth_height)}
    types, frames, counts, duration = read_bag(o.bag)

    print(f"\n  {o.bag}")
    print(f"  checking against:  colour {o.color_width}x{o.color_height},  "
          f"depth {o.depth_width}x{o.depth_height},  "
          f"colour/depth {o.fps} fps,  imu {o.imu_fps} Hz  "
          f"(rate floor {int(o.fps_floor_frac * 100)}%)")

    if duration <= 0:
        print("\n  !! bag duration is zero -- cannot compute framerate\n",
              file=sys.stderr)
        sys.exit(1)

    fails = 0

    # --- resolution: colour streams vs colour size, depth vs depth size ---
    print("\n  RESOLUTION")
    for topic in sorted(types):
        group = classify(topic, types[topic])
        if group not in ("color", "depth"):
            continue
        want = want_res[group]
        raw = frames.get(topic)
        if raw is None:
            print(f"  FAIL {topic:44s} {'no frames':>11s}  "
                  f"(want {group} {want[0]}x{want[1]})")
            fails += 1
            continue
        got = image_dims(types[topic], raw)
        ok = got == want
        fails += not ok
        got_s = f"{got[0]}x{got[1]}" if got else "??"
        print(f"  {'OK  ' if ok else 'FAIL'} {topic:44s} {got_s:>11s}  "
              f"(want {group} {want[0]}x{want[1]})")

    # --- framerate: every colour and depth stream vs the one fps value ----
    print("\n  FRAMERATE  (average = message count / bag duration)")
    floor = o.fps * o.fps_floor_frac
    for topic in sorted(types):
        group = classify(topic, types[topic])
        if group not in ("color", "depth"):
            continue
        avg = counts.get(topic, 0) / duration
        ok = avg >= floor
        fails += not ok
        print(f"  {'OK  ' if ok else 'FAIL'} {topic:44s} {avg:7.1f} fps  "
              f"(want {o.fps}, min {floor:.1f})")

    # --- imu: present, non-empty, averaging its rate ----------------------
    print("\n  IMU")
    imu_topics = [t for t in sorted(types) if classify(t, types[t]) == "imu"]
    imu_floor = o.imu_fps * o.fps_floor_frac
    if not imu_topics:
        print("  FAIL no IMU topic recorded  "
              "(want an /ego/.../imu topic present and non-empty)")
        fails += 1
    for topic in imu_topics:
        n = counts.get(topic, 0)
        avg = n / duration
        ok = n > 0 and avg >= imu_floor
        fails += not ok
        print(f"  {'OK  ' if ok else 'FAIL'} {topic:44s} "
              f"{n:>7d} msgs {avg:7.1f} Hz  "
              f"(want non-empty, {o.imu_fps} Hz, min {imu_floor:.0f})")

    if fails:
        print(f"\n  {fails} check(s) FAILED\n")
        sys.exit(1)
    print("\n  all checks passed\n")


if __name__ == "__main__":
    main()
