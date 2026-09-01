import ctypes
import glob

from launch import LaunchDescription
from launch_ros.actions import Node


# On a USB 2 link the D435i silently drops 1280x720x30 and the only symptom is
# "Frames didn't arrive within 5 seconds", forever. A USB 2 port or a worn
# cable both do it. Linux reports the negotiated speed, so catch it up front.
D435I_USB_ID = ('8086', '0b3a')
MIN_USB_SPEED_MBPS = 5000        # 480 = USB2, 5000 = USB3 gen1, 10000 = gen2


# Publish-side transport whitelist. 'image_transport/raw' is the built-in raw
# transport (registered as 'raw_pub'; the whitelist strips '_pub').
# Cannot be [] -- launch_ros can't infer a type for an empty list and errors.
ONLY_RAW = ['image_transport/raw']


# Post-processing assumes the 16UC1 depth values are millimetres. That scale
# is the camera's Depth Units option: no ROS parameter exposes it, and
# json_file_path fails to apply it on this stack (set_xu error), so it is
# checked here instead -- readable only while the sensor is closed, i.e.
# exactly now. A wrong value: replug the camera (resets to the firmware
# default, 0.001) or set it in realsense-viewer -> Stereo Module (1000 value).
EXPECTED_DEPTH_UNITS = 0.001 # 1mm

# Stable across librealsense versions -- its enums are append-only.
RS2_OPTION_DEPTH_UNITS = 28
RS2_CAMERA_INFO_SERIAL_NUMBER = 1


def check_usb_link():
    """Abort the launch if a D435i came up on a USB 2 link."""
    speeds = []
    for dev in glob.glob('/sys/bus/usb/devices/*/'):
        try:
            with open(dev + 'idVendor') as f:
                vid = f.read().strip()
            with open(dev + 'idProduct') as f:
                pid = f.read().strip()
            if (vid, pid) != D435I_USB_ID:
                continue
            with open(dev + 'speed') as f:
                speeds.append(int(f.read().strip()))
        except (OSError, ValueError):
            continue

    if not speeds:
        raise RuntimeError(
            'D435i not detected on USB. Is the ego camera plugged in?')

    if min(speeds) < MIN_USB_SPEED_MBPS:
        raise RuntimeError(
            f'D435i negotiated a {min(speeds)} Mbps USB link, needs '
            f'{MIN_USB_SPEED_MBPS}+ (found: {speeds} Mbps).\n'
            '  Reconnect the ego camera: a blue/SS USB 3 port AND the cable '
            'that shipped with it.\n'
            '  A USB 2 cable in a USB 3 port still negotiates USB 2.\n'
            '  Verify with: rs-enumerate-devices | grep "Usb Type"')


def check_depth_units(serial: str):
    """Abort the launch unless the D435i reports EXPECTED_DEPTH_UNITS.

    ctypes straight into librealsense's C API -- the same calls
    rs-sensor-control makes behind its menus. rs2_get_option returns the
    CURRENT value (rs-enumerate-devices -o only ever shows the default).
    """
    rs = ctypes.CDLL('librealsense2.so')
    # ctypes assumes int returns; 64-bit pointers must be declared or they
    # get truncated and segfault.
    for fn in ('rs2_create_context', 'rs2_query_devices', 'rs2_create_device',
               'rs2_query_sensors', 'rs2_create_sensor'):
        getattr(rs, fn).restype = ctypes.c_void_p
    rs.rs2_get_error_message.restype = ctypes.c_char_p
    rs.rs2_get_device_info.restype = ctypes.c_char_p
    rs.rs2_get_option.restype = ctypes.c_float

    err = ctypes.c_void_p()

    def ok(result):
        if err.value:
            raise RuntimeError('librealsense: '
                               + rs.rs2_get_error_message(err).decode())
        return result

    # Ask the library its own version rather than pinning one here: the
    # context rejects mismatched callers, and setup.md only demands >= 2.57.
    api = ok(rs.rs2_get_api_version(ctypes.byref(err)))
    ctx = ok(rs.rs2_create_context(api, ctypes.byref(err)))
    devs = ok(rs.rs2_query_devices(ctypes.c_void_p(ctx), ctypes.byref(err)))
    n = ok(rs.rs2_get_device_count(ctypes.c_void_p(devs), ctypes.byref(err)))

    for i in range(n):
        dev = ok(rs.rs2_create_device(
            ctypes.c_void_p(devs), i, ctypes.byref(err)))
        found = ok(rs.rs2_get_device_info(
            ctypes.c_void_p(dev), RS2_CAMERA_INFO_SERIAL_NUMBER,
            ctypes.byref(err)))
        if found.decode() != serial:
            continue

        sensors = ok(rs.rs2_query_sensors(
            ctypes.c_void_p(dev), ctypes.byref(err)))
        sn = ok(rs.rs2_get_sensors_count(
            ctypes.c_void_p(sensors), ctypes.byref(err)))
        for j in range(sn):
            s = ok(rs.rs2_create_sensor(
                ctypes.c_void_p(sensors), j, ctypes.byref(err)))
            if not ok(rs.rs2_supports_option(
                    ctypes.c_void_p(s), RS2_OPTION_DEPTH_UNITS,
                    ctypes.byref(err))):
                continue

            units = ok(rs.rs2_get_option(
                ctypes.c_void_p(s), RS2_OPTION_DEPTH_UNITS,
                ctypes.byref(err)))
            # c_float: 0.001 comes back as float32, not exactly 0.001.
            if abs(units - EXPECTED_DEPTH_UNITS) > 1e-6:
                raise RuntimeError(
                    f'D435i depth units are {units:.6f} m/unit, expected '
                    f'{EXPECTED_DEPTH_UNITS} -- recorded depth would NOT be '
                    'millimetres.\n'
                    '  Replug the camera (resets to the firmware default) '
                    'or fix it in realsense-viewer -> Stereo Module.')
            return

        raise RuntimeError('D435i has no sensor exposing depth units')
    raise RuntimeError(f'no RealSense with serial {serial} '
                       f'({n} device(s) present)')


def make_realsense_node(node_name: str, serial: str, ns: str, frame_prefix: str):
    return Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name=node_name,
        namespace=ns,
        output='screen',
        parameters=[{
            'serial_no': serial,

            # Unique TF frames
            'base_frame_id': frame_prefix + 'link',
            'depth_frame_id': frame_prefix + 'depth_frame',
            'depth_optical_frame_id': frame_prefix + 'depth_optical_frame',
            'color_frame_id': frame_prefix + 'color_frame',
            'color_optical_frame_id': frame_prefix + 'color_optical_frame',

            # Streams
            'enable_color': True,
            'enable_depth': True,
            'enable_infra1': False,
            'enable_infra2': False,

            'align_depth.enable': False,
            'pointcloud.enable': False,
            'enable_sync': True,

            'enable_gyro': True, #200hz
            'enable_accel': True, #200hz

            'gyro_fps': 200,     
            'accel_fps': 200,    
            'unite_imu_method': 2,   

            # image_transport bolts /compressed, /compressedDepth and /theora
            # onto every image topic, from plugins that arrived as ROS
            # metapackage dependencies. Free while unsubscribed -- but any
            # subscriber (rqt_image_view, a stray echo, a loose -e when
            # recording) starts an encode inside THIS node, mid-session.
            #
            # Publisher side only, so rqt_image_view still decodes the C922
            # JPEGs; apt-removing the plugins would have broken that.
            #
            # Prefix is the NODE NAME, not the namespace, and a wrong prefix is
            # ignored in silence. Re-check after a rename:
            #   ros2 param list /ego/d435i_ego | grep enable_pub_plugins
            f'{node_name}.color.image_raw.enable_pub_plugins': ONLY_RAW,
            f'{node_name}.depth.image_rect_raw.enable_pub_plugins': ONLY_RAW,

            # Only declared while align_depth.enable is True; ignored otherwise.
            f'{node_name}.aligned_depth_to_color.image_raw.'
            f'enable_pub_plugins': ONLY_RAW,

            'rgb_camera.color_profile': '1280x720x30',
            'depth_module.depth_profile': '848x480x30',

            # Colour: exposure and gain are the only controls -- the lens is
            # fixed focus. Live parameters, tunable without relaunching, then
            # paste what you settle on back here:
            #   ros2 param set /ego/d435i_ego rgb_camera.exposure 156
            #
            # 100us steps, same as the C922, so 333 is one frame period at
            # 30 fps. Range 1..10000 (C922 is 3..2047).
            #
            # Auto, deliberately: the ego camera moves, swings towards windows
            # and lamps, and has people step between it and the light.
            'rgb_camera.enable_auto_exposure': True,

            # False = hold the framerate, never lengthen the shutter past one
            # frame period. This is what stops auto exposure silently halving
            # us to 15 fps, the trap that forces c922.launch.py to run manual.
            'rgb_camera.auto_exposure_priority': False,

            # Depth has its own exposure, in single microseconds -- 100x finer
            # than colour above. Auto: manual usually costs depth quality.
            'depth_module.enable_auto_exposure': True,

            # Depth has no auto_exposure_priority, so bound it explicitly.
            # 33000us is one frame period at 30 fps. The toggle enforces it and
            # defaults to False, so the limit alone would do nothing.
            #
            # Launch-time only -- "will not take effect until next streaming
            # session", so unlike the colour controls this is not live.
            'depth_module.auto_exposure_limit': 33000,
            'depth_module.auto_exposure_limit_toggle': True,
        }],
    )


# (node name, serial, namespace, TF frame prefix).
EGO = ('d435i_ego', '405622076405', 'ego', 'ego')


def generate_launch_description():
    check_usb_link()
    check_depth_units(EGO[1])
    return LaunchDescription([make_realsense_node(*EGO)])
