import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


# ---------------------------------------------------------------------------
# Brings up every camera in one go, so a single rosbag can record them all.
#
# This file deliberately holds NO camera settings. Each driver keeps its own,
# because the two are configured in completely different ways:
#
#   four_d435i.launch.py   RealSense. Settings are native ROS parameters and
#                          are live -- ros2 param set works while running.
#                          No focus control: fixed-focus lens.
#
#   c922.launch.py         Logitech. Settings go through v4l2-ctl subprocesses
#                          because usb_cam's own control parameters are broken
#                          on kernel 6.8. Edit the CAMERAS/DEFAULTS blocks at
#                          the top of that file, then rebuild.
#
# To add or remove a camera, edit the file that owns it -- not this one.
#
#   ros2 launch d435i_multicam_launch all_cameras.launch.py
# ---------------------------------------------------------------------------
LAUNCH_DIR = os.path.join(
    get_package_share_directory('d435i_multicam_launch'), 'launch')


def include(launch_file: str):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(LAUNCH_DIR, launch_file)))


def generate_launch_description():
    return LaunchDescription([
        include('four_d435i.launch.py'),
        include('c922.launch.py'),
    ])
