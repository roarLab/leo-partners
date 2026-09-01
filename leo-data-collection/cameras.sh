#!/usr/bin/env bash
# TERMINAL 1 -- the cameras. Leave it up for the whole day.
#
#     ./cameras.sh
#
# Executed, not sourced: it becomes the launch rather than changing your
# shell. Ctrl-C here stops the cameras, and does not touch a running session.
#
# The launch refuses to start on a USB 2 link, wrong depth units or a missing
# C922. Read what it prints -- that is the check, not this script.
set -eo pipefail

# No -u: ROS's setup files read unset variables and die under it.
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash

if [ ! -f "$WS/install/setup.bash" ]; then
    echo "" >&2
    echo "  no install/setup.bash in $WS -- the workspace is not built" >&2
    echo "  run:  ./rebuild.sh" >&2
    echo "" >&2
    exit 1
fi
source "$WS/install/setup.bash"
cd "$WS" || exit 1

exec ros2 launch d435i_multicam_launch all_cameras.launch.py
