#!/usr/bin/env bash
# TERMINAL 1 (exo calibration) -- the four exo C922s only, no ego D435i.
#
#     ./exo_calib_launch.sh
#
# Calibrating the exo cameras: brings up c922.launch.py alone, so the ego is
# neither started nor drawing USB bandwidth. Pair with ./exo_calib_record.sh.
#
# Executed, not sourced: it becomes the launch rather than changing your
# shell. Ctrl-C here stops the cameras, and does not touch a running session.
#
# The launch refuses to start on a missing or already-open C922. Read what it
# prints -- that is the check, not this script.
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

exec ros2 launch d435i_multicam_launch c922.launch.py
