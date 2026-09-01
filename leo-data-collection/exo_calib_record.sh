#!/usr/bin/env bash
# TERMINAL 4 (exo calibration) -- record ONE exo-only bag, auto-numbered.
#
#     ./exo_calib_record.sh 23   ->  2026-08-04_23_1
#     ./exo_calib_record.sh 23   ->  2026-08-04_23_2   (next run)
#
# Records only the four C922 topics, no ego. The recorder's startup watch is
# scoped to the C922s too, so it will not stall waiting on a missing ego.
#
# ./exo_calib_launch.sh first. Bags land in BAG_ROOT (default: the workspace):
#     export BAG_ROOT=/mnt/data
# Ctrl-C ends the episode; let it close the bag before you close the window.
set -eo pipefail

if [ $# -eq 0 ]; then
    echo "" >&2
    echo "  usage: ./exo_calib_record.sh <episode-code>" >&2
    echo "" >&2
    exit 1
fi

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

export BAG_ROOT="${BAG_ROOT:-$WS}"

# --calib-exo drops the /ego/ topics from both the bag and the silence watch.
exec ros2 run d435i_multicam_launch record_session "$1" --calib-exo
