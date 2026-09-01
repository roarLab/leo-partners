#!/usr/bin/env bash
# TERMINAL 4 -- record ONE episode. Each run writes one bag, auto-numbered.
#
#     ./record.sh 23   ->  2026-08-04_23_1
#     ./record.sh 23   ->  2026-08-04_23_2   (next run)
#     ./record.sh 24   ->  2026-08-04_24_1   (own count)
#
# One command per episode, not a running session loop, so that:
#   * you label each bag as it records -- a retake or deviation gets its own
#     code on the spot, which a fixed 1,2,3 loop cannot do; and
#   * 'ros2 bag record' keeps sole use of this terminal's keys (pause/resume);
#     a loop waiting on your keypress would fight it for them.
# The cost is typing the code each time -- free, since you are at the keyboard.
#
# You type only the code; the number is (highest for today + this code) + 1.
# Bags land in BAG_ROOT (default: the workspace):  export BAG_ROOT=/mnt/data
#
# ./cameras.sh first.
# Ctrl-C ends the episode; let it close the bag before you close the window.
set -eo pipefail

if [ $# -eq 0 ]; then
    echo "" >&2
    echo "  usage: ./record.sh <episode-code>" >&2
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

# ROOT (which tree bags live under) is the shell's job -- only it knows $WS.
# The NAME (date + code + number) is the node's job. Hand it both: the root
# via BAG_ROOT, the code as the argument.
export BAG_ROOT="${BAG_ROOT:-$WS}"

exec ros2 run d435i_multicam_launch record_session "$1"
