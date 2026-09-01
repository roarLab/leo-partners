#!/usr/bin/env bash
# UNUSED: retired marker mechanism, kept for reference. Nothing records
# /episode_markers, and the runbook does not start this.
#
# TERMINAL 3 -- episode markers.
#
#     ./marker.sh
#
#     s  start an episode
#     e  end the open episode
#     q  quit
#
# Two keys, not one toggle: every toggle sequence is well-formed by
# construction, so a missed press could never be detected. See the docstring
# in episode_marker.py.
#
# Start it BEFORE ./record.sh, so /episode_markers exists when the bag opens.
#
# It must have this terminal to itself: it reads single keypresses, and
# anything else reading the same terminal takes some of them instead.
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

exec ros2 run d435i_multicam_launch episode_marker
