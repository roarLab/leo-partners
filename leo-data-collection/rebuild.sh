#!/usr/bin/env bash
# Rebuild the workspace.
#
#     ./rebuild.sh
#
# Building is all this does. The ./ scripts source ROS and install/setup.bash
# themselves every time, so nothing here has to set up your shell.
#
# Any extra arguments are passed through to colcon, e.g.
#     ./rebuild.sh --packages-select d435i_multicam_launch

# The workspace is wherever this script lives, so it works from any directory
# and survives the repo being moved or cloned elsewhere.
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /opt/ros/humble/setup.bash

# colcon writes build/ install/ log/ relative to the working directory.
cd "$WS" || exit 1

if colcon build "$@"; then
    echo ""
    echo "  build OK"
    echo ""
    echo "  Next, one terminal each, in this order:"
    echo ""
    echo "    1.  ./cameras.sh             cameras -- leave it up all day"
    echo "    2.  rqt --perspective-file leo.perspective   all feeds -- leave it up"
    echo "    3.  ./record.sh <code>       one bag per episode, into <date>_<code>_N/"
    echo ""
    echo "  Run them with ./, not source"
else
    echo ""
    echo "  BUILD FAILED - the previous build is still in place"
fi
