#!/usr/bin/env bash
# Mark a recorded episode with one or more mistake codes.
#
# Run it from the directory that holds the episode folders, so the
# <episode> name tab-completes -- you recognise it, you don't recall it.
#
#     ./oops.sh 2026-08-04_23_2 m1          # one code
#     ./oops.sh 2026-08-04_23_2 m1 m3       # several
#
# A code becomes an EMPTY marker file at <episode>/oops/<code>. The
# extractor reads the filename; the file content is unused.
# Undo a mark:  rm <episode>/oops/<code>
set -eo pipefail

if [ $# -lt 2 ]; then
    echo "  usage: ./oops.sh <episode-folder> <code> [<code> ...]" >&2
    exit 1
fi

dir="$1"; shift

if [ ! -d "$dir" ]; then
    echo "  no such episode folder: $dir" >&2
    exit 1
fi

# Validate every code BEFORE writing any marker, so a typo can't leave
# the episode half-marked. Format: a lowercase letter then alphanumerics
# (e.g. m1). This also refuses spaces, slashes and dots that would break
# out of the oops/ folder.
for code in "$@"; do
    if ! [[ "$code" =~ ^[a-z][a-z0-9]*$ ]]; then
        echo "  bad code: '$code' -- want lowercase alphanumeric like m1" >&2
        exit 1
    fi
done

mkdir -p "$dir/oops"
for code in "$@"; do
    touch "$dir/oops/$code"
done

echo "  marked ${dir%/}: $*"
