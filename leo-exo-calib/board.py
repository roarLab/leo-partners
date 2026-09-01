"""Shared ChArUco board definition and OpenCV-version compatibility.

Every script here reads the SAME board.json, which is written by make_board.py
when the board is generated. That is deliberate: the single most common way a
calibration silently produces garbage is measuring with one square size and
solving with another. If board.json travels with the printed board, they cannot
drift apart.

OpenCV moved this API twice. 4.6 and earlier use CharucoBoard_create and
interpolateCornersCharuco; 4.7+ use the CharucoBoard class; 4.8+ add
CharucoDetector; and 5.0 REMOVED calibrateCameraCharuco and
estimatePoseCharucoBoard entirely. Rather than depend on any of those, we take
the board's own 3D corner positions from getChessboardCorners() and feed plain
calibrateCamera / solvePnP, which have been stable for a decade.
"""

import json
import os

import cv2
import numpy as np

DEFAULT = {
    "squares_x": 7,
    "squares_y": 5,
    "square_mm": 100.0,
    "marker_mm": 75.0,
    "dictionary": "DICT_5X5_100",
}

# Rotation from the board's own frame to the world, for a board lying FLAT ON
# THE FLOOR, printed side up. (The printed arrow is only a "this way up" handling
# cue for the operator; it no longer encodes a world direction.)
#
# Not the identity, which is the trap. OpenCV lays the board out with x right
# and y DOWN the printed page, so z = x cross y points INTO the sheet, away
# from the face you can see. A board printed-side-up therefore has its z axis
# pointing down into the floor, and its y axis pointing along world -y.
#
# Getting this wrong is close to undetectable by eye — it mirrors the board,
# and a mirrored ArUco marker simply fails to decode, so the symptom is "no
# detections" rather than "wrong answer". The test suite pins this down.
FLAT_ON_FLOOR = np.diag([1.0, -1.0, -1.0])


def load_spec(path):
    with open(path) as fh:
        return json.load(fh)


def save_spec(path, spec):
    with open(path, "w") as fh:
        json.dump(spec, fh, indent=2)


def make_board(spec):
    """(board, dictionary) for this spec, on any OpenCV from 4.6 to 5.x."""
    d = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, spec["dictionary"]))
    sq = spec["square_mm"] / 1000.0          # OpenCV wants metres
    mk = spec["marker_mm"] / 1000.0
    nx, ny = spec["squares_x"], spec["squares_y"]
    if hasattr(cv2.aruco, "CharucoBoard"):
        try:
            return cv2.aruco.CharucoBoard((nx, ny), sq, mk, d), d
        except TypeError:
            pass                              # 4.7 took a different signature
    return cv2.aruco.CharucoBoard_create(nx, ny, sq, mk, d), d


def corner_xyz(board):
    """3D positions of the chessboard corners, in board coordinates (metres).

    Origin is the board's first inner corner, x along the long side, y along
    the short side, z out of the board face.
    """
    if hasattr(board, "getChessboardCorners"):
        return np.asarray(board.getChessboardCorners(), np.float32)
    return np.asarray(board.chessboardCorners, np.float32)   # <= 4.6


def detect(gray, board, dictionary):
    """(corners Nx1x2 float32, ids Nx1 int) or (None, None) if not found."""
    if hasattr(cv2.aruco, "CharucoDetector"):
        cc, ci, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
        if ci is None or len(ci) == 0:
            return None, None
        return np.asarray(cc, np.float32), np.asarray(ci).reshape(-1, 1)

    # Legacy path, OpenCV <= 4.7.
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        params = cv2.aruco.DetectorParameters_create()
    else:
        params = cv2.aruco.DetectorParameters()
    mc, mi, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    if mi is None or len(mi) == 0:
        return None, None
    n, cc, ci = cv2.aruco.interpolateCornersCharuco(mc, mi, gray, board)
    if n is None or n < 1:
        return None, None
    return np.asarray(cc, np.float32), np.asarray(ci).reshape(-1, 1)


def object_points(board, ids):
    """The 3D corners matching these detected ids."""
    return corner_xyz(board)[np.asarray(ids).reshape(-1)]


def video_streams(episode_dir):
    """(name, path) for every video in an episode, exo cameras first.

    Camera naming drifts between sessions — the 08-04 and 08-05 rigs recorded
    cam1/cam2/cam3, the 08-06 rig exo_cam1..exo_cam4 — so match on anything
    that is not the ego camera rather than on a fixed list.

    Looks in <episode_dir>/videos/ (the standard episode layout); if there is no
    videos/ subfolder, falls back to <episode_dir> itself, so a flat folder of
    per-camera clips (e.g. a dedicated intrinsics capture with no timestamps)
    also works.
    """
    vdir = os.path.join(episode_dir, "videos")
    if not os.path.isdir(vdir):
        vdir = episode_dir
    if not os.path.isdir(vdir):
        return []
    out = []
    for f in sorted(os.listdir(vdir)):
        if not f.endswith((".mp4", ".avi", ".mkv")):
            continue
        name = os.path.splitext(f)[0]
        if "ego" in name.lower():
            continue
        out.append((name, os.path.join(vdir, f)))
    return out
