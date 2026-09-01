"""L1 -- the OpenCV boundary. Render one real frame of the printed board through
the homography, then run the genuine detector and PnP solver on it. This is the
only layer that exercises detect()/board_pose(); the happy-path E2E covers them
again in context, but here a failure points straight at the detector or the
solve rather than the whole pipeline."""

import cv2
import numpy as np

from board import detect
from exo_extrinsics import board_pose

K_TRUE = np.array([[1350.0, 0, 960.0], [0, 1350.0, 540.0], [0, 0, 1]])
D_ZERO = np.zeros(5)


def test_detect_finds_board_corners(one_frame):
    frame, _, rig = one_frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cc, ci = detect(gray, rig.board, None)
    assert ci is not None
    assert len(ci) >= 10          # a head-on full-frame board -> most corners


def test_board_pose_recovers_known_pose(one_frame):
    frame, T_true, rig = one_frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cc, ci = detect(gray, rig.board, None)
    T, err = board_pose(cc, ci, K_TRUE, D_ZERO, rig.board)
    assert T is not None
    assert err < 1.0              # clean synthetic corners -> sub-pixel reproj
    # board pose in the camera == the T_cam_board we rendered at
    assert np.allclose(T[:3, 3], T_true[:3, 3], atol=0.01)
    assert np.allclose(T[:3, :3], T_true[:3, :3], atol=0.02)
