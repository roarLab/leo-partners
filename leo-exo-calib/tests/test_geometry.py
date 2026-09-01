"""L0 -- pure geometry core. No OpenCV video, no files: hand-built arrays with a
known truth. These are the functions a silent bug corrupts every result through,
so they get the densest, cheapest coverage."""

import numpy as np
import pytest

from exo_extrinsics import avg_pose, build_rig, kabsch, pair_frames


def _rot(rx, ry, rz):
    import cv2
    R, _ = cv2.Rodrigues(np.array([rx, ry, rz]))
    return R


def _xf(t, R=None):
    T = np.eye(4)
    T[:3, :3] = np.eye(3) if R is None else R
    T[:3, 3] = t
    return T


# ------------------------------------------------------------------- kabsch


def test_kabsch_recovers_known_transform():
    rng = np.random.default_rng(1)
    R_true, t_true = _rot(0.3, -0.5, 0.2), np.array([1.0, -2.0, 0.5])
    P = rng.uniform(-1, 1, (12, 3))
    Q = (R_true @ P.T).T + t_true
    R, t = kabsch(P, Q)
    assert np.allclose(R, R_true, atol=1e-9)
    assert np.allclose(t, t_true, atol=1e-9)


def test_kabsch_returns_proper_rotation_not_reflection():
    # Coplanar points (z=0) are the classic case where a naive SVD fit can flip
    # to a determinant -1 reflection. kabsch must always hand back a rotation.
    rng = np.random.default_rng(2)
    P = np.column_stack([rng.uniform(-1, 1, (8, 2)), np.zeros(8)])
    R_true = _rot(0.0, 0.0, 0.7)
    Q = (R_true @ P.T).T + np.array([0.5, 0.5, 0.0])
    R, _ = kabsch(P, Q)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)


# ------------------------------------------------------------------ avg_pose


def test_avg_pose_rejects_translation_outlier():
    inliers = [_xf(np.array([1.0, 0.0, 0.0])) for _ in range(5)]
    outlier = _xf(np.array([1.0, 0.0, 5.0]))       # 5 m off in z
    pose, n_kept, scatter = avg_pose(inliers + [outlier])
    assert n_kept == 5
    assert np.allclose(pose[:3, 3], [1.0, 0.0, 0.0], atol=1e-9)
    assert scatter < 1e-9


def test_avg_pose_output_is_orthonormal():
    Ts = [_xf(np.array([0.0, 0.0, 1.0]), _rot(0.1 * i, -0.05 * i, 0.02 * i))
          for i in range(6)]
    pose, _, _ = avg_pose(Ts)
    R = pose[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)


def test_avg_pose_all_outliers_falls_back_to_all():
    # Two poses 1 m apart: the median sits between them, so both are beyond the
    # 0.10 m reject radius -> keep empties -> fallback keeps all, no crash.
    Ts = [_xf(np.array([0.0, 0.0, 0.0])), _xf(np.array([1.0, 0.0, 0.0]))]
    pose, n_kept, scatter = avg_pose(Ts, reject=0.10)
    assert n_kept == 2
    assert np.allclose(pose[:3, 3], [0.5, 0.0, 0.0], atol=1e-9)
    assert scatter > 0


# --------------------------------------------------------------- pair_frames


def test_pair_frames_matches_in_phase():
    a = [(1.0, "a0"), (2.0, "a1"), (3.0, "a2")]
    b = [(1.0, "b0"), (2.0, "b1"), (3.0, "b2")]
    assert pair_frames(a, b, sync_ms=5.0) == [("a0", "b0"), ("a1", "b1"),
                                              ("a2", "b2")]


def test_pair_frames_matches_out_of_phase_within_window():
    # b lags a by 2 ms; sync_ms=5 -> still the nearest, still paired.
    a = [(1.000, "a0"), (2.000, "a1")]
    b = [(1.002, "b0"), (2.002, "b1")]
    assert pair_frames(a, b, sync_ms=5.0) == [("a0", "b0"), ("a1", "b1")]


def test_pair_frames_drops_aliased_beyond_window():
    # b lags a by 20 ms, sync_ms=5 -> nothing pairs rather than mispairing.
    a = [(1.000, "a0"), (2.000, "a1")]
    b = [(1.020, "b0"), (2.020, "b1")]
    assert pair_frames(a, b, sync_ms=5.0) == []


def test_pair_frames_empty_series_b():
    assert pair_frames([(1.0, "a0")], [], sync_ms=5.0) == []


# ----------------------------------------------------------------- build_rig


def _edge(t):
    """A relative-pose edge (translation-only) and its inverse."""
    return _xf(np.array(t)), _xf(-np.array(t))


def test_build_rig_links_all_and_chains_poses():
    # 0--1 (strong), 1--2 (medium): a chain. Root should be the most-connected
    # camera, and camera 2's pose the composition through 1.
    r01, r10 = _edge([1.0, 0.0, 0.0])
    r12, r21 = _edge([0.0, 1.0, 0.0])
    rel = {(0, 1): r01, (1, 0): r10, (1, 2): r12, (2, 1): r21}
    covis = {(0, 1): 50, (1, 2): 30}
    T_root, links = build_rig(3, rel, covis)
    assert set(T_root) == {0, 1, 2}
    # camera 1 is most connected (50+30) -> root at identity
    assert np.allclose(T_root[1], np.eye(4))
    assert np.allclose(T_root[0][:3, 3], [-1.0, 0.0, 0.0])   # via r10
    assert np.allclose(T_root[2][:3, 3], [0.0, 1.0, 0.0])    # via r12
    # strongest edge attaches first
    assert links[0][:2] == (0, 1)


def test_build_rig_leaves_disconnected_camera_out():
    r01, r10 = _edge([1.0, 0.0, 0.0])
    rel = {(0, 1): r01, (1, 0): r10}          # camera 2 has no edges
    covis = {(0, 1): 40}
    T_root, links = build_rig(3, rel, covis)
    assert 2 not in T_root
    assert set(T_root) == {0, 1}


def test_build_rig_single_camera():
    T_root, links = build_rig(1, {}, {})
    assert set(T_root) == {0}
    assert np.allclose(T_root[0], np.eye(4))
    assert links == []
