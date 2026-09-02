"""Extrinsics from the walk-around, by multi-camera bundle.

During the walk-around the board is carried through the room and is co-visible to
camera PAIRS in many frames, at varied angles. For one such frame the board is a
single physical object, so

    T_i_j = T_i_board @ inv(T_j_board)          # camera j's pose in camera i

recovers the RELATIVE pose of the two cameras with the board's own pose cancelled
out. Averaged over the shared frames this is tight to a few centimetres. Chaining
the pairs gives one rigid 4-camera rig; the operator's tape-measured camera
positions then anchor that rig into the bed/world frame (Kabsch).

    python exo_extrinsics.py --walk capture/calib-walk \\
        --calib calib/intrinsics.json --camera_positions cams.json \\
        --board board/board.json --out calib/calib.json

Intrinsics are read from --calib (produced by exo_intrinsics.py); this script
recomputes the extrinsics and writes the anchored result -- the single final
calibration file, carrying intrinsics + extrinsics + all metrics.

The geometry core is factored into pure, importable functions (stamps,
board_pose, avg_pose, kabsch, pair_frames, build_rig, cameras_to_solve,
build_extrinsics_output) so the toolchain can be unit-tested without a real
capture; main() wires them to the CLI. See tests/.
"""

import argparse
import csv
import json
import os

import cv2
import numpy as np

from board import detect, make_board, object_points, video_streams
from inputs import load_json, require_dir, require_exo_cams, require_file

parser = argparse.ArgumentParser()
parser.add_argument("--walk", required=True, help="walk-around episode dir")
parser.add_argument("--calib", default="calib/calib.json",
                    help="intrinsics JSON from exo_intrinsics.py: the intrinsics "
                         "used to solve board poses")
parser.add_argument("--board", required=True,
                    help="path to board.json -- the metric ruler that sets "
                         "scale. Required, never defaulted: a wrong or missing "
                         "spec would silently rescale every distance.")
parser.add_argument("--out", default="calib",
                    help="output FILE (.../name.json) or a directory (writes "
                         "calib_bundle.json inside it).")
parser.add_argument("--camera_positions", required=True,
                    help="JSON of {cam_name: [x,y,z]} measured camera positions "
                         "in the bed frame; the Kabsch anchor fits the rig to "
                         "these, setting the world frame.")
parser.add_argument("--stride", type=int, default=2)
parser.add_argument("--min_corners", type=int, default=10,
                    help="min board corners to accept a frame. Loosen (6) only if "
                         "a small/far board in a pair's overlap zone is rejected.")
parser.add_argument("--max_reproj", type=float, default=2.0,
                    help="reject a board pose whose reprojection error exceeds "
                         "this (px). Raise (3-4) only for small/far boards whose "
                         "corners are noisier but still usable once averaged.")
parser.add_argument("--sync_ms", type=float, default=33.0,
                    help="two frames count as simultaneous within this many ms. "
                         "Default 33 = half a frame interval at 30fps/stride 2, "
                         "so out-of-phase cameras still pair (20 aliased cam1 "
                         "out). Rule: sync_ms >= (1000/fps) * stride / 2.")
parser.add_argument("--min_covis", type=int, default=8,
                    help="a camera pair needs at least this many co-visible "
                         "frames to trust its relative pose")


# ---------------------------------------------------------------- pure helpers


def stamps(path):
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            out[int(row["index"])] = float(row["ros_time_s"])
    return out


def board_pose(cc, ids, K, D, board):
    """Board pose in the camera, lower-error planar (IPPE) branch, or None."""
    obj = object_points(board, ids).astype(np.float32)
    img = cc.reshape(-1, 2).astype(np.float32)
    try:
        n, rv, tv, er = cv2.solvePnPGeneric(obj, img, K, D,
                                            flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return None, None
    if not n:
        return None, None
    k = min(range(n), key=lambda j: float(np.ravel(er[j])[0]))
    R, _ = cv2.Rodrigues(rv[k])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, tv[k].ravel()
    return T, float(np.ravel(er[k])[0])


def avg_pose(Ts, reject=0.10):
    """Robust average of 4x4 poses: drop translation outliers vs the median,
    then mean-and-reorthonormalise what remains.
    Returns (pose, n_kept, scatter_m) where scatter is how tightly the kept
    positions agree (1-sigma), i.e. the precision of this estimate."""
    P = np.stack([T[:3, 3] for T in Ts])
    med = np.median(P, axis=0)
    keep = [T for T in Ts if np.linalg.norm(T[:3, 3] - med) <= reject]
    if not keep:
        keep = Ts
    kt = np.stack([T[:3, 3] for T in keep])
    t = kt.mean(axis=0)
    scatter = float(np.linalg.norm(kt.std(axis=0))) if len(kt) > 1 else 0.0
    M = np.mean([T[:3, :3] for T in keep], axis=0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = R, t
    return out, len(keep), scatter


def kabsch(P, Q):
    """Rigid transform R,t with R@P_i + t ~= Q_i. P,Q: (N,3)."""
    cp, cq = P.mean(0), Q.mean(0)
    H = (P - cp).T @ (Q - cq)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cq - R @ cp


def pair_frames(series_a, series_b, sync_ms):
    """Match frames co-visible within sync_ms. Each series is a list of
    (timestamp, payload); returns [(payload_a, payload_b), ...], each element of
    series_a matched to its nearest-in-time element of series_b when the two are
    within sync_ms. Pure timestamp logic -- payloads are opaque, so tests can
    pass plain markers instead of poses.

    This is the aliasing guard: out-of-phase cameras still pair (nearest within
    the window), but frames further apart than sync_ms are dropped rather than
    mispaired."""
    tb = np.array([t for t, _ in series_b])
    out = []
    for t, pa in series_a:
        if not len(tb):
            break
        k = int(np.argmin(np.abs(tb - t)))
        if abs(tb[k] - t) * 1000.0 <= sync_ms:
            out.append((pa, series_b[k][1]))
    return out


def build_rig(n, rel, covis):
    """Assemble one rigid rig from pairwise relative poses via a max-co-visibility
    spanning tree over n cameras (indices 0..n-1).

    rel[(a, b)] = camera b's pose in camera a's frame (4x4); covis[(i, j)] with
    i<j = shared-frame count for that pair (the edge trust weight). Roots at the
    most-connected camera, then repeatedly attaches the unplaced camera reachable
    by the strongest edge, chaining T_root[b] = T_root[a] @ rel[(a, b)].

    Returns (T_root, links): T_root maps camera index -> pose in the root frame;
    links is the ordered [(child, parent, shared_frames), ...] chosen, for the
    caller to log. A camera with no co-visibility path is absent from T_root --
    that is how a camera ends up failed_extrinsics. Pure: no I/O, no OpenCV."""
    deg = {a: sum(covis.get((min(a, b), max(a, b)), 0)
                  for b in range(n) if b != a) for a in range(n)}
    root = max(deg, key=deg.get)
    T_root, links = {root: np.eye(4)}, []
    while True:
        best = None
        for a in list(T_root):
            for b in range(n):
                if b in T_root or (a, b) not in rel:
                    continue
                w = covis[(min(a, b), max(a, b))]
                if best is None or w > best[0]:
                    best = (w, a, b)
        if best is None:
            break
        w, a, b = best
        T_root[b] = T_root[a] @ rel[(a, b)]
        links.append((b, a, w))
    return T_root, links


def cameras_to_solve(cams_all, names):
    """Split the roster into (solvable, no_lens). A camera whose lens never
    calibrated carries intrinsics=None (status failed_intrinsics) -- the extrinsics
    stage cannot solvePnP for it, so it is skipped in detection here and later
    written straight through as failed_intrinsics. Pure: unit-testable without a
    capture, so the skip decision is covered without running the pipeline."""
    solvable = [n for n in names if cams_all[n].get("intrinsics") is not None]
    no_lens = [n for n in names if cams_all[n].get("intrinsics") is None]
    return solvable, no_lens


def build_extrinsics_output(calib, names, T_root, T_world_root,
                            rel_scatter, pair_scatter, residuals, shared,
                            fit_rms_m):
    """Assemble the final calibration dict from already-solved pieces.

    Pure: no I/O, no OpenCV -- feed it synthetic inputs to exercise the three
    status branches. Each camera's status is the worst stage it reached:
      - intrinsics is null (lens never calibrated)   -> "failed_intrinsics"
      - intrinsics ok but absent from T_root (no rig link) -> "failed_extrinsics"
      - linked to the rig                              -> "ok" (carries world pose)
    The intrinsics failure wins: such a camera is also absent from T_root, so the
    null-intrinsics check must come first. `shared` (the anchor cameras) sets
    whether absolute accuracy has a cross-check (>= 4 points)."""
    cams_all = calib["cameras"]
    N = len(shared)

    out = {"scene_details": dict(calib["scene_details"]),
           "cameras": {}, "accuracy": {}}
    for a, name in enumerate(names):
        intrinsics = cams_all[name]["intrinsics"]
        if intrinsics is None:
            out["cameras"][name] = {"intrinsics": None, "extrinsics": None,
                                    "status": "failed_intrinsics"}
            continue
        if a not in T_root:
            out["cameras"][name] = {"intrinsics": intrinsics, "extrinsics": None,
                                    "status": "failed_extrinsics"}
            continue
        T_wc = T_world_root @ T_root[a]
        out["cameras"][name] = {
            "intrinsics": intrinsics,
            "status": "ok",
            "extrinsics": {"T_world_cam": T_wc.tolist(),
                           "position_m": T_wc[:3, 3].tolist()},
        }

    out["accuracy"] = {
        "relative": {
            "describes": ("how tightly the cameras agree with each other -- "
                          "triangulation precision. metres, 1-sigma; lower is "
                          "better. Blind to scale: a mis-scaled rig can still "
                          "agree with itself."),
            "worst_m": round(max(rel_scatter), 4) if rel_scatter else None,
            "per_pair_m": {k: round(v["agree_m"], 4)
                           for k, v in pair_scatter.items()},
        },
        "absolute": {
            "describes": ("how well the solved rig fits the tape-measured camera "
                          "positions -- bed-placement accuracy. metres, RMS; lower "
                          "is better. The only metric that sees a scale error. "
                          + ("4 points give a cross-check."
                             if N >= 4 else "only 3 points -- no cross-check.")),
            "rms_m": round(fit_rms_m, 4),
            "per_camera_m": {n: round(residuals[n], 4) for n in shared},
        },
        # per-camera lens quality, carried through unchanged from --calib
        "intrinsics": calib["accuracy"]["intrinsics"],
        "cameras_solved": f"{len(T_root)}/{len(names)}",
    }
    return out


# ------------------------------------------------------------------------ main


def main(args):
    board_json = args.board
    require_dir(args.walk, "--walk", "exo_extrinsics")
    spec = load_json(board_json, "--board", "exo_extrinsics")
    board, dictionary = make_board(spec)
    print(f"[board] {board_json}: {spec.get('squares_x')}x{spec.get('squares_y')} "
          f"squares, {spec.get('square_mm')}mm, {spec.get('dictionary')}, "
          f"print_scale {spec.get('print_scale', 1.0)}")
    calib = load_json(args.calib, "--calib", "exo_extrinsics")
    cams_all = calib["cameras"]

    # -------------------------------------------- per-camera board pose series
    print("[1/4] detecting the board through the walk-around")
    streams = require_exo_cams(video_streams(args.walk), args.walk,
                               "--walk", "exo_extrinsics")
    names = [n for n, _ in streams if n in cams_all]
    solvable, no_lens = cameras_to_solve(cams_all, names)
    det = {n: [] for n in no_lens}    # failed_intrinsics -> no poses, skip detection
    for n in no_lens:
        print(f"  {n:<12} lens failed (failed_intrinsics) -- skipped")
    for name, path in streams:
        if name not in solvable:
            continue
        ts_csv = os.path.join(args.walk, "timestamps", name + ".csv")
        require_file(ts_csv, "--walk timestamps", "exo_extrinsics")
        ts = stamps(ts_csv)
        K = np.array(cams_all[name]["intrinsics"]["K"])
        D = np.array(cams_all[name]["intrinsics"]["dist"])
        cap = cv2.VideoCapture(path)
        series, i = [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % args.stride or i not in ts:
                i += 1
                continue
            cc, ci = detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                            board, dictionary)
            if ci is not None and len(ci) >= args.min_corners:
                T, err = board_pose(cc, ci, K, D, board)
                if T is not None and err < args.max_reproj:
                    series.append((ts[i], T))
            i += 1
        cap.release()
        det[name] = series
        print(f"  {name:<12} {len(series):>4} frames with a clean board pose")

    # ------------------------------------------------ pairwise relative poses
    print("\n[2/4] relative camera poses from co-visible frames")
    print("  'agree to' = how tightly the shared frames pin this pair's geometry "
          "(1-sigma). want a few cm.")
    rel = {}          # (i, j) -> T_i_j  (camera j expressed in camera i's frame)
    covis = {}
    rel_scatter = []
    pair_scatter = {}   # "ni-nj" -> {"agree_m": scatter, "shared_frames": covis}
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            ni, nj = names[a], names[b]
            matched = pair_frames(det[ni], det[nj], args.sync_ms)
            pairs = [Ti @ np.linalg.inv(Tj) for Ti, Tj in matched]
            covis[(a, b)] = len(pairs)
            if len(pairs) >= args.min_covis:
                T_ij, kept, sc = avg_pose(pairs)
                rel[(a, b)] = T_ij
                rel[(b, a)] = np.linalg.inv(T_ij)
                rel_scatter.append(sc)
                pair_scatter[f"{ni}-{nj}"] = {"agree_m": sc,
                                              "shared_frames": len(pairs)}
                tag = "GOOD" if sc < 0.03 else "OK" if sc < 0.08 else "HIGH-check"
                print(f"  {ni}-{nj}: {len(pairs):>3} frames, agree to {sc*100:4.1f} "
                      f"cm [{tag:<10}] baseline {np.linalg.norm(T_ij[:3,3]):.2f} m")
            else:
                print(f"  {ni}-{nj}: {len(pairs):>3} co-visible  -- too few, "
                      "skipped")

    # ------------------------------------------------ chain into one rig
    print("\n[3/4] assembling the rig (max-co-visibility spanning tree)")
    T_root, links = build_rig(len(names), rel, covis)
    for b, a, w in links:
        print(f"  {names[b]} <- {names[a]}  (via {w} shared frames)")
    if len(T_root) < len(names):
        missing = [names[a] for a in range(len(names)) if a not in T_root]
        print(f"  [warn] not linked to the rig: {missing} -- no co-visibility path")

    # ------------------------------------------------ anchor the rig to world
    name_idx = {n: a for a, n in enumerate(names)}

    # The rig gives each camera's position in the rig frame; the operator
    # tape-measured each camera's position in the bed/world frame. Same points in
    # two frames -> one rigid transform (Kabsch) fits world<-rig directly. Robust
    # to grazing angles, which is where the board placements fail.
    print("\n[4/4] anchoring to the bed/world frame via measured camera positions")
    measured = load_json(args.camera_positions, "--camera_positions",
                         "exo_extrinsics")
    shared = [n for n in names if name_idx[n] in T_root and n in measured]
    if len(shared) < 3:
        raise SystemExit(
            f"need >= 3 cameras both linked to the rig and present in "
            f"{args.camera_positions}; only {len(shared)} matched: {shared}")
    P = np.stack([T_root[name_idx[n]][:3, 3] for n in shared])      # rig frame
    Q = np.stack([np.asarray(measured[n], float) for n in shared])  # world frame
    R, t = kabsch(P, Q)
    T_world_root = np.eye(4)
    T_world_root[:3, :3], T_world_root[:3, 3] = R, t

    # Per-camera residual = fitted position vs the measured one. One large
    # residual with the rest small means THAT measurement is likely wrong.
    residuals = {}
    print(f"\n  {'camera':<12}{'measured (x y z)':>25}"
          f"{'fitted (x y z)':>25}{'resid':>8}")
    for n in shared:
        fitted = (T_world_root @ T_root[name_idx[n]])[:3, 3]
        q = np.asarray(measured[n], float)
        residuals[n] = float(np.linalg.norm(fitted - q))
        mpos = f"[{q[0]:6.2f} {q[1]:6.2f} {q[2]:6.2f}]"
        fpos = f"[{fitted[0]:6.2f} {fitted[1]:6.2f} {fitted[2]:6.2f}]"
        flag = "  <-- large, likely a bad measurement" if residuals[n] > 0.15 else ""
        print(f"  {n:<12}{mpos:>25}{fpos:>25}{residuals[n]*100:>6.1f}cm{flag}")
    N = len(shared)
    fit_rms_m = float(np.sqrt(np.mean([d * d for d in residuals.values()])))
    anchor_cm = fit_rms_m * 100
    atag = "GOOD" if anchor_cm < 5 else "OK" if anchor_cm < 15 else "HIGH -- loose"
    xchk = "" if N >= 4 else " (only 3 points -- no cross-check)"
    print(f"  anchored to {N} measured camera positions, fit RMS "
          f"{anchor_cm:.1f} cm [{atag}]{xchk}")

    # ------------------------------------------------ write the result
    out = build_extrinsics_output(calib, names, T_root, T_world_root,
                                  rel_scatter, pair_scatter, residuals, shared,
                                  fit_rms_m)
    print(f"\n{'camera':<12}{'linked':>8}   position (x y z)")
    for name, entry in out["cameras"].items():
        if entry.get("extrinsics") is None:
            print(f"{name:<12}{'no':>8}   {entry['status']}")
        else:
            p = entry["extrinsics"]["position_m"]
            print(f"{name:<12}{'yes':>8}   [{p[0]:6.2f} {p[1]:6.2f} {p[2]:6.2f}]")

    # --out is either a name.json FILE or a directory (then calib_bundle.json in).
    out_arg = os.path.abspath(args.out)
    if out_arg.lower().endswith(".json"):
        dst, out_dir = out_arg, os.path.dirname(out_arg)
    else:
        out_dir, dst = out_arg, os.path.join(out_arg, "calib_bundle.json")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(out, open(dst, "w"), indent=2, ensure_ascii=False)
    print(f"\n[out] {dst}")
    print("Positions are metres in the simulation's world frame: x along the bed "
          "with the headboard at +x, y across it, z up. Sanity-check against where "
          "the cameras physically are.")

    # Headline: the plain-language bottom line for whoever ran this.
    rel_cm = max(rel_scatter) * 100 if rel_scatter else float("nan")
    rtag = "GOOD" if rel_cm < 3 else "OK" if rel_cm < 8 else "HIGH"
    n_linked = len(T_root)
    print("\n" + "=" * 70)
    print(f"HEADLINE: {n_linked}/{len(names)} cameras solved. They agree with each "
          f"other to ~{rel_cm:.0f} cm [{rtag}],")
    if N < 4:
        print(f"          anchored to {N} measured camera positions -- absolute "
              f"placement UNVERIFIED (only 3 points).")
        print(f"  A point seen by 2+ cameras triangulates to about {rel_cm:.0f} "
              f"cm; feed a 4th")
        print("  measured camera position to cross-check the absolute placement.")
    else:
        print(f"          and fit {N} measured camera positions to ~{anchor_cm:.0f} "
              f"cm RMS [{atag.split()[0]}].")
        print(f"  A point seen by 2+ cameras triangulates to about {rel_cm:.0f} "
              f"cm; the whole rig's")
        print(f"  absolute placement matches the tape measure to about "
              f"{anchor_cm:.0f} cm.")
    print("=" * 70)


if __name__ == "__main__":
    main(parser.parse_args())
