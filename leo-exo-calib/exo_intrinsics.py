"""Solve exo-camera intrinsics from a dedicated intrinsics capture.

Lens only -- K + distortion per camera, no world reference. Feed this a close-up
capture where the board FILLS each camera's frame and reaches its edges/corners
at varied tilt and distance -- NOT the walk-around clip, which never pins focal
length (that is the fx-drift trap). The extrinsics are anchored afterwards by
exo_extrinsics.py, which reads this file via --calib.

    python exo_intrinsics.py --intrinsics capture/intrinsics \\
        --board board/board.json --out calib/intrinsics.json

The view sampling and output assembly are factored into pure, importable
functions (evenly, coverage_and_tilt, build_intrinsics_output) so they can be
unit-tested without a real capture; main() wires them to the CLI. See tests/.
"""

import argparse
import json
import os

import cv2
import numpy as np

from board import detect, make_board, object_points, video_streams
from inputs import load_json, require_dir, require_exo_cams

parser = argparse.ArgumentParser()
parser.add_argument("--intrinsics", required=True,
                    help="episode dir of the dedicated intrinsics capture "
                         "(board filling each camera's frame, at its edges, "
                         "tilted). NOT the walk-around clip.")
parser.add_argument("--board", required=True,
                    help="path to board.json -- the metric ruler that sets "
                         "scale. Required, never defaulted: a wrong or missing "
                         "spec would silently rescale every distance.")
parser.add_argument("--out", default="calib",
                    help="output FILE (.../name.json) or a directory (writes "
                         "calib.json inside it).")
parser.add_argument("--stride", type=int, default=3)
parser.add_argument("--min_corners", type=int, default=8)
parser.add_argument("--max_views", type=int, default=120,
                    help="cap on frames used for intrinsics, sampled EVENLY "
                         "across the whole clip; more is slower with little gain "
                         "once coverage is high")
parser.add_argument("--free_k3", action="store_true",
                    help="fit the 3rd radial distortion term k3 instead of "
                         "pinning it to 0. Use ONLY on a proper edge-filling "
                         "capture -- on weak footage k3 overfits and drags fx. "
                         "Default off; for this rig its influence is negligible.")


# ---------------------------------------------------------------- pure helpers


def views(path, board, dictionary, stride, min_corners):
    """(corners, ids) for every detected frame across the WHOLE clip, plus size.

    Reads the entire clip (no early stop) so the caller can subsample views
    spread evenly across it -- see evenly(). Taking the first N instead would
    only see the coverage of the clip's opening seconds."""
    cap = cv2.VideoCapture(path)
    out, size, i = [], None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride:
            i += 1
            continue
        i += 1
        if size is None:
            size = (frame.shape[1], frame.shape[0])
        cc, ci = detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                        board, dictionary)
        if ci is not None and len(ci) >= min_corners:
            out.append((np.asarray(cc, np.float32).reshape(-1, 1, 2),
                        np.asarray(ci)))
    cap.release()
    return out, size


def evenly(vs, k):
    """k views spread evenly across the clip (not the first k), so the intrinsics
    see the whole clip's frame coverage at a fixed solve cost."""
    if len(vs) <= k:
        return vs
    idx = np.linspace(0, len(vs) - 1, k).round().astype(int)
    idx = sorted(set(idx.tolist()))
    return [vs[j] for j in idx]


def coverage_and_tilt(vs, size):
    """Capture-quality metrics over the views actually used.
    `coverage` = fraction of a 6x8 image grid
    the board corners ever reached (edge coverage -- what pins fx/distortion);
    `angle_spread` = variety of the board's apparent tilt (10th-90th pct of its
    width/height ratio). Low coverage or low spread = a degenerate capture whose
    fx is not to be trusted, however low the RMS."""
    grid = np.zeros((6, 8), bool)
    ratios = []
    for cc, _ in vs:
        pts = cc.reshape(-1, 2)
        gx = np.clip((pts[:, 0] / size[0] * 8).astype(int), 0, 7)
        gy = np.clip((pts[:, 1] / size[1] * 6).astype(int), 0, 5)
        grid[gy, gx] = True
        span = pts.max(axis=0) - pts.min(axis=0)
        ratios.append(span[0] / max(span[1], 1e-6))
    cov = round(float(grid.mean()), 3)
    spread = (round(float(np.percentile(ratios, 90) - np.percentile(ratios, 10)),
                    3) if len(ratios) > 4 else 0.0)
    return cov, spread


def calib_flags(free_k3):
    """OpenCV calibrateCamera flags for the constrained c922 lens model. Square
    pixels (fx=fy) and no tangential term are ALWAYS enforced -- that is what the
    hardware physically is. k3 (3rd radial term) is pinned to 0 unless free_k3:
    on weak footage a free k3 overfits (it once ran to -34 and dragged fx), so it
    is only safe to fit on a proper edge-filling capture. Pure: unit-testable
    without a capture."""
    flags = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST
    if not free_k3:
        flags |= cv2.CALIB_FIX_K3
    return flags


def build_intrinsics_output(intr, spec):
    """Assemble the intrinsics-only calib dict from per-camera solves.

    Pure: no I/O, no OpenCV -- feed synthetic per-camera entries to exercise the
    fx_spread computation and the output schema. `intr` is
    {name: {K, dist, image_size, rms_px, n_views, coverage, angle_spread}} for a
    solved camera, or {name: {"failed": True}} for one whose lens never
    calibrated. K is a 3x3 nested list. A failed camera is written with
    status="failed_intrinsics" and intrinsics=null (present-but-null, never a
    missing key) and is left out of the fx_spread/per_camera quality table.
    Intrinsics only; exo_extrinsics.py anchors the extrinsics and copies the
    accuracy.intrinsics block straight through into the final file."""
    out = {"scene_details": {"board": spec}, "cameras": {}, "accuracy": {}}
    for name, entry in intr.items():
        if entry.get("failed"):
            out["cameras"][name] = {"intrinsics": None, "extrinsics": None,
                                    "status": "failed_intrinsics"}
            continue
        out["cameras"][name] = {
            "intrinsics": {"K": entry["K"], "dist": entry["dist"],
                           "image_size": entry["image_size"]},
            "extrinsics": None,
            "status": "ok",
        }

    solved = {n: e for n, e in intr.items() if not e.get("failed")}
    fx = {n: e["K"][0][0] for n, e in solved.items()}
    lo, hi = (min(fx.values()), max(fx.values())) if fx else (0.0, 0.0)
    fx_spread = round((hi - lo) / lo * 100, 1) if len(fx) > 1 and lo > 0 else 0.0
    out["accuracy"]["intrinsics"] = {
        "describes": (
            "Per-camera lens solve. Two metrics decide whether it is trustworthy: "
            "coverage = fraction of the frame the board's corners reached (the "
            "edges are what pin fx and distortion; want >0.45, ideally >0.9), and "
            "fx_spread_pct = how far apart the identical lenses' fx are (want "
            "<~2%; a camera whose fx sits far from the others had a weak capture). "
            "The rest are supporting only: rms_px = reprojection error in px, which "
            "checks the corners were detected cleanly but does NOT prove fx is "
            "right (a degenerate capture can be low-rms and wrong); angle_spread = "
            "board tilt variety (low tilt under-constrains fx); n_views = frames "
            "used (context, not quality)."),
        "fx_spread_pct": fx_spread,
        "per_camera": {
            n: {"fx": round(e["K"][0][0], 1), "coverage": e["coverage"],
                "rms_px": round(e["rms_px"], 3),
                "angle_spread": e["angle_spread"], "n_views": e["n_views"]}
            for n, e in solved.items()
        },
    }
    return out


# ------------------------------------------------------------------------ main


def main(args):
    board_json = args.board
    require_dir(args.intrinsics, "--intrinsics", "exo_intrinsics")
    spec = load_json(board_json, "--board", "exo_intrinsics")
    board, dictionary = make_board(spec)
    print(f"[board] {board_json}: {spec.get('squares_x')}x{spec.get('squares_y')} "
          f"squares, {spec.get('square_mm')}mm, {spec.get('dictionary')}, "
          f"print_scale {spec.get('print_scale', 1.0)}")
    if abs(spec.get("print_scale", 1.0) - 1.0) < 1e-9:
        print("[warn] board.json holds the NOMINAL square size -- if the print was "
              "scaled, every distance below is wrong by the same factor.")

    print("[1/1] intrinsics from the 4 camera videos")
    streams = require_exo_cams(video_streams(args.intrinsics), args.intrinsics,
                               "--intrinsics", "exo_intrinsics")
    intr = {}
    for name, path in streams:
        all_vs, size = views(path, board, dictionary, args.stride,
                             args.min_corners)
        if len(all_vs) < 12:
            print(f"  {name:<12} only {len(all_vs)} usable views -- lens NOT "
                  f"calibrated (failed_intrinsics). Re-shoot this camera.")
            intr[name] = {"failed": True}
            continue
        vs = evenly(all_vs, args.max_views)
        objp = [object_points(board, ids).astype(np.float32).reshape(-1, 1, 3)
                for _, ids in vs]
        imgp = [cc for cc, _ in vs]
        # Constrain the lens model to what a square-pixel webcam physically is:
        # fx=fy, no tangential term, and no k3. The walk views never reach the
        # frame edges (measured: 0/24 edge cells), so an unconstrained k3 overfits
        # -- it ran to -34 on one camera and dragged focal length with it, which is
        # what made the focal swing between recordings. Fixing these gives a stable
        # focal and sane distortion; run a dedicated edge-filling clip for the best.
        K0 = np.array([[1500.0, 0, size[0] / 2.0],
                       [0, 1500.0, size[1] / 2.0],
                       [0, 0, 1.0]])
        flags = calib_flags(args.free_k3)
        rms, K, D, _, _ = cv2.calibrateCamera(objp, imgp, size, K0, None,
                                              flags=flags)
        cov, tilt = coverage_and_tilt(vs, size)
        intr[name] = {"K": K.tolist(), "dist": D.ravel().tolist(),
                      "image_size": list(size), "rms_px": float(rms),
                      "n_views": len(vs), "coverage": cov, "angle_spread": tilt}
        # RMS gates detection quality; coverage+tilt gate whether fx is trustworthy.
        flags_txt = []
        if rms >= 1.0:
            flags_txt.append("rms high (focus/flatness)")
        if cov < 0.45:
            flags_txt.append("coverage low (board never reached the edges)")
        if tilt < 0.25:
            flags_txt.append("tilt low (board too flat-on)")
        flag = "" if not flags_txt else "   <-- " + "; ".join(flags_txt)
        print(f"  {name:<12} fx {K[0][0]:6.1f}   {len(vs):>3}/{len(all_vs):<4} "
              f"views   rms {rms:.3f} px   coverage {cov * 100:>3.0f}%   "
              f"tilt {tilt:.2f}{flag}")

    solved = [n for n, e in intr.items() if not e.get("failed")]
    if not solved:
        raise SystemExit("no camera calibrated -- nothing to do")

    out = build_intrinsics_output(intr, spec)
    fx_spread = out["accuracy"]["intrinsics"]["fx_spread_pct"]
    if len(solved) > 1:
        tag = ("GOOD" if fx_spread < 2 else "OK" if fx_spread < 10 else
               "HIGH -- reshoot")
        print(f"\n  fx spread across {len(solved)} cameras: {fx_spread:.1f}% "
              f"[{tag}]")

    # --out is either a name.json FILE or a directory (then calib.json inside it).
    out_arg = os.path.abspath(args.out)
    if out_arg.lower().endswith(".json"):
        dst, out_dir = out_arg, os.path.dirname(out_arg)
    else:
        out_dir, dst = out_arg, os.path.join(out_arg, "calib.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"\n[out] {dst}")
    print("Intrinsics only (K + distortion per camera). Anchor the extrinsics with "
          "exo_extrinsics.py --calib <this file> --camera_positions <positions>.")


if __name__ == "__main__":
    main(parser.parse_args())
