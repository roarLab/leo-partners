#!/usr/bin/env python3
# ==============================================================================
# groundtruth_check.py
#
# One-shot ground-truth check for the offline aligner (rosbag_process_depth_v3.py).
# NOT a pytest unit test -- named *_check (not *_test) on purpose so pytest never
# collects it, because it runs against a real recorded hardware bag. It owns the
# whole A/B comparison, because only a specially-recorded bag carries a
# ground-truth topic (the production aligner has NO validation code).
#
# WHAT IT CHECKS (and what "ground truth" really is)
#   The offline aligner is a faithful transcription of librealsense's GENERIC
#   (scalar) rs2::align. The SDK's live `aligned_depth_to_color` in the bag is
#   produced by the SDK's SSE-optimized align path (align_sse on x86), which
#   rounds the sub-pixel projection slightly differently. Because the D435i depth
#   (848x480) is upsampled to color (1280x720), each depth pixel's fill-rect is
#   only ~1.5 px, so those tiny projection differences reshuffle which depth
#   pixel wins the min-z at rect boundaries: ~1 mm on smooth surfaces, metres at
#   occlusion edges. Both results are valid copies of real depth -- verified: at
#   every disagreeing pixel BOTH values exist in the raw depth frame, and no
#   filtering is applied. So the SDK output is NOT a bit-exact oracle; a 0-mm
#   match is not achievable against an SSE build.
#
#   Therefore this test does not demand bit-exactness. It measures, separately:
#     - FIDELITY  : on pixels where BOTH sides have depth, the distribution of
#                   |diff| (median / p95 / p99). A real bug (wrong scale, units,
#                   extrinsics, column-major order, resolution) shifts the MEDIAN
#                   off 0 -- that is the load-bearing check.
#     - COVERAGE  : do both sides fill the same pixels (IoU). Catches gross
#                   resolution / half-frame faults; tolerates rect-edge px.
#   Values of 0 and 65535 (the SDK invalid marker) are treated as "no depth".
#
# THE SPECIAL RECORDING (do this ONCE)
#   Record with the RealSense node's `align_depth.enable:=True`, which ADDS the
#   live-aligned topic on top of the raw streams. Production bags use =False
#   (no ground-truth topic) -- that is why the offline aligner exists.
#
# PASS GATE  (all thresholds are config below, set with margin over a good bag)
#   median_abs_mm <= MEDIAN_MAX_MM     # required correctness anchor (0 = bulk exact)
#   p95_abs_mm    <= P95_MAX_MM
#   p99_abs_mm    <= P99_MAX_MM
#   coverage_iou  >= COVERAGE_IOU_MIN
#   overlap_px    >= MIN_OVERLAP_PX    # guards a bag that paired nothing
#   (max_abs_mm is reported for info only -- one occlusion-edge swap is metres.)
#
# USAGE
#   python groundtruth_check.py       # from data-processing-scripts-v3/
# ==============================================================================
import sys
from pathlib import Path

import numpy as np
import h5py
import pandas as pd
from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage

import rosbag_process_depth_v3 as ead

# --- CONFIG -------------------------------------------------------------------
BAG = "/home/user/leo_ws/2026-08-03_depth-groundtruth2"  # recorded with align_depth.enable:=True
CAMERA = "ego"
OUT_DIR = "/home/user/test-output"          # scratch output root (isolated from prod)
REPORT_FILEPATH = "groundtruth-report.xlsx"

RUN_ALIGN = True        # True: (re)align the bag first (tests current code); False: reuse OUT_DIR h5
LIMIT = 0               # cap color frames when aligning (0 = all); for a quick smoke test
INVALID_VAL = 65535     # SDK invalid marker; treated as "no depth" (same as 0)

# PASS thresholds, calibrated with margin over the known-good bag
# 2026-07-31_depth-groundtruth-test (full run: median 0, p95 11mm, p99 38mm,
# IoU 0.991, mean 2.8mm). median==0 is the real correctness anchor; the rest are
# guardrails a genuine regression (median off 0, p95/p99 in the hundreds) blows past.
MEDIAN_MAX_MM = 0       # required correctness anchor (bulk bit-exact)
P95_MAX_MM = 25         # good bag ~11
P99_MAX_MM = 100        # good bag ~38
COVERAGE_IOU_MIN = 0.97 # good bag ~0.991
MIN_OVERLAP_PX = 1_000_000

HIST_MAX_MM = 1000      # |diff| histogram range for percentiles; above -> overflow bin
PAIR_TOLERANCE_MS = 16.0  # SDK-frame <-> color-frame window for indexing into our color-indexed h5

# topic discovery (None = auto-discover via suffix + camera)
ALIGNED_SUFFIX = "/aligned_depth_to_color/image_raw"
ALIGNED_TOPIC = "/ego/d435i_ego/aligned_depth_to_color/image_raw"
COLOR_SUFFIX = "/color/image_raw"
COLOR_TOPIC = None


def _discover(reader, suffix, override, kind) -> str:
    if override:
        return override
    return ead.pick_topic(list(reader.connections), suffix, CAMERA, None, kind)


def _pct_from_hist(hist: np.ndarray, total: int, p: float) -> int:
    """Smallest 1-mm bin whose cumulative count reaches the p-th percentile.
    Returns mm; HIST_MAX_MM+1 means '> HIST_MAX_MM' (overflow bin)."""
    if total <= 0:
        return 0
    return int(np.searchsorted(np.cumsum(hist), p / 100.0 * total))


def compare(h5_path: Path, bag: Path, sdk_topic: str, color_topic: str):
    """Diff our color-indexed h5 against the SDK aligned topic, frame by frame.
    Returns a metrics dict, or None on a hard resolution mismatch."""
    tol_s = PAIR_TOLERANCE_MS / 1000.0
    hist = np.zeros(HIST_MAX_MM + 2, dtype=np.int64)   # bins 0..HIST_MAX_MM + overflow
    both = only_sdk = only_ours = 0
    sum_abs = 0.0
    max_abs = 0
    n_frames = n_blank_skipped = n_unpaired = 0

    with h5py.File(h5_path, "r") as h5:
        dset = h5["data"]
        n_color = dset.shape[0]
        with AnyReader([bag], default_typestore=ead.typestore) as reader:
            cstamps = ead.collect_stamps(reader, color_topic, 1)
            order = np.argsort(cstamps, kind="stable")
            sstamps = cstamps[order]

            conns = [c for c in reader.connections if c.topic == sdk_topic]
            for conn, _t, raw in reader.messages(connections=conns):
                msg = reader.deserialize(raw, conn.msgtype)
                ss = ead._stamp_s(msg.header)
                j = int(np.searchsorted(sstamps, ss))
                best_k, best = -1, np.inf
                for k in (j - 1, j):
                    if 0 <= k < len(sstamps):
                        dt = abs(sstamps[k] - ss)
                        if dt < best:
                            best, best_k = dt, k
                if best_k < 0 or best > tol_s:
                    n_unpaired += 1                       # SDK frame with no color partner
                    continue
                cidx = int(order[best_k])
                if cidx >= n_color:
                    n_unpaired += 1
                    continue
                sdk = message_to_cvimage(msg).astype(np.int32)
                ours = dset[cidx].astype(np.int32)
                if sdk.shape != ours.shape:
                    print(f"[FAIL] shape mismatch {sdk.shape} vs {ours.shape}")
                    return None
                vsdk = (sdk > 0) & (sdk != INVALID_VAL)   # 0 and 65535 -> no depth
                vours = (ours > 0) & (ours != INVALID_VAL)
                ov = vsdk & vours
                if not ov.any():
                    n_blank_skipped += 1                  # one/both sides empty: pairing, not align
                    continue
                both += int(ov.sum())
                only_sdk += int((vsdk & ~vours).sum())
                only_ours += int((vours & ~vsdk).sum())
                d = np.abs(sdk[ov] - ours[ov])
                sum_abs += float(d.sum())
                max_abs = max(max_abs, int(d.max()))
                hist += np.bincount(np.minimum(d, HIST_MAX_MM + 1),
                                    minlength=HIST_MAX_MM + 2)
                n_frames += 1

    base = {"frames_compared": int(n_frames), "blank_skipped": int(n_blank_skipped),
            "unpaired_sdk": int(n_unpaired), "overlap_px": int(both),
            "only_sdk_px": int(only_sdk), "only_ours_px": int(only_ours),
            "max_abs_mm": int(max_abs)}
    if both == 0:
        base.update({"coverage_iou": 0.0, "exact_pct": 0.0, "within_1mm_pct": 0.0,
                     "within_5mm_pct": 0.0, "median_abs_mm": 0, "p95_abs_mm": 0,
                     "p99_abs_mm": 0, "mean_abs_mm": 0.0})
        return base
    base.update({
        "coverage_iou": round(both / (both + only_sdk + only_ours), 4),
        "exact_pct": round(100.0 * hist[0] / both, 3),
        "within_1mm_pct": round(100.0 * hist[:2].sum() / both, 3),
        "within_5mm_pct": round(100.0 * hist[:6].sum() / both, 3),
        "median_abs_mm": _pct_from_hist(hist, both, 50),
        "p95_abs_mm": _pct_from_hist(hist, both, 95),
        "p99_abs_mm": _pct_from_hist(hist, both, 99),
        "mean_abs_mm": round(sum_abs / both, 4),
    })
    return base


def _write_report(metrics: dict, verdict: str, reasons: list) -> None:
    row = {"bag": BAG, "camera": CAMERA, "verdict": verdict,
           "reasons": "; ".join(reasons), **metrics,
           "median_max_mm": MEDIAN_MAX_MM, "p95_max_mm": P95_MAX_MM,
           "p99_max_mm": P99_MAX_MM, "coverage_iou_min": COVERAGE_IOU_MIN,
           "min_overlap_px": MIN_OVERLAP_PX}
    pd.DataFrame([row]).to_excel(REPORT_FILEPATH, index=False)
    print(f"[groundtruth] wrote {REPORT_FILEPATH}")


def gate(m: dict) -> list:
    """Return the list of failed conditions (empty == PASS)."""
    reasons = []
    if m["overlap_px"] < MIN_OVERLAP_PX:
        reasons.append(f"overlap_px={m['overlap_px']} < {MIN_OVERLAP_PX}")
    if m["median_abs_mm"] > MEDIAN_MAX_MM:
        reasons.append(f"median_abs_mm={m['median_abs_mm']} > {MEDIAN_MAX_MM}")
    if m["p95_abs_mm"] > P95_MAX_MM:
        reasons.append(f"p95_abs_mm={m['p95_abs_mm']} > {P95_MAX_MM}")
    if m["p99_abs_mm"] > P99_MAX_MM:
        reasons.append(f"p99_abs_mm={m['p99_abs_mm']} > {P99_MAX_MM}")
    if m["coverage_iou"] < COVERAGE_IOU_MIN:
        reasons.append(f"coverage_iou={m['coverage_iou']} < {COVERAGE_IOU_MIN}")
    return reasons


def main() -> int:
    bag = Path(BAG)
    if not bag.exists():
        print(f"[FAIL] bag not found: {bag}")
        return 1

    with AnyReader([bag], default_typestore=ead.typestore) as reader:
        try:
            sdk_topic = _discover(reader, ALIGNED_SUFFIX, ALIGNED_TOPIC, "aligned")
            color_topic = _discover(reader, COLOR_SUFFIX, COLOR_TOPIC, "color")
        except SystemExit as e:
            print(f"[FAIL] topic discovery: {e}")
            print("       Did you record with align_depth.enable:=True? "
                  "Else set ALIGNED_TOPIC / COLOR_TOPIC explicitly.")
            return 1
    print(f"[groundtruth] SDK ground-truth topic = {sdk_topic}")
    print(f"[groundtruth] color topic (pairing)  = {color_topic}")

    stem = f"{CAMERA or 'cam'}_aligned_depth_to_color"
    h5_path = Path(OUT_DIR) / "depth_frames" / f"{stem}.h5"
    if RUN_ALIGN:
        ead.LIMIT = LIMIT
        ead.main(bag=bag, out_dir=OUT_DIR, camera=CAMERA)
    if not h5_path.exists():
        print(f"[FAIL] aligned h5 not found: {h5_path} (set RUN_ALIGN=True)")
        return 1

    m = compare(h5_path, bag, sdk_topic, color_topic)
    if m is None:
        _write_report({}, verdict="FAIL", reasons=["resolution mismatch"])
        return 1

    reasons = gate(m)
    verdict = "PASS" if not reasons else "FAIL"

    print("\n=========== GROUND-TRUTH TEST (vs SDK SSE align) ===========")
    print(f"  bag                  : {bag}")
    print(f"  frames compared      : {m['frames_compared']}  "
          f"(blank-skipped {m['blank_skipped']}, unpaired-sdk {m['unpaired_sdk']})")
    print(f"  overlap pixels       : {m['overlap_px']:,}")
    print("  -- fidelity (overlapping depth pixels) --")
    print(f"  median |diff|        : {m['median_abs_mm']} mm   (<= {MEDIAN_MAX_MM})")
    print(f"  p95 |diff|           : {m['p95_abs_mm']} mm   (<= {P95_MAX_MM})")
    print(f"  p99 |diff|           : {m['p99_abs_mm']} mm   (<= {P99_MAX_MM})")
    print(f"  within 1mm / 5mm     : {m['within_1mm_pct']}% / {m['within_5mm_pct']}%")
    print(f"  exact / mean / max   : {m['exact_pct']}% / {m['mean_abs_mm']} mm / "
          f"{m['max_abs_mm']} mm (max = info only)")
    print("  -- coverage --")
    print(f"  IoU                  : {m['coverage_iou']}   (>= {COVERAGE_IOU_MIN})")
    print(f"  only-SDK / only-ours : {m['only_sdk_px']:,} / {m['only_ours_px']:,} px")
    print(f"  VERDICT              : {verdict}")
    for r in reasons:
        print(f"    - {r}")
    print("===========================================================")

    _write_report(m, verdict=verdict, reasons=reasons)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
