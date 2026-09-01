"""Generate the printable ChArUco board, plus the board.json that describes it.

Run this once, print the PNG at 100% scale, then MEASURE a printed square with
a ruler and pass the true value back in --measured_mm. Printers scale silently,
and a 3% scale error becomes a 3% error in every distance we recover from the
cameras — which is 3 cm on a metre.

    python make_board.py --out board/
    # print board/charuco_a1.pdf at 100% / "Actual size", measure one square:
    python make_board.py --out board/ --measured_mm 98.4

The board is 7x5 squares of 100 mm = 700 x 500 mm, which fits A1 with a margin.
Bigger is better: these cameras are several metres from the bed, and a board
that spans only a few dozen pixels gives a pose no better than a guess.
"""

import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image

from board import DEFAULT, make_board, save_spec

# A1 is the target print size: 594 x 841 mm. The board is placed on it at its
# TRUE physical size, centred, so a printer set to "Actual size" (100%, NOT
# "fit to page") reproduces every square at exactly its nominal millimetre width.
A1_MM = (594.0, 841.0)


def save_true_size_pdf(path, sheet_bgr, dpi):
    """Write sheet_bgr onto an A1 page at 100% physical scale.

    The board's pixel dimensions were computed as mm * dpi / 25.4, so pasting it
    unscaled onto an A1 canvas built at the SAME dpi makes its printed size exact:
    physical_mm = pixels / dpi. No resampling happens anywhere, which is the whole
    point -- a silent rescale here becomes a metric error in every later distance.
    """
    px_per_mm = dpi / 25.4
    sheet_h, sheet_w = sheet_bgr.shape[:2]
    sheet_w_mm, sheet_h_mm = sheet_w / px_per_mm, sheet_h / px_per_mm

    # Choose the A1 orientation the board fits in; fail loudly if it fits neither
    # rather than silently shrinking it.
    for page_w_mm, page_h_mm in (A1_MM, A1_MM[::-1]):
        if sheet_w_mm <= page_w_mm and sheet_h_mm <= page_h_mm:
            break
    else:
        raise SystemExit(
            f"board is {sheet_w_mm:.0f} x {sheet_h_mm:.0f} mm and does not fit "
            f"A1 ({A1_MM[0]:.0f} x {A1_MM[1]:.0f} mm). Use fewer/smaller squares "
            f"or print on a larger sheet.")

    page_w = int(round(page_w_mm * px_per_mm))
    page_h = int(round(page_h_mm * px_per_mm))
    canvas = np.full((page_h, page_w, 3), 255, np.uint8)
    ox = (page_w - sheet_w) // 2
    oy = (page_h - sheet_h) // 2
    canvas[oy:oy + sheet_h, ox:ox + sheet_w] = sheet_bgr

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    # resolution= stamps the DPI into the PDF, which is what fixes the page's
    # physical size (page_mm = page_px / dpi) for a 100% print.
    Image.fromarray(rgb).save(path, "PDF", resolution=float(dpi))
    return page_w_mm, page_h_mm

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="board")
parser.add_argument("--squares_x", type=int, default=DEFAULT["squares_x"])
parser.add_argument("--squares_y", type=int, default=DEFAULT["squares_y"])
parser.add_argument("--square_mm", type=float, default=DEFAULT["square_mm"])
parser.add_argument("--marker_mm", type=float, default=DEFAULT["marker_mm"])
parser.add_argument("--dictionary", default=DEFAULT["dictionary"])
parser.add_argument("--dpi", type=int, default=300)
parser.add_argument("--measured_mm", type=float, default=None,
                    help="the ACTUAL width of one printed square, measured "
                         "with a ruler. Rewrites board.json to match; the "
                         "image is not regenerated.")
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)
spec_path = os.path.join(args.out, "board.json")

if args.measured_mm is not None:
    if not os.path.exists(spec_path):
        raise SystemExit(f"no {spec_path} yet — generate the board first")
    with open(spec_path) as fh:
        spec = json.load(fh)
    nominal = spec.get("nominal_square_mm", spec["square_mm"])
    scale = args.measured_mm / nominal
    spec["square_mm"] = args.measured_mm
    spec["marker_mm"] = spec.get("nominal_marker_mm",
                                 spec["marker_mm"]) * scale
    spec["print_scale"] = scale
    save_spec(spec_path, spec)
    print(f"[ok] measured square {args.measured_mm} mm vs nominal {nominal} mm"
          f"  -> print scale {scale:.4f}")
    print(f"     marker length corrected to {spec['marker_mm']:.2f} mm")
    if abs(scale - 1.0) > 0.05:
        print("[warn] that is more than 5% off. Check the print was at 100% "
              "and not 'fit to page'.")
    raise SystemExit(0)

spec = {
    "squares_x": args.squares_x,
    "squares_y": args.squares_y,
    "square_mm": args.square_mm,
    "marker_mm": args.marker_mm,
    "nominal_square_mm": args.square_mm,
    "nominal_marker_mm": args.marker_mm,
    "dictionary": args.dictionary,
    "print_scale": 1.0,
}
board, _ = make_board(spec)

px_per_mm = args.dpi / 25.4
w = int(round(args.squares_x * args.square_mm * px_per_mm))
h = int(round(args.squares_y * args.square_mm * px_per_mm))
margin = int(round(10 * px_per_mm))          # 10 mm quiet zone
img = board.generateImage((w, h), marginSize=0)
if img.ndim == 2:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Side gutters (OUTSIDE the 10 mm quiet zone, which must stay clear for the
# detector) each carry a tall "this way up" arrow up the edge of the board. The
# board's world orientation doesn't affect the solve -- the rig is anchored on
# measured CAMERA positions, not the board's pose -- so this is purely an
# ergonomic cue that keeps the operator holding the board one consistent way up.
gut = int(round(26 * px_per_mm))              # side gutter width, for the arrows
bottom = int(round(20 * px_per_mm))           # bottom caption strip
sheet = np.full((h + 2 * margin + bottom, w + 2 * margin + 2 * gut, 3),
                255, np.uint8)
bx, by = gut + margin, margin                 # the board's top-left on the sheet
sheet[by:by + h, bx:bx + w] = img

# a tall up-arrow in each gutter, spanning most of the board height
thick = max(3, int(round(1.5 * px_per_mm)))
for gx in (gut // 2, sheet.shape[1] - gut // 2):
    cv2.arrowedLine(sheet, (gx, by + h - int(20 * px_per_mm)),
                    (gx, by + int(20 * px_per_mm)),            # points UP
                    (0, 0, 0), thick, tipLength=0.03)

# "THIS WAY UP" rotated to read up the left gutter, beside its arrow
label = np.full((int(round(9 * px_per_mm)), int(round(80 * px_per_mm)), 3),
                255, np.uint8)
cv2.putText(label, "THIS WAY UP", (int(2 * px_per_mm), int(7 * px_per_mm)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.95 * px_per_mm / 4, (0, 0, 0), 2)
label = cv2.rotate(label, cv2.ROTATE_90_COUNTERCLOCKWISE)
lh, lw = label.shape[:2]
ly, lx = by + (h - lh) // 2, gut // 2 + int(2 * px_per_mm)
sheet[ly:ly + lh, lx:lx + lw] = label

cv2.putText(sheet, "ORIGIN CORNER", (bx, by - int(2 * px_per_mm)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9 * px_per_mm / 4, (0, 0, 0), 2)
cv2.putText(sheet,
            f"LEO calib  {args.squares_x}x{args.squares_y}  "
            f"square {args.square_mm:g}mm  marker {args.marker_mm:g}mm  "
            f"{args.dictionary}  PRINT AT 100%",
            (bx, by + h + int(13 * px_per_mm)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75 * px_per_mm / 4, (0, 0, 0), 2)

pdf = os.path.join(args.out, "charuco_a1.pdf")
page_w_mm, page_h_mm = save_true_size_pdf(pdf, sheet, args.dpi)
save_spec(spec_path, spec)

print(f"[out] {pdf}   board {args.squares_x * args.square_mm:.0f} x "
      f"{args.squares_y * args.square_mm:.0f} mm on A1 "
      f"({page_w_mm:.0f} x {page_h_mm:.0f} mm) at {args.dpi} dpi")
print(f"[out] {spec_path}")
print()
print("Next: print at 100% / 'Actual size' (NOT 'fit to page'), on A1 stock.")
print("Mount it on something rigid")
print("and flat — foam board or stiff card. A board that bends is a board that")
print("lies about where its corners are. Then measure one printed square and")
print(f"run:  python make_board.py --out {args.out} --measured_mm <value>")
