# Exo-camera calibration — setup & use

**What we need from you: about 60 minutes at the rig — two short captures, four
tape measurements, and the folders back.**

This file is everything needed to **set up and run** the calibration.

---

## Set up the environment

Create a conda env on **Python 3.12** (the pinned `numpy 2.5.2` requires ≥3.12)
and install the pinned dependencies:

```bash
conda create -n exo-calib python=3.12 -y
conda activate exo-calib
pip install -r requirements.txt
```

This installs the exact hashed versions in `requirements.txt`
(numpy 2.5.2, opencv-contrib-python-headless 5.0.0.93, pillow 12.3.0).
Run everything below inside this env, from the repository root.

---

## 1. Print the board

```bash
python make_board.py --out board/
```

This writes `board/charuco_a1.pdf` (700 × 500 mm board on an A1 page) and
`board/board.json`.

- Print at **100% scale** / "Actual size". Not "fit to page", not "shrink to
  margins". The PDF carries its true physical size, so an A1 printer set to 100%
  reproduces every square exactly. A2 works but the board will be small in a
  camera four metres away.
- **Mount it on something rigid and flat** — foam board, stiff card, a
  clipboard backing. A board that bends lies about where its corners are, and
  the error goes straight into every measurement afterwards.
- Take a ruler, **measure one printed square**, and record the true value:

```bash
python make_board.py --out board/ --measured_mm 98.4
```

This step matters more than it looks. Printers rescale silently. A 3% scale
error becomes a 3% error in every distance we recover — 3 cm on a metre.

---

## ⚠️ Before you record — assume it drifted

- **Do not trust "nobody touched it."** Even when the rig looks untouched,
  people knock cables and brush past the cameras and bed between sessions —
  millimetres of movement you cannot see.
- That movement goes **straight into the calibration** and silently corrupts
  every distance we recover.
- **Before every session, re-measure each camera's distance from the origin**
  (the bed reference corner, section 3) and the bed's own position, then compare
  against your last `camera_positions.json`.
- If anything changed, **re-measure and update the numbers before recording** —
  a calibration only describes the cameras *where they are now*.

---

## 2. Record — two captures, plus measurements

Record these **exactly the way you record normal episodes** 
- Same rig
- same camera settings: exposure, gain, focus
- same lighting in the room 
- same camera resolution

Calibration has **two jobs that need nearly opposite things**, which is why
there are **two separate captures**: the lenses need the board *big and in one
camera's corners*; the rig needs the board *shared between camera pairs*. Trying
to serve both from one walk clip is exactly what leaves focal length
under-determined.

**Lock the camera focus first.** The C922 autofocus will otherwise change the
lens between shots and invalidate the calibration. Set focus once and use the
**same locked focus for both captures below.**

### Capture 1 — intrinsics (for the lenses) — ~30 s per camera

This capture pins each lens, and it is the one that has gone wrong before, so do
it deliberately — **one camera at a time**:

- Stand close enough that the board **fills a third-to-half of *that* camera's
  frame**.
- **Trace the board around the image border** so its corners reach every edge and
  corner of the frame — top-left, along the top, top-right, down the right edge,
  and so on. Distortion (and focal length) live at the edges; if the board never
  gets there, the lens cannot be measured — this is the single most common cause
  of a bad calibration.
- **Tilt it ±30–45°** as you go — tipped left/right, top/bottom. Flat-on views
  carry almost no information.
- Do it at **two distances** (close, then a step back).
- Move **slowly** — motion blur destroys corner detection.

All four cameras can be recording at once; just make sure **each camera gets its
own close-up pass**. Name this take e.g. `calib_intrinsics`.

### Capture 2 — walk-around (for the rig) — one take, 2–3 minutes

Carry the board slowly through the room so **adjacent camera pairs see it at the
same time** — that shared view is what links the cameras into one rig.

- Cover the whole space, especially around and over the bed.
- Spend time so **each pair of neighbouring cameras** sees the board together.
- Moderate distance is fine here; the board does **not** need to fill the frame.
- Move **slowly**; keep the board fully inside at least one camera's frame at all
  times.

Name this take e.g. `calib_walk`.

**This take is a rosbag — extract it before solving.** Unlike Capture 1 (OBS,
already `.mp4`), the walk is recorded through the ROS rig, so it must be
extracted to per-camera `videos/*.mp4` with the extraction repo (a separate
tool with its own docs) before the calibration can read it. The scripts read
mp4s, not rosbags.

---

## 3. Measurements to write down

Tape-measure **each camera's position `(x, y, z)`** in metres from the bed
reference corner (section 3), into `camera_positions.json`. This is what places
the rig in the room — it replaces the old board-on-the-floor placements entirely.

Everything is measured from **one physical corner of the bed**: the corner of
the **bed frame** (not the mattress) at the **foot end**, on the **side the
demonstrator stands on**. Call it the reference corner. The axes are:

- **x** — along the bed, positive x toward the headboard (positive toward the headboard).
- **y** — if facing the headboard positive y moves toward the right
- **z** — up from the floor.

Measure **each camera's `(x, y, z)`** — to its lens — in metres, and fill them
into `camera_positions.json`:

```json
{
  "exo_cam1": [2.67, 2.315, 1.99],
  "exo_cam2": [2.67, -1.44, 1.99],
  "exo_cam3": [-1.2, -1.44, 1.99],
  "exo_cam4": [-1.2, 2.315, 1.99]
}
```

Measure all four in **one consistent frame** — the calibration fits the rig onto
exactly these points, so a wrong or inconsistent number moves that camera and
nothing else corrects it. Nearest centimetre is fine.

---

## 4. Send back

- both capture folders — `calib_intrinsics` (OBS mp4) and `calib_walk`
  (**extracted** to mp4 via the extraction repo)
- `board/board.json` — **including the measured square size**
- `camera_positions.json` with the four measured camera positions


---

## 5. Run the calibration

Two scripts, run one after the other. Both are deterministic; inputs are **full
paths**. `--out` accepts a `name.json` file or a directory.

**Step 1 — intrinsics** (the lenses, from the dedicated intrinsics capture):

```bash
python exo_intrinsics.py \
  --intrinsics /full/path/to/<session>_calib-intrinsics_1 \
  --board      board/board.json \
  --out        calib/intrinsics-<session>.json
```

Before going further, **check the four `fx` agree within a few percent** (the
script prints `fx spread across N cameras`). A >10% spread means the close
passes weren't close/edge-covering enough — the intrinsics are junk, so
re-shoot rather than proceed.

*Advanced:* `--free_k3` fits the k3 distortion term instead of pinning it to 0.
Leave it off for this rig (its influence is negligible and it can destabilise
`fx`); only use it on a close-range, corner-filling capture where you've checked
it keeps the `fx` spread tight.

**Step 2 — extrinsics** (camera poses from the walk, anchored to the measured
positions, reusing Step 1's intrinsics):

```bash
python exo_extrinsics.py \
  --walk             /full/path/to/<session>_calib-walk_1 \
  --calib            calib/intrinsics-<session>.json \
  --camera_positions /full/path/to/camera_positions-<session>.json \
  --board            board/board.json \
  --out              calib/calib-<session>.json
```

**`--board` is required and never defaulted** — a wrong or missing ruler would
silently rescale every distance.

**Reuse the intrinsics.** Focal length is the metric scale of the whole
reconstruction, and a rig-only walk pins it only weakly (the same lens has come
out `fx≈1450` on one walk and `fx≈2141` on another). To rebuild the rig from a
new walk — a camera moved, the lenses did not — re-run **only Step 2** with the
**same** `--calib intrinsics-<session>.json`. Never let a walk re-solve `fx`.

Step 2 writes the single final `calib-<session>.json`, carrying intrinsics +
extrinsics + all metrics.

---

## 6. Read the output

`calib/calib-<session>.json` has three top-level keys:

- **`scene_details.board`** — the board spec the solve used.
- **`cameras[name]`** — per camera:
  - `status`: one of `"ok"`, `"failed_intrinsics"` (the lens never calibrated —
    too few clean close-up views), or `"failed_extrinsics"` (the lens is fine but
    the camera couldn't be linked into the rig). A camera key is only *absent* if
    that camera did not exist in the capture at all — a failed one is present with
    the failed status.
  - `intrinsics`: `K = [[fx,0,cx],[0,fy,cy],[0,0,1]]`, `dist = k1 k2 p1 p2 k3`,
    `image_size = [1920, 1080]` — or `null` when `status` is `failed_intrinsics`.
  - `extrinsics`: `T_world_cam` and `position_m` when `status` is `ok`, otherwise
    `null`.
    `T_world_cam` is a 4×4 homogeneous transform (an array of 4 rows, each 4
    numbers) that maps a point **from the camera frame into the world (board)
    frame**:

    ```
    [ R R R | tx ]   top-left 3×3 = rotation (the camera's x/y/z axes
    [ R R R | ty ]     expressed in world coordinates → which way it faces)
    [ R R R | tz ]   last column, top 3 = camera position in metres
    [ 0 0 0 |  1 ]   bottom row is [0,0,0,1] padding (makes the 4×4
    ```                 chainable/invertible; carries no camera info)

    So: **where is the camera** → last column top-3 (also mirrored as
    `position_m` for convenience); **which way does it face** → top-left 3×3;
    **bottom row** → ignore. Apply it as `p_world = T_world_cam · [x, y, z, 1]`.
- **`accuracy`** — each sub-block carries a `describes` string explaining its own
  metric, plus:
  - `relative.worst_m` / `per_pair_m` — how tightly the cameras agree with **each
    other** (triangulation precision, 1-sigma). Target ≤ ~3 cm. **Blind to
    scale** — a wrongly-scaled rig can still agree with itself, so a good relative
    does *not* clear the intrinsics.
  - `absolute.rms_m` / `per_camera_m` — how well the rig fits your tape
    (bed-placement accuracy, RMS). The **only** metric that sees a scale error.
    GOOD < 5 cm, OK < 15 cm, suspect > 15 cm.
  - `intrinsics.fx_spread_pct` / `per_camera` — the lens-solve quality carried
    through from Step 1.
  - `cameras_solved` — e.g. `"4/4"`.

The classic failure is **relative small + absolute large**: the rig is internally
consistent but wrongly scaled, which points at the intrinsics `fx`. `per_camera_m`
ranks *which tape measurement to re-check* — a residual > ~2× the others (or
> 15 cm) is the one to re-measure. Tape-measuring high cameras has a ~5 cm floor.

---

## Common ways this goes wrong

| Symptom | Cause |
|---|---|
| Board barely detected | printed too small, or motion blur — slow down, print bigger |
| High reprojection error | board not flat (mounted on something floppy), or the printed square was never measured |
| Whole rig too big/small (absolute error tens of cm, but cameras agree with each other) | bad lens focal length — the intrinsics capture didn't fill the frame edges. Re-shoot Capture 1 up close, into the corners |
| One camera position off | that camera's tape measurement is wrong — `exo_extrinsics.py` prints the per-camera residual, so you can tell which |
| Everything looks fine but is subtly off | print scale. Measure the square. |
