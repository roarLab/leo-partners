# Readme

How to **configure, run, and troubleshoot** the rosbag extraction/validation pipeline.
This is the *operator* doc: get a session processed and fix the errors you hit.

- **Machine / Python environment** (conda + dependency lockfile) → [../readme.md](../readme.md).

One **bag** = one **episode**. A **session** = a folder of episode bags. You process a whole
session (or several) in one run.

> **Versioning:** `v3` in the scripts/filenames is the version of the **extraction pipeline**,
> not the dataset. The dataset version is tracked separately (in each episode's `metadata.json`).

---

## 1. Prerequisites

Create the project conda env (**Python 3.12**) and install the locked dependencies:

```bash
conda create -n leo-process python=3.12 -y
conda activate leo-process
pip install pip-tools
pip-sync requirements.txt        # installs exactly the locked deps
# or 
pip install -r requirements.txt
```

Full dependency workflow (updating/adding packages) → [../readme.md](../readme.md).

- **No ROS install needed.** The pipeline reads bags with the pure-Python `rosbags[image]` library
  (`AnyReader`), not `rclpy`/`cv_bridge` — so it runs on macOS. (The ROS2-Humble note in the
  root readme is about a *different*, upstream splitting step, not this pipeline.)
- If you get `ModuleNotFoundError` for the reader, the color script prints the exact fix:
  `python -m pip install "rosbags[image]" opencv-python numpy pandas`.

---

## 2. There is no CLI — you configure by editing `wrapper.py`

The pipeline takes **no command-line arguments**. All configuration lives in the
`if __name__ == "__main__":` block at the bottom of [wrapper.py](wrapper.py). You edit that
block, then run the file. Everything below refers to that block.

### 2a. The run list `ls` — the one thing you must set every time

Each row is one session. Fields:

```python
ls = [
    # (session_dir, output_root, meta_override, cameras_override, calib_override)
    ("/Volumes/TRANSCEND/session-2026-08-20", "./out/session-2026-08-20", None, None, None),
]
```

| Field | Meaning |
|---|---|
| `session_dir` | Folder of episode-bag subfolders. Each episode nests under `output_root/<episode>`. (A single bag folder also works → writes straight into `output_root`.) |
| `output_root` | Where extracted videos/timestamps/metadata/summary are written. |
| `meta_override` | `None` → use `DEFAULT_META`; or a partial dict to override per session. |
| `cameras_override` | `None` → use `DEFAULT_CAMERAS`; or a partial patch (see 2c). |
| `calib_override` | `None` → use `DEFAULT_CALIB_JSON`; or a per-session calib path. |

Add more rows to batch multiple sessions in one run. Every row is pre-flight-checked
**before any bag opens** (see §4).

### 2b. The stream layout `DEFAULT_CAMERAS` — set once for your rig

Declares what the rig recorded. Streams are *discovered in the bag by topic suffix*.

- `ego` — the D435i color stream, labelled `cam_ego` (`singleton: True` → one topic). The rig
  records ego color **compressed** (JPEG), so the default is
  `suffix: ".../color/image_raw/compressed", compressed: True`. For an older **raw** ego bag,
  switch that group back to `suffix: ".../color/image_raw", compressed: False`. Depth aligns to
  whichever the bag carries (raw or compressed) automatically — no depth edit needed.
- `exo` — the webcams. `ids: [1, 2, 3, 4]` is the **set of device numbers to expect**:
  - a declared id **not found** in the bag → `missing_stream` flag;
  - a found id **not declared** → `extra_stream` flag.
  - To add a webcam cleanly, add its id. To skip one (e.g. cam3 is not used), omit it: `[1, 2, 4]`.
- `depth`, `imu` — the ego non-color streams, `present: True` (assume the rig recorded them).

`present: False` on any stream **skips its extraction, its sanity requirement, and its
validator, and does NOT flag its absence** (its absence was declared, not lost). Left
`present: True`, a missing/empty topic is flagged `missing_stream`.

### 2c. Per-session override (don't edit the defaults for a one-off)

To change one session without touching the canonical defaults, use the override fields in its
`ls` row:

```python
# Session recorded with no IMU, and a different exo label:
("…/sess-A", "./out/sess-A", None,
 {"imu": {"present": False}, "exo": {"label": "side_cam"}}, None),
```

`cameras_override` is a shallow per-group patch merged onto `DEFAULT_CAMERAS`; `meta_override`
is a shallow patch onto `DEFAULT_META`.

### 2d. Calibration `DEFAULT_CALIB_JSON`

Points at the session's exo calibration file. The calib step **merges** those intrinsics/
extrinsics into each episode's `metadata.json`. **Exo calibration is mandatory** — set this to
your real `calib-<date>.json` (or override per session via the 5th `ls` field). The shipped
`"filepath.json"` is a placeholder; leave it unset/wrong and the run **fails loudly at
startup**, it does not silently skip.

Two failure gates enforce this:

- **File gate (config-plane, pre-flight):** `validate_calib_rows` resolves every session's
  calib path before any bag opens. A path that is unset, missing, or malformed JSON **fails the
  whole batch at startup** — a loud config crash, naming the offending session.
- **Coverage gate (data-plane, per-episode):** during the merge, every **recorded** exo camera
  must have an entry in the calib file. A recorded exo cam with **no calib entry fails that
  episode** (a per-bag crash + `pipeline_error.txt`; the batch continues). A cam that was
  *expected but never recorded* is a separate `missing_stream` flag, not a calib failure.

One tolerated shortfall: a camera **present in the calib but with a failed solve**
(`status != "ok"`) is **not** a failure — its `K`/`D`/pose are written `null` with the `status`
carried to explain why, plus an informational `steps.exo_calib_notes` entry. A bad solve limits
downstream 3D use but does not corrupt the recorded data, so it does not fail the episode.

### 2e. Other knobs

- `CAMERA = "ego"` — substring identifying the ego D435i. Leave unless the rig id changes.
- `QUIET` — `False`: per-bag step chatter prints inline with wrapper progress. `True`: chatter
  goes to `<out_dir>/pipeline.log`, console shows only wrapper progress.
- `DEFAULT_META` — descriptive fields written into `metadata.json["metadata"]` (dataset name/
  version, environment, subject, bed_type, …). Fill `subject` etc. for the session.
- Advanced depth knobs (`rpd.PAIR_TOLERANCE_MS`, `rpd.HOLE_FILL`) — commented out; uncomment to change.

---

## 3. Run it

```bash
conda activate leo-process
cd data-processing-scripts-v3
python wrapper.py
```

Per session, the wrapper runs every bag through the step loop, writes per-episode outputs, then
writes one summary. Outputs under each `output_root/<episode>/`:

| Path | What |
|---|---|
| `videos/cam_ego.mp4`, `videos/exo_cam<N>.mp4` | extracted color video per camera |
| `depth_frames/…​.h5`, `timestamps/…​.csv` | aligned depth + per-stream timestamps |
| `imu/cam_ego_imu.csv` | imu samples |
| `metadata.json` | the spine: metadata, intrinsics/extrinsics, stream records, termination verdict |
| `pipeline.log` | step chatter (only when `QUIET=True`) |
| `pipeline_error.txt` | written only if sanity finds a declared file missing |

Per session: `session-summary.csv` — one row per episode, error columns as booleans. This is
your first stop for "did the session come out clean?" (column meanings in
[PIPELINE.md](PIPELINE.md) §5).

---

## 4. Troubleshooting

Two failure classes, handled oppositely — knowing which you're looking at is half the fix.

### 4a. Config-plane crashes — the whole batch aborts before any bag opens

`validate_camera_configs()` pre-flights every `ls` row at startup. A self-contradictory config is
an **operator mistake**, so it crashes **loudly and early** (a `ValueError`) rather than part-
way through. The three causes and their fixes:

| `ValueError` | Cause | Fix |
|---|---|---|
| depth present while ego color absent | You set `ego present: False` but left `depth present: True`. Depth aligns *onto* ego color — it can't run without it. | Either turn depth off too, or keep ego color on. |
| missing required color group | `ego` or `exo` group removed/renamed away — the color script subscripts them. | Restore the `ego` and `exo` keys in `DEFAULT_CAMERAS` (use `present: False` to disable, don't delete the key). |
| all-off config | ego color, exo color, depth, imu **all** `present: False` — nothing to extract. | Turn at least one stream on. |

Note: **imu is not constrained against color** (it's a standalone stream). Turning imu off
alone is fine.

### 4b. Data-plane flags — the run finished, but the data was flagged

A *valid* config met an *imperfect* bag (a declared camera wasn't in it, a stream lost frames,
timestamps gapped). These **flag and continue** — no crash. You read them back from
`session-summary.csv`:

- `completed = True` but an error column tripped → the pipeline ran fine, but that stream has a
  data-quality problem. Open the episode's `metadata.json` for the full message (the CSV cell is
  just a boolean).
- `missing_stream_error` / `extra_stream_error` → a declared stream wasn't in the bag, or an
  undeclared one was. Check the id set in `exo.ids` against what the rig actually recorded.
- `exo_calib` is an **enum** (`full` / `partial` / `none_usable` / `null`), not a boolean — it
  reports how much exo calibration merged. `null` usually means `DEFAULT_CALIB_JSON` points at a
  non-existent file (calibration off).
- `completed = False` → a step **crashed** or a declared file went missing. Look for
  `pipeline_error.txt` in the episode folder, and (if `QUIET=True`) `pipeline.log`.

Full error taxonomy — every trigger, the token it writes, and whether it flips `completed` —
is in [PIPELINE.md](PIPELINE.md) §2.
