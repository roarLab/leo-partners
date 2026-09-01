# Operating the Multi-Camera Rig

## Primer — read this first

**What this is:** launch + record scripts for a 5-camera rig, into one ROS bag
per take. This file is the operator runbook — how to set up a machine and run
recording days. The companion [maintenance.md](maintenance.md) is for *editing*
the code.

> **Run every command from the repo root** — the directory holding this file.
> `cd` there in each terminal you open. Every path in this guide is relative to
> it: the `./` scripts, `leo.perspective`, and the `<date>_<code>_N` bag folders
> all live here. (The `./` scripts self-source ROS and the workspace, so you
> never `source install/setup.bash` yourself, and a bag lands next to this file.)
> To run from elsewhere, give a full path — e.g.
> `~/repos/leo-data-collection/cameras.sh` — or set `BAG_ROOT` for where bags go.

**The mental model — four nouns, nested:**

- **day** — the cameras are up and stay up; a day holds one or more sessions.
- **session** — the set of **episodes** covering one workflow. You record all of
  a session's episodes together, then move to the next session. Multiple
  sessions in a day.
- **episode** — one recording within a session. You give it a short **code**
  (e.g. `23`); a retake reuses the code and gets the next number.
- **bag** — the file one episode records into, named `<date>_<code>_<N>`
  automatically. You never name or number it yourself.

> The bag name carries the **date, code and take number** — not the session.
> A session is how *you* group the episodes you shoot for one workflow; 

**The flow, once per machine then once per day:**

```
section 0   INSTALL (once per machine) — ROS, apt, OBS, verify SDK 2.57
   │
   ▼  ─────────────── each recording day ───────────────
1  PLUG IN + confirm 5 cameras                  (§1)
2  SWAP/ADD a camera — only if hardware changed  (§2)  → edit launch → rebuild
3  TUNE for the light — sunny / rainy / new room (§3)  → edit launch → rebuild
4  ./rebuild.sh   after any edit                 (§4)
5  ./cameras.sh   terminal 1, leave up           (§5)
6  rqt (leo.perspective)  terminal 2, leave up    (§6)
   │
   ▼  ── per session (a workflow), terminals 1 & 2 stay up ──
   │     ▼  ── per episode in the session ──
7  ./record.sh <code>   record one bag, Ctrl-C to stop   (§7a)
   ./oops.sh <bag> <code>   flag a bad take (optional)   (§7b)
   │     └─ repeat for each episode; next session = keep going, same cameras
8  inspect_bag   sanity-check the bag (res/fps/imu) before teardown  (§8)
```

**Four key points that bite people:**

- **Nothing runs from `src/`.** Every script runs `install/`. Edit a launch
  file and it changes *nothing* until `./rebuild.sh` (§4).
- **The launch refuses to start on a real problem** (wrong USB link, wrong depth
  units, missing/held camera) and names it. Read terminal 1, not the script.
- **`Ctrl-C` to stop a recording, then let it close.** Closing the window
  (SIGHUP) leaves the bag unreadable.
- **`ros2 bag info` cannot show C922 resolution** — the frames are opaque JPEG.
  `inspect_bag` (§8) is the post-hoc sanity check: resolution, framerate and IMU
  against the rig spec, read from the bytes on disk.

---

## The rig

- **1× RealSense D435i** — the **ego** camera. Colour 1280×720@30 and depth
  848×480@30, both recorded **raw** (depth is not aligned to colour on the rig;
  that alignment is done offline). Its **IMU is on** — gyro and accel at 200 Hz.
  Needs a USB 3 (5 Gbps) link.
- **4× Logitech C922** — the **exo** cameras, `c922_1`…`c922_4`. Colour
  1280×720@30, MJPEG, recorded compressed. USB 2 (480 Mbps) is correct for
  these.

That is **6 image streams plus the IMU** in one bag.

**Three terminals.** Two run scripts and stay up all day; the third comes and
goes once per episode.

| | Terminal | Runs | Lifetime |
|---|---|---|---|
| 1 | cameras | `./cameras.sh` | up for the whole day |
| 2 | monitor | `rqt` (leo.perspective) | up for the whole day |
| 3 | recorder | `./record.sh <code>` | one episode, then it ends |

---

## 0. First-time machine setup

One-time, per machine. Skip the whole section on a machine that already streams.

### 0a. Install ROS 2 Humble

Follow the official guide through the "Install ROS 2 Packages" step:
https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html#install-ros-2-packages

The rig is built and tested on this stack. These versions are **pinned for
stability** — do not upgrade them because a newer release exists.

| | Version |
|---|---|
| OS | Ubuntu 22.04.5 LTS (jammy) |
| Kernel | 6.8.0-x-generic (Ubuntu HWE on jammy) |
| ROS | Humble |
| librealsense2 | >= **2.57.7** (apt) |
| realsense2_camera | >= **4.57.7** (apt) |

### 0b. Install everything else via apt

```bash
sudo apt update
sudo apt install ros-humble-librealsense2 \
                 ros-humble-realsense2-camera \
                 ros-humble-realsense2-camera-msgs \
                 ros-humble-realsense2-description \
                 ros-humble-gscam \
                 gstreamer1.0-plugins-good \
                 gstreamer1.0-plugins-base \
                 gstreamer1.0-tools \
                 v4l-utils \
                 ros-humble-rosbag2-storage-mcap \
                 ros-humble-rqt-image-view \
                 ros-humble-rqt-gui
```

Why each of the non-obvious ones is required is in
[maintenance.md](maintenance.md); the short version:

- **`v4l-utils`** — the C922 exposure/gain/focus are applied by shelling out to
  `v4l2-ctl`. Without the binary the cameras stream but every setting is
  silently ignored.
- **`ros-humble-gscam`** — the C922 driver. Publishes the camera's MJPEG
  untouched, no decode/re-encode.
- **`gstreamer1.0-plugins-good`** — provides `v4l2src`, the element that opens
  the camera. gscam does **not** pull it in on its own.
- **`ros-humble-rosbag2-storage-mcap`** — the recorder backend. The default
  sqlite3 backend drops frames at these rates.
- **`ros-humble-rqt-gui` + `ros-humble-rqt-image-view`** — the monitor (§6).
  `rqt-gui` provides the `rqt` shell that loads `leo.perspective`; `rqt-image-view`
  provides the Image View plugin each panel uses. rviz2 cannot show these
  compressed streams in Humble.

### 0c. Install OBS separately

Not in the apt block above, on purpose — OBS needs its own apt repository:
https://obsproject.com/download

You need it. OBS is the exposure/gain/focus tuning tool: its V4L2 capture
source talks straight to the camera hardware, so the preview is what the driver
will record. Tuning itself is a per-conditions job (section 3).

### 0d. Verify the RealSense SDK is 2.57

The IMU is on, and the kernel-6.8 IMU fix only landed in librealsense **2.57**
(the why is in [maintenance.md](maintenance.md)). Confirm:

```bash
dpkg -l | grep -E "librealsense2|realsense2-camera "
```

Expect `ros-humble-librealsense2  2.57.7-1jammy...` and
`ros-humble-realsense2-camera  4.57.7-...`. If you see 2.56.x, `sudo apt upgrade`.
Then confirm the wrapper is linked against it:

```bash
ldd /opt/ros/humble/lib/librealsense2_camera.so | grep realsense
```

Must show `librealsense2.so.2.57`. If it shows `.so.2.56`, the upgrade did not
take.

### 0e. USB topology

- The **D435i** needs a USB 3 link and moves a few hundred MB/s. On a USB 2
  link it silently offers fewer profiles instead of erroring.
- Each **C922** is USB 2.0 and cannot do better — that is correct, MJPEG is
  already compressed.
- A single USB 3 controller's real throughput is shared by everything on it.
  **Where possible, keep the D435i on a bus the four webcams are not using**,
  or you get `Frames didn't arrive within 5 seconds`.

### 0f. Build and confirm

```bash
./rebuild.sh
./cameras.sh
```

Expect five camera nodes to come up: `/ego/d435i_ego`, `/c922_1`…`/c922_4`.
From here the rig is a per-day concern — the rest of this file.

---

## 1. Plug in

### D435i — USB 3.1 Gen 1 (5 Gbps), no exceptions

- **Blue port, or one marked `SS`.** The launch refuses to start below
  5000 Mbps.
- **Use the cable that shipped with the camera.** A USB 2 cable in a USB 3
  port still negotiates USB 2, and the camera responds by silently offering
  fewer stream profiles rather than by erroring.
- **Any long run needs an *active* cable.** Passive USB 3 cables lose signal
  past ~3 m and drop to USB 2 — or hold 5 Gbps and corrupt frames.

### C922 — USB 2.0 is fine

- USB 2.0 devices. They negotiate 480 Mbps and cannot do better, even in a blue
port. 1280×720@30 fits because the wire format is MJPEG, already compressed.
- **Any long run needs an *active* cable.** This prevents data corruption and
maintains the data speed.

### Confirm all five

```bash
lsusb | grep -iE "realsense|c922"     # expect 5 lines: 1 RealSense + 4 C922
```

Matching on the device **name** — not a hardcoded product ID — means this still
works if your camera revision reports a different ID, and it prints each unit so
you see *what* is there, not just a count.

- **Names, not vendor IDs.** `lsusb` resolves names from its USB database, so the
  RealSense and the C922s show up by name. If a name does not resolve (an
  unusual revision, a minimal system), read the devices straight from the
  hardware instead — no ID needed: `rs-enumerate-devices` (RealSense) and
  `v4l2-ctl --list-devices` (C922).
- **The exact IDs, for reference**, are `8086:0b3a` (D435i) and `046d:085c`
  (C922) — model IDs, identical on every unit, *not* serials. The D435i one is
  what the launch's `D435I_USB_ID` guard matches
  ([maintenance.md](maintenance.md) §2); the C922 launch finds cameras by
  `/dev/v4l/by-id` path, not by ID. Per-unit **serials**, which differ between
  your cameras, are a §2 concern, not this one.

If that is not 5, find the missing one before going further — the launch refuses
to start anyway, and section 5 tells you what its message means.

---

## 2. Swap, add or move a camera

**Rare — usually a one-time job when the rig is built or a camera is replaced.**
Skip it entirely unless the *hardware* actually changed. (For a change in the
*light*, you want section 3, not this.)

Both drivers find cameras by **hardware serial**, never by `/dev/videoN` —
those renumber on every replug. Serials are baked into the launch files, so a
new or swapped camera will not appear until you put its serial in and rebuild.

**RealSense:**

```bash
rs-enumerate-devices | grep "Serial Number"
```

```
    Serial Number                 : 	405622076405     <- use this one
    Asic Serial Number            : 	252843060866     <- NOT this one
```

Put it in the `EGO` tuple in
[four_d435i.launch.py:225](src/d435i_multicam_launch/launch/four_d435i.launch.py#L225):

```python
# (node name, serial, namespace, TF frame prefix)
EGO = ('d435i_ego', '405622076405', 'ego', 'ego')
```

That tuple is the only place the serial appears — the node and its checks are
built from it, so they cannot drift apart.

> **Trap:** the camera's `/dev/v4l/by-id/` path contains the *ASIC* serial, a
> different number. Use the one labelled `Serial Number`, or the camera
> enumerates fine and never starts.

**Logitech C922:**

```bash
ls /dev/v4l/by-id/ | grep -i c922
```

```
usb-046d_C922_Pro_Stream_Webcam_11815FDF-video-index0    <- use this
usb-046d_C922_Pro_Stream_Webcam_11815FDF-video-index1    <- metadata node, no video
```

Paste the **whole path** into a block in `CAMERAS` in
[c922.launch.py:29](src/d435i_multicam_launch/launch/c922.launch.py#L29):

```python
{
    'name': 'c922_1',
    'device': '/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_11815FDF-video-index0',
},
```

- **`name` is the stable contract — keep the convention, do not rename.** It
  becomes the node, the namespace and the topic prefix, so `c922_1` publishes on
  `/c922_1/image_raw/compressed`. **Everything downstream is built from this
  name, not from the serial:** the recorded topics in `BAG_TOPICS`, the six
  panels in `leo.perspective` (§6), and the monitor all key off `c922_1`…`c922_4`
  and `d435i_ego`. A different serial or a different PC changes nothing as long
  as the names hold; **renaming a camera silently breaks the perspective and
  `BAG_TOPICS`.** So when you add cameras, keep counting `c922_5`, `c922_6`, …
  rather than inventing new names.
- Always `-video-index0`. Index 1 is a metadata node and will never stream.
- **Add a camera:** paste another block, and add its two topics to `BAG_TOPICS`
  in [record_session.py:69](src/d435i_multicam_launch/d435i_multicam_launch/record_session.py#L69)
  (the silence watch is derived from that list, so it gets watched
  automatically). **Remove one:** delete its block and its `BAG_TOPICS` lines.

**Which physical camera is which?** Four identical C922s give you four serials
and no clue. Unplug the one you care about and diff the list:

```bash
before=$(ls /dev/v4l/by-id/)
# physically unplug ONE camera, wait 2s
diff <(echo "$before") <(ls /dev/v4l/by-id/)   # lines that vanished are that camera
```

Then rebuild — section 4.

---

## 3. Tune for the light

**Variable — do this whenever the light has changed, not just when hardware
did.** The room is a different scene on a bright sunny morning than under
overcast rain or artificial light in the evening, and a C922 tuned for one is
wrong in another — the cameras will not tell you, they just hand you dark or
blown-out footage. Re-check at the start of a day, and again if the weather or
lighting shifts noticeably.

**Why the C922s are manual:** auto-exposure lengthens the shutter in dim light,
and a shutter longer than one frame period silently drops you below 30 fps.
Manual exposure is the only way to guarantee the framerate.

**Do this in OBS.** Add a **Video Capture Device (V4L2)** source, pick the
camera, open its properties, expand **Camera Controls**. Those sliders talk
straight to the camera hardware, so the OBS preview is what the driver will
record. **This is also where you look at the picture properly** — the only step
with a big live preview. For each camera check it is aimed where you want, in
focus, unobstructed, neither dark nor blown out.

Work top to bottom:

- **Turn auto exposure off first.** Set `auto_exposure` to *Manual Mode*. Until
  you do, the exposure slider is locked.
- **Set exposure, then gain.** Exposure is the shutter; open it as far as you
  can before touching gain, because gain buys brightness with noise.
- **Set focus.** The C922 autofocus hunts; the rig runs it fixed.
- **Repeat for all four cameras** in the same scene and light. If one needs a
  wildly different number, it is pointed somewhere different.

| Slider | Range | What it means |
|---|---|---|
| `exposure_time_absolute` | 3 – 2047 (default 166) | Shutter, in 100 µs units. **333 is one frame period at 30 fps and your hard ceiling** — above it the camera silently drops to 15 fps. 1/60 s → 167, 1/125 s → 80 |
| `gain` | 0 – 255 (default 25) | Sensor gain. 0 cleanest, 255 brightest and noisiest. Needing more than ~100 means fix the lighting instead |
| `focus_absolute` | 0 – 250 (default 0) | Fixed focus, steps of 5. 0 = far, 250 = near |

**The D435i needs nothing here.** Its colour runs on auto exposure with the
framerate protected (`auto_exposure_priority: False`) — it is the ego camera:
it moves, swings towards windows and lamps, and has people step in front of it,
so no one fixed shutter suits every heading. Do not touch the depth module.

**Write the numbers back** into
[c922.launch.py](src/d435i_multicam_launch/launch/c922.launch.py):

| Camera | Goes in |
|---|---|
| All four C922 the same | `DEFAULTS` ([c922.launch.py:52](src/d435i_multicam_launch/launch/c922.launch.py#L52)) |
| One C922 differing | add `'exposure'` / `'gain'` / `'focus'` to that camera's block in `CAMERAS` — it overrides `DEFAULTS` |
| D435i | nothing — colour is on auto exposure |

> **Then rebuild — section 4 — and close OBS.** What you set in OBS sticks on
> the camera until it is replugged, so the next launch can look right while the
> file still holds the old numbers, and the launch overwrites your settings a
> few seconds in anyway. **If it is not in the launch file, it does not exist.**
> And a camera open in OBS cannot be opened by the launch.

---

## 4. Build

```bash
./rebuild.sh
```

- **Rebuild after any edit** to a launch file, to `BAG_TOPICS`, or to the
  exposure/gain/focus numbers in section 3. Editing alone changes nothing:
  everything runs what is in `install/`, never `src/`.
- Building is all it does — the other scripts source ROS and the workspace
  themselves, every time.
- Wait for `build OK`. It then prints the terminals to run next.
- A failed build leaves the previous build in place and says so.

---

## 5. Terminal 1 — start the cameras

```bash
./cameras.sh
```

Leave this terminal alone for the rest of the day — it is your camera log.
`Ctrl-C` here stops the cameras and does not touch a running recording.

### If it refuses to start

Several checks run before any camera node starts, and **any one of them kills
the whole launch, all cameras included** — the checks run while the launch
description is built, so nothing has started and nothing survives. You get the
message and your prompt back, never a half-running rig.

| What it prints | What to do |
|---|---|
| `no install/setup.bash ... the workspace is not built` | `./rebuild.sh` — section 4 |
| `D435i not detected on USB. Is the ego camera plugged in?` | Plug the ego camera in. |
| `D435i negotiated a 480 Mbps USB link, needs 5000+` | Wrong port or wrong cable. Blue/`SS` port, stock cable, no passive extension — section 1. |
| `D435i depth units are 0.002000 m/unit, expected 0.001` | Depth scale was changed and stuck. **Replug the D435i** — it resets to 0.001. If it persists, set it in `realsense-viewer` → Stereo Module → Depth Units, then close the viewer. |
| `C922 not found. Unplugged, or its serial changed` | It names the device path. Replug it, or its serial changed — section 2, then rebuild. |
| `C922 already open. A camera opens in one program at a time` | It names the program and pid. Close it — usually OBS. |
| `C922 will not deliver the requested resolution ... driver offered ...` | The size in `DEFAULTS`/`CAMERAS` is not a mode this camera — or this cable/port — can produce. It names asked-vs-offered per camera. Pick a mode the camera lists (`v4l2-ctl -d <device> --list-formats-ext`), or fix the link — section 1. |

**These are presence checks, not health checks.** A camera that enumerates,
opens, then never delivers a frame gets past all of them. `./record.sh` catches
that one — it refuses to open a bag until every stream is live.

> The D435i has no "already open" check; only the C922s do. If you opened
> `realsense-viewer` to fix depth units, **close it** — otherwise the camera
> node fails to open the device with a less obvious message.

### Otherwise

Startup takes ~8 seconds. The C922 exposure, gain and focus are applied a few
seconds *after* each stream opens, not at launch — UVC resets those controls
when streaming starts, so anything set earlier would be wiped. Until then you
are seeing the camera's own defaults, so **the first few seconds of image look
wrong. That is normal.**

**Ignore these:**

- `Camera calibration file .../c922_N.yaml not found` and
  `Unable to open camera calibration file` — the C922s are uncalibrated; the
  stream is unaffected.
- `Using gstreamer config from rosparam: "v4l2src device=..."` — the pipeline
  being accepted, not a warning.
- `JSON file is not provided`, `re-enable the stream for the change to take effect`

**Stop and fix these** — the rig came up, but a camera is not healthy:

- `Depth stream start failure ... Hardware Error` — replug the D435i, ideally
  onto a port not shared with the webcams.
- `Frames didn't arrive within 5 seconds` — bandwidth. Move the D435i to
  another bus.

---

## 6. Terminal 2 — watch the feeds

Start this once terminal 1 has settled — the topics do not exist until the
cameras are up.

```bash
rqt --perspective-file leo.perspective
```

`leo.perspective` (in the repo root) is a saved **rqt** layout with one Image
View panel per camera, so all six feeds come up at once:

```
/ego/d435i_ego/color/image_raw
/ego/d435i_ego/depth/image_rect_raw
/c922_1/image_raw/compressed
/c922_2/image_raw/compressed
/c922_3/image_raw/compressed
/c922_4/image_raw/compressed
```

- **Use `rqt`, not `rqt_image_view`.** `rqt_image_view` shows only one topic at
  a time (a dropdown to switch); the perspective loads all six panels together.
  If the `--perspective-file` flag is not recognised on your install, open plain
  `rqt` and load it by hand: **Perspectives → Import → `leo.perspective`**.
- **The perspective is pinned to the camera *names*, not serials — so keep the
  naming convention.** Its panels subscribe to `/c922_1`…`/c922_4` and
  `/ego/d435i_ego/…`, which are built from the `name`/namespace in the launch
  files (§2), never from a serial. That is why it works unchanged on a different
  PC or after swapping in cameras with different serials. It only breaks if you
  **rename** a camera or **change the count** — then a panel goes blank, and you
  re-export the layout (**Perspectives → Export → `leo.perspective`**). Keep the
  `c922_N` / `d435i_ego` names and it stays portable.
- if the perspective fails to load, you can still create your own workspace for all 6 feeds.
- Leave it up for the day. You tuned and inspected these feeds in OBS in
  section 3; this is the check that nothing moved *since* — a bumped camera, a
  knocked cable, a lens cap.
- If a view looks laggy, measure rather than eyeball:

```bash
ros2 topic hz /c922_1/image_raw/compressed      # expect ~30
```

---

## 7. Record a session

Terminals 1 and 2 stay up. A **session** is the set of episodes covering one
workflow; you record its episodes one after another, then move on to the next
session with the same cameras still up. Each episode is one `./record.sh` run
and one bag.

### 7a. Terminal 3 — record one episode

```bash
./record.sh 23        # -> 2026-08-26_23_1/
./record.sh 23        # -> 2026-08-26_23_2/   (next take of the same episode)
./record.sh 24        # -> 2026-08-26_24_1/   (new episode, own count)
```

- You type only the **code**; the bag is named `<date>_<code>_<N>`, where `N`
  is number of takes. You never overwrite a bag and
  never have to remember which number you are on.
- Bags land in `BAG_ROOT` (default: the workspace).
  `export BAG_ROOT=/mnt/data` before the run to write elsewhere.

It checks free space, waits for every camera stream, and only then creates the
bag:

```
  waiting for all 6 streams (up to 30s)...
  all 6 stream(s) live after 7.4s

  recording episode 23-1 -> 2026-08-26_23_1
  16 topics, watching 6 for silence every 5s
  Ctrl-C to stop. Do NOT close this window -- SIGHUP leaves the bag unreadable.
```

- Recording ends within **~7 seconds of a camera disconnect**.
- **A camera that never starts means no bag at all.** You get
  `CAMERAS NOT READY -- NOTHING RECORDED`; nothing is written; fix the camera
  and run the command again.
- Below **100 GB** free it warns; below **1 GB** it refuses. A full rosbag runs
  **~10 GB per minute**, so 100 GB is roughly 10 minutes of headroom — check
  `df -h .` before a long session.
- **Change what is recorded** in `BAG_TOPICS`
  ([record_session.py:69](src/d435i_multicam_launch/d435i_multicam_launch/record_session.py#L69)),
  then rebuild. The silence watch is derived from that list.

**Stop the episode:**

- **`Ctrl-C` in terminal 3, then let it finish.** It prints
  `closing the bag -- do not interrupt`, then `still closing... 5s` ticks, and
  ends with `ros2 bag info` output. **Those ticks are your bag being written** —
  a multi-GB mcap writes its index at close. Interrupting there is how a bag
  becomes unreadable.
- **Never close the window to stop.** That sends SIGHUP, not SIGINT, and the
  bag never closes.

A recording can also end on its own — a camera died, the recorder died, the
disk filled, or the watch itself broke. Each prints a full-width banner naming
the cause, and each still closes the bag before exiting.

### 7b. Mark a bad take — after the fact

There are no live markers. If an episode was a fumble, a retake, or otherwise
flawed, tag the **bag folder** afterwards:

```bash
./oops.sh 2026-08-26_23_2 m1          # one code
./oops.sh 2026-08-26_23_2 m1 m2       # several
```

- Run it from the directory holding the episode folders, so the folder name
  tab-completes.
- Each code becomes an **empty marker file** at `<episode>/oops/<code>`; the
  downstream extractor reads the filename. Codes are lowercase alphanumeric
  (e.g. `m1`).
- **Undo a mark:** `rm 2026-08-26_23_2/oops/m1`.

### 7c. Next episode, and next session

Terminals 1 and 2 are still up — nothing to restart, no ~8 s relaunch.

- **Next episode in this session:** run `./record.sh` again with the next code.
- **Next session:** nothing special to do — the cameras stay up across sessions.
  Just carry on with that session's episode codes. The only thing separating one
  session from the next is which episodes you group together; the rig does not
  track it for you.

---

## 8. Before tearing down the scene

Re-shooting is cheap now. `./record.sh` already printed the `ros2 bag info` at
close — read it rather than skipping it:

```bash
ros2 bag info 2026-08-26_23_1
```

- Duration matches what you recorded.
- Every topic present. For each image topic, message count ≈ `duration × 30`;
  roughly half that means frames were dropped — the disk could not keep up.
- The **IMU** (`/ego/d435i_ego/imu`) and the three `extrinsics/*` topics are
  present, alongside the six image/`camera_info` pairs.

**Then run the full bag sanity check.** `ros2 bag info` is not enough — the C922
frames are opaque JPEG blobs whose `camera_info` records 0×0, so a wrong size is
invisible there. `inspect_bag` reads what actually landed in the bag and checks
every stream against the rig's configured spec:

```bash
ros2 run d435i_multicam_launch inspect_bag 2026-08-26_23_1
```

```
  2026-08-26_23_1
  checking against:  colour 1280x720,  depth 848x480,  colour/depth 30 fps,  imu 200 Hz  (rate floor 90%)

  RESOLUTION
  OK   /c922_1/image_raw/compressed                     1280x720  (want color 1280x720)
  OK   /ego/d435i_ego/color/image_raw                   1280x720  (want color 1280x720)
  FAIL /ego/d435i_ego/depth/image_rect_raw              1280x720  (want depth 848x480)

  FRAMERATE  (average = message count / bag duration)
  OK   /ego/d435i_ego/color/image_raw                      29.8 fps  (want 30, min 27.0)
  FAIL /c922_2/image_raw/compressed                        14.9 fps  (want 30, min 27.0)

  IMU
  OK   /ego/d435i_ego/imu                                12043 msgs   199.4 Hz  (want non-empty, 200 Hz, min 180)

  2 check(s) FAILED
```

- **What it's for:** to confirm a bag is usable before you tear the scene down —
  right resolution, right framerate, IMU present — reading the *bytes on disk*,
  not what a config file claims.
- **What it checks, and fails on** (`FAIL` on any → non-zero exit):
  - **Resolution** — the four **C922** and the **ego colour** against
    **1280×720**; the **ego depth** against **848×480**. C922 sizes are decoded
    from inside the JPEG (the only place the truth lives); ego sizes are read
    from the raw message.
  - **Framerate** — every colour and depth stream, as `message count ÷ duration`,
    against **30 fps** (must be at least 90% of it — a half-rate bag means the
    disk dropped frames).
  - **IMU** — `/ego/d435i_ego/imu` is present, non-empty, and averaging
    **~200 Hz**. Catches an IMU that silently stopped streaming (e.g. an SDK
    below 2.57).
- **It prints what each stream is checked against**, so a `FAIL` line shows both
  what recorded and what was expected. Every expected value has a flag to
  override it (`--fps`, `--depth-width/-height`, `--color-width/-height`,
  `--imu-fps`); with no flags it checks against the shipped defaults, which
  mirror the launch config.
- **When to run it:** on a fresh bag after any camera, cable, port or launch
  change. The launch-time guards already refuse a bad *start*; this catches
  anything that changed the bytes *after* that — bandwidth, a swapped camera, a
  marginal cable, a dropped IMU.

### Watch a recording back

`inspect_bag` proves the numbers; to actually *see* the footage, play the bag
and view it in **rqt**.

**Stop the cameras first** — `Ctrl-C` terminal 1 (`cameras.sh`). Playback
republishes the recorded topics under their **original names**, so if the live
cameras are still up you get two publishers on every topic and rqt shows the
live feed and the recording interleaved. Stop `cameras.sh` and the only thing on
those topics is the bag.

```bash
# terminal inside the root of the ROS node workspace.
# terminal 2
rqt # ensure rqt is present for you to view the play back
# terminal 1 -- Ctrl-C cameras.sh FIRST, then:
ros2 bag play 2026-08-26_23_1            # add --loop to repeat
```

Then watch the same topics in `rqt` (the §6 perspective) — the feeds are the
recorded ones now, not live. The C922 streams play back compressed, and the
perspective's panels are already pointed at those topics.

- **This costs the ~8 s `cameras.sh` relaunch** to resume recording, so play
  back when you are done shooting a session — or genuinely need to review a take
  — not between every episode. For a quick "did it record right" mid-session,
  `inspect_bag` plus the live monitor are enough.

---

## 9. Exo-only calibration

To calibrate the four exo C922s with the ego camera down (so it neither streams
nor draws USB bandwidth), there is a parallel two-terminal flow:

```bash
./exo_calib_launch.sh          # terminal 1 -- the four C922s only, no D435i
./exo_calib_record.sh walkabout    # terminal 3 -- C922-only bag, -> 2026-08-26_walkabout_1
```

`exo_calib_record.sh` passes `--calib-exo`, which drops every `/ego/` topic
from both the bag and the silence watch — so a missing ego is expected, not a
startup abort. Everything else (auto-numbering, `BAG_ROOT`, stop-on-Ctrl-C) is
identical to section 7.

---

## Quick fixes

| Symptom | Cause |
|---|---|
| `no install/setup.bash ... not built` | `./rebuild.sh`. |
| `package 'd435i_multicam_launch' not found` | Same — the workspace was never built. |
| Edited a launch file, nothing changed | Did not rebuild. Everything runs `install/`, never `src/`. Section 4. |
| A C922 is missing at launch | Its `/dev/v4l/by-id/` path changed, or the camera is new. Section 2. |
| `C922 already open ... held by obs (pid N)` | Close whatever the message names. Usually OBS. |
| RealSense enumerates but never starts | Wrong serial — you used the ASIC serial. Section 2. |
| `D435i depth units are ...` | Replug the D435i. Section 5. |
| D435i fails to open, nothing else wrong | `realsense-viewer` still has it. Close it. |
| IMU topic empty / no `/ego/d435i_ego/imu` | SDK below 2.57, or the camera needs a replug after an SDK change. Section 0d. |
| C922 at 15 fps | Exposure above 333. Section 3. |
| Image black or blown out | Exposure/gain left over from different lighting. Section 3. |
| Tuned it, looks wrong next launch | Numbers never made it into the launch file, or no rebuild. Section 3. |
| `CAMERAS NOT READY`, no bag written | A camera never delivered a frame. Check terminal 1. Section 7a. |
| `C922 will not deliver the requested resolution` | The size in `DEFAULTS`/`CAMERAS` is not a mode the camera or cable can do. `v4l2-ctl -d <dev> --list-formats-ext`. Section 5. |
| Bag looks fine but recorded the wrong size | gscam ignored the requested resolution. `inspect_bag` shows the truth; the launch guard aborts on it at startup. Section 8. |
| `inspect_bag` FAILs on RESOLUTION | A stream recorded the wrong size. C922 → cable/bandwidth/swapped camera (§8); ego → a launch profile edit that didn't take (`camera_info` check, maintenance.md §2). |
| `inspect_bag` FAILs on FRAMERATE | Dropped frames — disk too slow (RealSense colour+depth dominate), or a C922 at half rate from exposure above 333. Section 3 / throughput. |
| `inspect_bag` FAILs on IMU | IMU empty or missing → SDK below 2.57, or a replug is needed after an SDK change. Section 0d. |
| Recording stopped by itself | A banner names the cause (camera silent / recorder died / watch broke). Section 7a. |
| Far fewer messages in the bag than expected | Disk too slow. The C922s are already compressed, so it is the RealSense colour + depth — shorter episodes, or trim `BAG_TOPICS`. |
| `!! no .../metadata.yaml -- the bag did not close cleanly` | The close was interrupted. Recover with `ros2 bag reindex -s mcap <bag>` — `./record.sh` prints the exact command. |
| Bad take needs flagging | `./oops.sh <episode> <code>`. Section 7b. |
| `No module named 'rosbag2_py._reader'` | You ran `python3` yourself in a ROS shell and conda's Python answered. Use `ros2 run`, or deactivate conda. |
| `pip` tries to uninstall ROS packages | Conda active in a ROS shell. Separate terminals. |

---

## Cheat sheet

```bash
# 0. first time on a machine only -- section 0
#    apt install (0b), OBS (0c), verify SDK 2.57 (0d):
dpkg -l | grep -E "librealsense2|realsense2-camera "

lsusb | grep -iE "realsense|c922"          # 1. expect 5 lines (1 D435i + 4 C922)
                                           #    name grep survives an ID change;
                                           #    else: rs-enumerate-devices /
                                           #    v4l2-ctl --list-devices
                                           #    D435i needs USB3 5Gbps; C922 USB2 ok

# 2. only if a camera changed (rare, ~one-time):
rs-enumerate-devices | grep "Serial Number"   # swapped camera -> serials into
ls /dev/v4l/by-id/ | grep -i c922             #   the launch files, then rebuild

# 3. when the light changes (sunny / rainy / new room):
#    tune exposure/gain/focus in OBS -> Camera Controls, look at every feed,
#    write numbers into c922.launch.py, then rebuild

./rebuild.sh                               # 4. after ANY edit above

./cameras.sh                               # 5. terminal 1 -- leave up
rqt --perspective-file leo.perspective     # 6. terminal 2 -- leave up, all 6 feeds

# --- per episode ---
./record.sh 23                             # 7a. terminal 3 -> 2026-08-26_23_1/
                                           #     Ctrl-C to stop, LET IT CLOSE
./oops.sh 2026-08-26_23_1 m1               # 7b. flag a bad take (optional)
# --- next episode: new code or same code, terminals 1 and 2 stay up ---

ros2 bag info 2026-08-26_23_1              # 8. before teardown
ros2 run d435i_multicam_launch inspect_bag 2026-08-26_23_1   #    sanity check:
                                           #    res + fps + imu vs the rig spec
# watch it back: Ctrl-C cameras.sh FIRST (else live + bag collide), then:
ros2 bag play 2026-08-26_23_1              #    view in rqt (the §6 perspective)

# exo-only calibration (section 9):
./exo_calib_launch.sh                      # terminal 1 -- C922s only
./exo_calib_record.sh walkabout            # terminal 4 -- C922-only bag
```
