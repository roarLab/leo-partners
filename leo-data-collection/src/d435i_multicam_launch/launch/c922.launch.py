import glob
import os
import subprocess

from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node


# ---------------------------------------------------------------------------
# EDIT HERE: one block per camera.
#
# Add a camera     -> paste another block
# Remove a camera  -> delete its block
# One camera only  -> leave a single block
#
# 'name' must be unique: it sets the node name, the namespace, camera_name and
# the TF frame, so /<name>/image_raw/compressed is where the images land.
#
# 'device' is the by-id symlink. The hex is the camera's hardware serial, so
# the path follows that physical camera into any USB port and survives replug:
#     ls -l /dev/v4l/by-id/ | grep -i c922
# Use -video-index0 (capture), not -video-index1 (metadata).
# Never use /dev/videoN -- it renumbers on every replug, and the RealSense
# cameras already occupy the low numbers.
#
# Any key from DEFAULTS below can be overridden per camera.
# ---------------------------------------------------------------------------
CAMERAS = [
    {
        'name': 'c922_1',
        'device': '/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_11815FDF-video-index0',
    },
    {
        'name': 'c922_2',
        'device': '/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_4EBC3FDF-video-index0',
    },
    {
        'name': 'c922_3',
        'device': '/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_4F9FFD8F-video-index0'
    },
    {
        'name':'c922_4',
        'device': '/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_FB833F6F-video-index0'
    }
]


# ---------------------------------------------------------------------------
# Applied to every camera unless a block above overrides the key.
# ---------------------------------------------------------------------------
DEFAULTS = {
    'width': 1280,
    'height': 720,
    'fps': 30.0,

    # The camera sends MJPEG over USB and gscam publishes those bytes
    # untouched as sensor_msgs/CompressedImage on
    #     /<name>/image_raw/compressed
    # Nothing decodes, nothing re-encodes, so there is no raw topic at all
    # and no generation of image quality is lost.
    # Measured per camera: 1.7% CPU idle, 3.0% while recording, 30.0 Hz.

    # Auto exposure lengthens the shutter in dim light and will quietly drop
    # you below 30 fps. For a guaranteed 30 fps set autoexposure False and
    # exposure <= 333 (one frame period).
    'autoexposure': False,

    # Shutter speed, in 100us units:  value = 10000 / denominator
    #     1/30 s -> 333      1/125 s -> 80
    #     1/60 s -> 167      1/250 s -> 40
    # Range 3 (~1/3333 s) .. 2047 (~1/4.9 s). Ignored while autoexposure.
    'exposure': 166,

    # The ISO analogue. There is no ISO control -- this is raw sensor gain,
    # 0 (cleanest) .. 255 (brightest, noisiest), uncalibrated. Raise it only
    # when a short enough shutter leaves the image too dark.
    'gain': 25,

    'autofocus': False,
    'focus': 0,           # 0 (far) .. 250 (near), steps of 5.
}


# gscam has no exposure or focus controls of its own -- it drives GStreamer,
# not the V4L2 control interface -- so those are applied by the v4l2-ctl calls
# below, exactly as before. That code is unchanged by the move off usb_cam
# because it always addressed the camera directly rather than through a driver.


def build_pipeline(cfg):
    """The GStreamer pipeline: pull MJPEG off the camera, and stop there.

    Naming 'image/jpeg' in the caps makes v4l2src negotiate the camera's MJPEG
    mode, which is what allows 1080p30 over USB 2 at all -- raw YUYV is
    bandwidth-capped at about 1080p5. No decoder element follows, so the JPEG
    reaches gscam untouched and is published as-is.

    The trailing jpegparse is mandatory, not cosmetic: gscam links its appsink
    to a pipeline ending in a bare capsfilter without honouring its width/height,
    so v4l2src falls back to the camera default (1080p) and the width/height
    above are silently ignored. jpegparse emits a fully-fixed image/jpeg stream
    that pins v4l2src to the requested size. It parses framing only -- no decode,
    no re-encode -- so the passthrough above still holds. Drop it and you record
    1080p regardless of what width/height say.

    gscam resolves nothing for us, and v4l2src wants a real device node, so the
    by-id symlink is resolved here. CAMERAS still holds the by-id path, which
    is what keeps a camera identified by its hardware serial across replugs.
    """
    return (
        f"v4l2src device={os.path.realpath(cfg['device'])} ! "
        f"image/jpeg,width={int(cfg['width'])},"
        f"height={int(cfg['height'])},framerate={int(cfg['fps'])}/1 ! jpegparse"
    )


def make_camera_node(cfg):
    name = cfg['name']

    return Node(
        package='gscam',
        executable='gscam_node',
        name=name,
        namespace=name,
        output='screen',
        parameters=[{
            'gscam_config': build_pipeline(cfg),

            # 'jpeg' is what selects the CompressedImage publisher. Any other
            # value makes gscam expect decoded frames and the pipeline above
            # will not match.
            'image_encoding': 'jpeg',

            'camera_name': name,
            'frame_id': name + '_optical_frame',

            # gscam blocks the pipeline on the sink clock when this is true,
            # which paces frames to GStreamer's clock rather than the camera's
            # and shows up as jitter. We want frames as the camera delivers
            # them.
            'sync_sink': False,
        }],
        remappings=[
            # gscam publishes under 'camera/...' by default, which would give
            # /c922_1/camera/image_raw/compressed. Strip the extra level so the
            # topics stay where every other file in this repo expects them:
            #   /<name>/image_raw/compressed  and  /<name>/camera_info
            ('camera/image_raw', 'image_raw'),
            ('camera/image_raw/compressed', 'image_raw/compressed'),
            ('camera/camera_info', 'camera_info'),
        ],
    )


def build_v4l2_controls(cfg):
    """Split the controls into (modes, values).

    exposure_time_absolute and focus_absolute are inactive while their auto
    mode is on, and the driver validates a whole v4l2-ctl batch against the
    state *before* applying any of it. Batching a mode with the value it
    unlocks makes the driver reject the entire call with EACCES, so the modes
    have to be applied first, by a separate process.
    """
    modes = []
    values = []

    # This camera only implements two of the four V4L2_EXPOSURE_* modes:
    #   1 = Manual Mode, 3 = Aperture Priority Mode (i.e. auto)
    # Confirm with: v4l2-ctl -d <dev> --list-ctrls-menus
    if cfg['autoexposure']:
        modes.append('auto_exposure=3')
    else:
        modes.append('auto_exposure=1')
        values.append(f"exposure_time_absolute={int(cfg['exposure'])}")
        # Auto exposure drives gain itself, so only set it in manual mode.
        values.append(f"gain={int(cfg['gain'])}")

    if cfg['autofocus']:
        modes.append('focus_automatic_continuous=1')
    else:
        modes.append('focus_automatic_continuous=0')
        values.append(f"focus_absolute={int(cfg['focus'])}")

    return modes, values


def make_control_actions(cfg):
    """Apply the modes, then the values, as two separate v4l2-ctl runs.

    The 3s delay is because UVC resets exposure and focus when streaming
    starts, so the controls only stick once the node has opened the device.
    The 1s gap between the two runs orders them: the second call needs the
    first to have already unlocked the controls it sets.
    """
    modes, values = build_v4l2_controls(cfg)

    def call(controls, period):
        return TimerAction(period=period, actions=[
            ExecuteProcess(
                cmd=['v4l2-ctl', '--device', cfg['device'],
                     '--set-ctrl', ','.join(controls)],
                output='screen',
            ),
        ])

    actions = [call(modes, 3.0)]
    if values:                    # empty when everything is left on auto
        actions.append(call(values, 4.0))
    return actions


def device_holder(device):
    """Return 'comm (pid N)' of whatever holds this device open, or None.

    A V4L2 capture device streams for one opener at a time, so anything
    holding it -- OBS, a second launch -- means this launch cannot have it.
    Scans /proc the way fuser does, so the launch stays free of subprocesses.

    Only sees processes you own: /proc/<pid>/fd is unreadable for other users
    without root, and those are skipped. fuser has the same blind spot.
    """
    target = os.path.realpath(device)

    for fd_dir in glob.glob('/proc/[0-9]*/fd'):
        pid = fd_dir.split('/')[2]
        try:
            for fd in os.listdir(fd_dir):
                if os.path.realpath(os.path.join(fd_dir, fd)) == target:
                    with open(f'/proc/{pid}/comm') as f:
                        return f'{f.read().strip()} (pid {pid})'
        except OSError:
            continue          # exited mid-scan, or not ours to read
    return None


def check_cameras_present():
    """Abort the launch if a webcam is missing or already open elsewhere.

    Presence only. A camera that enumerates but never delivers frames --
    bandwidth, a marginal cable, an exposure above 333 -- gets past this.
    """
    missing = []
    held = []

    for camera in CAMERAS:
        device = camera['device']
        if not os.path.exists(device):
            missing.append(f"    {camera['name']}: {device}")
            continue
        holder = device_holder(device)
        if holder:
            held.append(f"    {camera['name']}: held by {holder}")

    problems = []
    if missing:
        problems.append(
            'C922 not found. Unplugged, or its serial changed and CAMERAS is '
            'stale:\n' + '\n'.join(missing)
            + '\n  Re-check with: ls /dev/v4l/by-id/ | grep -i c922')
    if held:
        problems.append(
            'C922 already open. A camera opens in one program at a time:\n'
            + '\n'.join(held)
            + '\n  Close it (usually OBS), then launch again.')

    if problems:
        raise RuntimeError('\n\n'.join(problems))


def try_format(device, width, height, pixelformat='MJPG'):
    """What the driver would ACTUALLY set for this request, without streaming.

    --try-fmt-video negotiates a format and echoes back the nearest mode the
    camera can honour -- it neither opens the device for capture nor fights the
    node that will. Ask for 1280x720 MJPG and get something else back, and that
    something else is what gscam would silently publish.
    """
    try:
        out = subprocess.run(
            ['v4l2-ctl', '--device', device,
             f'--try-fmt-video=width={width},height={height},'
             f'pixelformat={pixelformat}'],
            capture_output=True, text=True, check=True).stdout
    except FileNotFoundError:
        raise RuntimeError(
            'v4l2-ctl not found -- install v4l-utils to launch the C922s.')
    except subprocess.CalledProcessError as err:
        raise RuntimeError(
            f'v4l2-ctl could not query {device}:\n  {err.stderr.strip()}')

    w = h = fmt = None
    for line in out.splitlines():
        if 'Width/Height' in line:
            w, h = (int(x) for x in line.split(':', 1)[1].strip().split('/'))
        elif 'Pixel Format' in line:
            # "'MJPG' (Motion-JPEG)" -> MJPG
            fmt = line.split("'")[1]
    return w, h, fmt


def check_resolutions():
    """Abort unless every C922 will deliver the requested MJPEG resolution.

    gscam pins fixed caps and cannot rescale, so a mode the camera will not
    negotiate does not downshift -- the node stays up publishing nothing, or
    (without jpegparse) quietly falls back to 1080p. This asks the driver up
    front and turns either outcome into a loud, launch-time abort.

    Resolution and MJPEG only. A camera that negotiates the mode but cannot
    sustain the framerate on a weak USB link is an fps fault -- caught by the
    record_session silence watch, not here.
    """
    problems = []
    for camera in CAMERAS:
        cfg = dict(DEFAULTS)
        cfg.update(camera)
        want = (int(cfg['width']), int(cfg['height']), 'MJPG')
        got = try_format(cfg['device'], want[0], want[1])
        if got != want:
            problems.append(
                f"    {cfg['name']}: asked {want[0]}x{want[1]} MJPG, "
                f"driver offered {got[0]}x{got[1]} {got[2]}")

    if problems:
        raise RuntimeError(
            'C922 will not deliver the requested resolution. gscam cannot '
            'rescale, so it would publish the wrong size or nothing:\n'
            + '\n'.join(problems)
            + '\n  Set width/height in CAMERAS/DEFAULTS to a mode the camera '
            'lists, or check the cable/port:\n'
            '  v4l2-ctl -d <device> --list-formats-ext')


def generate_launch_description():
    check_cameras_present()
    check_resolutions()

    actions = []

    for camera in CAMERAS:
        cfg = dict(DEFAULTS)
        cfg.update(camera)

        actions.append(make_camera_node(cfg))
        actions.extend(make_control_actions(cfg))

    return LaunchDescription(actions)
