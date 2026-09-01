#!/usr/bin/env python3
"""Records one episode and closes it itself when a camera goes silent.

    ros2 run d435i_multicam_launch record_session 23  ->  $BAG_ROOT/2026-08-04_23_1/

The one thing you run in terminal 4. Terminal 1 keeps the cameras up and is
not touched by this -- after a failure you Ctrl-C that window yourself.

Why this owns the recorder rather than asking it to stop: Humble's recorder
exposes pause/resume/snapshot and no stop service, so the only way to end a
bag is SIGINT, and the only way to know which process to signal is to have
started it. Hence subprocess.Popen -- the pid comes with the parenthood.

'ros2 bag record' stays in the data path because it is C++ and this rig moves
~110 MB/s. Nothing here touches a camera message; it only holds the pid.

Order matters: every camera must deliver a frame BEFORE the bag is created, so
a dead camera costs a refusal instead of a session you cannot use.

Four ways a session ends, all through stop_recorder():
    a camera goes silent  -> banner, exit 1
    you press Ctrl-C      -> normal stop, exit 0
    the recorder dies     -> disk full or crash, exit 3
    the watch breaks      -> bag closed rather than left unwatched, exit 4

The watch measures silence and nothing else. Measured on this rig, do not
undo:

* Silence only, never framerate. Measured rates move with machine load, so a
  rate check fired on healthy hardware every single run.
* Sample, don't stay subscribed. Staying subscribed costs 4.6% of a core;
  sampling costs 0.16%.
* Idle by sleeping, never by spinning. Nothing is subscribed between probes,
  so spin_once() there woke 20x a second to process nothing -- that one loop
  was ~1% of a core, more than everything else combined.
* Leave the early exit in _listen(). It ends a probe once every topic has
  reported, so a healthy probe costs ~0.5s whatever PROBE_WINDOW says. That
  is what makes the probe rate nearly free: 5s apart costs 0.16%, 30s apart
  0.12%.
* PROBE_WINDOW is a ceiling, not a cost -- it only bounds how long a QUIET
  topic is given before being called dead. A re-created subscription took up
  to 0.493s to start receiving, so 2s is 4x margin. Below that, a slow
  re-subscribe starts looking like a dead camera.
* camera_info, not image_raw. Both publish once per frame, but camera_info is
  a few hundred bytes against ~6MB -- image subs would risk causing the
  bandwidth failure the watch looks for.
"""
import glob
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import date

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo

# ---------------------------------------------------------------------------
# EDIT HERE: what goes in the bag.
#
# The watch list is derived from this -- every */camera_info. Add a camera
# here, which you have to do anyway, and it is watched too. No second list.
# ---------------------------------------------------------------------------
BAG_TOPICS = [
    '/ego/d435i_ego/color/image_raw',
    '/ego/d435i_ego/color/camera_info',
    '/ego/d435i_ego/depth/image_rect_raw',
    '/ego/d435i_ego/depth/camera_info',
    '/ego/d435i_ego/imu',
    '/ego/d435i_ego/extrinsics/depth_to_gyro',
    '/ego/d435i_ego/extrinsics/depth_to_accel',
    '/ego/d435i_ego/extrinsics/depth_to_color',
    '/c922_1/image_raw/compressed',
    '/c922_1/camera_info',
    '/c922_2/image_raw/compressed',
    '/c922_2/camera_info',
    '/c922_3/image_raw/compressed',
    '/c922_3/camera_info',
    '/c922_4/image_raw/compressed',
    '/c922_4/camera_info'
]
# Every ego topic shares this prefix, so the exo (C922-only) set is just
# BAG_TOPICS without it -- one list stays canonical, no second one to sync.
EGO_PREFIX = '/ego/'
# Seconds.
#
# STARTUP_GRACE is a deadline, not a wait: the check returns the moment every
# camera has delivered, so a healthy rig clears it in ~8s whatever this says.
# It only sets how long you sit before being told something is broken.
#
# A dead camera mid-session is caught in PROBE_PERIOD + PROBE_WINDOW at worst, so
# the bag ends with up to 7s missing a camera. Probing harder is nearly free
# -- see the early-exit note in the docstring -- so the limit is PROBE_WINDOW,
# not CPU.
STARTUP_GRACE = 30.0
PROBE_WINDOW = 2.0
PROBE_PERIOD = 5.0

# Seconds allowed to flush and index on SIGINT before escalating. A multi-GB
# mcap writes its summary at close; escalating early is what leaves a bag with
# no metadata.yaml, so this is far longer than it should ever need.
FLUSH_TIMEOUT = 120

MIN_FREE_GB = 100        # warn below
HARD_FREE_GB =1         # refuse below

HINT = """  RealSense (/ego/...):
    * USB 3 port on a bus the webcams are not using: lsusb -t
    * A long passive cable cannot hold 5 Gbps. Use an active cable.
    * 'Depth stream start failure' in terminal 1 means bandwidth or cable.

  Logitech (/c922_N/...):
    * Still open in OBS? A camera opens in one program at a time.
    * Replugged? Re-check its /dev/v4l/by-id path in CAMERAS in c922.launch.py.
    * Half rate (~15 Hz)? An exposure above 333 cannot sustain 30 fps."""

WIDTH = 78
HASH = '#' * WIDTH

# Watch verdicts.
OK = 'ok'              # stopped for a reason that is not a camera fault
SILENT = 'silent'      # a topic stopped producing -- the banner has been shown
ABORTED = 'aborted'    # abort() fired, i.e. the recorder died
ERROR = 'error'        # the watch itself threw


def select_topics(argv):
    """Which topics to record: the full list, or C922-only under --calib-exo.

    --calib-exo is for calibrating the exo cameras with the ego down. The
    subset drives both the bag and the watch, so a missing ego is expected
    rather than a startup abort.
    """
    if '--calib-exo' in argv:
        return [t for t in BAG_TOPICS if not t.startswith(EGO_PREFIX)]
    return BAG_TOPICS


def watch_topics(topics):
    """camera_info only -- see the transport note in the module docstring."""
    return [t for t in topics if t.endswith('/camera_info')]


class StreamWatch(Node):
    """Answers one question, repeatedly: is every camera still producing?

    Reports only. It holds no pid and stops nothing -- main() acts on the
    verdict.
    """

    def __init__(self, topics, label):
        super().__init__('record_session_watch')
        self.topics = list(topics)
        self.label = label
        self.counts = {}
        self.subs = []

    def _subscribe(self):
        self.counts = {t: 0 for t in self.topics}
        self.subs = [
            self.create_subscription(
                CameraInfo, t, lambda _m, t=t: self._bump(t), 1)
            for t in self.topics
        ]

    def _bump(self, topic):
        self.counts[topic] += 1

    def _unsubscribe(self):
        for sub in self.subs:
            self.destroy_subscription(sub)
        self.subs = []

    def _idle_for(self, seconds, abort=None):
        """Wait between probes. Sleeps rather than spins.

        Nothing is subscribed during the gap, so spin_once() would wake 20x a
        second to process nothing. Measured, that idle loop was ~1% of a core
        -- most of the watch's entire cost. Ctrl-C still interrupts a sleep.

        False if abort() fired -- the caller wants out before the timer.
        """
        end = time.monotonic() + seconds
        while time.monotonic() < end and rclpy.ok():
            if abort is not None and abort():
                return False
            time.sleep(min(0.25, end - time.monotonic()))
        return True

    def _listen(self, seconds, abort=None):
        """Spin until every topic has reported, or the window runs out.

        The early exit fires only when ALL topics report, so a quiet one still
        gets the full window and a silent verdict is still backed by `seconds`
        of listening. On a healthy rig this returns in ~0.5s instead of 5s,
        which is where most of the sampling cost went.

        False if abort() fired.
        """
        end = time.monotonic() + seconds
        while time.monotonic() < end and rclpy.ok():
            if all(self.counts.values()):
                break
            if abort is not None and abort():
                return False
            rclpy.spin_once(self, timeout_sec=0.05)
        return True

    def wait_for_startup(self, abort=None):
        """Silence is normal until every camera has produced a first frame."""
        self._subscribe()
        began = time.monotonic()
        if not self._listen(STARTUP_GRACE, abort):
            return ABORTED
        waited = time.monotonic() - began

        silent = [t for t, n in self.counts.items() if n == 0]
        if silent:
            self.shout(silent, f'never started (waited {waited:.0f}s)')
            return SILENT

        print(f'  all {len(self.topics)} stream(s) live after {waited:.1f}s',
              flush=True)
        return OK

    def probe_loop(self, abort=None):
        """Return OK, SILENT, or ABORTED.

        abort() is polled while waiting, so a recorder that dies is noticed
        immediately instead of at the next probe.
        """
        if not self.topics:
            self.get_logger().warn('no topics configured, nothing to watch')
            return OK

        while rclpy.ok():
            self._unsubscribe()
            if not self._idle_for(PROBE_PERIOD, abort):
                return ABORTED
            if not rclpy.ok():
                return OK

            self._subscribe()
            if not self._listen(PROBE_WINDOW, abort):
                return ABORTED

            silent = [t for t in self.topics if self.counts.get(t, 0) == 0]
            if silent:
                self.shout(silent, f'no frames in {PROBE_WINDOW:.0f}s')
                return SILENT
        return OK

    def shout(self, silent, why):
        """Print the failure at a size that cannot scroll past unnoticed."""
        lines = ['', HASH, HASH, '', '        ####   CAMERA SILENT   ####', '']
        lines.extend(f'        >>>  {topic}' for topic in silent)
        lines.extend(['', f'        {why}', '', f'        EPISODE: {self.label}',
                      '', HASH, ''])
        print('\n'.join(lines), file=sys.stderr, flush=True)


def banner(title, *lines):
    """Print at a size that cannot be missed, and ring the terminal bell."""
    print('\a', end='')
    out = ['', HASH, HASH, '', f'        ####   {title}   ####', '']
    out.extend(lines)
    out.extend(['', HASH, HASH, ''])
    print('\n'.join(out), flush=True)


def die(message):
    sys.exit(f'  {message}')


def next_episode(root, prefix):
    """Highest existing <prefix>_N under root, plus one.

    The glob's trailing '_' scopes to this exact code; the regex then keeps
    only a pure-integer tail, so a hand-renamed dir like '<prefix>_backup'
    can't corrupt the count.
    """
    nums = []
    for path in glob.glob(os.path.join(root, f'{prefix}_*')):
        m = re.fullmatch(rf'{re.escape(prefix)}_(\d+)', os.path.basename(path))
        if m:
            nums.append(int(m.group(1)))
    return max(nums, default=0) + 1


def preflight(out, root):
    os.makedirs(root, exist_ok=True)
    if os.path.exists(out):
        die(f'{out} already exists -- should not happen with auto-numbering')

    free_gb = shutil.disk_usage(root).free // 2**30
    if free_gb < HARD_FREE_GB:
        die(f'only {free_gb}GB free on {root} -- not enough for a usable episode')
    if free_gb < MIN_FREE_GB:
        print(f'  !! only {free_gb}GB free on {root} (want {MIN_FREE_GB}GB+)')


def start_recorder(out, topics):
    """Start the bag and return the process, whose pid is the whole point.

    start_new_session puts it in its own process group, so Ctrl-C in this
    terminal reaches only us. Nothing closes the bag except stop_recorder().
    """
    return subprocess.Popen(
        ['ros2', 'bag', 'record', '-s', 'mcap', '-o', out, *topics],
        start_new_session=True)


def stop_recorder(proc):
    """SIGINT and wait. True if it closed on its own, False if forced."""
    if proc.poll() is not None:
        return False                       # already gone; it did not close

    print('  closing the bag -- do not interrupt', flush=True)
    proc.send_signal(signal.SIGINT)

    # Progress, not silence. A silent wait is indistinguishable from a hang,
    # and interrupting the close is how a whole session is lost.
    waited = 0
    while proc.poll() is None and waited < FLUSH_TIMEOUT:
        time.sleep(1)
        waited += 1
        if waited % 5 == 0:
            print(f'    still closing... {waited}s', flush=True)

    if proc.poll() is None:
        print(f'  !! recorder still up {FLUSH_TIMEOUT}s after SIGINT '
              f'-- escalating', file=sys.stderr, flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        return False

    print(f'  bag closed after {waited}s', flush=True)
    return True


def probe(watch, abort):
    """Run the probe loop. ERROR if the watch itself broke.

    A bug in the watch must not be reported as a camera fault, so its own
    failure gets a verdict of its own.
    """
    try:
        return watch.probe_loop(abort=abort)
    except KeyboardInterrupt:
        return OK
    except Exception:
        traceback.print_exc()
        return ERROR


def report(verdict, out, elapsed):
    """Say what happened and what it means for the session. Returns exit code."""
    if verdict == SILENT:
        banner('CAMERA SILENT -- RECORDING STOPPED',
               '  The banner above names the topic that stopped.', '',
               HINT, '',
               f'  Bag closed after {elapsed}s: {out}',
               f'  Its last {int(PROBE_PERIOD + PROBE_WINDOW)}s are missing '
               f'that camera.',
               '  Terminal 1 is STILL RUNNING -- Ctrl-C that window too.')
        code = 1
    elif verdict == ABORTED:
        banner('RECORDER DIED -- NOT YOUR DOING',
               f'  ros2 bag record exited on its own after {elapsed}s.',
               '  Disk full, or it crashed. Check: df -h .', '',
               '  Anything filmed since then was NOT recorded.',
               f'  Bag: {out}')
        code = 3
    elif verdict == ERROR:
        banner('WATCH BROKE -- RECORDING STOPPED TO BE SAFE',
               '  The traceback above is the bug. Nothing was monitoring the',
               '  cameras, so the bag was closed rather than left unwatched.',
               '', f'  Bag closed after {elapsed}s: {out}')
        code = 4
    else:
        print(f'\n  stopped after {elapsed}s -- {out}\n')
        code = 0

    # Proof, not assumption. No metadata.yaml means the bag did not close.
    if os.path.isfile(os.path.join(out, 'metadata.yaml')):
        subprocess.run(['ros2', 'bag', 'info', out], check=False)
    else:
        print(f'  !! no {out}/metadata.yaml -- the bag did not close cleanly\n'
              f'  !! recover it with: ros2 bag reindex -s mcap {out}',
              file=sys.stderr)
        code = 5
    return code


def main(args=None):
    argv = sys.argv[1:] if args is None else args
    codes = [a for a in argv if not a.startswith('-')]
    if not codes:
        die('usage: ros2 run d435i_multicam_launch record_session '
            '<episode-code> [--calib-exo]')

    code = codes[0]
    topics = select_topics(argv)
    watched = watch_topics(topics)
    root = os.environ.get('BAG_ROOT', '.')
    prefix = f'{date.today().isoformat()}_{code}'
    n = next_episode(root, prefix)
    out = os.path.join(root, f'{prefix}_{n}')
    preflight(out, root)

    rclpy.init()
    watch = StreamWatch(watched, f'{code}-{n}')
    try:
        # Before the bag exists: a dead camera costs a refusal, not a session,
        # and leaves no empty directory behind to block the retry.
        print(f'\n  waiting for all {len(watched)} streams '
              f'(up to {STARTUP_GRACE:.0f}s)...', flush=True)
        if watch.wait_for_startup() != OK:
            banner('CAMERAS NOT READY -- NOTHING RECORDED',
                   '  The banner above names what never started.', '',
                   HINT, '',
                   '  No bag was created. Fix it and run this again.')
            sys.exit(2)

        proc = start_recorder(out, topics)
        print(f'\n  recording episode {code}-{n} -> {out}\n'
              f'  {len(topics)} topics, watching {len(watched)} '
              f'for silence every {PROBE_PERIOD:.0f}s\n'
              f'  Ctrl-C to stop. Do NOT close this window -- SIGHUP leaves '
              f'the bag unreadable.\n', flush=True)

        began = time.monotonic()
        try:
            verdict = probe(watch, abort=lambda: proc.poll() is not None)
        except KeyboardInterrupt:
            verdict = OK
        finally:
            # Unconditional: however this ends, the recorder is never left
            # behind holding an unreadable bag.
            stop_recorder(proc)
    finally:
        watch.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    sys.exit(report(verdict, out, int(time.monotonic() - began)))


if __name__ == '__main__':
    main()
