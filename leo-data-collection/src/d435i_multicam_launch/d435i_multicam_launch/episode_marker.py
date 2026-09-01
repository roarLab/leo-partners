#!/usr/bin/env python3
"""UNUSED: retired marker mechanism -- nothing records /episode_markers and no
script launches this node. Kept for reference; not wired into the runbook.

Publish episode start/stop markers into the running recording.

Run this in its own terminal, BEFORE starting the recorder:

    ros2 run d435i_multicam_launch episode_marker

    s  start an episode
    e  end the open episode
    q  quit

It cannot be put in a launch file: launch-managed nodes have their stdin
redirected, so there is nothing to type into.

Two keys rather than one toggle, and do not "simplify" it back: a missed press
here leaves an ILLEGAL sequence -- a start with no end, or an end with no
start -- which export_episodes writes into the CSV as MISSING_START or
MISSING_END and check_episodes then refuses to pass. Every sequence of toggles
is well-formed by construction, so a toggle cannot be checked at all: a missed
press just runs two episodes together and nothing downstream can tell.

An illegal press publishes nothing. 's' while an episode is open, or 'e' with
none open, is refused with a warning, so a leaned-on key cannot fill the bag
with episodes and the marker stream stays checkable.
"""

import sys
import termios
import time
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header

TOPIC = 'episode_markers'

COLS = ('index', 'action', 'timestamp', 'episode_index', 'duration')
WIDTHS = (7, 8, 12, 14, 10)


def row(values) -> str:
    return '  ' + ''.join(f'{str(v):<{w}}' for v, w in zip(values, WIDTHS))


class EpisodeMarker(Node):
    """Turns start and end keypresses into timestamped markers.

    It publishes exactly what was pressed and never repairs a mistake -- but it
    will not publish a press that cannot be true, so what reaches the bag is
    always a sequence the exporter can check.
    """

    def __init__(self):
        super().__init__('episode_marker')

        # Guards against held-key repeat (~33 ms intervals), not against the
        # operator. A fast double-tap could be a real short episode, so only
        # a repeat of the key within this window is swallowed.
        self.declare_parameter('debounce_sec', 0.4)

        # Reliable so a marker is not dropped under recording load; volatile
        # (the default) so markers from a previous run cannot leak into a
        # new bag.
        self.pub = self.create_publisher(Header, TOPIC, 10)

        self.event_index = 0          # row count printed so far
        self.episode_index = 0        # episodes started so far
        self.inside_episode = False
        self.episode_start_at = None  # monotonic, for duration

        self.last_key = None
        self.last_key_at = 0.0

    @property
    def debounce(self) -> float:
        return self.get_parameter('debounce_sec').value

    def is_repeat(self, key: str) -> bool:
        now = time.monotonic()
        repeat = key == self.last_key and (now - self.last_key_at) < self.debounce
        self.last_key = key
        self.last_key_at = now
        return repeat

    def publish(self, role: str) -> None:
        msg = Header()
        msg.stamp = self.get_clock().now().to_msg()
        # Index is written down, never implied by position: a dropped
        # message shows up as a gap in the numbering rather than silently
        # shifting everything after it.
        msg.frame_id = f'{role}_{self.episode_index}'
        self.pub.publish(msg)

    def start(self) -> None:
        """Open an episode. Refused, and nothing published, if one is open."""
        if self.inside_episode:
            warn(f"IGNORED 's' -- episode {self.episode_index} is still open. "
                 "Press 'e' to end it first")
            return

        self.episode_index += 1
        self.episode_start_at = time.monotonic()
        self.publish('start')
        self.event_index += 1
        self.print_row('start', duration='-')
        self.inside_episode = True

    def end(self) -> None:
        """Close the open episode. Refused if there is none."""
        if not self.inside_episode:
            warn("IGNORED 'e' -- no episode is open. Press 's' to start one")
            return

        duration = time.monotonic() - self.episode_start_at
        self.publish('end')
        self.event_index += 1
        self.print_row('end', duration=f'{duration:.1f}s')
        self.inside_episode = False
        if duration < 1.0:
            warn(f'episode {self.episode_index} lasted {duration:.1f}s -- '
                 'check that was a real episode, not a double-press')

    def print_row(self, action: str, duration: str) -> None:
        stamp = time.strftime('%H:%M:%S')
        print(row((self.event_index, action, stamp, self.episode_index,
                    duration)))


def warn(message: str) -> None:
    print(f'  !! {message}')


BANNER = """\
episode_marker -- publishing on /{topic}

  s  start an episode
  e  end the open episode
  q  quit

Start the recorder now if you have not already, and include /{topic}
in its topic list. Nothing here is undoable: markers go straight into the
bag, mistakes included.

's' with an episode already open, or 'e' with none open, is refused and
publishes nothing -- the table below only ever grows on a press that counted.
Everything between an 'e' and the next 's' belongs to no episode, which is
the point: keep scene resets and chat out of the data.
"""


def print_header() -> None:
    print(row(COLS))
    print('  ' + '-' * sum(WIDTHS))


def loop(node: EpisodeMarker) -> None:
    while True:
        key = sys.stdin.read(1).lower()

        if key in ('s', 'e'):
            # Two guards, for two different mistakes: this one swallows a held
            # key, node.start()/end() refuse a press that contradicts the
            # state.
            if node.is_repeat(key):
                warn(f'IGNORED {key!r} -- repeat within '
                     f'{node.debounce}s (key-repeat guard)')
                continue
            node.start() if key == 's' else node.end()

        elif key == 'q':
            if node.inside_episode:
                warn(f'quitting with episode {node.episode_index} still '
                     'open -- no end was recorded for it')
            return


def main(args=None) -> None:
    if not sys.stdin.isatty():
        sys.exit('episode_marker needs a terminal. Run it with "ros2 run", '
                 'not from a launch file.')

    rclpy.init(args=args)
    node = EpisodeMarker()
    print(BANNER.format(topic=TOPIC))
    print_header()

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        # cbreak, not raw: clears ICANON and ECHO so a bare keypress
        # registers immediately, but leaves ISIG alone so Ctrl-C still works.
        tty.setcbreak(fd)
        loop(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Unconditional. Skipping this leaves the shell with echo off -- you
        # type and see nothing and have to run "reset" blind.
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        print(f'\nstopped after {node.episode_index} episode(s)')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()