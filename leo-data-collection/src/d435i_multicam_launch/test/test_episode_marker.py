"""Unit tests for the marker node's state machine.

The load-bearing property here is that the node publishes what was pressed and
never tidies it up. A node that silently completed a missing pair would make
every sequence well-formed, and a well-formed sequence is exactly what nobody
can audit.
"""

from d435i_multicam_launch.episode_marker import EpisodeMarker
import pytest
import rclpy
from rclpy.parameter import Parameter


class FakePublisher:
    """Captures markers instead of putting them on the wire."""

    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


@pytest.fixture
def node():
    """Build a marker node whose publisher captures instead of transmitting."""
    rclpy.init()
    marker = EpisodeMarker()
    marker.pub = FakePublisher()
    yield marker
    marker.destroy_node()
    rclpy.shutdown()


def markers(node):
    return [message.frame_id for message in node.pub.messages]


# --- the happy path --------------------------------------------------------

def test_a_start_and_end_make_a_matched_pair(node):
    node.mark_start()
    node.mark_end()

    assert markers(node) == ['start_1', 'end_1']
    assert not node.inside_episode


def test_the_index_advances_once_per_episode(node):
    node.mark_start()
    node.mark_end()
    node.mark_start()
    node.mark_end()

    assert markers(node) == ['start_1', 'end_1', 'start_2', 'end_2']


def test_the_index_is_written_into_the_message_not_implied(node):
    """A dropped marker must leave a visible hole in the numbering."""
    node.mark_start()

    assert node.pub.messages[0].frame_id == 'start_1'


def test_a_marker_carries_the_time_it_was_pressed(node):
    node.mark_start()
    stamp = node.pub.messages[0].stamp

    assert stamp.sec > 0


# --- malformed input must survive intact -----------------------------------

def test_a_start_while_open_is_published_literally(node, capsys):
    """No synthetic end is inserted; the mistake goes into the bag as made."""
    node.mark_start()
    node.mark_start()

    assert markers(node) == ['start_1', 'start_2']
    assert 'still open' in capsys.readouterr().out


def test_an_end_with_nothing_open_is_still_published(node, capsys):
    node.mark_end()

    assert markers(node) == ['end_0']
    assert 'no episode open' in capsys.readouterr().out


def test_a_repeated_end_is_published_twice(node, capsys):
    node.mark_start()
    node.mark_end()
    node.mark_end()

    assert markers(node) == ['start_1', 'end_1', 'end_1']
    assert 'no episode open' in capsys.readouterr().out


def test_quitting_mid_episode_leaves_the_episode_open(node):
    node.mark_start()

    assert node.inside_episode


# --- the key-repeat guard --------------------------------------------------

def test_the_first_press_of_a_key_is_never_a_repeat(node):
    assert node.is_repeat('s') is False


def test_the_same_key_again_immediately_is_a_repeat(node):
    node.is_repeat('s')

    assert node.is_repeat('s') is True


def test_a_different_key_immediately_is_not_a_repeat(node):
    """A fast s-then-e is a real short episode, not a bounce."""
    node.is_repeat('s')

    assert node.is_repeat('e') is False


def test_the_same_key_after_the_window_is_deliberate(node):
    node.is_repeat('s')
    node.last_key_at -= node.debounce + 0.1

    assert node.is_repeat('s') is False


def test_the_window_comes_from_a_ros_parameter(node):
    node.set_parameters(
        [Parameter('debounce_sec', Parameter.Type.DOUBLE, 5.0)])

    assert node.debounce == 5.0


def test_the_guard_does_not_publish_anything(node):
    node.is_repeat('s')
    node.is_repeat('s')

    assert node.pub.messages == []
