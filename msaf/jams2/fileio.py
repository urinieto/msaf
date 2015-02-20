"""CPickle style methods for reading / writing JAMS.

TODO(ejhumphrey@nyu.edu): merge / append definition, for safely writing the
    union of two JAMS to disk.
"""
import json
from .pyjams import Jams


def load(filepath):
    """Load a JSON formatted stream from a file."""
    fpointer = open(filepath, 'r')
    return Jams(**json.load(fpointer))


def dump(jam, filepath):
    """Serialize jam as a JSON formatted stream to a file."""
    fpointer = open(filepath, 'w')
    json.dump(jam, fpointer, indent=2)
    fpointer.close()


def append(jam, filepath, new_filepath=None, on_conflict='fail'):
    """writeme"""
    old_jam = load(filepath)
    old_jam.add(jam, on_conflict=on_conflict)
    if new_filepath is None:
        new_filepath = filepath
    dump(old_jam, new_filepath)
