#!/usr/bin/env python
#
# Run me as follows:
# cd tests/
# nosetests

import jams
import librosa
from nose.tools import raises
import numpy as np
import os

# Msaf imports
import msaf
from msaf.exceptions import WrongAlgorithmID
from msaf.input_output import FileStruct

# Global vars
audio_file = os.path.join("fixtures", "chirp.mp3")
sr = msaf.config.sample_rate


def test_read_hier_references():
    one_jams = os.path.join("..", "datasets", "Sargon", "references",
                            "01-Sargon-Mindless.jams")
    three_jams = os.path.join("..", "datasets", "SALAMI", "references",
                              "SALAMI_200.jams")

    audio, fs = librosa.load(audio_file, sr=sr)
    y_harmonic, y_percussive = librosa.effects.hpss(audio)

    # One level file
    hier_bounds, hier_labels, hier_levels = \
        msaf.io.read_hier_references(one_jams)
    assert len(hier_bounds) == len(hier_labels) and \
        len(hier_labels) == len(hier_levels)
    assert len(hier_levels) == 1

    # Three level file
    hier_bounds, hier_labels, hier_levels = \
        msaf.io.read_hier_references(three_jams)
    assert len(hier_bounds) == len(hier_labels) and \
        len(hier_labels) == len(hier_levels)
    assert len(hier_levels) == 3


def test_find_estimation():
    est_file = os.path.join("fixtures", "01-Sargon-Mindless-ests.jams")
    jam = jams.load(est_file)
    params = {"hier": False}
    ann = msaf.io.find_estimation(jam, "sf", None, params)
    assert len(ann.data) == 21


@raises(WrongAlgorithmID)
def test_find_estimation_wrong():
    est_file = os.path.join("fixtures", "01-Sargon-Mindless-ests.jams")
    jam = jams.load(est_file)
    params = {"hier": False}
    msaf.io.find_estimation(jam, "sf", "caca", params)


def test_find_estimation_multiple():
    est_file = os.path.join("fixtures", "01-Sargon-Mindless-est-multiple.jams")
    jam = jams.load(est_file)
    params = {"hier": True}
    ann = msaf.io.find_estimation(jam, "sf", None, params)
    assert len(ann.data) == 86


@raises(AssertionError)
def test_save_estimations_hier_wrong():
    file_struct = FileStruct("dummy")
    file_struct.features_file = os.path.join("fixtures", "01_-_Come_Together.json")

    # Wrong times and labels (don't match)
    times = [np.arange(0, 10, 2), np.arange(0, 10, 1)]
    labels = [['A', 'B'], ['a', 'a', 'b']]

    # Should raise assertion error
    msaf.io.save_estimations(file_struct, times, labels, None, None)

