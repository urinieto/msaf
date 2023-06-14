#!/usr/bin/env python
#
# Run me as follows:
# cd tests/
# nosetests

import jams
import librosa
from pytest import raises
import numpy as np
import os
import shutil

# Msaf imports
import msaf
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


def test_find_estimation_multiple():
    est_file = os.path.join("fixtures", "01-Sargon-Mindless-est-multiple.jams")
    jam = jams.load(est_file)
    params = {"hier": True}
    ann = msaf.io.find_estimation(jam, "sf", None, params)
    assert len(ann.data) == 86


def test_save_estimations_hier_wrong():
    file_struct = FileStruct("dummy")
    file_struct.features_file = os.path.join("fixtures",
                                             "01_-_Come_Together.json")

    # Wrong times and labels (don't match)
    times = [np.arange(0, 10, 2), np.arange(0, 10, 1)]
    labels = [['A', 'B'], ['a', 'a', 'b']]

    # Should raise assertion error
    with raises(AssertionError):
        msaf.io.save_estimations(file_struct, times, labels, None, None)


def test_save_estimations_existing():
    # Copy estimations file temporarily
    est_file = "tmp.jams"
    shutil.copy(os.path.join("fixtures", "01-Sargon-Mindless-ests.jams"),
                est_file)

    # First, find estimation
    jam = jams.load(est_file)
    params = {"hier": False}
    ann = msaf.io.find_estimation(jam, "sf", None, params)
    assert len(ann.data) == 21

    # Add to estimation which will replace it
    file_struct = FileStruct("dummy")
    file_struct.est_file = est_file
    file_struct.features_file = os.path.join("fixtures",
                                             "01_-_Come_Together.json")
    times = np.array([0, 10, 20, 30])
    labels = np.array([-1] * (len(times) - 1))
    msaf.io.save_estimations(file_struct, times, labels, "sf", None, **params)
    jam = jams.load(est_file)
    ann = msaf.io.find_estimation(jam, "sf", None, params)
    assert len(ann.data) == len(times) - 1

    # Add to estimation which will add a new one
    times2 = np.array([0, 10, 20, 30, 40])
    labels2 = np.array([-1] * (len(times2) - 1))
    params2 = {"sf_param": 0.1, "hier": False}
    msaf.io.save_estimations(file_struct, times2, labels2, "sf", None,
                             **params2)

    # Make sure the old one is the same
    jam = jams.load(est_file)
    ann = msaf.io.find_estimation(jam, "sf", None, params)
    assert len(ann.data) == len(times) - 1

    # Make sure the new one is the same
    ann = msaf.io.find_estimation(jam, "sf", None, params2)
    assert len(ann.data) == len(times2) - 1

    # Add hierarchical
    times3 = [np.array([0, 40]), np.array([0, 10, 20, 30, 40])]
    labels3 = [np.array([-1] * (len(times3[0]) - 1)),
               np.array([-1] * (len(times3[1]) - 1))]
    params3 = {"sf_param": 0.1, "hier": True}
    msaf.io.save_estimations(file_struct, times3, labels3,
                             "sf", None, **params3)
    jam = jams.load(est_file)
    ann = msaf.io.find_estimation(jam, "sf", None, params3)
    assert len(ann.data) == 5
    assert ann.data[0].value["level"] == 0
    assert ann.data[1].value["level"] == 1
    assert ann.data[2].value["level"] == 1
    assert ann.data[3].value["level"] == 1
    assert ann.data[4].value["level"] == 1

    # Cleanup
    os.remove(est_file)


def test_write_mirex():
    times = np.array([0, 10, 20, 30])
    labels = np.array([0, 1, 2])
    out_file = "out_mirex.txt"
    msaf.io.write_mirex(times, labels, out_file)

    # Check that results is correct
    inters = msaf.utils.times_to_intervals(times)
    with open(out_file, "r") as f:
        lines = f.readlines()
    assert len(lines) == 3
    for line, inter, label in zip(lines, inters, labels):
        saved_inter = [0, 0]
        saved_inter[0], saved_inter[1], saved_label = line.split('\t')
        assert float(saved_inter[0]) == inter[0]
        assert float(saved_inter[1]) == inter[1]
        assert float(saved_label) == label

    # Cleanup
    os.remove(out_file)


def test_align_times():
    times = np.array([0, 10, 20, 30])
    frames = np.array([0, 12, 19, 25, 31])
    aligned_times = msaf.io.align_times(times, frames)
    assert len(times) == len(aligned_times)
