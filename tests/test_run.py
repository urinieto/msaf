#!/usr/bin/env python
#
# Run me as follows:
# cd tests/
# nosetests

# For plotting and testing
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
import matplotlib.style
matplotlib.style.use('seaborn-ticks')

from pytest import raises
import numpy.testing as npt
import os
from types import ModuleType

# Msaf imports
import msaf
from msaf.features import Features
from msaf.exceptions import (NoHierBoundaryError, FeaturesNotFound,
                             NoAudioFileError)


# Global vars
audio_file = os.path.join("fixtures", "chirp.mp3")
long_audio_file = os.path.join("fixtures", "Sargon_test", "audio",
                               "Mindless_cut.mp3")
fake_module_name = "fake_name_module"


def test_get_boundaries_module():
    # Check that it returns modules for all the existing MSAF boundaries algos
    bound_ids = msaf.io.get_all_boundary_algorithms()
    for bound_id in bound_ids:
        bound_module = msaf.run.get_boundaries_module(bound_id)
        assert isinstance(bound_module, ModuleType)

    # Check that "gt" returns None
    assert msaf.run.get_boundaries_module("gt") is None

    # Check that a AttributeError is raised when calling it with non-existent
    # boundary id
    with raises(RuntimeError):
        msaf.run.get_boundaries_module(fake_module_name)

    # Check that a RuntimeError is raised when calling it with invalid
    # boundary id
    with raises(RuntimeError):
        msaf.run.get_boundaries_module("fmc2d")


def test_get_labels_module():
    # Check that it returns modules for all the existing MSAF boundaries algos
    label_ids = msaf.io.get_all_label_algorithms()
    for label_id in label_ids:
        label_module = msaf.run.get_labels_module(label_id)
        assert isinstance(label_module, ModuleType)

    # Check that None returns None
    assert msaf.run.get_labels_module(None) is None

    # Check that a AttributeError is raised when calling it with non-existent
    # labels id
    with raises(RuntimeError):
        msaf.run.get_labels_module(fake_module_name)

    # Check that a RuntimeError is raised when calling it with invalid
    # labels id
    with raises(RuntimeError):
        msaf.run.get_labels_module("foote")


def test_run_algorithms():
    """Test running all the algorithms."""
    bound_ids = msaf.io.get_all_boundary_algorithms()
    label_ids = msaf.io.get_all_label_algorithms()

    # Add ground truth to boundary id
    bound_ids += ["gt"]

    # Add None to labels
    label_ids += [None]

    # Config params
    feature = "pcp"
    annot_beats = False
    framesync = False
    file_struct = msaf.io.FileStruct(audio_file)
    file_struct.features_file = msaf.config.features_tmp_file

    # Running all algorithms on a file that is too short
    for bound_id in bound_ids:
        for label_id in label_ids:
            print("bound_id: %s,\tlabel_id: %s" % (bound_id, label_id))
            config = msaf.io.get_configuration(feature, annot_beats, framesync,
                                               bound_id, label_id)
            config["hier"] = False
            config["features"] = Features.select_features(
                feature, file_struct, annot_beats, framesync)
            est_times, est_labels = msaf.run.run_algorithms(
                file_struct, bound_id, label_id, config)
            assert len(est_times) == 2
            assert len(est_labels) == 1
            npt.assert_almost_equal(est_times[0], 0.0, decimal=2)
            npt.assert_almost_equal(est_times[-1], config["features"].dur,
                                    decimal=2)

    # Commpute and save features for long audio file
    file_struct = msaf.io.FileStruct(long_audio_file)
    file_struct.features_file = msaf.config.features_tmp_file

    def _test_run_msaf(bound_id, label_id, hier=False):
        print("bound_id: %s,\tlabel_id: %s" % (bound_id, label_id))
        config = msaf.io.get_configuration(feature, annot_beats, framesync,
                                           bound_id, label_id)
        config["hier"] = hier
        config["features"] = Features.select_features(
            feature, file_struct, annot_beats, framesync)
        est_times, est_labels = msaf.run.run_algorithms(
            file_struct, bound_id, label_id, config)

        # Take the first level if hierarchy algorithm
        if hier:
            est_times = est_times[0]
            est_labels = est_labels[0]

        npt.assert_almost_equal(est_times[0], 0.0, decimal=2)
        assert len(est_times) - 1 == len(est_labels)
        npt.assert_almost_equal(est_times[-1], config["features"].dur,
                                decimal=2)

    # Running all boundary algorithms on a relatively long file
    # Combining boundaries with labels
    for bound_id in bound_ids:
        if bound_id == "gt":
            continue
        for label_id in label_ids:
            _test_run_msaf(bound_id, label_id, False)

    # Test the hierarchical algorithms
    hier_ids = ["olda", "scluster"]
    for hier_bounds_id in hier_ids:
        for hier_labels_id in hier_ids:
            if hier_labels_id == "olda":
                hier_labels_id = "fmc2d"
            _test_run_msaf(hier_bounds_id, hier_labels_id, True)


def test_no_bound_hierarchical():
    with raises(NoHierBoundaryError):
        msaf.run.run_hierarchical(None, None, None, None, None)


def test_no_gt_flat_bounds():
    """Make sure the results are empty if there is not ground truth found."""
    feature = "pcp"
    annot_beats = False
    framesync = False
    file_struct = msaf.io.FileStruct(audio_file)
    file_struct.features_file = msaf.config.features_tmp_file

    config = {}
    config["features"] = Features.select_features(
        feature, file_struct, annot_beats, framesync)
    est_times, est_labels = msaf.run.run_flat(file_struct, None, None,
                                              None, config, 0)
    assert(not est_times)
    assert(not est_labels)


def test_process_track():
    bounds_id = "foote"
    labels_id = None
    file_struct = msaf.io.FileStruct(audio_file)
    file_struct.features_file = msaf.config.features_tmp_file
    file_struct.est_file = "tmp.json"

    config = {}
    config["feature"] = "pcp"
    config["annot_beats"] = False
    config["framesync"] = False
    config["hier"] = False
    est_times, est_labels = msaf.run.process_track(
        file_struct, bounds_id, labels_id, config)

    assert os.path.isfile(file_struct.est_file)
    os.remove(file_struct.est_file)


def test_process_with_gt():
    bounds_id = "gt"
    labels_id = "fmc2d"
    est_times, est_labels = msaf.run.process(
        long_audio_file, boundaries_id=bounds_id, labels_id=labels_id)
    assert est_times[0] == 0
    assert len(est_times) == len(est_labels) + 1


def test_process_wrong_feature():
    feature = "caca"
    with raises(FeaturesNotFound):
        est_times, est_labels = msaf.run.process(long_audio_file, feature=feature)


def test_process_wrong_path():
    wrong_path = "caca.mp3"
    with raises(NoAudioFileError):
        est_times, est_labels = msaf.run.process(wrong_path)


def test_process():
    est_times, est_labels = msaf.run.process(long_audio_file)
    assert est_times[0] == 0
    assert len(est_times) == len(est_labels) + 1


def test_process_sonify():
    out_wav = "out_wav.wav"
    est_times, est_labels = msaf.run.process(long_audio_file,
                                             sonify_bounds=True,
                                             out_bounds=out_wav)
    assert os.path.isfile(out_wav)
    os.remove(out_wav)


def test_process_dataset():
    ds_path = os.path.join("fixtures", "Sargon_test")
    res = msaf.run.process(ds_path)
    est_times, est_labels = res[0]
    assert est_times[0] == 0
    assert len(est_times) == len(est_labels) + 1
