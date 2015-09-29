#!/usr/bin/env python
#
# Run me as follows:
# cd tests/
# nosetests

from nose.tools import assert_raises
from types import ModuleType
import numpy.testing as npt
import os

# Msaf imports
import msaf

# Global vars
audio_file = os.path.join("fixtures", "chirp.mp3")
long_audio_file = os.path.join("..", "datasets", "Sargon", "audio",
                               "01-Sargon-Mindless.mp3")
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
    assert_raises(RuntimeError,
                  msaf.run.get_boundaries_module, fake_module_name)

    # Check that a RuntimeError is raised when calling it with invalid
    # boundary id
    assert_raises(RuntimeError,
                  msaf.run.get_boundaries_module, "fmc2d")


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
    assert_raises(RuntimeError,
                  msaf.run.get_labels_module, fake_module_name)

    # Check that a RuntimeError is raised when calling it with invalid
    # labels id
    assert_raises(RuntimeError,
                  msaf.run.get_labels_module, "foote")


def test_run_algorithms():
    bound_ids = msaf.io.get_all_boundary_algorithms()
    label_ids = msaf.io.get_all_label_algorithms()

    # Add ground truth to boundary id
    bound_ids += ["gt"]

    # Add None to labels
    label_ids += [None]

    # Config params
    feature = "hpcp"
    annot_beats = False
    framesync = False
    file_struct = msaf.io.FileStruct(audio_file)
    all_features = msaf.featextract.compute_features_for_audio_file(audio_file)
    msaf.utils.ensure_dir(os.path.dirname(file_struct.features_file))
    msaf.featextract.save_features(file_struct.features_file,
                                   all_features)

    # Running all algorithms on a file that is too short
    for bound_id in bound_ids:
        for label_id in label_ids:
            config = msaf.io.get_configuration(feature, annot_beats, framesync,
                                               bound_id, label_id)
            config["features"] = msaf.io.get_features(
                audio_file, annot_beats, framesync)
            config["hier"] = False
            est_times, est_labels = msaf.run.run_algorithms(
                audio_file, bound_id, label_id, config)
            assert len(est_times) == 2
            assert len(est_labels) == 1
            npt.assert_almost_equal(est_times[0], 0.0, decimal=2)
            npt.assert_almost_equal(est_times[-1],
                                    config["features"]["anal"]["dur"],
                                    decimal=2)

    def _test_run_msaf(bound_id, label_id):
        config = msaf.io.get_configuration(feature, annot_beats, framesync,
                                           bound_id, label_id)
        config["features"] = msaf.io.get_features(
            audio_file, annot_beats, framesync)
        config["hier"] = False
        est_times, est_labels = msaf.run.run_algorithms(long_audio_file,
                                                        bound_id,
                                                        label_id,
                                                        config)
        npt.assert_almost_equal(est_times[0], 0.0, decimal=2)
        assert len(est_times) - 1 == len(est_labels)
        npt.assert_almost_equal(
            est_times[-1], config["features"]["anal"]["dur"], decimal=2)

    # Running all boundary algorithms on a relatively long file
    for bound_id in bound_ids:
        if bound_id == "gt":
            continue
        yield (_test_run_msaf, bound_id, None)

    # Combining boundaries with labels
    for bound_id in bound_ids:
        if bound_id == "gt":
            continue
        for label_id in label_ids:
            yield (_test_run_msaf, bound_id, label_id)
