#!/usr/bin/env python
#
# Run me as follows:
# cd tests/
# nosetests

import glob
import json
import librosa
from nose.tools import nottest, eq_, raises, assert_equals, assert_raises
from types import ModuleType
import numpy.testing as npt
import os

# Msaf imports
import msaf
from msaf.input_output import FileStruct

# Global vars
audio_file = os.path.join("data", "chirp.mp3")


def test_get_boundaries_module():
    # Check that it returns modules for all the existing MSAF boundaries algos
    bound_ids = msaf.io.get_all_boundary_algorithms(msaf.algorithms)
    for bound_id in bound_ids:
        bound_module = msaf.run.get_boundaries_module(bound_id)
        assert isinstance(bound_module, ModuleType)

    # Check that "gt" returns None
    assert msaf.run.get_boundaries_module("gt") is None

    # Check that a AttributeError is raised when calling it with non-existent
    # boundary id
    assert_raises(RuntimeError,
                  msaf.run.get_boundaries_module, "cacadelavaca")

    # Check that a RuntimeError is raised when calling it with invalid
    # boundary id
    assert_raises(RuntimeError,
                  msaf.run.get_boundaries_module, "fmc2d")


def test_get_labels_module():
    # Check that it returns modules for all the existing MSAF boundaries algos
    label_ids = msaf.io.get_all_label_algorithms(msaf.algorithms)
    for label_id in label_ids:
        label_module = msaf.run.get_labels_module(label_id)
        assert isinstance(label_module, ModuleType)

    # Check that None returns None
    assert msaf.run.get_labels_module(None) is None

    # Check that a AttributeError is raised when calling it with non-existent
    # labels id
    assert_raises(RuntimeError,
                  msaf.run.get_labels_module, "cacadelavaca")

    # Check that a RuntimeError is raised when calling it with invalid
    # labels id
    assert_raises(RuntimeError,
                  msaf.run.get_labels_module, "foote")
