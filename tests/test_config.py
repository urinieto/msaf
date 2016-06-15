#!/usr/bin/env python

import json
import librosa
from nose.tools import assert_equals, raises
import numpy as np
import numpy.testing as npt
import os

# Msaf imports
import msaf

from msaf.configparser import (AddConfigVar, BoolParam, ConfigParam, EnumStr,
                               FloatParam, IntParam, StrParam,
                               MsafConfigParser, MSAF_FLAGS_DICT)


def test_add_var():
    """Adds a config variable and checks that it's correctly stored."""
    val = 10
    AddConfigVar('test.new_var', "Test Variable only for unit testing",
                 IntParam(val))

    assert msaf.config.test.new_var == val


@raises(AttributeError)
def test_no_var():
    """Raises an AttributeError if the variable doesn't exist in the config."""
    _ = msaf.config.wrong_variable_name


@raises(ValueError)
def test_wrong_value():
    """Raises error if the variable type is wrong."""
    msaf.config.cqt.ref_power = 10


def test_warnings():
    """Tests that warnings are raised when needed."""
    MSAF_FLAGS = "what is this"
    msaf.configparser.parse_config_string(MSAF_FLAGS, issue_warnings=True)


def test_bool_var():
    """Adds a boolean variable."""
    AddConfigVar('test.my_new_bool', "Test bool variable only for testing",
                 BoolParam(True))
    assert msaf.config.test.my_new_bool


def test_none_str():
    """Adds a None String variable."""
    AddConfigVar('test.my_none_str', "Test None string",
                 StrParam(None))
    assert msaf.config.test.my_none_str is None


def test_config():
    """All the features should be in the features register."""
    print(msaf.config)
    print(msaf.config.sample_rate)
    print(msaf.config.cqt.bins)
    print(str(msaf.config.cqt.ref_power))
