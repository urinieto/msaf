#!/usr/bin/env python
from nose.tools import raises

# Msaf imports
import msaf

from msaf.configparser import (AddConfigVar, BoolParam, EnumStr,
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
    msaf.config.wrong_variable_name


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


def test_empty_config_val():
    """Tests an empty config value in the string."""
    MSAF_FLAGS = "param1=3, ,param2=10"
    conf = msaf.configparser.parse_config_string(MSAF_FLAGS,
                                                 issue_warnings=True)
    assert "param1" in conf.keys()
    assert "param2" in conf.keys()


def test_override_config_val():
    """Tests overriding value in the string."""
    MSAF_FLAGS = "param1=3,param1=10"
    conf = msaf.configparser.parse_config_string(MSAF_FLAGS,
                                                 issue_warnings=True)
    assert len(conf.keys()) == 1
    assert conf["param1"] == "10"


@raises(KeyError)
def test_fetch_nonexisting_config_val():
    """Tests fetching non-existing value in the conf."""
    msaf.configparser.fetch_val_for_key("caca", delete_key=False)


@raises(KeyError)
def test_fetch_too_many_sections():
    """Tests fetching a key with too many sections."""
    msaf.configparser.fetch_val_for_key("caca.merda.merdeta",
                                        delete_key=False)


@raises(AttributeError)
def test_add_existing_config_var():
    """Tests adding a var that already exists."""
    AddConfigVar('sample_rate', "doc", IntParam(1))


def test_add_filter_config_var():
    """Tests adding a var that already exists."""
    configparam = EnumStr(".wav", ".mp3", ".aif",
                          convert=lambda x: x.replace("wav", "aif"))
    AddConfigVar('new_var', "doc", configparam)
    assert msaf.config.new_var == ".aif"


def test_config():
    """All the features should be in the features register."""
    print(msaf.config)
    print(msaf.config.sample_rate)
    print(msaf.config.cqt.bins)
    print(str(msaf.config.cqt.ref_power))
