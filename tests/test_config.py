#!/usr/bin/env python
# For plotting and testing
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

from pytest import raises
from collections import namedtuple

# Msaf imports
import msaf

from msaf.configparser import (AddConfigVar, BoolParam, EnumStr,
                               IntParam, StrParam, ListParam)


def test_print_config():
    """Tests that we can print the main config."""
    print(msaf.config)


def test_add_var():
    """Adds a config variable and checks that it's correctly stored."""
    val = 10
    AddConfigVar('test.new_var', "Test Variable only for unit testing",
                 IntParam(val))
    assert msaf.config.test.new_var == val


def test_no_var():
    """Raises an AttributeError if the variable doesn't exist in the config."""
    with raises(AttributeError):
        msaf.config.wrong_variable_name


def test_wrong_value():
    """Raises error if the variable type is wrong."""
    with raises(ValueError):
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


def test_fetch_nonexisting_config_val():
    """Tests fetching non-existing value in the conf."""
    with raises(KeyError):
        msaf.configparser.fetch_val_for_key("caca", delete_key=False)


def test_fetch_too_many_sections():
    """Tests fetching a key with too many sections."""
    with raises(KeyError):
        msaf.configparser.fetch_val_for_key("caca.merda.merdeta",
                                            delete_key=False)


def test_wrong_addconfig_root():
    """Tests adding a var that already exists in the root."""
    with raises(AttributeError):
        AddConfigVar('sample_rate', "doc", IntParam(1), root=msaf.config)


def test_wrong_addconfig_root_multsections():
    """Tests adding a var that already exists in the root with
    multiple sections and wrong root."""
    conf = namedtuple('conf', ['cqt'])
    with raises(TypeError):
        AddConfigVar('cqt.bins', "doc", IntParam(1), root=conf)


def test_add_existing_config_var():
    """Tests adding a var that already exists."""
    with raises(AttributeError):
        AddConfigVar('sample_rate', "doc", IntParam(1))


def test_wrong_callable_arg():
    """Tests adding a var with a wrong callable default param."""
    # We can add it
    AddConfigVar('my_int', "doc", IntParam(sorted))
    # but it should fail when retrieving it
    with raises(TypeError):
        msaf.config.my_int


def test_add_filter_config_var():
    """Tests adding a var that already exists."""
    configparam = EnumStr(".wav", ".mp3", ".aif",
                          convert=lambda x: x.replace("wav", "aif"))
    AddConfigVar('new_var', "doc", configparam)
    assert msaf.config.new_var == ".aif"


def test_allowoverride_fail():
    """Tests overriding a variable that can't be overridden."""
    configparam = EnumStr("caca", "merda", allow_override=False)
    AddConfigVar('new_var2', "doc", configparam)
    with raises(Exception):
        msaf.config.new_var2 = "caca2"  # Raise Exception


def test_allowoverride():
    """Tests overriding a variable that can be overridden."""
    configparam = EnumStr("caca", "merda", allow_override=True)
    AddConfigVar('new_var3', "doc", configparam)
    msaf.config.new_var3 = "merda"
    assert msaf.config.new_var3 == "merda"


def test_invalid_enumstr():
    """Tests invalid enumstrings."""
    with raises(ValueError):
        AddConfigVar('new_var4', "doc", EnumStr("caca", 42, "merda"))


def test_false_boolparam():
    """Tests a false boolean param."""
    AddConfigVar('new_var4', "doc", BoolParam("false"))
    assert not msaf.config.new_var4


def test_wrong_boolparam():
    """Tests a wrong boolean param."""
    with raises(ValueError):
        AddConfigVar('new_var5', "doc", BoolParam("falsitto"))


def test_empty_list_param():
    """Tests an empty list param."""
    with raises(ValueError):
        AddConfigVar('new_var6', "doc", ListParam([]))


def test_wrong_list_param():
    """Tests a wrong list param."""
    with raises(ValueError):
        AddConfigVar('new_var7', "doc", ListParam(42))
