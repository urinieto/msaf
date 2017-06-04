"""This script contains relevant functions to read the configuration for
MSAF.
A bunch of it is basically shamefully copy pasted from the almighty theano."""

from __future__ import absolute_import, print_function, division
try:
    from configparser import (ConfigParser, NoOptionError, NoSectionError,
                              InterpolationError)
except ImportError:
    from six.moves.configparser import (ConfigParser, NoOptionError, NoSectionError,
                                        InterpolationError)
import os
import shlex
from six import StringIO
from six import string_types
import sys
import warnings

import msaf


MSAF_FLAGS = os.getenv(msaf.MSAF_FLAGS_VAR, "")
# The MSAF_FLAGS environment variable should be a list of comma-separated
# [section.]option=value entries. If the section part is omitted, there should
# be only one section that contains the given option.


class MsafConfigWarning(Warning):
    def warn(cls, message, stacklevel=0):
        warnings.warn(message, cls, stacklevel=stacklevel + 3)
    warn = classmethod(warn)


def parse_config_string(config_string, issue_warnings=True):
    """
    Parses a config string (comma-separated key=value components) into a dict.
    """
    config_dict = {}
    my_splitter = shlex.shlex(config_string, posix=True)
    my_splitter.whitespace = ','
    my_splitter.whitespace_split = True
    for kv_pair in my_splitter:
        kv_pair = kv_pair.strip()
        if not kv_pair:
            continue
        kv_tuple = kv_pair.split('=', 1)
        if len(kv_tuple) == 1:
            if issue_warnings:
                MsafConfigWarning.warn(
                    ("Config key '%s' has no value, ignoring it" %
                     kv_tuple[0]), stacklevel=1)
        else:
            k, v = kv_tuple
            # subsequent values for k will override earlier ones
            config_dict[k] = v
    return config_dict

MSAF_FLAGS_DICT = parse_config_string(MSAF_FLAGS, issue_warnings=True)


# MSAFRC can contain a colon-delimited list of config files, like
# MSAFRC=~rincewind/.msafrc:~/.msafrc
# In that case, definitions in files on the right (here, ~/.msafrc) have
# precedence over those in files on the left.
def config_files_from_msafrc():
    rval = [os.path.expanduser(s) for s in
            os.getenv(msaf.MSAFRC_VAR, msaf.MSAFRC_FILE).split(os.pathsep)]
    if os.getenv(msaf.MSAFRC_VAR) is None and sys.platform == "win32":
        # to don't need to change the filename and make it open easily
        rval.append(os.path.expanduser(msaf.MSAFRC_WIN_FILE))
    return rval


config_files = config_files_from_msafrc()
msaf_cfg = ConfigParser(
    {'USER': os.getenv("USER", os.path.split(os.path.expanduser('~'))[-1]),
     'LSCRATCH': os.getenv("LSCRATCH", ""),
     'TMPDIR': os.getenv("TMPDIR", ""),
     'TEMP': os.getenv("TEMP", ""),
     'TMP': os.getenv("TMP", ""),
     'PID': str(os.getpid()),
     }
)
msaf_cfg.read(config_files)
# Having a raw version of the config around as well enables us to pass
# through config values that contain format strings.
# The time required to parse the config twice is negligible.
msaf_raw_cfg = ConfigParser()
msaf_raw_cfg.read(config_files)


def fetch_val_for_key(key, delete_key=False):
    """Return the overriding config value for a key.
    A successful search returns a string value.
    An unsuccessful search raises a KeyError
    The (decreasing) priority order is:
    - MSAF_FLAGS
    - ~./msafrc
    """

    # first try to find it in the FLAGS
    try:
        if delete_key:
            return MSAF_FLAGS_DICT.pop(key)
        return MSAF_FLAGS_DICT[key]
    except KeyError:
        pass

    # next try to find it in the config file

    # config file keys can be of form option, or section.option
    key_tokens = key.rsplit('.', 1)
    if len(key_tokens) == 2:
        section, option = key_tokens
    else:
        section, option = 'global', key
    try:
        try:
            return msaf_cfg.get(section, option)
        except InterpolationError:
            return msaf_raw_cfg.get(section, option)
    except (NoOptionError, NoSectionError):
        raise KeyError(key)

_config_var_list = []


def _config_print(thing, buf, print_doc=True):
    for cv in _config_var_list:
        print(cv, file=buf)
        if print_doc:
            print("    Doc: ", cv.doc, file=buf)
        print("    Value: ", cv.__get__(True, None), file=buf)
        print("", file=buf)


class MsafConfigParser(object):
    # properties are installed by AddConfigVar
    _i_am_a_config_class = True

    def __str__(self, print_doc=True):
        sio = StringIO()
        _config_print(self.__class__, sio, print_doc=print_doc)
        return sio.getvalue()

# N.B. all instances of MsafConfigParser give access to the same properties.
config = MsafConfigParser()


# The data structure at work here is a tree of CLASSES with
# CLASS ATTRIBUTES/PROPERTIES that are either a) INSTANTIATED
# dynamically-generated CLASSES, or b) ConfigParam instances.  The root
# of this tree is the MsafConfigParser CLASS, and the internal nodes
# are the SubObj classes created inside of AddConfigVar().
# Why this design ?
# - The config object is a true singleton.  Every instance of
#   MsafConfigParser is an empty instance that looks up attributes/properties
#   in the [single] MsafConfigParser.__dict__
# - The subtrees provide the same interface as the root
# - ConfigParser subclasses control get/set of config properties to guard
#   against craziness.
def AddConfigVar(name, doc, configparam, root=config):
    """Add a new variable to msaf.config

    Parameters
    ----------
    name: str
        String of the form "[section0.[section1.[etc]]]option", containing the
        full name for this configuration variable.
    string: str
        What does this variable specify?
    configparam: `ConfigParam`
        An object for getting and setting this configuration parameter.
    root: object
        Used for recursive calls -- do not provide an argument for this
        parameter.
    """

    # This method also performs some of the work of initializing ConfigParam
    # instances

    if root is config:
        # only set the name in the first call, not the recursive ones
        configparam.fullname = name
    sections = name.split('.')
    if len(sections) > 1:
        # set up a subobject
        if not hasattr(root, sections[0]):
            # every internal node in the config tree is an instance of its own
            # unique class
            class SubObj(object):
                _i_am_a_config_class = True
            setattr(root.__class__, sections[0], SubObj())
        newroot = getattr(root, sections[0])
        if (not getattr(newroot, '_i_am_a_config_class', False) or
                isinstance(newroot, type)):
            raise TypeError(
                'Internal config nodes must be config class instances',
                newroot)
        return AddConfigVar('.'.join(sections[1:]), doc, configparam,
                            root=newroot)
    else:
        if hasattr(root, name):
            raise AttributeError('This name is already taken',
                                 configparam.fullname)
        configparam.doc = doc
        # Trigger a read of the value from config files and env vars
        # This allow to filter wrong value from the user.
        if not callable(configparam.default):
            configparam.__get__(root, type(root), delete_key=True)
        else:
            # We do not want to evaluate now the default value
            # when it is a callable.
            try:
                fetch_val_for_key(configparam.fullname)
                # The user provided a value, filter it now.
                configparam.__get__(root, type(root), delete_key=True)
            except KeyError:
                pass
        setattr(root.__class__, sections[0], configparam)
        _config_var_list.append(configparam)


class ConfigParam(object):

    def __init__(self, default, filter=None, allow_override=True):
        """
        If allow_override is False, we can't change the value after the import
        of Theano. So the value should be the same during all the execution.
        """
        self.default = default
        self.filter = filter
        self.allow_override = allow_override
        self.is_default = True
        # N.B. --
        # self.fullname  # set by AddConfigVar
        # self.doc       # set by AddConfigVar

        # Note that we do not call `self.filter` on the default value: this
        # will be done automatically in AddConfigVar, potentially with a
        # more appropriate user-provided default value.
        # Calling `filter` here may actually be harmful if the default value is
        # invalid and causes a crash or has unwanted side effects.

    def __get__(self, cls, type_, delete_key=False):
        if cls is None:
            return self
        if not hasattr(self, 'val'):
            try:
                val_str = fetch_val_for_key(self.fullname,
                                            delete_key=delete_key)
                self.is_default = False
            except KeyError:
                if callable(self.default):
                    val_str = self.default()
                else:
                    val_str = self.default
            self.__set__(cls, val_str)
        # print "RVAL", self.val
        return self.val

    def __set__(self, cls, val):
        if not self.allow_override and hasattr(self, 'val'):
            raise Exception(
                "Can't change the value of this config parameter "
                "after initialization!")
        # print "SETTING PARAM", self.fullname,(cls), val
        if self.filter:
            self.val = self.filter(val)
        else:
            self.val = val


class EnumStr(ConfigParam):
    def __init__(self, default, *options, **kwargs):
        self.default = default
        self.all = (default,) + options

        # All options should be strings
        for val in self.all:
            if not isinstance(val, string_types) and val is not None:
                raise ValueError('Valid values for an EnumStr parameter '
                                 'should be strings or `None`', val, type(val))

        convert = kwargs.get("convert", None)

        def filter(val):
            # uri: We want to keep None values
            if val is None:
                return val

            if convert:
                val = convert(val)
            if val in self.all:
                return val
            else:
                raise ValueError((
                    'Invalid value ("%s") for configuration variable "%s". '
                    'Valid options are %s'
                    % (val, self.fullname, self.all)))
        over = kwargs.get("allow_override", True)
        super(EnumStr, self).__init__(default, filter, over)

    def __str__(self):
        return '%s (%s) ' % (self.fullname, self.all)


class ListParam(ConfigParam):
    def __init__(self, default, *options, **kwargs):
        self.default = default
        try:
            assert len(default) > 0
        except AssertionError:
            raise ValueError("List is empty")
        except TypeError:
            raise ValueError("The parameter is not a list.")

        over = kwargs.get("allow_override", True)
        super(ListParam, self).__init__(default, None, over)

    def __str__(self):
        return '%s (%s) ' % (self.fullname, self.default)


class TypedParam(ConfigParam):
    def __init__(self, default, mytype, is_valid=None, allow_override=True):
        self.mytype = mytype

        def filter(val):
            # uri: We want to keep None values
            if val is None:
                return val
            cast_val = mytype(val)
            if callable(is_valid):
                if is_valid(cast_val):
                    return cast_val
                else:
                    raise ValueError(
                        'Invalid value (%s) for configuration variable '
                        '"%s".'
                        % (val, self.fullname), val)
            return cast_val

        super(TypedParam, self).__init__(default, filter,
                                         allow_override=allow_override)

    def __str__(self):
        return '%s (%s) ' % (self.fullname, self.mytype)


def StrParam(default, is_valid=None, allow_override=True):
    return TypedParam(default, str, is_valid, allow_override=allow_override)


def IntParam(default, is_valid=None, allow_override=True):
    return TypedParam(default, int, is_valid, allow_override=allow_override)


def FloatParam(default, is_valid=None, allow_override=True):
    return TypedParam(default, float, is_valid, allow_override=allow_override)


def BoolParam(default, is_valid=None, allow_override=True):
    # see comment at the beginning of this file.

    def booltype(s):
        if s in ['False', 'false', '0', False]:
            return False
        elif s in ['True', 'true', '1', True]:
            return True

    def is_valid_bool(s):
        if s in ['False', 'false', '0', 'True', 'true', '1', False, True]:
            return True
        else:
            return False

    if is_valid is None:
        is_valid = is_valid_bool

    return TypedParam(default, booltype, is_valid,
                      allow_override=allow_override)
