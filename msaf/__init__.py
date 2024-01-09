"""Top-level module for MSAF."""
import numpy as np

from .version import version as __version__

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2016, Music and Audio Research Lab (MARL)"
__license__ = "MIT"
__email__ = "oriol.nieto@gmail.com"

# Default configuration files and environment variables
MSAFRC_VAR = "MSAFRC"
MSAF_FLAGS_VAR = "MSAF_FLAGS"
MSAFRC_FILE = "~/.msafrc"
MSAFRC_WIN_FILE = "~/.msafrc.txt"

# Get config
from msaf.configdefaults import config

# Import all submodules
from . import algorithms, eval, features
from . import input_output as io
from . import plotting, run, utils
from .base import features_registry
from .input_output import get_all_boundary_algorithms, get_all_label_algorithms
from .run import process

# TODO: Include this in algorithms
feat_dict = {
    'sf': 'pcp',
    'levy': 'pcp',
    'foote': 'pcp',
    'siplca': '',
    'olda': '',
    'cnmf': 'pcp',
    '2dfmc': '',
    'cbm': 'log_mel',
}
