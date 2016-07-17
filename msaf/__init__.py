"""Top-level module for MSAF."""
from .version import version as __version__
import numpy as np

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
from . import features
from . import input_output as io
from . import eval
from . import plotting
from . import utils
from . import algorithms
from . import run
from .base import features_registry
from .run import process
from .input_output import get_all_boundary_algorithms
from .input_output import get_all_label_algorithms

# TODO: Include this in algorithms
feat_dict = {
    'sf': 'pcp',
    'levy': 'pcp',
    'foote': 'pcp',
    'siplca': '',
    'olda': '',
    'cnmf': 'pcp',
    '2dfmc': ''
}
