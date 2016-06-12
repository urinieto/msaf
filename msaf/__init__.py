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

results_dir = "results"
results_ext = ".csv"
out_boundaries_ext = "-bounds.wav"
minimum__frames = 10
features_tmp_file = ".features_msaf_tmp.json"


class Dataset():
    # Directories
    audio_dir = "audio"
    estimations_dir = "estimations"
    features_dir = "features"
    references_dir = "references"

    # Extensions
    estimations_ext = ".jams"
    features_ext = ".json"
    references_ext = ".jams"
    audio_exts = [".wav", ".mp3", ".aif"]


feat_dict = {
    'sf': 'pcp',
    'levy': 'pcp',
    'foote': 'pcp',
    'siplca': '',
    'olda': '',
    'cnmf': 'pcp',
    '2dfmc': ''
}
