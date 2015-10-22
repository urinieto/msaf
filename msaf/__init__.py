"""Top-level module for MSAF."""

__author__      = "Oriol Nieto"
__copyright__   = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__     = "MIT"
__version__     = "0.0.3"
__email__       = "oriol.nieto@gmail.com"


# Analysis Params
class Anal():
    sample_rate = 22050
    frame_size = 2048
    hop_size = 512
    mfcc_coeff = 14
    n_mels = 128
    window_type = "blackmanharris62"
    n_octaves = 6
    f_min = 27.5   # Minimum frequency for chroma
    cqt_bins = 84

# Default algorithms for msaf
DEFAULT_BOUND_ID = "sf"
DEFAULT_LABEL_ID = None

# Import all submodules (for each task)
from . import featextract
from . import input_output as io
from . import eval
from . import plotting
from . import utils
from . import algorithms
from . import run
from .run import process
from .input_output import get_all_boundary_algorithms
from .input_output import get_all_label_algorithms

# Global Config
prefix_dict = {
    "Cerulean"      : "large_scale",
    "Epiphyte"      : "function",
    "Isophonics"    : "function",
    "SALAMI"        : "large_scale",
    "SPAM"          : "large_scale"
}

results_dir = "results"
results_ext = ".csv"
out_boundaries_ext = "-bounds.wav"
minimum__frames = 10


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
    'sf'    :   'hpcp',
    'levy'  :   'hpcp',
    'foote' :   'hpcp',
    'siplca':   '',
    'olda'  :   '',
    'kmeans':   'hpcp',
    'cnmf'  :   'hpcp',
    'cnmf2' :   'hpcp',
    'cnmf3' :   'hpcp',
    '2dfmc' :   ''
}

AVAILABLE_FEATS = ["hpcp", "mfcc", "cqt", "tonnetz"]
