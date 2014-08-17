"""Top-level module for MSAF."""

# Import all submodules (for each task)
from . import eval
from . import featextract
from . import input_output as io
from . import plotting
from . import utils

__version__ = '0.0.1'

# Global Config
feat_dict = {
    'serra' :   'mix',
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

prefix_dict = {
    "Cerulean"      : "large_scale",
    "Epiphyte"      : "function",
    "Isophonics"    : "function",
    "SALAMI"        : "large_scale"
}

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
