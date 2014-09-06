"""Top-level module for MSAF."""

# Import all submodules (for each task)
import eval
import featextract
import input_output as io
import plotting
import utils
import algorithms
from run import process

__version__ = '0.0.1'

# Global Config
prefix_dict = {
    "Cerulean"      : "large_scale",
    "Epiphyte"      : "function",
    "Isophonics"    : "function",
    "SALAMI"        : "large_scale"
}

results_dir = "results"
results_ext = ".csv"
out_boundaries_ext = "-bounds.wav"


# Analysis Params
class Anal():
    sample_rate = 11025
    frame_size = 2048
    hop_size = 1024
    mfcc_coeff = 14
    window_type = "blackmanharris62"


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
    audio_exts = [".wav", "mp3", ".aif"]


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
