"""Default configuration parameters for MSAF."""
import logging
import numpy as np

from msaf.configparser import \
    (AddConfigVar, BoolParam, ConfigParam, EnumStr, FloatParam,
     IntParam, StrParam, ListParam, MsafConfigParser)


_logger = logging.getLogger('msaf.configdefaults')

config = MsafConfigParser()

# Globals
AddConfigVar('default_bound_id', "Default boundary detection algorithm",
             EnumStr("sf", "cnmf", "foote", "olda", "scluster", "gt"))
AddConfigVar('default_label_id', "Default label detection algorithm",
             EnumStr(None, "cnmf", "fmc2d", "scluster"))

# Global analysis parameters
AddConfigVar('sample_rate', "Default Sample Rate to be used.", IntParam(22050))
AddConfigVar('n_fft', "FFT size", IntParam(4096))
AddConfigVar('hop_size', "Hop length in samples", IntParam(1024))


# Files and dirs
AddConfigVar('results_dir', "Default directory to store results.",
             StrParam("results"))
AddConfigVar('results_ext', "Default extension for the results file.",
             StrParam(".csv"))
AddConfigVar('out_boundaries_ext', "Default extension for output audio "
             "bounds.", StrParam("-bounds.wav"))
AddConfigVar('minimum_frames', "Minimum number of frames to activate "
             "algorithms.", IntParam(10))
AddConfigVar('features_tmp_file', "Default temporary file for features.",
             StrParam(".features_msaf_tmp.json"))

# Dataset files and dirs
AddConfigVar('dataset.audio_dir', "Default audio directory.",
             StrParam("audio"))
AddConfigVar('dataset.estimations_dir', "Default estimations directory.",
             StrParam("estimations"))
AddConfigVar('dataset.features_dir', "Default features directory.",
             StrParam("features"))
AddConfigVar('dataset.references_dir', "Default references directory.",
             StrParam("references"))
AddConfigVar('dataset.audio_exts', "Available audio files.",
             ListParam([".wav", ".mp3", ".aif"]))
AddConfigVar('dataset.estimations_ext', "Extension for the estimation files.",
             StrParam(".jams"))
AddConfigVar('dataset.features_ext', "Extension for the features files.",
             StrParam(".json"))
AddConfigVar('dataset.references_ext', "Extension for the reference files.",
             StrParam(".jams"))


# CQT Features
AddConfigVar('cqt.bins', "Number of frequency bins for the CQT features.",
             IntParam(84))
AddConfigVar('cqt.norm',
             "Type of norm to use for basis function normalization.",
             FloatParam(np.inf))
AddConfigVar('cqt.filter_scale',
             "Type of norm to use for basis function normalization.",
             FloatParam(1.0))
AddConfigVar('cqt.ref_power',
             "Reference function to use for the logarithm power.",
             EnumStr("max", "min", "median"))

# MFCC Features
AddConfigVar('mfcc.n_mels', "Number of mel filters.", IntParam(128))
AddConfigVar('mfcc.n_mfcc', "Number of mel coefficients.", IntParam(14))
AddConfigVar('mfcc.ref_power',
             "Reference function to use for the logarithm power.",
             EnumStr("max", "min", "median"))

# PCP Features
AddConfigVar('pcp.bins', "Number of frequency bins for the CQT.",
             IntParam(84))
AddConfigVar('pcp.norm', "Normalization parameter.", FloatParam(np.inf))
AddConfigVar('pcp.f_min', "Minimum frequency.", FloatParam(27.5))
AddConfigVar('pcp.n_octaves', "Number of octaves.", IntParam(6))

# Tonnetz Features
AddConfigVar('tonnetz.bins', "Number of frequency bins for the CQT.",
             IntParam(84))
AddConfigVar('tonnetz.norm', "Normalization parameter.", FloatParam(np.inf))
AddConfigVar('tonnetz.f_min', "Minimum frequency.", FloatParam(27.5))
AddConfigVar('tonnetz.n_octaves', "Number of octaves.", IntParam(6))

# Tempogram Features
AddConfigVar('tempogram.win_length',
             "The size of the window of the tempogram.", IntParam(192))
