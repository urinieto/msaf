"""Default configuration parameters for MSAF."""
import logging
import numpy as np

import msaf
from msaf.configparser import (AddConfigVar, BoolParam, ConfigParam, EnumStr,
                               FloatParam, IntParam, StrParam,
                               MsafConfigParser, MSAF_FLAGS_DICT)


_logger = logging.getLogger('msaf.configdefaults')

config = MsafConfigParser()

# Globals
AddConfigVar('default_bound_id', "Default boundary detection algorithm",
             StrParam("sf"))
AddConfigVar('default_label_id', "Default label detection algorithm",
             StrParam(None))

# Global analysis parameters
AddConfigVar('sample_rate', "Default Sample Rate to be used.", IntParam(22050))
AddConfigVar('n_fft', "FFT size", IntParam(4096))
AddConfigVar('hop_size', "Hop length in samples", IntParam(1024))


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
