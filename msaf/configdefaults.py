import logging

import msaf
from msaf.configparser import (AddConfigVar, BoolParam, ConfigParam, EnumStr,
                               FloatParam, IntParam, StrParam,
                               MsafConfigParser, MSAF_FLAGS_DICT)


_logger = logging.getLogger('msaf.configdefaults')

config = TheanoConfigParser()



AddConfigVar('sample_rate',
             "Default Sample Rate to be used.",
             IntParam(22050))

AddConfigVar('cqt.bins',
             "Number of frequency bins for the CQT features.",
             IntParam(84))
