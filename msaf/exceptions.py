'''Exception classes for msaf'''


class MSAFError(Exception):
    '''The root msaf exception class'''


class NoReferencesError(MSAFError):
    '''Exception class for trying evaluations without references'''


class WrongFeaturesFormatError(MSAFError):
    '''Exception class for worngly formatted features files'''


class NoFeaturesFileError(MSAFError):
    '''Exception class for missing features file'''


class FeaturesNotFound(MSAFError):
    '''Exception class for missing specific features in a file'''


class FeatureTypeNotFound(MSAFError):
    '''Exception class for feature type missing'''


class FeatureParamsError(MSAFError):
    '''Exception class for feature parameters missing'''


class NoAudioFileError(MSAFError):
    '''Exception class for audio file not found'''


class NoHierBoundaryError(MSAFError):
    '''Exception class for missing hierarchical boundary algorithm'''


class NoEstimationsError(MSAFError):
    '''Exception class for missing estimations'''


class WrongAlgorithmID(MSAFError):
    '''This algorithm was not found in msaf'''
