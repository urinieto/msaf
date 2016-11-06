'''Exception classes for msaf'''


class MSAFError(Exception):
    '''The root msaf exception class'''
    pass


class NoReferencesError(MSAFError):
    '''Exception class for trying evaluations without references'''
    pass


class WrongFeaturesFormatError(MSAFError):
    '''Exception class for worngly formatted features files'''
    pass


class NoFeaturesFileError(MSAFError):
    '''Exception class for missing features file'''
    pass


class FeaturesNotFound(MSAFError):
    '''Exception class for missing specific features in a file'''
    pass


class FeatureTypeNotFound(MSAFError):
    '''Exception class for feature type missing'''
    pass


class FeatureParamsError(MSAFError):
    '''Exception class for feature parameters missing'''
    pass


class NoAudioFileError(MSAFError):
    '''Exception class for audio file not found'''
    pass


class NoHierBoundaryError(MSAFError):
    '''Exception class for missing hierarchical boundary algorithm'''
    pass


class NoEstimationsError(MSAFError):
    '''Exception class for missing estimations'''
    pass


class WrongAlgorithmID(MSAFError):
    '''This algorithm was not found in msaf'''
    pass
