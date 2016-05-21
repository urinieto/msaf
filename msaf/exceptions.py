'''Exception classes for msaf'''


class MSAFError(Exception):
    '''The root msaf exception class'''
    pass


class NoReferencesError(MSAFError):
    '''Exception class for trying evaluations without references'''
    pass
