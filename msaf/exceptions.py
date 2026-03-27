"""Exception classes for msaf."""
from __future__ import annotations


class MSAFError(Exception):
    """The root msaf exception class."""


class NoReferencesError(MSAFError):
    """Exception class for trying evaluations without references."""


class FeatureTypeNotFound(MSAFError):
    """Exception class for feature type missing."""


class NoAudioFileError(MSAFError):
    """Exception class for audio file not found."""


class NoHierBoundaryError(MSAFError):
    """Exception class for missing hierarchical boundary algorithm."""


class NoEstimationsError(MSAFError):
    """Exception class for missing estimations."""


class WrongAlgorithmID(MSAFError):
    """This algorithm was not found in msaf."""
