"""Top-level module for MSAF."""

from .version import version as __version__

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2024, Oriol Nieto"
__license__ = "MIT"
__email__ = "oriol@nyu.edu"

# Get config
from msaf.config import config

# Import all submodules
from . import algorithms, eval, features
from . import input_output as io
from . import plotting, run, utils
from .base import ProcessingContext, features_registry
from .input_output import get_all_boundary_algorithms, get_all_label_algorithms
from .run import process

__all__ = [
    "algorithms",
    "config",
    "eval",
    "features",
    "features_registry",
    "ProcessingContext",
    "get_all_boundary_algorithms",
    "get_all_label_algorithms",
    "io",
    "plotting",
    "process",
    "run",
    "utils",
]
