"""Top-level module for MSAF."""

# Import all submodules (for each task)
from . import input_output as io
from . import utils
from . import eval

__version__ = '0.0.1'

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
