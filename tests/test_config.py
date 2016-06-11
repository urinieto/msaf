#!/usr/bin/env python

import json
import librosa
from nose.tools import assert_equals, raises
import numpy as np
import numpy.testing as npt
import os

# Msaf imports
import msaf


def test_config():
    """All the features should be in the features register."""
    print(msaf.config)
    print(msaf.config.sample_rate)
