#!/usr/bin/env python
#
# Run me as follows:
# cd tests/
# nosetests

import glob
import json
import librosa
from nose.tools import nottest, eq_, raises, assert_equals
import numpy.testing as npt
import os

# Msaf imports
import msaf.featextract
from msaf.input_output import FileStruct

# Global vars
audio_file = os.path.join("data", "chirp.mp3")
sr = msaf.Anal.sample_rate
audio, fs = librosa.load(audio_file, sr=sr)
y_harmonic, y_percussive = librosa.effects.hpss(audio)


def test_get_features():
    #TODO
    pass
