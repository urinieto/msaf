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
import msaf
from msaf.input_output import FileStruct

# Global vars
audio_file = os.path.join("data", "chirp.mp3")
sr = msaf.Anal.sample_rate
audio, fs = librosa.load(audio_file, sr=sr)
y_harmonic, y_percussive = librosa.effects.hpss(audio)


def test_get_features():
    #TODO
    pass

def test_read_hier_references():
    # One level file
    one_jams = os.path.join("..", "datasets", "Sargon", "references",
                            "01-Sargon-Mindless.jams")
    hier_bounds, hier_labels, hier_levels = \
        msaf.io.read_hier_references(one_jams)
