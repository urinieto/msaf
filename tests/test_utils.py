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

# Global vars
audio_file = os.path.join("data", "chirp.mp3")
sr = msaf.Anal.sample_rate
audio, fs = librosa.load(audio_file, sr=sr)
y_harmonic, y_percussive = librosa.effects.hpss(audio)


def test_synchronize_labels():
    old_bound_idxs = [0, 82, 150, 268, 342, 353, 463, 535, 616, 771, 833, 920,
                      979, 1005]
    new_bound_idxs = [0, 229, 337, 854, 929, 994, 1004]
    labels = [4, 6, 2, 0, 0, 2, 5, 3, 0, 5, 1, 5, 1]
    N = 1005
    new_labels = msaf.utils.synchronize_labels(new_bound_idxs,
                                               old_bound_idxs,
                                               labels,
                                               N)
    assert len(new_labels) == len(new_bound_idxs) - 1
