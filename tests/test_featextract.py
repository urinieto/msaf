#!/usr/bin/env python
#
# Run me as follows:
# cd tests/
# nosetests

import glob
from nose.tools import nottest, eq_, raises
import os

# Msaf imports
import msaf.featextract


def test_save_features():
    tmp_file = "temp.json"
    audio_file = os.path.join("data", "chirp.mp3")
    features = msaf.featextract.compute_features_for_audio_file(audio_file)
    msaf.featextract.save_features(tmp_file, features)
