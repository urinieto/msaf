#!/usr/bin/env python
#
# Run me as follows:
# cd tests/
# nosetests

import glob
import json
from nose.tools import nottest, eq_, raises, assert_equals
import numpy.testing as npt
import os

# Msaf imports
import msaf.featextract


def test_save_features():
    # Read audio and compute features
    tmp_file = "temp.json"
    audio_file = os.path.join("data", "chirp.mp3")
    features = msaf.featextract.compute_features_for_audio_file(audio_file)
    msaf.featextract.save_features(tmp_file, features)

    # Check that the json file is actually readable
    with open(tmp_file, "r") as f:
        features = json.load(f)
    npt.assert_almost_equal(features["analysis"]["dur"], 10, decimal=1)

    # Clean up
    os.remove(tmp_file)
