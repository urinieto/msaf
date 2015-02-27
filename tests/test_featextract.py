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

# Global vars
audio_file = os.path.join("data", "chirp.mp3")
sr = 44100
audio, fs = librosa.load(audio_file, sr=sr)
y_harmonic, y_percussive = librosa.effects.hpss(audio)


def test_compute_beats():
    beats_idx, beats_times = msaf.featextract.compute_beats(y_percussive)
    assert len(beats_idx) > 0
    assert len(beats_idx) == len(beats_times)


def test_compute_features():
    mfcc, hpcp, tonnetz = msaf.featextract.compute_features(audio, y_harmonic)
    assert mfcc.shape[1] == msaf.Anal.mfcc_coeff
    assert hpcp.shape[1] == 12
    assert tonnetz.shape[1] == 6
    assert_equals(mfcc.shape[0], hpcp.shape[0])
    assert_equals(hpcp.shape[0], tonnetz.shape[0])


def test_save_features():
    # Read audio and compute features
    tmp_file = "temp.json"
    features = msaf.featextract.compute_features_for_audio_file(audio_file)
    msaf.featextract.save_features(tmp_file, features)

    # Check that the json file is actually readable
    with open(tmp_file, "r") as f:
        features = json.load(f)
    npt.assert_almost_equal(features["analysis"]["dur"], 10, decimal=1)

    # Clean up
    os.remove(tmp_file)


def test_compute_beat_sync_features():
    # Compute features
    features = msaf.featextract.compute_features_for_audio_file(audio_file)

    # Beat track
    beats_idx, beats_times = msaf.featextract.compute_beats(y_percussive)

    # Compute beat sync feats
    bs_mfcc, bs_hpcp, bs_tonnetz = \
        msaf.featextract.compute_beat_sync_features(features, beats_idx)
    assert_equals(bs_mfcc.shape[0],  len(beats_idx) - 1)
    assert_equals(bs_mfcc.shape[0], bs_hpcp.shape[0])
    assert_equals(bs_hpcp.shape[0], bs_tonnetz.shape[0])


def test_compute_features_for_audio_file():
    features = msaf.featextract.compute_features_for_audio_file(audio_file)
    keys = ["mfcc", "hpcp", "tonnetz", "beats_idx", "beats", "bs_mfcc",
            "bs_hpcp", "bs_tonnetz", "anal"]
    anal_keys = ["frame_rate", "hop_size", "mfcc_coeff", "sample_rate",
                 "window_type", "n_mels", "dur"]
    for key in keys:
        assert key in features.keys()
    for anal_key in anal_keys:
        assert anal_key in features["anal"].keys()
