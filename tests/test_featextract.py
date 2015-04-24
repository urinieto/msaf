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


def test_compute_beats():
    beats_idx, beats_times = msaf.featextract.compute_beats(y_percussive)
    assert len(beats_idx) > 0
    assert len(beats_idx) == len(beats_times)


def test_compute_features():
    mfcc, hpcp, tonnetz, cqt = msaf.featextract.compute_features(audio, y_harmonic)
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

    # Check that new file exists
    assert os.path.isfile(tmp_file)

    # Check that the json file is actually readable
    with open(tmp_file, "r") as f:
        features = json.load(f)
    npt.assert_almost_equal(features["analysis"]["dur"], len(audio) / float(sr),
                            decimal=1)

    # Clean up
    os.remove(tmp_file)


def test_compute_beat_sync_features():
    # Compute features
    features = msaf.featextract.compute_features_for_audio_file(audio_file)

    # Beat track
    beats_idx, beats_times = msaf.featextract.compute_beats(y_percussive,
                                                            sr=sr)

    # Compute beat sync feats
    bs_mfcc, bs_hpcp, bs_tonnetz, bs_cqt = \
        msaf.featextract.compute_beat_sync_features(features, beats_idx)
    assert_equals(bs_mfcc.shape[0],  len(beats_idx) - 1)
    assert_equals(bs_mfcc.shape[0], bs_hpcp.shape[0])
    assert_equals(bs_hpcp.shape[0], bs_tonnetz.shape[0])


def test_compute_features_for_audio_file():
    features = msaf.featextract.compute_features_for_audio_file(audio_file)
    keys = ["mfcc", "hpcp", "tonnetz", "cqt", "beats_idx", "beats", "bs_mfcc",
            "bs_hpcp", "bs_tonnetz", "bs_cqt", "anal"]
    anal_keys = ["frame_rate", "hop_size", "mfcc_coeff", "sample_rate",
                 "window_type", "n_mels", "dur"]
    for key in keys:
        assert key in features.keys()
    for anal_key in anal_keys:
        assert anal_key in features["anal"].keys()


def test_compute_all_features():
    # Create file struct
    file_struct = FileStruct(audio_file)

    # Set output file
    feat_file = "tmp.json"
    beats_file = "beats.wav"
    file_struct.features_file = feat_file

    # Remove previously computed outputs if exist
    if os.path.isfile(feat_file):
        os.remove(feat_file)
    if os.path.isfile(beats_file):
        os.remove(beats_file)

    # Call main function
    msaf.featextract.compute_all_features(file_struct, sonify_beats=False,
                                          overwrite=False)
    assert os.path.isfile(feat_file)

    # Call again main function (should do nothing, since feat_file exists)
    msaf.featextract.compute_all_features(file_struct, sonify_beats=False,
                                          overwrite=False)
    assert os.path.isfile(feat_file)

    # Overwrite
    msaf.featextract.compute_all_features(file_struct, sonify_beats=False,
                                          overwrite=True)
    assert os.path.isfile(feat_file)

    # Sonify
    msaf.featextract.compute_all_features(file_struct, sonify_beats=True,
                                          overwrite=True, out_beats=beats_file)
    assert os.path.isfile(feat_file) and os.path.isfile(beats_file)

    # Clean up
    os.remove(feat_file)
    os.remove(beats_file)


def test_process():
    # Set output file
    feat_file = "tmp.json"
    beats_file = "beats.wav"

    # Remove previously computed outputs if exist
    if os.path.isfile(feat_file):
        os.remove(feat_file)
    if os.path.isfile(beats_file):
        os.remove(beats_file)

    # Call main function
    msaf.featextract.process(audio_file, out_file=feat_file)
    assert os.path.isfile(feat_file)

    # Call again main function (should do nothing, since feat_file exists)
    msaf.featextract.process(audio_file, out_file=feat_file)
    assert os.path.isfile(feat_file)

    # Overwrite
    msaf.featextract.process(audio_file, out_file=feat_file, overwrite=True)
    assert os.path.isfile(feat_file)

    # Sonify
    msaf.featextract.process(audio_file, out_file=feat_file, overwrite=True,
                             sonify_beats=True, out_beats=beats_file)
    assert os.path.isfile(feat_file) and os.path.isfile(beats_file)

    # Clean up
    os.remove(feat_file)
    os.remove(beats_file)
