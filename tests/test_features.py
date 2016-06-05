#!/usr/bin/env python
#
# Run me as follows:
# cd tests/
# nosetests

import json
import librosa
from nose.tools import assert_equals
import numpy as np
import numpy.testing as npt
import os

# Msaf imports
import msaf
from msaf.base import FeatureTypes
from msaf.features import CQT, PCP, Tonnetz, MFCC, Tempogram
from msaf.input_output import FileStruct

# Global vars
audio_file = os.path.join("fixtures", "chirp.mp3")
file_struct = FileStruct(audio_file)
msaf.utils.ensure_dir("features")
features_file = os.path.join("features", "chirp.json")
file_struct.features_file = features_file
os.remove(features_file)


def test_registry():
    """All the features should be in the features register."""
    assert(CQT.get_id() in msaf.base.features_registry.keys())
    assert(PCP.get_id() in msaf.base.features_registry.keys())
    assert(Tonnetz.get_id() in msaf.base.features_registry.keys())
    assert(MFCC.get_id() in msaf.base.features_registry.keys())
    assert(Tempogram.get_id() in msaf.base.features_registry.keys())


def run_framesync(features_class):
    """Runs the framesync features for the given features class with the
    default parameters, and checks that it correctly saved the information
    in the json file."""
    feat_type = FeatureTypes.framesync
    feats = features_class(file_struct, feat_type).features
    assert (os.path.isfile(file_struct.features_file))
    with open(file_struct.features_file) as f:
        data = json.load(f)
    assert(features_class.get_id() in data.keys())
    read_feats = np.array(data[features_class.get_id()]["framesync"])
    assert(np.array_equal(feats, read_feats))


def test_standard_cqt():
    """CQT features should run and create the proper entry in the json file."""
    run_framesync(CQT)


def test_standard_pcp():
    """PCP features should run and create the proper entry in the json file."""
    run_framesync(PCP)


def test_standard_mfcc():
    """MFCC features should run and create the proper entry in the json
    file."""
    run_framesync(MFCC)


def test_standard_tonnetz():
    """Tonnetz features should run and create the proper entry in the json
    file."""
    run_framesync(Tonnetz)


def test_standard_tempogram():
    """Tempogram features should run and create the proper entry in the json
    file."""
    run_framesync(Tempogram)


def test_metadata():
    """The metadata of the json file should be correct."""
    # Note: The json file should have been created with previous tests
    with open(file_struct.features_file) as f:
        data = json.load(f)
    assert("metadata" in data.keys())
    metadata = data["metadata"]
    assert("timestamp" in metadata.keys())
    assert(metadata["versions"]["numpy"] == np.__version__)
    assert(metadata["versions"]["msaf"] == msaf.__version__)
    assert(metadata["versions"]["librosa"] == librosa.__version__)

"""
def test_compute_beats():
    beats_idx, beats_times = msaf.featextract.compute_beats(y_percussive)
    assert len(beats_idx) > 0
    assert len(beats_idx) == len(beats_times)


def test_compute_features():
    mfcc, hpcp, tonnetz, cqt = msaf.featextract.compute_features(audio,
                                                                 y_harmonic)
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
    npt.assert_almost_equal(features["analysis"]["dur"],
                            len(audio) / float(sr), decimal=1)

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
"""
