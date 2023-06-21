#!/usr/bin/env python

from enum import Enum
import json
import librosa
from pytest import raises
import numpy as np
import os

# Msaf imports
import msaf
from msaf.base import FeatureTypes
from msaf.exceptions import (NoAudioFileError, FeatureParamsError,
                             FeatureTypeNotFound)
from msaf.features import CQT, PCP, Tonnetz, MFCC, Tempogram, Features
from msaf.input_output import FileStruct

# Global vars
audio_file = os.path.join("fixtures", "chirp.mp3")
file_struct = FileStruct(audio_file)
file_struct.ref_file = os.path.join("fixtures", "chirp.jams")
msaf.utils.ensure_dir("features")
features_file = os.path.join("features", "chirp.json")
file_struct.features_file = features_file
try:
    os.remove(features_file)
except OSError:
    pass


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
    assert("framesync" in data[features_class.get_id()].keys())
    assert("est_beatsync" in data[features_class.get_id()].keys())
    assert("ann_beatsync" in data[features_class.get_id()].keys())
    read_feats = np.array(data[features_class.get_id()]["framesync"])
    assert(np.array_equal(feats, read_feats))


def run_ref_power(features_class):
    feats = features_class(file_struct, FeatureTypes.framesync,
                           ref_power="max")
    assert(feats.ref_power.__code__.co_code == np.max.__code__.co_code)
    feats = features_class(file_struct, FeatureTypes.framesync,
                           ref_power="min")
    assert(feats.ref_power.__code__.co_code == np.min.__code__.co_code)
    feats = features_class(file_struct, FeatureTypes.framesync,
                           ref_power="median")
    assert(feats.ref_power.__code__.co_code == np.median.__code__.co_code)


def test_standard_cqt():
    """CQT features should run and create the proper entry in the json file."""
    run_framesync(CQT)


def test_ref_power_cqt():
    """Test for different possible parameters for the ref_power of the cqt."""
    run_ref_power(CQT)


def test_wrong_ref_power_cqt():
    """Test for wrong parameters for ref_power of the cqt."""
    with raises(FeatureParamsError):
        CQT(file_struct, FeatureTypes.framesync, ref_power="caca")


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


def test_ref_power_mfcc():
    """Test for different possible parameters for the ref_power of the mfcc."""
    run_ref_power(MFCC)


def test_wrong_ref_power_mfcc():
    """Test for wrong parameters for ref_power of the mfcc."""
    with raises(FeatureParamsError):
        MFCC(file_struct, FeatureTypes.framesync, ref_power="caca")


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


def test_change_local_cqt_paramaters():
    """The features should be correctly updated if parameters of CQT are
    updated."""
    feat_type = FeatureTypes.framesync
    CQT(file_struct, feat_type, n_bins=70).features
    assert (os.path.isfile(file_struct.features_file))
    with open(file_struct.features_file) as f:
        data = json.load(f)
    assert(CQT.get_id() in data.keys())

    # These should be here from previous tests
    assert(MFCC.get_id() in data.keys())
    assert(Tonnetz.get_id() in data.keys())
    assert(Tempogram.get_id() in data.keys())
    assert(PCP.get_id() in data.keys())
    assert("framesync" in data[CQT.get_id()].keys())
    assert("est_beatsync" in data[CQT.get_id()].keys())
    assert("ann_beatsync" in data[CQT.get_id()].keys())


def test_change_global_paramaters():
    """The features should be correctly updated if global parameters
    updated."""
    feat_type = FeatureTypes.framesync
    CQT(file_struct, feat_type, sr=11025).features
    assert (os.path.isfile(file_struct.features_file))
    with open(file_struct.features_file) as f:
        data = json.load(f)
    assert(CQT.get_id() in data.keys())

    # These should have disappeared, since we now have new global parameters
    assert(MFCC.get_id() not in data.keys())
    assert(Tonnetz.get_id() not in data.keys())
    assert(Tempogram.get_id() not in data.keys())
    assert(PCP.get_id() not in data.keys())
    assert("framesync" in data[CQT.get_id()].keys())
    assert("est_beatsync" in data[CQT.get_id()].keys())
    assert("ann_beatsync" in data[CQT.get_id()].keys())


def test_no_audio():
    """The features should be returned even without having an audio file if
    they have been previously been computed."""
    # This file doesn't exist
    no_audio_file_struct = FileStruct("fixtures/chirp_noaudio.mp3")
    no_audio_file_struct.features_file = "features/chirp_noaudio.json"
    feat_type = FeatureTypes.framesync
    CQT(no_audio_file_struct, feat_type, sr=22050).features
    assert (os.path.isfile(no_audio_file_struct.features_file))
    with open(no_audio_file_struct.features_file) as f:
        data = json.load(f)
    assert(CQT.get_id() in data.keys())


def test_no_audio_no_params():
    """The features should raise a NoFileAudioError if different parameters
    want to be explored and no audio file is found."""
    # This file doesn't exist
    no_audio_file_struct = FileStruct("fixtures/chirp_noaudio.mp3")
    no_audio_file_struct.features_file = "features/chirp_noaudio.json"
    feat_type = FeatureTypes.framesync
    with raises(NoAudioFileError):
        CQT(no_audio_file_struct, feat_type, sr=11025).features


def test_no_audio_no_features():
    """The features should raise a NoFileAudioError if no features nor
    audio files are found."""
    # This file doesn't exist
    no_audio_file_struct = FileStruct("fixtures/caca.mp3")
    feat_type = FeatureTypes.framesync
    with raises(NoAudioFileError):
        CQT(no_audio_file_struct, feat_type, sr=11025).features


def test_ann_features():
    """Testing the annotated beat synchornized features."""
    CQT(file_struct, FeatureTypes.ann_beatsync, sr=11025).features


def test_wrong_ann_features():
    """Trying to get annotated features when no annotated beats are found."""
    my_file_struct = FileStruct(os.path.join("fixtures", "chirp.mp3"))
    my_file_struct.features_file = os.path.join("features", "no_file.json")
    cqt = CQT(my_file_struct, FeatureTypes.ann_beatsync, sr=11025)
    with raises(FeatureTypeNotFound):
        cqt.features


def test_wrong_ann_frame_times():
    """Trying to get annotated frametimes when no annotated beats are found."""
    my_file_struct = FileStruct(os.path.join("fixtures", "chirp.mp3"))
    my_file_struct.features_file = os.path.join("features", "no_file.json")
    cqt = CQT(my_file_struct, FeatureTypes.ann_beatsync, sr=11025)
    with raises(FeatureTypeNotFound):
        cqt.frame_times


def test_wrong_type_features():
    """Trying to use custom type for features."""
    my_file_struct = FileStruct(os.path.join("fixtures", "chirp.mp3"))
    my_file_struct.features_file = os.path.join("features", "no_file.json")
    FeatureTypes2 = Enum('FeatureTypes',
                         'framesync1 est_beatsync ann_beatsync')
    cqt = CQT(my_file_struct, FeatureTypes2.framesync1, sr=11025)
    with raises(FeatureTypeNotFound):
        cqt.features


def test_wrong_type_frame_times():
    """Trying to use custom type for frame times."""
    my_file_struct = FileStruct(os.path.join("fixtures", "chirp.mp3"))
    my_file_struct.features_file = os.path.join("features", "no_file.json")
    FeatureTypes2 = Enum('FeatureTypes',
                         'framesync1 est_beatsync ann_beatsync')
    cqt = CQT(my_file_struct, FeatureTypes2.framesync1, sr=11025)
    with raises(FeatureTypeNotFound):
        cqt.frame_times


def test_read_ann_beats_old_jams():
    """Trying to read an old jams file."""
    my_file_struct = FileStruct(os.path.join("fixtures", "chirp.mp3"))
    my_file_struct.ref_file = os.path.join("fixtures", "old_jams.jams")
    pcp = PCP(my_file_struct, FeatureTypes.ann_beatsync, sr=11025)
    times, frames = pcp.read_ann_beats()
    assert times is None
    assert frames is None


def test_frame_times_old_jams():
    """Trying to use invalid jams file."""
    my_file_struct = FileStruct(os.path.join("fixtures", "chirp.mp3"))
    my_file_struct.ref_file = os.path.join("fixtures", "old_jams.jams")
    pcp = PCP(my_file_struct, FeatureTypes.ann_beatsync, sr=11025)
    with raises(FeatureTypeNotFound):
        pcp.frame_times


def test_frame_times_framesync():
    """Checking frame times of framesync type of features"""
    my_file_struct = FileStruct(os.path.join("fixtures", "chirp.mp3"))
    pcp = PCP(my_file_struct, FeatureTypes.framesync, sr=11025)
    times = pcp.frame_times
    assert(isinstance(times, np.ndarray))


def test_frame_times_no_annotations():
    """Checking frame times when there are no beat annotations."""
    my_file_struct = FileStruct(os.path.join("fixtures", "chirp.mp3"))
    my_file_struct.ref_file = os.path.join("fixtures", "old_jams.jams")
    pcp = PCP(my_file_struct, FeatureTypes.ann_beatsync, sr=11025)
    with raises(FeatureTypeNotFound):
        pcp.frame_times


def test_wrong_frame_times():
    my_file_struct = FileStruct(os.path.join("fixtures", "chirp.mp3"))
    pcp = PCP(my_file_struct, FeatureTypes.ann_beatsync, sr=11025)
    pcp.feat_types = "wrong"
    with raises(FeatureTypeNotFound):
        pcp.frame_times


def test_global_get_id():
    with raises(NotImplementedError):
        Features.get_id()


def test_global_compute_features():
    my_file_struct = FileStruct(os.path.join("fixtures", "chirp.mp3"))
    feats = Features(my_file_struct, 11025, 1024, FeatureTypes.framesync)
    with raises(NotImplementedError):
        feats.compute_features()


def test_select_features():
    my_file_struct = FileStruct(os.path.join("fixtures", "chirp.mp3"))
    feature = Features.select_features("pcp", my_file_struct, False, True)
    assert isinstance(feature, PCP)
    assert feature.feat_type == FeatureTypes.framesync

    feature = Features.select_features("mfcc", my_file_struct, False, False)
    assert isinstance(feature, MFCC)
    assert feature.feat_type == FeatureTypes.est_beatsync

    feature = Features.select_features("cqt", my_file_struct, True, False)
    assert isinstance(feature, CQT)
    assert feature.feat_type == FeatureTypes.ann_beatsync


def test_wrong_select_features():
    with raises(FeatureTypeNotFound):
        Features.select_features("cqt", None, True, True)


def test_read_features_wrong_fun():
    my_file_struct = FileStruct("01_-_Come_Together.wav")
    my_file_struct.features_file = os.path.join(
        "fixtures", "01_-_Come_Together.json")
    mfcc = MFCC(my_file_struct, FeatureTypes.est_beatsync, sr=22050)
    mfcc.ref_power = np.mean
    with raises(FeatureParamsError):
        mfcc.read_features()
