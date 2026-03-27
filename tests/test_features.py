import os
from enum import Enum

import numpy as np
from pytest import raises

import msaf
from msaf.base import FeatureTypes
from msaf.exceptions import FeatureTypeNotFound, NoAudioFileError
from msaf.features import CQT, MFCC, PCP, Features, Tempogram, Tonnetz
from msaf.input_output import FileStruct

# Global vars
audio_file = os.path.join("fixtures", "chirp.mp3")
file_struct = FileStruct(audio_file)
file_struct.ref_file = os.path.join("fixtures", "chirp.jams")


def run_features(features_class):
    """Runs features for the given class and checks shape/dtype."""
    feat_type = FeatureTypes.framesync
    feats = features_class(file_struct, feat_type).features
    assert isinstance(feats, np.ndarray)
    assert feats.ndim == 2
    assert feats.shape[0] > 0
    assert feats.shape[1] > 0


def run_ref_power(features_class):
    feats = features_class(file_struct, FeatureTypes.framesync, ref_power="max")
    assert feats.ref_power == np.amax
    feats = features_class(file_struct, FeatureTypes.framesync, ref_power="min")
    assert feats.ref_power == np.amin
    feats = features_class(file_struct, FeatureTypes.framesync, ref_power="median")
    assert feats.ref_power == np.median


def test_registry():
    """All the features should be in the features register."""
    assert CQT.get_id() in msaf.base.features_registry.keys()
    assert PCP.get_id() in msaf.base.features_registry.keys()
    assert Tonnetz.get_id() in msaf.base.features_registry.keys()
    assert MFCC.get_id() in msaf.base.features_registry.keys()
    assert Tempogram.get_id() in msaf.base.features_registry.keys()


def test_standard_cqt():
    """CQT features should compute and return proper array."""
    run_features(CQT)


def test_ref_power_cqt():
    """Test for different possible parameters for the ref_power of the cqt."""
    run_ref_power(CQT)


def test_wrong_ref_power_cqt():
    """Test for wrong parameters for ref_power of the cqt."""
    with raises(ValueError):
        CQT(file_struct, FeatureTypes.framesync, ref_power="caca")


def test_standard_pcp():
    """PCP features should compute and return proper array."""
    run_features(PCP)


def test_standard_mfcc():
    """MFCC features should compute and return proper array."""
    run_features(MFCC)


def test_standard_tonnetz():
    """Tonnetz features should compute and return proper array."""
    run_features(Tonnetz)


def test_ref_power_mfcc():
    """Test for different possible parameters for the ref_power of the mfcc."""
    run_ref_power(MFCC)


def test_wrong_ref_power_mfcc():
    """Test for wrong parameters for ref_power of the mfcc."""
    with raises(ValueError):
        MFCC(file_struct, FeatureTypes.framesync, ref_power="caca")


def test_standard_tempogram():
    """Tempogram features should compute and return proper array."""
    run_features(Tempogram)


def test_no_audio():
    """Features should raise NoAudioFileError when no audio file exists."""
    no_audio_file_struct = FileStruct("fixtures/nonexistent.mp3")
    feat_type = FeatureTypes.framesync
    with raises(NoAudioFileError):
        CQT(no_audio_file_struct, feat_type, sr=22050).features


def test_ann_features():
    """Testing the annotated beat synchronized features."""
    CQT(file_struct, FeatureTypes.ann_beatsync, sr=11025).features


def test_wrong_ann_features():
    """Trying to get annotated features when no annotated beats are found."""
    my_file_struct = FileStruct(os.path.join("fixtures", "chirp.mp3"))
    cqt = CQT(my_file_struct, FeatureTypes.ann_beatsync, sr=11025)
    with raises(FeatureTypeNotFound):
        cqt.features


def test_wrong_ann_frame_times():
    """Trying to get annotated frametimes when no annotated beats are found."""
    my_file_struct = FileStruct(os.path.join("fixtures", "chirp.mp3"))
    cqt = CQT(my_file_struct, FeatureTypes.ann_beatsync, sr=11025)
    with raises(FeatureTypeNotFound):
        cqt.frame_times


def test_wrong_type_features():
    """Trying to use custom type for features."""
    my_file_struct = FileStruct(os.path.join("fixtures", "chirp.mp3"))
    FeatureTypes2 = Enum("FeatureTypes", "framesync1 est_beatsync ann_beatsync")
    cqt = CQT(my_file_struct, FeatureTypes2.framesync1, sr=11025)
    with raises(FeatureTypeNotFound):
        cqt.features


def test_wrong_type_frame_times():
    """Trying to use custom type for frame times."""
    my_file_struct = FileStruct(os.path.join("fixtures", "chirp.mp3"))
    FeatureTypes2 = Enum("FeatureTypes", "framesync1 est_beatsync ann_beatsync")
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
    """Checking frame times of framesync type of features."""
    my_file_struct = FileStruct(os.path.join("fixtures", "chirp.mp3"))
    pcp = PCP(my_file_struct, FeatureTypes.framesync, sr=11025)
    times = pcp.frame_times
    assert isinstance(times, np.ndarray)


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


def test_beatsync_features():
    """Beat-synchronous features should compute and return proper array."""
    # Use longer audio file if available, otherwise just check type
    long_audio = os.path.join("fixtures", "Sargon_test", "audio", "Mindless_cut.mp3")
    if os.path.isfile(long_audio):
        fs = FileStruct(long_audio)
        feat_type = FeatureTypes.est_beatsync
        feats = PCP(fs, feat_type).features
        assert isinstance(feats, np.ndarray)
        assert feats.ndim == 2
        assert feats.shape[0] > 0
