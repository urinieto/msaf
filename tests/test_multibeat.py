import json
import os

import numpy as np
import numpy.testing as npt
from pytest import fixture, raises

import msaf
from msaf.base import FeatureTypes
from msaf.exceptions import FeatureTypeNotFound, FramePerBeatTooHigh
from msaf.features import CQT, PCP
from msaf.input_output import FileStruct

# Move to __file__ path
os.chdir(os.path.dirname(__file__))

# Global vars
audio_file = os.path.join("fixtures", "chirp.mp3")
long_audio_file = os.path.join("fixtures", "Sargon_test", "audio", "Mindless_cut.mp3")
file_struct = FileStruct(audio_file)
file_struct.ref_file = os.path.join("fixtures", "chirp.jams")
msaf.utils.ensure_dir("features")
features_file = os.path.join("features", "chirp.json")
file_struct.features_file = features_file
try:
    os.remove(features_file)
except OSError:
    pass


@fixture
def feat_class():
    return CQT(file_struct, FeatureTypes.est_multibeat)


@fixture
def long_file_struct(tmp_path):
    fs = FileStruct(long_audio_file)
    fs.features_file = str(tmp_path / "Mindless_cut.json")
    return fs


def test_compute_multibeat(feat_class):
    assert feat_class._compute_multibeat(None) is None
    assert len(feat_class._compute_multibeat([])) == 0
    assert len(feat_class._compute_multibeat([10])) == 0
    frame_beats = [k * 100 for k in range(10)]
    multibeat_frames = feat_class._compute_multibeat(frame_beats)
    assert isinstance(multibeat_frames, np.ndarray)
    assert multibeat_frames.shape[0] == feat_class.frames_per_beat * (
        len(frame_beats) - 1
    )
    npt.assert_array_equal(multibeat_frames[:4], [0, 33, 66, 100])

    # Beats as short as frames_per_beat frames are fine
    feat_class.frames_per_beat = 100
    npt.assert_array_equal(feat_class._compute_multibeat([0, 100]), np.arange(100))
    feat_class.frames_per_beat = 101
    with raises(FramePerBeatTooHigh):
        feat_class._compute_multibeat(frame_beats)


def test_shape_beatwise(feat_class):
    assert feat_class._shape_beatwise(None) is None
    assert feat_class._shape_beatwise(np.array([])).shape[0] == 0
    multibeat_feature = np.array([[k * 2, k * 2 + 1] for k in range(51)])
    multibeat_shaped = feat_class._shape_beatwise(multibeat_feature)
    assert isinstance(multibeat_shaped, np.ndarray)
    assert multibeat_shaped.shape == (17, 2 * feat_class.frames_per_beat)
    npt.assert_array_equal(multibeat_shaped[0], [0, 1, 2, 3, 4, 5])
    with raises(ValueError):
        feat_class._shape_beatwise(multibeat_feature[:50])


def test_multibeat_rows_are_beat_subdivisions(feat_class):
    """Each row holds the subdivisions of one beat, from that beat to the next."""
    n_frames, n_feats = 40, 2
    feat_class._framesync_features = np.arange(n_frames * n_feats, dtype=float).reshape(
        n_frames, n_feats
    )
    beat_frames = np.array([5, 11, 20, 29])
    beat_times = np.array([0.5, 1.1, 2.0, 2.9])

    feats, times = feat_class._compute_multibeat_features(beat_frames, beat_times)

    assert feats.shape == (3, feat_class.frames_per_beat * n_feats)
    npt.assert_array_equal(times, beat_times[:-1])
    # First beat spans frames 5-10, split into [5, 6], [7, 8] and [9, 10]
    F = feat_class._framesync_features
    expected = np.concatenate([F[5:7].mean(0), F[7:9].mean(0), F[9:11].mean(0)])
    npt.assert_allclose(feats[0], expected)
    # Last beat ends right before the last beat frame
    npt.assert_allclose(feats[-1, -n_feats:], F[26:29].mean(0))


def test_multibeat_features_and_times_match(long_file_struct):
    feats = PCP(long_file_struct, FeatureTypes.est_multibeat)
    X, times = feats.features, feats.frame_times
    beats = feats._est_beats_times

    assert X.shape == (len(beats) - 1, 12 * feats.frames_per_beat)
    npt.assert_allclose(times, beats[:-1])


def test_multibeat_features_are_cached(long_file_struct):
    X = PCP(long_file_struct, FeatureTypes.est_multibeat).features
    with open(long_file_struct.features_file) as f:
        data = json.load(f)
    assert data["globals"]["frames_per_beat"] == 3
    assert "est_multibeat_times" in data

    cached = PCP(long_file_struct, FeatureTypes.est_multibeat)
    cached.read_features()  # Raises if the multibeat features can't be read
    npt.assert_allclose(cached.features, X)

    # A different frames_per_beat must not reuse the cached features
    other = PCP(long_file_struct, FeatureTypes.est_multibeat, frames_per_beat=4)
    assert other.features.shape == (X.shape[0], 12 * 4)


def test_short_beats_only_fail_multibeat(long_file_struct):
    """frames_per_beat larger than a beat must not break other features."""
    beatsync = PCP(long_file_struct, FeatureTypes.est_beatsync, frames_per_beat=1000)
    assert beatsync.features.shape[1] == 12

    long_file_struct.features_file += ".multibeat.json"
    multibeat = PCP(long_file_struct, FeatureTypes.est_multibeat, frames_per_beat=1000)
    with raises(FramePerBeatTooHigh):
        multibeat.features


def test_old_features_file_without_multibeat(tmp_path):
    """Feature files written before multibeat existed can still be read."""
    fs = FileStruct("fixtures/chirp_noaudio.mp3")
    fs.features_file = str(tmp_path / "chirp_noaudio.json")
    with open("features/chirp_noaudio.json") as f:
        data = json.load(f)
    assert "frames_per_beat" not in data["globals"]
    with open(fs.features_file, "w") as f:
        json.dump(data, f)

    feats = CQT(fs, FeatureTypes.est_beatsync)
    feats.read_features()


def test_process_multibeat():
    est_times, est_labels = msaf.process(
        long_audio_file, boundaries_id="foote", labels_id="fmc2d", multibeat=True
    )
    assert est_times[0] == 0
    assert len(est_times) == len(est_labels) + 1


def test_olda_rejects_multibeat():
    with raises(FeatureTypeNotFound):
        msaf.process(long_audio_file, boundaries_id="olda", multibeat=True)
