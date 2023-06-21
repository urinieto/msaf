# Run me as follows:
# cd tests/
# nosetests -v -s test_utils.py
import copy
import librosa
import numpy as np
import os

# Msaf imports
import msaf

# Global vars
audio_file = os.path.join("fixtures", "chirp.mp3")
sr = msaf.config.sample_rate
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


def test_get_num_frames():
    dur = 320.2
    anal = {"sample_rate": 22050, "hop_size": 512}
    n_frames = msaf.utils.get_num_frames(dur, anal)
    assert n_frames == int(dur * anal["sample_rate"] / anal["hop_size"])


def test_get_time_frames():
    dur = 1
    anal = {"sample_rate": 22050, "hop_size": 512}
    n_frames = msaf.utils.get_time_frames(dur, anal)
    assert n_frames.shape[0] == 43
    assert n_frames[0] == 0.0
    assert n_frames[-1] == 1.0


def test_align_end_hierarchies():
    def _test_equal_hier(hier_orig, hier_new):
        for layer_orig, layer_new in zip(hier_orig, hier_new):
            assert layer_orig == layer_new

    hier1 = [[0, 10, 20, 30], [0, 30]]
    hier2 = [[0, 5, 40, 50], [0, 50]]
    hier1_orig = copy.deepcopy(hier1)
    hier2_orig = copy.deepcopy(hier2)

    msaf.utils.align_end_hierarchies(hier1, hier2)

    _test_equal_hier(hier1_orig, hier1)
    _test_equal_hier(hier2_orig, hier2)


def test_lognormalize():
    # Just check that we're not overwriting data
    X = np.random.random((300, 10))
    Y = msaf.utils.lognormalize(X)
    assert not np.array_equal(X, Y)


def test_min_max_normalize():
    # Just check that we're not overwriting data
    X = np.random.random((300, 10))
    Y = msaf.utils.min_max_normalize(X)
    assert not np.array_equal(X, Y)
