# Run me as follows:
# cd tests/
# nosetests -v -s test_utils.py

import librosa
from nose.tools import nottest, raises
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
