import json
import os
import pytest
from pytest import fixture
from enum import Enum

import librosa
import numpy as np
from pytest import raises

import msaf
from msaf.base import FeatureTypes
from msaf.exceptions import FramePerBeatTooHigh
from msaf.features import Features, CQT
from msaf.input_output import FileStruct

# Move to __file__ path
os.chdir(os.path.dirname(__file__))

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

multibeat_feature = np.array([[k*2, k*2+1] for k in range(10)])
print(multibeat_feature)

@fixture
def feat_class():
    return CQT(file_struct, FeatureTypes.est_multibeat)

def test_compute_multibeat(feat_class):
    assert (feat_class._compute_multibeat(None) is None)
    assert (feat_class._compute_multibeat([]) == [])
    frame_beats = [k*100 for k in range (10)]
    assert (isinstance(feat_class._compute_multibeat(frame_beats), np.ndarray))
    assert (feat_class._compute_multibeat(frame_beats).shape[0] == feat_class.frames_per_beat * (len(frame_beats) - 1))
    feat_class.frames_per_beat = 100
    with raises(FramePerBeatTooHigh):
        feat_class._compute_multibeat(frame_beats)

def test_shape_beatwise(feat_class):
    assert (feat_class._shape_beatwise(None) is None)
    assert (feat_class._shape_beatwise(np.array([])).shape[0] == 0)
    multibeat_feature = np.array([[k*2, k*2+1] for k in range(50)])
    multibeat_shaped = feat_class._shape_beatwise(multibeat_feature)
    assert(isinstance(multibeat_shaped, np.ndarray))
    assert(multibeat_shaped.shape[0] == len(multibeat_feature)//feat_class.frames_per_beat)
    assert(multibeat_shaped.shape[1] == 2 * feat_class.frames_per_beat)
    assert(np.equal(multibeat_shaped[0],np.array([0,1,2,3,4,5])).all())


