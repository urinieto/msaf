from enum import Enum
from nose.tools import raises
import numpy as np
import os

# Msaf imports
from msaf.algorithms import fmc2d


def test_get_feat_segments():
    F = np.random.random((100, 10))
    bound_idxs = np.array([0, 10, 40, 70])
    feat_segments = fmc2d.get_feat_segments(F, bound_idxs)
    assert len(feat_segments) == len(bound_idxs) - 1


@raises(AssertionError)
def test_get_feat_segments_outbounds_down():
    F = np.random.random((100, 10))
    bound_idxs = np.array([-10, 10, 40, 70])
    fmc2d.get_feat_segments(F, bound_idxs)


@raises(AssertionError)
def test_get_feat_segments_outbounds_up():
    F = np.random.random((100, 10))
    bound_idxs = np.array([0, 10, 40, 70, 120])
    fmc2d.get_feat_segments(F, bound_idxs)


@raises(AssertionError)
def test_get_feat_segments_empty():
    F = np.random.random((100, 10))
    bound_idxs = np.array([])
    fmc2d.get_feat_segments(F, bound_idxs)
