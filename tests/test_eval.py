#!/usr/bin/env python
"""
Unit tests for eval.
"""

from nose.tools import assert_raises
from types import ModuleType
import numpy.testing as npt
import numpy as np

import msaf.eval as E


def test_compute_results():
    ann_times = [3, 12, 20, 40]
    est_times = [1, 4, 20]
    ann_inter = np.asarray(list(zip(ann_times[:-1], ann_times[1:])))
    est_inter = np.asarray(list(zip(est_times[:-1], est_times[1:])))
    res = E.compute_results(ann_inter, est_inter, None, None, 251, "")
    npt.assert_almost_equal(res["HitRate_3F"], 0.5714285714285715, decimal=6)
    npt.assert_almost_equal(res["HitRate_3P"], 0.6666666666666666, decimal=6)
    npt.assert_almost_equal(res["HitRate_3R"], 0.5, decimal=6)
