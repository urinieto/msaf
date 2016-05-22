#!/usr/bin/env python
"""
Unit tests for eval.
"""

import unittest
import numpy as np

import msaf.eval as E


class TestEval(unittest.TestCase):

    def test_compute_results(self):
        ann_times = [3, 12, 20, 40]
        est_times = [1, 4, 20]
        ann_inter = np.asarray(list(zip(ann_times[:-1], ann_times[1:])))
        est_inter = np.asarray(list(zip(est_times[:-1], est_times[1:])))
        res = E.compute_results(ann_inter, est_inter, None, None, 251, "")
        np.testing.assert_almost_equal(res["HitRate_3F"], 0.5714285714285715,
                                       decimal=6)
        np.testing.assert_almost_equal(res["HitRate_3P"], 0.6666666666666666,
                                       decimal=6)
        np.testing.assert_almost_equal(res["HitRate_3R"], 0.5, decimal=6)

if __name__ == '__main__':
    unittest.main()
