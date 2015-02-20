#!/usr/bin/env python
"""
Unit tests for eval.
"""

import sys
import unittest
import numpy as np

import msaf
import msaf.eval as E


class TestEval(unittest.TestCase):

    def test_conditional_entropy(self):
        ann_times = [3, 12, 20, 40]
        est_times = [4, 20]
        ann_inter = np.asarray(zip(ann_times[:-1], ann_times[1:]))
        est_inter = np.asarray(zip(est_times[:-1], est_times[1:]))
        res = E.compute_results(ann_inter, est_inter, None, None, 251, "")

if __name__ == '__main__':
    unittest.main()
