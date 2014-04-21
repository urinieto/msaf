#!/usr/bin/env python
"""
Some unit tests for the MSAF input and output module.

Written by Oriol Nieto (oriol@nyu.edu), 2014
"""

import pickle
import sys
import unittest
import numpy as np

sys.path.append("..")
import msaf_io


class IOTests(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_read_boundaries(self):
        results = pickle.load(open(
            "data/Cerulean_50_Cent-Window_Shopper-serra.pk", "r"))
        est_file = "data/Cerulean_50_Cent-Window_Shopper.json"
        alg_id = "serra"
        annot_beats = False
        bounds = msaf_io.read_boundaries(est_file, alg_id, annot_beats)
        self.assertEqual(np.array_equal(bounds, results), True)


if __name__ == "__main__":
    unittest.main()
