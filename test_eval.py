#!/usr/bin/env python
"""
Unit tests for eval.
"""

import unittest

import eval as E


class TestEval(unittest.TestCase):

    def test_conditional_entropy(self):
        ann_times = [3, 12, 20, 40]
        est_times = [4, 20]
        self.assertEqual(E.compute_conditional_entropy(ann_times, est_times),
                         1)
        self.assertEqual(E.compute_conditional_entropy(est_times, est_times),
                         0)

if __name__ == '__main__':
    unittest.main()
