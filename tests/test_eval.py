#!/usr/bin/env python
"""
Unit tests for eval.
"""

from nose.tools import assert_raises
from types import ModuleType
import numpy.testing as npt
import numpy as np
import pandas as pd

import msaf.eval as E


def test_compute_boundary_results():
    ann_times = [3, 12, 20, 40]
    est_times = [1, 4, 20]
    ann_inter = np.asarray(list(zip(ann_times[:-1], ann_times[1:])))
    est_inter = np.asarray(list(zip(est_times[:-1], est_times[1:])))
    res = E.compute_results(ann_inter, est_inter, None, None, 251, "")

    npt.assert_almost_equal(res["HitRate_3F"], 0.5714285714285715, decimal=6)
    npt.assert_almost_equal(res["HitRate_3P"], 0.6666666666666666, decimal=6)
    npt.assert_almost_equal(res["HitRate_3R"], 0.5, decimal=6)
    npt.assert_almost_equal(res["HitRate_w3P"], 0.6666666666666666, decimal=6)
    npt.assert_almost_equal(res["HitRate_t3F"], 0.0, decimal=6)
    npt.assert_almost_equal(res["HitRate_t3P"], 0.0, decimal=6)
    npt.assert_almost_equal(res["HitRate_t3R"], 0.0, decimal=6)
    npt.assert_almost_equal(res["HitRate_wt3P"], 0.0, decimal=6)

    npt.assert_almost_equal(res["HitRate_0.5F"], 0.28571428571428575,
                            decimal=6)
    npt.assert_almost_equal(res["HitRate_0.5P"], 0.33333333333, decimal=6)
    npt.assert_almost_equal(res["HitRate_0.5R"], 0.25, decimal=6)
    npt.assert_almost_equal(res["HitRate_w0.5P"], 0.33333333333, decimal=6)
    npt.assert_almost_equal(res["HitRate_t0.5F"], 0.0, decimal=6)
    npt.assert_almost_equal(res["HitRate_t0.5P"], 0.0, decimal=6)
    npt.assert_almost_equal(res["HitRate_t0.5R"], 0.0, decimal=6)
    npt.assert_almost_equal(res["HitRate_wt0.5P"], 0.0, decimal=6)

    npt.assert_almost_equal(res["DevE2R"], 1.0, decimal=6)
    npt.assert_almost_equal(res["DevR2E"], 4.5, decimal=6)
    npt.assert_almost_equal(res["DevtE2R"], 8.0, decimal=6)
    npt.assert_almost_equal(res["DevtR2E"], 12.0, decimal=6)


def test_compute_label_results():
    ann_times = [3, 12, 20, 40]
    est_times = [1, 4, 20]
    ann_inter = np.asarray(list(zip(ann_times[:-1], ann_times[1:])))
    est_inter = np.asarray(list(zip(est_times[:-1], est_times[1:])))
    ann_labels = ["a", "b", "c"]
    est_labels = ["a", "b"]
    res = E.compute_results(ann_inter, est_inter, ann_labels, est_labels,
                            251, "")
    npt.assert_almost_equal(res["Su"], 0.76556390622295, decimal=6)
    npt.assert_almost_equal(res["So"], 0.90894734356069651, decimal=6)
    npt.assert_almost_equal(res["Sf"], 0.83111687541927404, decimal=6)
    npt.assert_almost_equal(res["PWP"], 0.80060422960725075, decimal=6)
    npt.assert_almost_equal(res["PWR"], 0.96363636363636362, decimal=6)
    npt.assert_almost_equal(res["PWF"], 0.87458745874587462, decimal=6)


def test_compute_label_results_wrong():
    def __test_dict_res(res):
        assert "Su" not in res.keys()
        assert "So" not in res.keys()
        assert "Sf" not in res.keys()
        assert "PWP" not in res.keys()
        assert "PWR" not in res.keys()
        assert "PWF" not in res.keys()

    ann_times = [3, 12, 20, 40]
    est_times = [1, 4, 20]
    ann_inter = np.asarray(list(zip(ann_times[:-1], ann_times[1:])))
    est_inter = np.asarray(list(zip(est_times[:-1], est_times[1:])))

    # Test the @ token
    ann_labels = ["a", "b", "c"]
    est_labels = ["a", "@"]
    res = E.compute_results(ann_inter, est_inter, ann_labels, est_labels,
                            251, "")
    yield (__test_dict_res, res)

    # Test the -1 token
    ann_labels = ["a", "b", "c"]
    est_labels = ["a", "-1"]
    res = E.compute_results(ann_inter, est_inter, ann_labels, est_labels,
                            251, "")
    yield (__test_dict_res, res)


def test_print_results():
    results = [
        {"HitRate_3F": 0.5, "HitRate_3P": 0.5, "HitRate_3R": 0.5},
        {"HitRate_3F": 0.32, "HitRate_3P": 0.8, "HitRate_3R": 0.2}]
    E.print_results(pd.DataFrame(results))
