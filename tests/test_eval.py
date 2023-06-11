#!/usr/bin/env python
"""
Unit tests for eval.
"""

from pytest import raises
from types import ModuleType
import numpy.testing as npt
import numpy as np
import os
import pandas as pd
import shutil

import msaf
import msaf.eval as E
from msaf.exceptions import NoEstimationsError, NoReferencesError
from msaf.input_output import FileStruct


def test_compute_boundary_results():
    ann_times = [3, 12, 20, 40]
    est_times = [1, 4, 20]
    ann_inter = np.asarray(list(zip(ann_times[:-1], ann_times[1:])))
    est_inter = np.asarray(list(zip(est_times[:-1], est_times[1:])))
    res = E.compute_results(ann_inter, est_inter, None, None, 251, "")

    npt.assert_almost_equal(res["HitRate_3F"], 0.5714285714285715, decimal=6)
    npt.assert_almost_equal(res["HitRate_3P"], 0.6666666666666666, decimal=6)
    npt.assert_almost_equal(res["HitRate_3R"], 0.5, decimal=6)
    npt.assert_almost_equal(res["HitRate_w3F"], 0.615058910162003, decimal=6)
    npt.assert_almost_equal(res["HitRate_t3F"], 0.0, decimal=6)
    npt.assert_almost_equal(res["HitRate_t3P"], 0.0, decimal=6)
    npt.assert_almost_equal(res["HitRate_t3R"], 0.0, decimal=6)
    npt.assert_almost_equal(res["HitRate_wt3F"], 0.0, decimal=6)

    npt.assert_almost_equal(res["HitRate_0.5F"], 0.28571428571428575,
                            decimal=6)
    npt.assert_almost_equal(res["HitRate_0.5P"], 0.33333333333, decimal=6)
    npt.assert_almost_equal(res["HitRate_0.5R"], 0.25, decimal=6)
    npt.assert_almost_equal(res["HitRate_w0.5F"], 0.307529455081001, decimal=6)
    npt.assert_almost_equal(res["HitRate_t0.5F"], 0.0, decimal=6)
    npt.assert_almost_equal(res["HitRate_t0.5P"], 0.0, decimal=6)
    npt.assert_almost_equal(res["HitRate_t0.5R"], 0.0, decimal=6)
    npt.assert_almost_equal(res["HitRate_wt0.5F"], 0.0, decimal=6)

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
    __test_dict_res(res)

    # Test the -1 token
    ann_labels = ["a", "b", "c"]
    est_labels = ["a", "-1"]
    res = E.compute_results(ann_inter, est_inter, ann_labels, est_labels,
                            251, "")
    __test_dict_res(res)


def test_print_results():
    # Empty results shouldn't crash
    E.print_results([])

    results = [
        {"HitRate_3F": 0.5, "HitRate_3P": 0.5, "HitRate_3R": 0.5},
        {"HitRate_3F": 0.32, "HitRate_3P": 0.8, "HitRate_3R": 0.2}]
    E.print_results(pd.DataFrame(results))


def test_compute_gt_results():
    def __compute_gt_results(est_file, ref_file, boundaries_id,
                             labels_id, config):
        res = E.compute_gt_results(est_file, ref_file, boundaries_id,
                                   labels_id, config)
        assert "HitRate_3F" in res.keys()
        assert "HitRate_3P" in res.keys()
        assert "HitRate_3R" in res.keys()

    def __compute_gt_results_hier(est_file, ref_file, boundaries_id,
                                  labels_id, config):
        res = E.compute_gt_results(est_file, ref_file, boundaries_id,
                                   labels_id, config)
        assert "t_measure10" in res.keys()
        assert "t_measure15" in res.keys()
        assert "t_precision10" in res.keys()
        assert "t_precision15" in res.keys()
        assert "t_recall10" in res.keys()
        assert "t_recall15" in res.keys()

    def __compute_gt_results_no_ests(est_file, ref_file, boundaries_id,
                                     labels_id, config):
        with raises(NoEstimationsError):
            E.compute_gt_results(est_file, ref_file, boundaries_id, labels_id,
                                config)

    def __compute_gt_results_wrong_file(est_file, ref_file, boundaries_id,
                                        labels_id, config):
        with raises(IOError):
            E.compute_gt_results(est_file, ref_file, boundaries_id, labels_id,
                                config)

    config = {"hier": True}
    __compute_gt_results_wrong_file("wrong.json", "wrong.jams", "sf",
           "fmc2d", config)
    config = {"hier": False}
    __compute_gt_results_wrong_file("wrong.json", "wrong.jams", "sf",
           "fmc2d", config)

    est_file = os.path.join("fixtures", "01-Sargon-Mindless-ests.jams")
    ref_file = os.path.join("fixtures", "01-Sargon-Mindless-refs.jams")
    config["feature"] = "wrong"
    __compute_gt_results_no_ests(est_file, ref_file, "foote", None,
           config)

    # Correct Flat
    config["feature"] = "pcp"
    __compute_gt_results(est_file, ref_file, "sf", None, config)
    # Correct Hierarchical
    config["hier"] = True
    __compute_gt_results_hier(est_file, ref_file, "olda", None,
           config)


def test_process_track():
    def __process_track_wrong_names(file_struct):
        with raises(AssertionError):
            E.process_track(file_struct, None, None, None)

    def __process_track_no_refs(file_struct):
        with raises(NoReferencesError):
            E.process_track(file_struct, None, None, None)

    def __process_track_correct(file_struct, bounds_id, labels_id, config):
        res = E.process_track(file_struct, bounds_id, labels_id, config)
        assert "HitRate_3F" in res.keys()

    # Wrong match
    no_match_fs = FileStruct("udontexist.mp3")
    no_match_fs.ref_file = "idontexist.mp3"
    __process_track_wrong_names(no_match_fs)

    # No References
    no_ref_fs = FileStruct("udontexist.mp3")
    no_ref_fs.ref_file = "udontexist.jams"
    __process_track_no_refs(no_ref_fs)

    # Correct
    audio_path = "fixtures/Sargon_test/audio/Mindless_cut.mp3"
    correct_fs = FileStruct(audio_path)
    config = {"hier": False}
    __process_track_correct(correct_fs, "sf", None, config)
    __process_track_correct(audio_path, "sf", None, config)


def test_get_results_file_name():
    config = {"hier": False}
    file_name = E.get_results_file_name("sf", None, config, 0)
    print(file_name)
    assert file_name == os.path.join(
        msaf.config.results_dir,
        "results_boundsEsf_labelsENone_annotatorE0_hierEFalse.csv")

    # Try with a file that's too long
    config["thisIsWayTooLongForCertainOSs"] = "I am Sargon, king of Akkad"
    config["thisIsWayTooLongForCertainOSspart2"] = "I am Sargon, the mighty " \
        "one"
    config["thisIsWayTooLongForCertainOSspart3"] = "You Are a Titan"
    config["thisIsWayTooLongForCertainOSspart4"] = "This is silly"
    file_name = E.get_results_file_name("sf", None, config, 0)
    assert len(file_name) == 255

    # Make sure the results folder was created
    assert os.path.isdir(msaf.config.results_dir)
    shutil.rmtree(msaf.config.results_dir)


def test_process():
    # Single File Mode
    res = E.process("fixtures/Sargon_test/audio/Mindless_cut.mp3")
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 1

    # Collection Mode
    res = E.process("fixtures/Sargon_test/")
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 2

    # Saving results to default file
    config = {"feature": "pcp",
              "framesync": False,
              "annot_beats": False}
    res = E.process("fixtures/Sargon_test/", config=config, save=True)
    out_file = E.get_results_file_name("sf", None, config, 0)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 2
    assert os.path.isfile(out_file)
    os.remove(out_file)

    # Saving results to custom file
    out_file = "my_fancy_file.csv"
    res = E.process("fixtures/Sargon_test/", config=config, save=True,
                    out_file=out_file)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 2
    assert os.path.isfile(out_file)

    # Do it again, this time it shouldn't compute anything, simply read file
    res = E.process("fixtures/Sargon_test/", config=config, save=True,
                    out_file=out_file)
    os.remove(out_file)
