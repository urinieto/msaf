#!/usr/bin/env python

import argparse
import logging
import time
import pandas as pd
import numpy as np

import msaf
from msaf import input_output as io
import msaf.algorithms as algorithms


def process(in_path, annot_beats=False, feature="mfcc", ds_name="*",
            framesync=False, boundaries_id="gt", labels_id=None, n_jobs=4,
            config=None):
    """Sweeps parameters across the specified algorithm."""

    results_file = "results_sweep_boundsE%s_labelsE%s.csv" % (boundaries_id,
                                                              labels_id)

    if labels_id == "cnmf3" or boundaries_id == "cnmf3":
        config = io.get_configuration(feature, annot_beats, framesync,
                                      boundaries_id, labels_id)

        hh = range(8, 33)
        RR = range(8, 65)
        ranks = range(3, 6)
        RR_labels = range(11, 12)
        ranks_labels = range(6, 7)
        all_results = pd.DataFrame()
        for rank in ranks:
            for h in hh:
                if rank == 3 and h <= 10:
                    continue
                for R in RR:
                    for rank_labels in ranks_labels:
                        for R_labels in RR_labels:
                            config["h"] = h
                            config["R"] = R
                            config["rank"] = rank
                            config["rank_labels"] = rank_labels
                            config["R_labels"] = R_labels
                            config["features"] = None

                            # Run process
                            msaf.run.process(in_path, ds_name=ds_name, n_jobs=n_jobs,
                                        boundaries_id=boundaries_id,
                                        labels_id=labels_id, config=config)

                            # Compute evaluations
                            results = msaf.eval.process(in_path, boundaries_id, labels_id,
                                                ds_name, save=True, n_jobs=n_jobs,
                                                config=config)

                            # Save avg results
                            new_columns = {"config_h": h, "config_R": R,
                                           "config_rank": rank,
                                           "config_R_labels": R_labels,
                                           "config_rank_labels": rank_labels}
                            results = results.append([new_columns],
                                                    ignore_index=True)
                            all_results = all_results.append(results.mean(),
                                                            ignore_index=True)
                            all_results.to_csv(results_file)

    elif labels_id is None and boundaries_id == "sf":
        config = io.get_configuration(feature, annot_beats, framesync,
                                      boundaries_id, labels_id)

        MM = range(8, 24)
        mm = range(2, 4)
        kk = np.arange(0.02, 0.1, 0.01)
        Mpp = range(16, 24)
        ott = np.arange(0.02, 0.1, 0.01)
        all_results = pd.DataFrame()
        for M in MM:
            for m in mm:
                for k in kk:
                    for Mp in Mpp:
                        for ot in ott:
                            config["M_gaussian"] = M
                            config["m_embedded"] = m
                            config["k_nearest"] = k
                            config["Mp_adaptive"] = Mp
                            config["offset_thres"] = ot
                            config["features"] = None

                            # Run process
                            msaf.run.process(in_path, ds_name=ds_name, n_jobs=n_jobs,
                                        boundaries_id=boundaries_id,
                                        labels_id=labels_id, config=config)

                            # Compute evaluations
                            results = msaf.eval.process(in_path, boundaries_id, labels_id,
                                                ds_name, save=True, n_jobs=n_jobs,
                                                config=config)

                            # Save avg results
                            new_columns = {"config_M": M, "config_m": m,
                                           "config_k": k, "config_Mp": Mp,
                                           "config_ot": ot}
                            results = results.append([new_columns],
                                                    ignore_index=True)
                            all_results = all_results.append(results.mean(),
                                                            ignore_index=True)
                            all_results.to_csv(results_file)

    else:
        logging.error("Can't sweep parameters for %s algorithm. "
                      "Implement me! :D")


def main():
    """Main function to sweep parameters of a certain algorithm."""
    parser = argparse.ArgumentParser(description=
        "Runs the speficied algorithm(s) on the MSAF formatted dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_path",
                        action="store",
                        help="Input dataset")
    parser.add_argument("-f",
                        action="store",
                        dest="feature",
                        default="hpcp",
                        type=str,
                        help="Type of features",
                        choices=["hpcp", "tonnetz", "mfcc"])
    parser.add_argument("-b",
                        action="store_true",
                        dest="annot_beats",
                        help="Use annotated beats",
                        default=False)
    parser.add_argument("-fs",
                        action="store_true",
                        dest="framesync",
                        help="Use frame-synchronous features",
                        default=False)
    parser.add_argument("-bid",
                        action="store",
                        help="Boundary algorithm identifier",
                        dest="boundaries_id",
                        default="gt",
                        choices=["gt"] +
                        io.get_all_boundary_algorithms(algorithms))
    parser.add_argument("-lid",
                        action="store",
                        help="Label algorithm identifier",
                        dest="labels_id",
                        default=None,
                        choices= io.get_all_label_algorithms(algorithms))
    parser.add_argument("-d",
                        action="store",
                        dest="ds_name",
                        default="*",
                        help="The prefix of the dataset to use "
                        "(e.g. Isophonics, SALAMI")
    parser.add_argument("-j",
                        action="store",
                        dest="n_jobs",
                        default=4,
                        type=int,
                        help="The number of threads to use")
    args = parser.parse_args()
    start_time = time.time()

    # Run the algorithm(s)
    process(args.in_path, annot_beats=args.annot_beats, feature=args.feature,
            ds_name=args.ds_name, framesync=args.framesync,
            boundaries_id=args.boundaries_id, labels_id=args.labels_id,
            n_jobs=args.n_jobs)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))


if __name__ == "__main__":
    main()
