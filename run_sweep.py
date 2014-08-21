#!/usr/bin/env python

import argparse
import eval
import logging
import time
import pandas as pd

import run
from msaf import input_output as io
import msaf.algorithms as algorithms


def process(in_path, annot_beats=False, feature="mfcc", ds_name="*",
            framesync=False, boundaries_id="gt", labels_id=None, n_jobs=4,
            config=None):
    """Sweeps parameters across the specified algorithm."""

    run_name = ds_name
    if ds_name == "Beatles":
        run_name = "Isophonics"

    results_file = "results_sweep_boundsE%s_labelsE%s.csv" % (boundaries_id,
                                                              labels_id)

    if labels_id == "cnmf3" and boundaries_id == "cnmf3":
        config = io.get_configuration(feature, annot_beats, framesync,
                                      boundaries_id, labels_id, algorithms)

        hh = range(4, 20)
        RR = range(4, 20)
        ranks = range(2, 5)
        all_results = pd.DataFrame()
        for rank in ranks:
            for h in hh:
                for R in RR:
                    config["h"] = h
                    config["R"] = R
                    config["rank"] = rank

                    # Run process
                    run.process(in_path, ds_name=run_name, n_jobs=n_jobs,
                                boundaries_id=boundaries_id,
                                labels_id=labels_id, config=config)

                    # Compute evaluations
                    results = eval.process(in_path, boundaries_id, labels_id,
                                           ds_name, save=True, n_jobs=n_jobs,
                                           config=config)

                    # Save avg results
                    new_columns = {"config_h": h, "config_R": R,
                                   "config_rank": rank}
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
