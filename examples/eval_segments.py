#!/usr/bin/env python
"""
Script to evaluate estimated segment results against ground truth (annotated
human data).
"""

import argparse
import logging
import time

# Local stuff
import msaf


def main():
    """Main function to parse the arguments and call the main process."""
    parser = argparse.ArgumentParser(description=
        "Evaluates the estimated results of the Segmentation dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_path",
                        action="store",
                        help="Input dataset")
    parser.add_argument("-bid",
                        action="store",
                        help="Boundary algorithm identifier",
                        dest="boundaries_id",
                        default="gt",
                        choices=["gt"] +
                        msaf.io.get_all_boundary_algorithms(msaf.algorithms))
    parser.add_argument("-lid",
                        action="store",
                        help="Label algorithm identifier",
                        dest="labels_id",
                        default=None,
                        choices= \
                        msaf.io.get_all_label_algorithms(msaf.algorithms))
    parser.add_argument("-d",
                        action="store",
                        dest="ds_name",
                        default="*",
                        help="The prefix of the dataset to use "
                        "(e.g. Isophonics, SALAMI")
    parser.add_argument("-b",
                        action="store_true",
                        dest="annot_beats",
                        help="Use annotated beats",
                        default=False)
    parser.add_argument("-fs",
                        action="store_true",
                        dest="framesync",
                        default=False,
                        help="Whether to use framesync features or not")
    parser.add_argument("-f",
                        action="store",
                        dest="feature",
                        default="hpcp",
                        type=str,
                        help="Type of features",
                        choices=["hpcp", "tonnetz", "mfcc"])
    parser.add_argument("-s",
                        action="store_true",
                        dest="save",
                        help="Whether to save the results or not",
                        default=False)
    parser.add_argument("-j",
                        action="store",
                        dest="n_jobs",
                        default=4,
                        type=int,
                        help="The number of processes to run in parallel")
    args = parser.parse_args()
    start_time = time.time()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                        level=logging.INFO)

    # Run the algorithm
    msaf.eval.process(args.in_path, args.boundaries_id, args.labels_id,
                      args.ds_name, args.annot_beats, save=args.save,
                      feature=args.feature, n_jobs=args.n_jobs,
                      framesync=args.framesync)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))

if __name__ == '__main__':
    main()
