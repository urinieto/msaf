#!/usr/bin/env python
"""
Runs one boundary algorithm and a label algorithm on a specified audio file or
dataset.
"""

import argparse
import time
import logging

# MSAF import
import msaf


def main():
    """Main function to parse the arguments and call the main process."""
    parser = argparse.ArgumentParser(description=
        "Runs the speficied algorithm(s) on the input file or MSAF formatted "
        "dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_path",
                        action="store",
                        help="Input audio file or dataset")
    parser.add_argument("-f",
                        action="store",
                        dest="feature",
                        default="hpcp",
                        type=str,
                        help="Type of features",
                        choices=["hpcp", "tonnetz", "mfcc"])
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
    parser.add_argument("-s",
                        action="store_true",
                        dest="sonify_bounds",
                        help="Sonifies the estimated boundaries on top of the "
                        "audio file", default=False)
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
    parser.add_argument("-j",
                        action="store",
                        dest="n_jobs",
                        default=1,
                        type=int,
                        help="The number of threads to use")
    parser.add_argument("-p",
                        action="store_true",
                        dest="plot",
                        help="Plots the current boundaries",
                        default=False)
    parser.add_argument("-d",
                        action="store",
                        dest="ds_name",
                        default="*",
                        help="The prefix of the dataset to use "
                        "(e.g. Isophonics, SALAMI)")
    args = parser.parse_args()
    start_time = time.time()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
        level=logging.INFO)

    # Run the algorithm(s)
    msaf.run.process(args.in_path, annot_beats=args.annot_beats,
                     feature=args.feature, ds_name=args.ds_name,
                     framesync=args.framesync, boundaries_id=args.boundaries_id,
                     labels_id=args.labels_id, n_jobs=args.n_jobs,
                     sonify_bounds=args.sonify_bounds, plot=args.plot)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))


if __name__ == '__main__':
    main()
