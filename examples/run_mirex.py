#!/usr/bin/env python
"""Runs one boundary algorithm and a label algorithm on a specified audio file
and outputs the results using the MIREX format."""
import argparse
import logging
import time

# MSAF import
import msaf


def main():
    """Main function to parse the arguments and call the main process."""
    parser = argparse.ArgumentParser(
        description="Runs the specified algorithm(s) on the input file and "
        "the results using the MIREX format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-bid",
        action="store",
        help="Boundary algorithm identifier",
        dest="boundaries_id",
        default=msaf.config.default_bound_id,
        choices=["gt"] + msaf.io.get_all_boundary_algorithms(),
    )
    parser.add_argument(
        "-lid",
        action="store",
        help="Label algorithm identifier",
        dest="labels_id",
        default=msaf.config.default_label_id,
        choices=msaf.io.get_all_label_algorithms(),
    )
    parser.add_argument("-i", action="store", dest="in_file", help="Input audio file")
    parser.add_argument(
        "-o",
        action="store",
        dest="out_file",
        help="Output file with the results",
        default="out.txt",
    )

    args = parser.parse_args()
    start_time = time.time()

    # Setup the logger
    logging.basicConfig(
        format="%(asctime)s: %(levelname)s: %(message)s", level=logging.INFO
    )

    # Run MSAF
    params = {
        "annot_beats": False,
        "feature": "cqt",
        "framesync": False,
        "boundaries_id": args.boundaries_id,
        "labels_id": args.labels_id,
        "n_jobs": 1,
        "hier": False,
        "sonify_bounds": False,
        "plot": False,
    }
    res = msaf.run.process(args.in_file, **params)
    msaf.io.write_mirex(res[0], res[1], args.out_file)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))


if __name__ == "__main__":
    main()
