#!/usr/bin/env python
"""
This script computes the features for a single or multiple audio files.


Examples:

    Single file mode:
        >> ./compute_features.py path_to_audio.mp3 -o my_features.json

    Collection mode:
        Run on 12 cores:
            >> ./compute_features.py path_to_dataset/ -j 12

        Run on 8 cores, and overwrite previous features:
            >> ./compute_features.py path_to_dataset/ -j 8 -ow

"""

import argparse
import time
import logging

# Local stuff
import msaf.featextract


def main():
    """Main function to parse the arguments and call the main process."""
    parser = argparse.ArgumentParser(description=
        "Extracts a set of features from the Segmentation dataset or a given "
        "audio file and saves them into the 'features' folder of the dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_path",
                        action="store",
                        help="Input dataset dir or audio file")
    parser.add_argument("-s",
                        action="store_true",
                        dest="sonify_beats",
                        help="Sonifies the estimated beats",
                        default=False)
    parser.add_argument("-j",
                        action="store",
                        dest="n_jobs",
                        type=int,
                        help="Number of jobs (only for collection mode)",
                        default=4)
    parser.add_argument("-ow",
                        action="store_true",
                        dest="overwrite",
                        help="Overwrite the previously computed features",
                        default=False)
    parser.add_argument("-o",
                        action="store",
                        dest="out_file",
                        type=str,
                        help="Output file (only for single file mode)",
                        default="out.json")
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

    # Run the algorithm
    msaf.featextract.process(args.in_path, sonify_beats=args.sonify_beats,
                             n_jobs=args.n_jobs, overwrite=args.overwrite,
                             out_file=args.out_file, ds_name=args.ds_name)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))

if __name__ == '__main__':
    main()
