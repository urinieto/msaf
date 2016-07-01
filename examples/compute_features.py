#!/usr/bin/env python
"""
This script computes the features for a single or multiple audio files.
It uses the default parameters. These parameters can be changed the
`.msafrc` config file.

Examples:

    Single file mode:
        >> ./compute_features.py path_to_audio.mp3 -o my_features.json

    Collection mode:
        Run on 12 cores on the specified dataset:
            >> ./compute_features.py path_to_dataset/ -j 12
"""

import argparse
import logging
import os
import time

# Local stuff
import msaf
from msaf.features import Features


def compute_all_features(file_struct):
    """Computes all features for the given file."""
    for feature_id in msaf.features_registry:
        logging.info("Computing %s for file %s" % (feature_id,
                                                   file_struct.audio_file))
        feats = Features.select_features(feature_id, file_struct, False, False)
        feats.features


def process(in_path, out_file, n_jobs):
    """Computes the features for the selected dataset or file."""
    if os.path.isfile(in_path):
        # Single file mode
        # Get (if they exitst) or compute features
        file_struct = msaf.io.FileStruct(in_path)
        file_struct.features_file = out_file
        compute_all_features(file_struct)
    else:
        # Collection mode
        file_structs = io.get_dataset_files(in_path)

        # Call in parallel
        return Parallel(n_jobs=n_jobs)(delayed(compute_all_features)(
            file_struct) for file_struct in file_structs)


def main():
    """Main function to parse the arguments and call the main process."""
    parser = argparse.ArgumentParser(
        description="Extracts a set of features from a given dataset "
        "or audio file and saves them into the 'features' folder of "
        "the dataset or the specified single file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_path",
                        action="store",
                        help="Input dataset dir or audio file")
    parser.add_argument("-j",
                        action="store",
                        dest="n_jobs",
                        type=int,
                        help="Number of jobs (only for collection mode)",
                        default=4)
    parser.add_argument("-o",
                        action="store",
                        dest="out_file",
                        type=str,
                        help="Output file (only for single file mode)",
                        default="out.json")
    args = parser.parse_args()
    start_time = time.time()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                        level=logging.INFO)

    # Run the main process
    process(args.in_path, out_file=args.out_file, n_jobs=args.n_jobs)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))

if __name__ == '__main__':
    main()
