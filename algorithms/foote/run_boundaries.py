#!/usr/bin/env python
'''Runs the Foote segmenter for boundaries across the Segmentation dataset

'''

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import sys
import glob
import os
import argparse
import time
import logging
import jams
import segmenter as S

import sys
sys.path.append("../../")
import msaf_io as MSAF


def process(in_path, annot_beats=False, feature="mfcc"):
    """Main process."""

    # Get relevant files
    ds_name = "*"
    jam_files = glob.glob(os.path.join(in_path, "annotations",
                                       "%s_*.jams" % ds_name))
    audio_files = glob.glob(os.path.join(in_path, "audio",
                                         "%s_*.[wm][ap][v3]" % ds_name))

    for audio_file, jam_file in zip(audio_files, jam_files):

        # Only analize files with annotated beats
        if annot_beats:
            jam = jams.load(jam_file)
            if jam.beats == []:
                continue
            if jam.beats[0].data == []:
                continue

        logging.info("Segmenting %s" % audio_file)

        # Foote segmenter call
        est_times = S.process(audio_file, feature=feature,
                              annot_beats=annot_beats)

        #print est_times
        # Save
        out_file = os.path.join(in_path, "estimations",
                                os.path.basename(audio_file)[:-4] + ".json")
        MSAF.save_estimations(out_file, est_times, annot_beats, "foote",
                             feature=feature)


def main():
    """Main function to parse the arguments and call the main process."""
    parser = argparse.ArgumentParser(description=
        "Runs the Foote segmenter across a the Segmentation dataset and "
        "stores the results in the estimations folder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_path",
                        action="store",
                        help="Input dataset")
    parser.add_argument("feature",
                        action="store",
                        help="Feature to be used (mfcc or hpcp)")
    parser.add_argument("-b",
                        action="store_true",
                        dest="annot_beats",
                        help="Use annotated beats",
                        default=False)
    args = parser.parse_args()
    start_time = time.time()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
        level=logging.INFO)

    # Run the algorithm
    process(args.in_path, annot_beats=args.annot_beats,
            feature=args.feature)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))


if __name__ == '__main__':
    main()
