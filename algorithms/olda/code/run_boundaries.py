#!/usr/bin/env python
'''Runs the OLDA segmenter for boundaries across the Segmentation dataset

'''

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import glob
import os
import argparse
import json
import numpy as np
import time
import logging
import datetime
import jams

import pylab as plt

import segmenter as S

import msaf

VERSION = "1.0"


def process(in_path, t_path, ds_prefix="SALAMI", annot_beats=False):
    """Main process."""
    W = S.load_transform(t_path)

    jam_files = glob.glob(os.path.join(in_path, "annotations",
                            "%s_*.jams" % ds_prefix))
    audio_files = glob.glob(os.path.join(in_path, "audio",
                            "%s_*.[wm][ap][v3]" % ds_prefix))

    for jam_file, audio_file in zip(jam_files, audio_files):
        assert os.path.basename(audio_file)[:-4] == \
            os.path.basename(jam_file)[:-5]

        # Only analize files with annotated beats
        if annot_beats:
            jam = jams.load(jam_file)
            if jam.beats == []:
                continue
            if jam.beats[0].data == []:
                continue

        logging.info("Segmenting %s" % audio_file)

        # Get audio features
        try:
            X, beats = S.features(audio_file, annot_beats)

            # Get segments
            kmin, kmax  = S.get_num_segs(beats[-1])
            segments    = S.get_segments(X, kmin=kmin, kmax=kmax)
            times       = beats[segments]
        except:
            # The audio file is too short, only beginning and end
            logging.warning("Audio file too short! Only start and end boundaries.")
            jam = jams.load(jam_file)
            times = [0, jam.metadata.duration]

        params = {
            "transform" : t_path
        }

        # Save segments
        out_file = os.path.join(in_path, "estimations",
            os.path.basename(jam_file).replace(".jams", ".json"))
        MSAF.save_estimations(out_file, times, annot_beats, "olda",
            version="1.0", **params)



def main():
    """Main function to parse the arguments and call the main process."""
    parser = argparse.ArgumentParser(description=
        "Runs the OLDA segmenter across a the Segmentation dataset and "\
            "stores the results in the estimations folder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_path",
                        action="store",
                        help="Input dataset")
    parser.add_argument("t_path",
                        action="store",
                        help="Path to the transform file")
    parser.add_argument("-p",
                        action="store",
                        dest="ds_prefix",
                        help="Prefix to the dataset to be computed "\
                            "(e.g. SALAMI, Isophonics)",
                        default="*")
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
    process(args.in_path, args.t_path, ds_prefix=args.ds_prefix,
            annot_beats=args.annot_beats)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))

if __name__ == '__main__':
    main()
