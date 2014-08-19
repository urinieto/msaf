#!/usr/bin/env python
'''Runs the 2D-FMC segment similarity algorithm in the Segments dataset.
'''

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import glob
import os
import argparse
import time
import logging
import jams2
import segmenter as S

from joblib import Parallel, delayed

import sys
sys.path.append("../../")
import msaf_io as MSAF


def process_track(in_path, audio_file, jam_file, annot_beats, ds_name,
                  xmeans, k):

    # Only analize files with annotated beats
    if annot_beats:
        jam = jams2.load(jam_file)
        if jam.beats == []:
            return
        if jam.beats[0].data == []:
            return

    logging.info("Segmenting %s" % audio_file)

    # 2D-FMC segmenter call
    est_times, est_labels = S.process(audio_file, annot_beats=annot_beats,
                                      xmeans=xmeans, k=k)

    # Save
    out_file = os.path.join(in_path, "estimations",
                            os.path.basename(audio_file)[:-4] + ".json")
    MSAF.save_estimations(out_file, est_labels, annot_beats, "2dfmc",
                          bounds=False, annot_bounds=True, feature="")


def process(in_path, annot_beats=False, feature="mfcc", ds_name="*", n_jobs=4,
            xmeans=False, k=5):
    """Main process."""

    # Get relevant files
    jam_files = glob.glob(os.path.join(in_path, "annotations",
                                       "%s_*.jams" % ds_name))
    audio_files = glob.glob(os.path.join(in_path, "audio",
                                         "%s_*.[wm][ap][v3]" % ds_name))

    # Call in parallel
    Parallel(n_jobs=n_jobs)(delayed(process_track)(
        in_path, audio_file, jam_file, annot_beats, ds_name, xmeans, k)
        for audio_file, jam_file in zip(audio_files, jam_files))


def main():
    """Main function to parse the arguments and call the main process."""
    parser = argparse.ArgumentParser(description=
        "Runs the version 3 of the C-NMF segmenter across a the Segmentation"
        " dataset and stores the results in the estimations folder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_path",
                        action="store",
                        help="Input dataset")
    parser.add_argument("-b",
                        action="store_true",
                        dest="annot_beats",
                        help="Use annotated beats",
                        default=False)
    parser.add_argument("-k",
                        action="store",
                        dest="k",
                        help="Number of unique segments",
                        default=6)
    parser.add_argument("-x",
                        action="store_true",
                        dest="xmeans",
                        help="Estimate K using the BIC method",
                        default=False)
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

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
        level=logging.INFO)

    # Run the algorithm
    process(args.in_path, annot_beats=args.annot_beats, ds_name=args.ds_name,
            n_jobs=args.n_jobs, xmeans=args.xmeans, k=args.k)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))


if __name__ == '__main__':
    main()
