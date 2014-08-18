#!/usr/bin/env python
'''Runs the new C-NMF segmenter (v3) for boundaries across the Segmentation
dataset
'''

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import copy
import glob
import os
import argparse
import time
import logging
import segmenter as S

from joblib import Parallel, delayed

import msaf
from msaf import jams2
from msaf import io
from msaf import utils
from config import *


def process_track(in_path, audio_file, jam_file, annot_beats, feature, ds_name,
                  framesync, boundaries_id):

    # Only analize files with annotated beats
    if annot_beats:
        jam = jams2.load(jam_file)
        if jam.beats == []:
            return
        if jam.beats[0].data == []:
            return

    logging.info("Segmenting %s" % audio_file)

    # C-NMF segmenter call
    est_times, est_labels = S.process(audio_file, feature=feature,
                                      annot_beats=annot_beats,
                                      boundaries_id=boundaries_id,
                                      framesync=framesync, **config)

    # Save
    out_file = os.path.join(in_path, msaf.Dataset.estimations_dir,
                            os.path.basename(audio_file)[:-4] +
                            msaf.Dataset.estimations_ext)
    logging.info("Writing results in: %s" % out_file)
    est_inters = utils.times_to_intervals(est_times)
    params = copy.deepcopy(config)
    params["annot_beats"] = annot_beats
    params["framesync"] = framesync
    params["feature"] = feature
    if boundaries_id is None:
        boundaries_id = algo_id
    io.save_estimations(out_file, est_inters, est_labels, boundaries_id,
                        algo_id, **params)


def process(in_path, annot_beats=False, feature="mfcc", ds_name="*",
            framesync=False, boundaries_id=None, n_jobs=4):
    """Main process."""

    # Get relevant files
    jam_files = glob.glob(os.path.join(in_path, msaf.Dataset.references_dir,
                                       ("%s_*" + msaf.Dataset.references_ext)
                                       % ds_name))
    audio_files = glob.glob(os.path.join(in_path, "audio",
                                         "%s_*.[wm][ap][v3]" % ds_name))

    # Call in parallel
    Parallel(n_jobs=n_jobs)(delayed(process_track)(
        in_path, audio_file, jam_file, annot_beats, feature, ds_name,
        framesync, boundaries_id)
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
    parser.add_argument("feature",
                        action="store",
                        help="Feature to be used (mfcc, hpcp, or tonnetz)")
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
                        dest="boundaries_id",
                        help="Algorithm id for the boundaries to use "
                        "(None for C-NMF, and \"gt\" for ground truth)",
                        default=None)
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
    process(args.in_path, annot_beats=args.annot_beats, feature=args.feature,
            ds_name=args.ds_name, framesync=args.framesync,
            boundaries_id=args.boundaries_id, n_jobs=args.n_jobs)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))


if __name__ == '__main__':
    main()
