#!/usr/bin/env python
'''Runs the Serra segmenter for boundaries across the Segmentation dataset

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
import json
import numpy as np
import time
import logging
import jams
import segmenter as S
import mir_eval

import sys
sys.path.append( "../../" )
import msaf_io as MSAF

from joblib import Parallel, delayed


def segment_track(audio_file, jam_file, in_path, annot_beats, feature,
        **params):
    """Segments the audio file with the corresponding given jam file."""
    # Only analize files with annotated beats
    if annot_beats:
        jam = jams.load(jam_file)
        if jam.beats == []:
            return
        if jam.beats[0].data == []:
            return 

    # Segment the Beatles only
    jam_file = os.path.join(in_path, "annotations", 
                os.path.basename(audio_file)[:-4] + ".jams")
    ann_times, ann_labels = mir_eval.input_output.load_jams_range(
                            jam_file, 
                            "sections", context="function")
    jam = jams.load(jam_file)
    if jam.metadata.artist != "The Beatles":
        return None

    logging.info("Segmenting %s" % audio_file)

    # Serra segmenter call
    est_times = S.process(audio_file, feature=feature, 
                                    annot_beats=annot_beats, **params)

    # Save
    out_file = os.path.join(in_path, "estimations", 
                                os.path.basename(audio_file)[:-4]+".json")
    MSAF.save_boundaries(out_file, est_times, annot_beats, "serra",
                                feature=feature)

    # Evaluate
    P, R, F = mir_eval.segment.boundary_detection(ann_times, est_times, 
                                                    window=3, trim=False)

    data = {
        "F" : F,
        "P" : P,
        "R" : R,
        "audio_file" : audio_file
    }
    return data



def process(in_path, annot_beats=False, feature="mfcc", n_jobs=1, **params):
    """Main process."""

    # Get relevant files
    ds_name = "Isophonics"
    jam_files = glob.glob(os.path.join(in_path, "annotations", "%s_*.jams" % ds_name))
    audio_files = glob.glob(os.path.join(in_path, "audio", "%s_*.[wm][ap][v3]" % ds_name))

    # Sweep parameters
    # for M in np.arange(8, 40):
    #     for m in np.arange(1,5,0.5):
    #         for k in np.arange(0.01, 0.1, 0.01):
    #             # Segment using joblib
    #             data = Parallel(n_jobs=n_jobs)(delayed(segment_track)( \
    #                 audio_file, jam_file, in_path, annot_beats, feature,
    #                 M=M, m=m, k=k) \
    #                 for audio_file, jam_file in zip(audio_files, jam_files))

    #             F, P, R = [], [], []
    #             for d in data:
    #                 if d is None:
    #                     continue
    #                 F.append(d["F"])
    #                 P.append(d["P"])
    #                 R.append(d["R"])
    #             F = np.asarray(F)
    #             P = np.asarray(P)
    #             R = np.asarray(R)

    #             with open("results.txt", "a") as f:
    #                 f.write("%.2f\t%.2f\t%.2f\t%d\t%.2f\t%.2f\n" % (100*F.mean(), 
    #                                 100*P.mean(), 100*R.mean(), M, m, k))

    # Segment using joblib
    data = Parallel(n_jobs=n_jobs)(delayed(segment_track)( \
        audio_file, jam_file, in_path, annot_beats, feature,
        M=params["M"], m=params["m"], k=params["k"]) \
        for audio_file, jam_file in zip(audio_files, jam_files))


    out_str = ""
    for d in data:
        if d is None:
            continue
        out_str += "%s\t%.2f\t%.2f\t%.2f\n" % (d["audio_file"], 
                    100*d["F"], 100*d["P"], 100*d["R"])

    with open("beatles_results.txt", "w") as f:
        f.write(out_str)


def main():
    """Main function to parse the arguments and call the main process."""
    parser = argparse.ArgumentParser(description=
        "Runs the Serra segmenter across a the Segmentation dataset and "\
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
    parser.add_argument("-j", 
                        action="store", 
                        dest="n_jobs",
                        type=int,
                        help="Number of jobs (threads)",
                        default=1)
    parser.add_argument("-M", 
                        action="store", 
                        dest="M",
                        help="Size of gaussian kernel in beats",
                        type=int,
                        default=17)
    parser.add_argument("-m", 
                        action="store", 
                        dest="m",
                        help="Number of embedded dimensions",
                        type=float,
                        default=4)
    parser.add_argument("-k", 
                        action="store", 
                        dest="k",
                        help="k*N-nearest neighbors for recurrence plot",
                        type=float,
                        default=0.08)
    args = parser.parse_args()
    start_time = time.time()
    
    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', 
        level=logging.INFO)

    # Run the algorithm
    process(args.in_path, annot_beats=args.annot_beats, 
            feature=args.feature, n_jobs=args.n_jobs, M=args.M, m=args.m,
            k=args.k)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))


if __name__ == '__main__':
    main()
