#!/usr/bin/env python
# CREATED:2013-08-22 12:20:01 by Brian McFee <brm2132@columbia.edu>
'''Runs the SIPLCA segmenter across the Segmentation dataset

If run as a program, usage is:

    ./run_segmenter.py dataset_dir/ transform.npy

'''


import sys
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

def create_estimation(times, t_path, annot_beats):
    """Creates a new estimation (dictionary)."""
    est = {}
    est["transform"] = t_path
    est["annot_beats"] = annot_beats
    est["timestamp"] = datetime.datetime.today().strftime("%Y/%m/%d %H:%M:%S")
    est["data"] = list(times)
    return est


def save_segments(out_file, times, t_path, annot_beats):
    """Saves the segment times in the out_file using a JSON format. If file
        exists, update with new annotation."""
    if os.path.isfile(out_file):
        # Add estimation
        f = open(out_file, "r")
        res = json.load(f)

        # Check if estimation already exists
        if "boundaries" in res.keys():
            if "olda" in res["boundaries"]:
                found = False
                for i, est in enumerate(res["boundaries"]["olda"]):
                    if est["transform"] == t_path and \
                            est["annot_beats"] == annot_beats:
                        found = True
                        res["boundaries"]["olda"][i] = \
                            create_estimation(times, t_path, annot_beats)
                        break
                if not found:
                    res["boundaries"]["olda"].append(create_estimation(times, 
                                                        t_path, annot_beats))

        else:
            res["boundaries"] = {}
            res["boundaries"]["olda"] = []
            res["boundaries"]["olda"].append(create_estimation(times, t_path,
                                                annot_beats))
        f.close()
    else:
        # Create new estimation
        res = {}
        res["boundaries"] = {}
        res["boundaries"]["olda"] = []
        res["boundaries"]["olda"].append(create_estimation(times, t_path,
                                            annot_beats))

    # Save dictionary to disk
    f = open(out_file, "w")
    json.dump(res, f, indent=2)
    f.close()


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
        # try:
        X, beats = S.features(audio_file, annot_beats)

        # Get segments
        kmin, kmax  = S.get_num_segs(beats[-1])
        segments    = S.get_segments(X, kmin=kmin, kmax=kmax)
        times       = beats[segments]
        # except:
        #     # The audio file is too short, only beginning and end
        #     logging.warning("Audio file too short! Only start and end boundaries.")
        #     jam = jams.load(jam_file)
        #     times = [0, jam.metadata.duration]

        # Save segments
        out_file = os.path.join(in_path, "estimations", 
            os.path.basename(jam_file).replace(".jams", ".json"))
        save_segments(out_file, times, t_path, annot_beats)


def main():
    """Main function to parse the arguments and call the main process."""
    parser = argparse.ArgumentParser(description=
        "Runs the OLDA segmenter across a the Segmentation dataset and "\
            "stores the results in the estimations folder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_path",
                        action="store",
                        help="Input dataset")
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
    process(args.in_path, args.t_path, annot_beats=args.annot_beats)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))


if __name__ == '__main__':
    main()
