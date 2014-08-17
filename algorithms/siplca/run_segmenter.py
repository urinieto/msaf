#!/usr/bin/env python
'''Runs the SIPLCA segmenter for boundaries and labels across the
a segmentation dataset.

'''

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import glob
import os
import argparse
import numpy as np
import time
import logging

from joblib import Parallel, delayed

# SI-PLCA segmenter
import segmenter as S

from msaf import input_output as io
from msaf import jams2
from msaf import utils as U


def process_track(in_path, audio_file, jam_file, annot_beats, annot_bounds,
                  features="hpcp"):
    assert os.path.basename(audio_file)[:-4] == \
        os.path.basename(jam_file)[:-5]

    # Only analize files with annotated beats
    if annot_beats:
        jam = jams2.load(jam_file)
        if jam.beats == []:
            return
        if jam.beats[0].data == []:
            return

    logging.info("Segmenting %s" % audio_file)

    # SI-PLCA Params (From MIREX)
    params = {
        "niter"             :   200,
        "rank"              :   15,
        "win"               :   60,
        "alphaZ"            :   -0.01,
        "normalize_frames"  :   True,
        "viterbi_segmenter" :   True,
        "min_segment_length":   1,
        "plotiter"          :   None,
        "feature"           :   features
    }
    if annot_bounds:
        feats, mfcc, beats, dur = io.get_features(
            audio_file, annot_beats=annot_beats)
        try:
            bound_idxs = io.read_annot_bound_frames(audio_file, beats)
        except:
            logging.warning("No annotation boundaries for %s" %
                            audio_file)
            return
        if features == "tonnetz":
            feats = U.chroma_to_tonnetz(feats)
        n_segments = len(bound_idxs) - 1
        max_beats_segment = np.max(np.diff(bound_idxs))
        print "Longest segment: %d", max_beats_segment

        # Inititalize the W and H matrices using the previously found bounds
        initW = np.zeros((feats.shape[1], n_segments, max_beats_segment))
        initH = np.zeros((n_segments, feats.shape[0]))
        for i in xrange(n_segments):
            dur = bound_idxs[i+1] - bound_idxs[i]
            initW[:, i, :dur] = feats[bound_idxs[i]:bound_idxs[i+1]].T
            initH[i, bound_idxs[i]] = 1

        # Update parameters
        params["win"] = max_beats_segment
        params["rank"] = n_segments
        params["initW"] = initW
        params["initH"] = initH

    segments, beattimes, frame_labels = S.segment_wavfile(
        audio_file, b=annot_beats, **params)

    # Convert segments to times
    lines = segments.split("\n")[:-1]
    times = []
    labels = []
    for line in lines:
        time = float(line.strip("\n").split("\t")[0])
        times.append(time)
        label = line.strip("\n").split("\t")[2]
        labels.append(ord(label))

    # Add last one and reomve empty segments
    times, idxs = np.unique(times, return_index=True)
    labels = np.asarray(labels)[idxs]
    times = np.concatenate((times,
                            [float(lines[-1].strip("\n").split("\t")[1])]))
    times = np.unique(times)

    if annot_bounds:
        labels = []
        start = bound_idxs[0]
        for end in bound_idxs[1:]:
            segment_labels = frame_labels[start:end]
            try:
                label = np.argmax(np.bincount(segment_labels))
            except:
                label = frame_labels[start]
            labels.append(label)
            start = end

        times = beats[bound_idxs]

    assert len(times) - 1 == len(labels)

    logging.info("Estimated boundaries: %s" % times)
    logging.info("Estimated labels: %s" % labels)

    # Save segments
    out_file = os.path.join(in_path, "estimations",
        os.path.basename(jam_file).replace(".jams", ".json"))
    logging.info("Writing results in: %s" % out_file)
    if not annot_bounds:
        io.save_estimations(out_file, times, annot_beats, "siplca",
            version="1.0", **params)

    # Remove paramaters that we don't want to store
    params.pop("initW", None)
    params.pop("initH", None)
    params.pop("plotiter", None)
    params.pop("win", None)
    params.pop("rank", None)

    # Save results
    io.save_estimations(out_file, labels, annot_beats, "siplca",
        bounds=False, annot_bounds=annot_bounds, version="1.0", **params)


def process(in_path, ds_name="*", n_jobs=4, annot_beats=False,
            annot_bounds=False, features="hpcp"):
    """Main process."""

    # Get relevant files
    jam_files = glob.glob(os.path.join(in_path, "annotations",
                                       "%s_*.jams" % ds_name))
    audio_files = glob.glob(os.path.join(in_path, "audio",
                                         "%s_*.[wm][ap][v3]" % ds_name))

    # Run jobs in parallel
    Parallel(n_jobs=n_jobs)(delayed(process_track)(
        in_path, audio_file, jam_file, annot_beats, annot_bounds, features)
        for jam_file, audio_file in zip(jam_files, audio_files)[:])


def main():
    """Main function to parse the arguments and call the main process."""
    parser = argparse.ArgumentParser(description=
        "Runs the OLDA segmenter across a the Segmentation dataset and "
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
    parser.add_argument("-bo",
                        action="store_true",
                        dest="annot_bounds",
                        help="Use annotated bounds",
                        default=False)
    parser.add_argument("-f",
                        action="store",
                        dest="features",
                        default="hpcp",
                        help="The type of features to be used.")
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
                        help="The number of processes to run in parallel")
    args = parser.parse_args()
    start_time = time.time()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
        level=logging.INFO)

    # Run the algorithm
    process(args.in_path, annot_beats=args.annot_beats, n_jobs=args.n_jobs,
            annot_bounds=args.annot_bounds, ds_name=args.ds_name,
            features=args.features)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))


if __name__ == '__main__':
    main()
