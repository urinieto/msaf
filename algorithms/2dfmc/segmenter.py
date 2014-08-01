#!/usr/bin/env python
# coding: utf-8
"""
This method labels segments using the 2D-FMC method described here:

Nieto, O., Bello, J.P., Music Segment Similarity Using 2D-Fourier Magnitude
    Coefficients. Proc. of the 39th IEEE International Conference on Acoustics,
    Speech, and Signal Processing (ICASSP). Florence, Italy, 2014.
"""

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import argparse
import json
import glob
import os
import sys
import time
import mir_eval
import logging
from joblib import Parallel, delayed

import numpy as np
import pylab as plt

import scipy.cluster.vq as vq

# Local stuff
import utils
from xmeans import XMeans

sys.path.append("../../")
import msaf_io as MSAF
#import eval as EV
import utils as U

# Globals
transform = dict()
audio_dir = "originalAudio"
gt_dir = "gt"
est_dir = "est"
pcp_dir = "pcp"

MIN_LEN = 4

# Various aggregation functions
aggregate_functions = {"max": np.max,
                       "median": np.median,
                       "mean": np.mean
                       }


class FileStruct(object):
    def __init__(self, audio, segments_gt, segments_est, beats_est, bounds_est,
                 pcp):
        self.audio = audio
        self.segments_gt = segments_gt
        self.segments_est = segments_est
        self.beats_est = beats_est
        self.bounds_est = bounds_est
        self.pcp = pcp

    def __str__(self):
        out_str = "Audio File: %s\n" % self.audio
        out_str += "Segments GT File: %s\n" % self.segments_gt
        out_str += "Segments Est File: %s\n" % self.segments_est
        out_str += "Beats Est File: %s\n" % self.beats_est
        out_str += "Bounds Est File: %s\n" % self.bounds_est
        out_str += "PCP File: %s\n" % self.pcp
        return out_str


def create_pcp(param_file):
    """Creates an AudioReaderPCP to be ready to compute PCPs."""
    with open(param_file, "r") as f:
        params = json.load(f)
        transform["params"] = params
        transform["pcp"] = AudioReaderPCP(
            q=params["q"], freq_min=params["freq_min"],
            octaves=params["octaves"], samplerate=params["samplerate"],
            bins_per_octave=params["bins_per_octave"],
            pitch_classes=params["pitch_classes"])


def audio_file_to_pcp(file_struct):
    """Compute the PCP for a file struct.

    Parameters
    ----------
    file_struct : File Struct

    Returns
    -------
    Nothing, but the output file is written in this call.
    """
    pcp = transform["pcp"]
    params = transform["params"]
    reader = FramedAudioReader(filepath=file_struct.audio,
                        samplerate=pcp.samplerate(),
                        framesize=pcp.framesize(),
                        framerate=params["framerate"],
                        alignment=params["alignment"],
                        channels=params["channels"])

    logging.info("Finished: %s" % file_struct.audio)

    print pcp
    pcp(reader)
    PCP = pcp(reader).squeeze()
    np.save(file_struct.pcp, PCP)
    return PCP


def bounds_to_bound_idxs(bounds, frame_times):
    """Given an array of bounds and the frame times, return bounds indeces with
        respect the frame times."""
    bounds_idxs = []
    k = 0
    for i, frame_time in enumerate(frame_times):
        if k >= len(bounds):
            break
        if bounds[k] <= frame_time:
            bounds_idxs.append(i)
            k += 1

    # Add last boundary if needed
    while len(bounds_idxs) != len(bounds):
        bounds_idxs.append(i)

    assert len(bounds_idxs) == len(bounds)

    return np.asarray(bounds_idxs)


def get_pcp_segments(PCP, bound_idxs):
    """Returns a set of segments defined by the bound_idxs."""
    pcp_segments = []
    for i in xrange(len(bound_idxs)-1):
        pcp_segments.append(PCP[bound_idxs[i]:bound_idxs[i+1], :])
    return pcp_segments


def pcp_segments_to_2dfmc_max(pcp_segments):
    """From a list of PCP segments, return a list of 2D-Fourier Magnitude
        Coefs using the maximumg segment size and zero pad the rest."""
    # Get maximum segment size
    max_len = max([pcp_segment.shape[0] for pcp_segment in pcp_segments])

    OFFSET = 4
    fmcs = []
    for pcp_segment in pcp_segments:
        # Zero pad if needed
        X = np.zeros((max_len, 12))
        #X[:pcp_segment.shape[0],:] = pcp_segment
        if pcp_segment.shape[0] <= OFFSET:
            X[:pcp_segment.shape[0], :] = pcp_segment
        else:
            X[:pcp_segment.shape[0]-OFFSET, :] = \
                pcp_segment[OFFSET/2:-OFFSET/2, :]

        # 2D-FMC
        fmcs.append(utils.compute_ffmc2d(X))

        # Normalize
        #fmcs[-1] = fmcs[-1] / fmcs[-1].max()

    return np.asarray(fmcs)


def compute_labels_kmeans(fmcs, k=6):
    # Removing the higher frequencies seem to yield better results
    fmcs = fmcs[:, fmcs.shape[1]/2:]

    fmcs = np.log1p(fmcs)
    wfmcs = vq.whiten(fmcs)

    dic, dist = vq.kmeans(wfmcs, k, iter=100)
    labels, dist = vq.vq(wfmcs, dic)

    return labels


def times_to_intervals(times):
    """A list of times to a list of intervals."""
    return np.asarray(zip(times[:-1], times[1:]))


def adjust_bounds_labels(bound_times, labels):
    """Times to intervals and remove reduntant."""
    if bound_times[1] <= bound_times[0]:
        bound_times = bound_times[1:]
        labels = labels[1:]
    if bound_times[-1] <= bound_times[-2]:
        bound_times = bound_times[:-1]
        labels = labels[:-1]
    bound_times = times_to_intervals(bound_times)
    return bound_times, list(labels)


def evaluate_results(labels_est, labels_gt, bounds_est, bounds_gt, name,
                     beats, output_file="results.txt"):
    """Evaluates the results and saves them into output_file."""
    a_inters, a_labels = adjust_bounds_labels(bounds_gt, labels_gt)
    e_inters, e_labels = adjust_bounds_labels(bounds_est, labels_est)
    a_inters, a_labels = mir_eval.util.adjust_intervals(
        a_inters, a_labels, t_min=0)
    e_inters, e_labels = mir_eval.util.adjust_intervals(
        e_inters, e_labels, t_min=0, t_max=a_inters.max())

    #e_sync_labels = evaluation.beat_sync_labels(labels_est, bounds_est, beats)
    #a_sync_labels = evaluation.beat_sync_labels(labels_gt, bounds_est, beats)
    #So, Su = evaluation.eval_segmentation_entropy(e_sync_labels, a_sync_labels)
    #Pf, Pp, Pr = evaluation.eval_similarity(labels_est, labels_gt, bounds_est,
                                         #beats, ver=True)

    try:
        Hp, Hr, Hf = mir_eval.boundary.detection(a_inters, e_inters, window=3)
        So, Su, Sf = mir_eval.structure.nce(a_inters, a_labels,
                                            e_inters, e_labels)
        Pp, Pr, Pf = mir_eval.structure.pairwise(a_inters, a_labels,
                                              e_inters, e_labels)

    except:
        logging.error("Error: Evaluation failed for file %s" % name)
        return

    with open(output_file, "a") as f:
        f.write("%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%s\n" %
                (Pf*100, Pp*100, Pr*100, So*100, Su*100,
                 Hf*100, Hp*100, Hr*100, name))


def enhance_segments(pcp_segments):
    """Enhances the segments with a power law expansion."""
    enhanced = []
    for seg in pcp_segments:
        enhanced.append(seg ** float(transform["params"]["power_exp"]))
        #seg = seg / seg.max()
        #enhanced.append(seg ** 1.96)
    return enhanced


def get_real_k(file_struct):
    """Returns the real K of a given file."""
    labels_gt = utils.json_to_labels(file_struct.segments_gt)
    return len(np.unique(labels_gt))


def compute_similarity(PCP, bound_idxs, xmeans=False, k=5):
    """Main function to compute the segment similarity of file file_struct."""

    # Get PCP segments
    pcp_segments = get_pcp_segments(PCP, bound_idxs)

    # Enchance segments
    pcp_segments = enhance_segments(pcp_segments)

    # Get the 2d-FMCs segments
    fmcs = pcp_segments_to_2dfmc_max(pcp_segments)

    # Compute the labels using kmeans
    if xmeans:
        xm = XMeans(fmcs, plot=False)
        k = xm.estimate_K_knee(th=0.01, maxK=8)
    labels_est = compute_labels_kmeans(fmcs, k=k)

    # Plot results
    #plot_pcp_wgt(PCP, bound_idxs)

    return labels_est


def process(in_path, annot_beats=False):
    """Main process.

    Parameters
    ----------
    in_path : str
        Path to audio file
    annot_beats : boolean
        Whether to use annotated beats or not
    feature : str
        Identifier of the features to use
    """
    # Read features
    chroma, mfcc, beats, dur = MSAF.get_features(in_path,
                                                 annot_beats=annot_beats)

    # Read annotated bounds
    try:
        bound_idxs = MSAF.read_annot_bound_frames(in_path, beats)[1:-1]
    except:
        logging.warning("No annotated boundaries in file %s" % in_path)

    # Use specific feature
    F = U.lognormalize_chroma(chroma)  # Normalize chromas

    # Find the labels using 2D-FMCs
    est_labels = compute_similarity(F, bound_idxs)

    # Add first and last boundary
    bound_idxs = np.concatenate(([0], bound_idxs)).astype(int)
    est_times = beats[bound_idxs]  # Index to times
    est_times = np.concatenate((est_times, [dur]))  # Last bound

    logging.info("Estimated times: %s" % est_times)
    logging.info("Estimated labels: %s" % est_labels)

    assert len(est_times) - 1 == len(est_labels)

    return est_times, est_labels


def main():
    """Main function to parse the arguments and call the main process."""
    parser = argparse.ArgumentParser(description=
        "Segments the given audio file using the new version of the C-NMF "
                                     "method.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_path",
                        action="store",
                        help="Input path to the audio file")
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
    process(args.in_path, annot_beats=args.annot_beats)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))

if __name__ == '__main__':
    main()

