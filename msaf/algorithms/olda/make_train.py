#!/usr/bin/env python

import argparse
import numpy as np
import glob
import librosa
import os
import sys
import time

from joblib import Parallel, delayed
import cPickle as pickle

import msaf
from msaf.base import Features

from segmenter import features


def align_segmentation(filename, beat_times, song):
    '''Load a ground-truth segmentation, and align times to the nearest
    detected beats.

    Arguments:
        filename -- str
        beat_times -- array
        song -- path to the audio file

    Returns:
        segment_beats -- array
            beat-aligned segment boundaries

        segment_times -- array
            true segment times

        segment_labels -- array
            list of segment labels
    '''
    try:
        segment_times, segment_labels = msaf.io.read_references(song)
    except:
        return None, None, None
    segment_times = np.asarray(segment_times)

    # Map to intervals
    segment_intervals = msaf.utils.times_to_intervals(segment_times)

    # Map beats to intervals
    beat_intervals = np.asarray(zip(beat_times[:-1], beat_times[1:]))

    # Map beats to segments
    beat_segment_ids = librosa.util.match_intervals(beat_intervals,
                                                    segment_intervals)

    segment_beats = []
    segment_times_out = []
    segment_labels_out = []

    # print segment_times, beat_segment_ids, len(beat_times),
    # len(beat_segment_ids)
    for i in range(segment_times.shape[0]):
        hits = np.argwhere(beat_segment_ids == i)
        if len(hits) > 0 and i < len(segment_intervals) and \
                i < len(segment_labels):
            segment_beats.extend(hits[0])
            segment_times_out.append(segment_intervals[i, :])
            segment_labels_out.append(segment_labels[i])

    # Pull out the segment start times
    segment_beats = list(segment_beats)
    # segment_times_out = np.asarray(
    #  segment_times_out)[:, 0].squeeze().reshape((-1, 1))

    # if segment_times_out.ndim == 0:
    #    segment_times_out = segment_times_out[np.newaxis]
    segment_times_out = segment_times

    return segment_beats, segment_times_out, segment_labels_out


def get_annotation(song, rootpath):
    return '%s/annotations/%s.jams' % (rootpath, os.path.basename(song)[:-4])


def import_data(file_struct, rootpath, output_path, annot_beats):
    msaf.utils.ensure_dir(output_path)
    msaf.utils.ensure_dir(os.path.join(output_path, "features"))
    data_file = '%s/features/%s_annotbeatsE%d.pickle' % \
        (output_path, os.path.splitext(
            os.path.basename(file_struct.audio_file))[0], annot_beats)

    if os.path.exists(data_file):
        with open(data_file, 'r') as f:
            Data = pickle.load(f)
            print file_struct.audio_file, 'cached!'
    else:
        X, dur = features(file_struct, annot_beats)
        pcp_obj = Features.select_features("pcp", file_struct, annot_beats,
                                           framesync=False)
        B = pcp_obj.frame_times[:pcp_obj.features.shape[0]]

        if X is None:
            return X

        Y, T, L = align_segmentation(
            get_annotation(file_struct.audio_file, rootpath), B,
            file_struct.audio_file)

        if Y is None:
            return Y

        Data = {'features': X,
                'beats': B,
                'filename': file_struct.audio_file,
                'segment_times': T,
                'segment_labels': L,
                'segments': Y}
        print file_struct.audio_file, 'processed!'

        with open(data_file, 'w') as f:
            pickle.dump(Data, f)

    return Data


def make_dataset(n=None, n_jobs=1, rootpath='', output_path='',
                 annot_beats=False):

    audio_files = msaf.io.get_dataset_files(rootpath)

    if n is None:
        n = len(audio_files)

    data = Parallel(n_jobs=n_jobs)(delayed(import_data)(
        file_struct, rootpath, output_path, annot_beats)
        for file_struct in audio_files[:n])

    X, Y, B, T, F, L = [], [], [], [], [], []
    for d in data:
        if d is None:
            continue
        X.append(d['features'])
        Y.append(d['segments'])
        B.append(d['beats'])
        T.append(d['segment_times'])
        F.append(d['filename'])
        L.append(d['segment_labels'])

    return X, Y, B, T, F, L


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Trains OLDA from the given dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("ds_path",
                        action="store",
                        help="Input dataset dir")
    parser.add_argument("output_path",
                        action="store",
                        help="Output dir to store the features")
    parser.add_argument("-n",
                        action="store",
                        dest="n",
                        type=int,
                        help="Number of files to process (`None` for all)",
                        default=None)
    parser.add_argument("-j",
                        action="store",
                        dest="n_jobs",
                        type=int,
                        help="Number of jobs (threads)",
                        default=1)
    parser.add_argument("-b",
                        action="store_true",
                        dest="annot_beats",
                        help="Use annotated beats",
                        default=False)
    args = parser.parse_args()
    start_time = time.time()
    X, Y, B, T, F, L = make_dataset(n=args.n,
                                    n_jobs=args.n_jobs,
                                    rootpath=args.ds_path,
                                    output_path=args.output_path,
                                    annot_beats=args.annot_beats)

    ds_path = args.ds_path
    if ds_path[-1] == "/":
        ds_path = ds_path[:-1]

    if args.annot_beats:
        out_path = '%s/AnnotBeats_%s_data.pickle' % (
            args.output_path, os.path.basename(ds_path))
    else:
        out_path = '%s/EstBeats_%s_data.pickle' % (
            args.output_path, os.path.basename(ds_path))
    with open(out_path, 'w') as f:
        pickle.dump((X, Y, B, T, F, L), f)
