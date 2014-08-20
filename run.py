#!/usr/bin/env python
"""
Runs one boundary algorithm and a label algorithm on a specified dataset.
"""

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import argparse
import glob
import os
import time
import logging

from joblib import Parallel, delayed

import msaf
from msaf import jams2
from msaf import input_output as io
from msaf import utils
import msaf.algorithms as algorithms


def get_boundaries_module(boundaries_id):
    if boundaries_id == "gt":
        return None
    module = eval(algorithms.__name__ + "." + boundaries_id)
    if not module.is_boundary_type:
        raise RuntimeError("Algorithm %s can not identify boundaries!" %
                           boundaries_id)
    return module


def get_labels_module(labels_id):
    if labels_id is None:
        return None
    module = eval(algorithms.__name__ + "." + labels_id)
    if not module.is_boundary_type:
        raise RuntimeError("Algorithm %s can not label segments!" %
                           labels_id)
    return module


def run_algorithms(audio_file, boundaries_id, labels_id, config):
    """Runs the algorithms with the specified identifiers on the audio_file."""
    # Get the corresponding modules
    bounds_module = get_boundaries_module(boundaries_id)
    labels_module = get_labels_module(labels_id)

    # Segment using the specified boundaries and labels
    if bounds_module is not None and labels_module is not None and \
            bounds_module.__name__ == labels_module.__name__:
        est_times, est_labels = bounds_module.Segmenter().process(audio_file,
                                                                  **config)
    else:
        # Identify segment boundaries
        if bounds_module is not None:
            est_times, est_labels = bounds_module.process(
                audio_file, in_labels=[], **config)
        else:
            est_times, est_labels = io.read_references(audio_file)

        # Label segments
        if labels_module is not None:
            est_times, est_labels = labels_module.process(
                audio_file, in_bound_times=est_times, **config)
        else:
            est_labels = [-1] * (len(est_times) - 1)

    return est_times, est_labels


def process_track(in_path, audio_file, jam_file, ds_name, boundaries_id,
                  labels_id, config):

    # Only analize files with annotated beats
    if config["annot_beats"]:
        jam = jams2.load(jam_file)
        if jam.beats == []:
            return
        if jam.beats[0].data == []:
            return

    logging.info("Segmenting %s" % audio_file)

    # Get estimations
    est_times, est_labels = run_algorithms(audio_file, boundaries_id, labels_id,
                                           config)

    # Save
    out_file = os.path.join(in_path, msaf.Dataset.estimations_dir,
                            os.path.basename(audio_file)[:-4] +
                            msaf.Dataset.estimations_ext)
    logging.info("Writing results in: %s" % out_file)
    est_inters = utils.times_to_intervals(est_times)
    io.save_estimations(out_file, est_inters, est_labels, boundaries_id,
                        labels_id, **config)


def process(in_path, annot_beats=False, feature="mfcc", ds_name="*",
            framesync=False, boundaries_id="gt", labels_id=None, n_jobs=4):
    """Main process."""

    # Set up configuration based on algorithms parameters
    config = io.get_configuration(feature, annot_beats, framesync,
                                  boundaries_id, labels_id, algorithms)
    # Get relevant files
    jam_files = glob.glob(os.path.join(in_path, msaf.Dataset.references_dir,
                                       ("%s_*" + msaf.Dataset.references_ext)
                                       % ds_name))
    audio_files = glob.glob(os.path.join(in_path, "audio",
                                         "%s_*.[wm][ap][v3]" % ds_name))

    # Call in parallel
    Parallel(n_jobs=n_jobs)(delayed(process_track)(
        in_path, audio_file, jam_file, ds_name, boundaries_id, labels_id,
        config) for audio_file, jam_file in zip(audio_files, jam_files))


def main():
    """Main function to parse the arguments and call the main process."""
    parser = argparse.ArgumentParser(description=
        "Runs the speficied algorithm(s) on the MSAF formatted dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_path",
                        action="store",
                        help="Input dataset")
    parser.add_argument("-f",
                        action="store",
                        dest="feature",
                        default="hpcp",
                        type=str,
                        help="Type of features",
                        choices=["hpcp", "tonnetz", "mfcc"])
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
                        help="Boundary algorithm identifier",
                        dest="boundaries_id",
                        default="gt",
                        choices=["gt"] +
                        io.get_all_boundary_algorithms(algorithms))
    parser.add_argument("-lid",
                        action="store",
                        help="Label algorithm identifier",
                        dest="labels_id",
                        default=None,
                        choices= io.get_all_label_algorithms(algorithms))
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

    # Run the algorithm(s)
    process(args.in_path, annot_beats=args.annot_beats, feature=args.feature,
            ds_name=args.ds_name, framesync=args.framesync,
            boundaries_id=args.boundaries_id, labels_id=args.labels_id,
            n_jobs=args.n_jobs)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))


if __name__ == '__main__':
    main()
