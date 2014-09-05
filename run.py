#!/usr/bin/env python
"""
Runs one boundary algorithm and a label algorithm on a specified audio file or
dataset.
"""

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2014, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import argparse
import time
import logging
import os
import numpy as np

from joblib import Parallel, delayed

import msaf
from msaf import jams2
from msaf import input_output as io
from msaf import utils
from msaf import featextract
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
    if not module.is_label_type:
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
        S = bounds_module.Segmenter(audio_file, **config)
        est_times, est_labels = S.process()
    else:
        # Identify segment boundaries
        if bounds_module is not None:
            S = bounds_module.Segmenter(audio_file, in_labels=[], **config)
            est_times, est_labels = S.process()
        else:
            est_times, est_labels = io.read_references(audio_file)

        # Label segments
        if labels_module is not None:
            S = labels_module.Segmenter(audio_file, in_bound_times=est_times,
                                        **config)
            est_times, est_labels = S.process()

    return est_times, est_labels


def process_track(file_struct, boundaries_id, labels_id, config):
    # Only analize files with annotated beats
    if config["annot_beats"]:
        jam = jams2.load(file_struct.ref_file)
        if len(jam.beats) > 0 and len(jam.beats[0].data) > 0:
            pass
        else:
            logging.warning("No beat information in file %s" %
                            file_struct.ref_file)
            return

    logging.info("Segmenting %s" % file_struct.audio_file)

    # Compute features if needed
    if not os.path.isfile(file_struct.features_file):
        featextract.compute_all_features(file_struct)

    # Get estimations
    est_times, est_labels = run_algorithms(file_struct.audio_file,
                                           boundaries_id, labels_id, config)

    # Save
    logging.info("Writing results in: %s" % file_struct.est_file)
    est_inters = utils.times_to_intervals(est_times)
    io.save_estimations(file_struct.est_file, est_inters, est_labels,
                        boundaries_id, labels_id, **config)

    return est_times, est_labels


def process(in_path, annot_beats=False, feature="mfcc", ds_name="*",
            framesync=False, boundaries_id="gt", labels_id=None,
            out_audio=False, n_jobs=4, config=None):
    """Main process to segment a file or a collection of files.

    Parameters
    ----------
    in_path: str
        Input path. If a directory, MSAF will function in collection mode.
        If audio file, MSAF will be in single file mode.
    annot_beats: bool
        Whether to use annotated beats or not. Only available in collection
        mode.
    feature: str
        String representing the feature to be used (e.g. hpcp, mfcc, tonnetz)
    ds_name: str
        Prefix of the dataset to be used (e.g. SALAMI, Isophonics)
    framesync: str
        Whether to use framesync features or not (default: False -> beatsync)
    boundaries_id: str
        Identifier of the boundaries algorithm (use "gt" for groundtruth)
    labels_id: str
        Identifier of the labels algorithm (use None to not compute labels)
    out_audio: bool
        Whether to write an output audio file with the annotated boundaries
        or not (only available in Single File Mode).
    n_jobs: int
        Number of processes to run in parallel. Only available in collection
        mode.
    config: dict
        Dictionary containing custom configuration parameters for the
        algorithms.  If None, the default parameters are used.

    Returns
    -------
    results : list
        List containing tuples of (est_times, est_labels) of estimated
        boundary times and estimated labels.
        If labels_id is None, est_labels will be a list of -1.
    """

    # Seed random to reproduce results
    np.random.seed(123)

    # Set up configuration based on algorithms parameters
    if config is None:
        config = io.get_configuration(feature, annot_beats, framesync,
                                      boundaries_id, labels_id, algorithms)

    if os.path.isfile(in_path):
        # Single file mode
        audio, features = featextract.compute_features_for_audio_file(in_path)
        config["features"] = features

        # And run the algorithms
        est_times, est_labels = run_algorithms(in_path, boundaries_id,
                                               labels_id, config)

        if out_audio:
            # TODO: Set a nicer output file name?
            #out_file = in_path[:-4] + msaf.out_boundaries_ext
            out_file = "out_boundaries.wav"
            logging.info("Sonifying boundaries in %s" % out_file)
            fs = 44100
            audio_hq = featextract.read_audio(in_path, fs)
            utils.write_audio_boundaries(audio_hq, np.delete(est_times, 1),
                                         out_file, fs)

        return est_times
    else:
        # Collection mode
        file_structs = io.get_dataset_files(in_path, ds_name)

        # Call in parallel
        return Parallel(n_jobs=n_jobs)(delayed(process_track)(
            file_struct, boundaries_id, labels_id, config)
            for file_struct in file_structs)


def main():
    """Main function to parse the arguments and call the main process."""
    parser = argparse.ArgumentParser(description=
        "Runs the speficied algorithm(s) on the MSAF formatted dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_path",
                        action="store",
                        help="Input audio file or dataset")
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
                        "(e.g. Isophonics, SALAMI)")
    parser.add_argument("-j",
                        action="store",
                        dest="n_jobs",
                        default=4,
                        type=int,
                        help="The number of threads to use")
    parser.add_argument("-a",
                        action="store_true",
                        dest="out_audio",
                        help="Output estimated boundaries with audio",
                        default=False)
    args = parser.parse_args()
    start_time = time.time()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
        level=logging.INFO)

    # Run the algorithm(s)
    process(args.in_path, annot_beats=args.annot_beats, feature=args.feature,
            ds_name=args.ds_name, framesync=args.framesync,
            boundaries_id=args.boundaries_id, labels_id=args.labels_id,
            n_jobs=args.n_jobs, out_audio=args.out_audio)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))


if __name__ == '__main__':
    main()
