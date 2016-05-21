#!/usr/bin/env python
"""
Saves all the TUT Beatles dataset into a jams. The structure of the TUT Beatles
dataset is:

/BeatlesTUT
    /Album
        /lab (or text) files

Example:

/TUT
    /01_-_Please_Please_Me
        /01_-_I_Saw_Her_Standing_There.lab

To parse the entire dataset, you simply need the path to the TUT Beatles
dataset and an output folder.

Example:
./tut_parser.py ~/datasets/BeatlesTUT -o ~/datasets/BeatlesTUT/outJAMS

"""

import argparse
import logging
import os
import time

import jams


__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2015, Music and Audio Research Lab (MARL)"
__license__ = "MIT"
__version__ = "1.1"
__email__ = "oriol@nyu.edu"

# Map of JAMS attributes to Isophonics directories.
ISO_ATTRS = {'beat': 'beat',
             'chord': 'chordlab',
             'key': 'keylab',
             'segment': 'seglab'}

# Namespace dictionary
NS_DICT = {'beat': 'beat',
           'chord': 'chord',
           'key': 'key_mode',
           'segment': 'segment_open'}

# Map chords that don't make much sense
CHORDS_DICT = {
    "E:4": "E:sus4",
    "Db:6": "Db:maj6",
    "F#min7": "F#:min7",
    "B:7sus": "B:maj7",
    "Db:6/2": "Db:maj6/2",
    "Ab:6": "Ab:maj6",
    "F:6": "F:maj6",
    "D:6": "D:maj6",
    "G:6": "G:maj6",
    "A:6": "A:maj6",
    "E:sus": "E",
    "E:7sus": "E:maj7"
}

# Map keys that don't make much sense
KEYS_DICT = {
    "C#:modal": "C#"
}


def fill_file_metadata(jam, artist, title):
    """Fills the global metada into the JAMS jam."""
    jam.file_metadata.artist = artist
    jam.file_metadata.duration = None
    jam.file_metadata.title = title


def get_duration_from_annot(annot):
    """Obtains the actual duration from a given annotation."""
    dur = annot.data.iloc[-1].time + annot.data.iloc[-1].duration
    return dur.total_seconds()


def fix_chord_labels(annot):
    """Fixes the name of the chords."""
    for i, label in enumerate(annot.data["value"]):
        annot.data.loc[i, "value"] = CHORDS_DICT.get(label, label)


def fix_key_labels(annot):
    """Fixes the name of the keys."""
    for i, label in enumerate(annot.data["value"]):
        annot.data.loc[i, "value"] = KEYS_DICT.get(label, label)


def fix_beats_values(annot):
    """Fixes the beat labels."""
    for i, value in enumerate(annot.data["value"]):
        try:
            annot.data.loc[i, "value"] = float(value)
        except ValueError:
            annot.data.loc[i, "value"] = None
    # Convert to float
    annot.data["value"] = annot.data["value"].astype("float")


def fix_ranges(annot):
    """Remove the empty ranges from the annotation."""
    idxs = []
    for i, dur in enumerate(annot.data["duration"]):
        if dur.total_seconds() <= 0:
            idxs.append(i)
    annot.data.drop(idxs, inplace=True)


def fix_silence(annot):
    """Removes the silences for the keys."""
    idxs = []
    for i, label in enumerate(annot.data["value"]):
        if label.lower() == "silence":
            idxs.append(i)
    annot.data.drop(idxs, inplace=True)


def process(in_dir, out_dir):
    """Converts the original Isophonic files into the JAMS format, and saves
    them in the out_dir folder."""
    all_jams = dict()
    output_paths = dict()
    all_labs = jams.util.find_with_extension(in_dir, 'lab', 5)
    all_labs += jams.util.find_with_extension(in_dir, 'txt', 4)

    for lab_file in all_labs:
        title = jams.util.filebase(lab_file)
        if title not in all_jams:
            all_jams[title] = jams.JAMS()
            parts = lab_file.replace(in_dir, '').strip('/').split('/')
            fill_file_metadata(all_jams[title], artist=parts[1], title=title)
            output_paths[title] = os.path.join(
                out_dir, *parts[1:]).replace(".lab", ".jams")
            logging.info("%s -> %s" % (title, output_paths[title]))

        jam = all_jams[title]
        tmp_jam, annot = jams.util.import_lab(NS_DICT['segment'], lab_file,
                                              jam=jam)
        fix_ranges(jam.annotations[-1])
        jam.file_metadata.duration = get_duration_from_annot(annot)

        # Add Metadata
        curator = jams.Curator(name="Matthias Mauch",
                               email="m.mauch@qmul.ac.uk")
        ann_meta = jams.AnnotationMetadata(curator=curator,
                                           version=1.0,
                                           corpus="Beatles TUT",
                                           annotator=None)
        jam.annotations[-1].annotation_metadata = ann_meta

    logging.info("Saving and validating JAMS...")
    for title in all_jams:
        out_file = output_paths[title]
        jams.util.smkdirs(os.path.split(out_file)[0])
        all_jams[title].save(out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Converts the Beatles TUT dataset to the JAMS format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_dir",
                        action="store",
                        help="Beatles TUT main folder")
    parser.add_argument("out_dir",
                        action="store",
                        help="Output JAMS folder")
    args = parser.parse_args()
    start_time = time.time()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)

    # Run the parser
    process(args.in_dir, args.out_dir)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))
