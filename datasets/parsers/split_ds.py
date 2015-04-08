#!/usr/bin/env python
'''
Splits the folder with the raw mp3 into 2 subfolders with the correct 
dataset format
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
import shutil

def ensure_dir(directory):
    """Makes sure that the directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_dataset(in_path, files):
    """Creates a dataset with the correct structure and the specified audio
        files."""

    # Creates dataset structure
    logging.info("Creating new dataset in %s" % in_path)
    audio_dir = os.path.join(in_path, "audio")
    ensure_dir(in_path)
    ensure_dir(audio_dir)
    ensure_dir(os.path.join(in_path, "estimations"))
    ensure_dir(os.path.join(in_path, "features"))

    # move files
    logging.info("Moving %d audio files to %s" % (len(files), audio_dir))
    for f in files:
        shutil.move(f, os.path.join(audio_dir, os.path.basename(f)))


def process(in_path):
    """Main process."""

    # Get relevant files
    audio_files = glob.glob(os.path.join(in_path, "*.mp3"))

    # Split files
    files1 = audio_files[::2]
    files2 = audio_files[1::2]

    # Create datasets
    create_dataset(os.path.join(in_path, "youtabs1"), files1)
    create_dataset(os.path.join(in_path, "youtabs2"), files2)
    

def main():
    """Main function to parse the arguments and call the main process."""
    parser = argparse.ArgumentParser(description=
        "Splits the mp3 into two subfolders with the correct dataset format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_path",
                        action="store",
                        help="Input dataset")
    args = parser.parse_args()
    start_time = time.time()
    
    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', 
        level=logging.INFO)

    # Run the algorithm
    process(args.in_path)

    # Done!
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))


if __name__ == '__main__':
    main()
