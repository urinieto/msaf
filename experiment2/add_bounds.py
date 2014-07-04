"""
Adds boundaries to a specific audio file.
"""

import numpy as np
import os
from scikits import audiolab
import subprocess


def ensure_dir(directory):
    """Checks if directory exists, and creates it otherwise."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def mp32wav(mp3file, wavfile, sr=44100):
    """Converts an mp3 to a wav file."""
    cmd = "sox -S %s -r %d -b 16 %s" % (mp3file, sr, wavfile)
    subprocess.call(cmd.split(" "))


def wav2mp3(wavfile, mp3file, bitrate=128):
    """Converts a wav file to an mp3 file."""
    cmd = "sox -c 1 %s -C %d %s" % (wavfile, bitrate, mp3file)
    subprocess.call(cmd.split(" "))


def read_wav(wavfile):
    """Reads the wav file and downsamples to 11025 Hz if needed.

        @param wavfile string: Path to the wave file.
        @return x np.array: Array of samples of the audio file.
        @return fs int: Sampling frequency.
    """
    assert os.path.isfile(wavfile), \
        'ERROR: wivefile file %s does not exist' % wavfile

    x, fs, enc = audiolab.wavread(wavfile)
    if len(x.shape) >= 2:
        x = x[:, 0]  # Make mono

    assert fs == 44100, \
        "ERROR: File %s is not sampled at 44100 Hz" % wavfile

    return x, fs


def read_gt_boundaries(labfile):

    """Reads the boundaries from a lab file and puts them in a numpy array."""
    f = open(labfile, "r")
    lines = f.readlines()
    f.close()

    bounds = []
    for line in lines:
        bound = line.split(" ")[0].split("\t")[0]
        bounds.append(float(bound))
    last_bound = lines[-1].split(" ")[0].split("\t")[0]
    bounds.append(float(last_bound))

    return np.asarray(bounds)


def add_boundaries(wavfile, boundaries, output='output.wav',
                   boundsound="sounds/bell.wav", start=0, end=None):
    """Adds a cowbell sound for each boundary and saves it into a new wav file.

        @param wavfile string: Input wav file (sampled at 11025Hz or 44100Hz).
        @param boundaries np.array: Set of times representing the boundaries
            (in seconds).
        @param output string: Name of the output wav file.
        @param boundsound string: Sound to add to the original file.
        @param start float: Start time (in seconds)
        @param end float: End time (in seconds)

    """

    OFFSET = 0.0  # offset time in seconds

    x, fs = read_wav(wavfile)
    xb, fsb = read_wav(boundsound)

    # Normalize
    x /= x.max()

    # Copy the input wav file to the output
    out = np.zeros(x.size + xb.size + 1000)
    out[:x.size] = x / 3.0

    # Add boundaries
    for bound in boundaries:
        start_idx = int((bound + OFFSET) * fs)
        end_idx = start_idx + xb.size
        read_frames = out[start_idx:end_idx].size
        out[start_idx:end_idx] += xb[:read_frames]

    # Cut track if needed
    start_time = start * fs
    if start_time < 0:
        start_time = 0
    if end is None:
        end_time = len(out)
    else:
        end_time = end * fs
        if end_time > len(out):
            end_time = len(out)

    out = out[int(start_time):int(end_time)]

    # Write output wav
    audiolab.wavwrite(out, output, fs=fs)

    # Convert to MP3 and delete wav
    dest_mp3 = output.replace(".wav", ".mp3")
    wav2mp3(output, dest_mp3)
    os.remove(output)

    print "Wrote %s" % dest_mp3
