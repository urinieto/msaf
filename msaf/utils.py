"""
Useful functions that are common in MSAF
"""
import librosa
import mir_eval
import numpy as np
import os
import scipy.io.wavfile
import six


def lognormalize(F, floor=0.1, min_db=-80):
    """Log-normalizes features such that each vector is between min_db to 0."""
    assert min_db < 0
    F = min_max_normalize(F, floor=floor)
    F = np.abs(min_db) * np.log10(F)  # Normalize from min_db to 0
    return F


def min_max_normalize(F, floor=0.001):
    """Normalizes features such that each vector is between floor to 1."""
    F += -F.min() + floor
    F = F / F.max(axis=0)
    return F


def normalize(X, norm_type, floor=0.0, min_db=-80):
    """Normalizes the given matrix of features.

    Parameters
    ----------
    X: np.array
        Each row represents a feature vector.
    norm_type: {"min_max", "log", np.inf, -np.inf, 0, float > 0, None}
        - `"min_max"`: Min/max scaling is performed
        - `"log"`: Logarithmic scaling is performed
        - `np.inf`: Maximum absolute value
        - `-np.inf`: Mininum absolute value
        - `0`: Number of non-zeros
        - float: Corresponding l_p norm.
        - None : No normalization is performed

    Returns
    -------
    norm_X: np.array
        Normalized `X` according the the input parameters.
    """
    if isinstance(norm_type, six.string_types):
        if norm_type == "min_max":
            return min_max_normalize(X, floor=floor)
        if norm_type == "log":
            return lognormalize(X, floor=floor, min_db=min_db)
    return librosa.util.normalize(X, norm=norm_type, axis=1)


def ensure_dir(directory):
    """Makes sure that the given directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def times_to_intervals(times):
    """Given a set of times, convert them into intervals.

    Parameters
    ----------
    times: np.array(N)
        A set of times.

    Returns
    -------
    inters: np.array(N-1, 2)
        A set of intervals.
    """
    return np.asarray(list(zip(times[:-1], times[1:])))


def intervals_to_times(inters):
    """Given a set of intervals, convert them into times.

    Parameters
    ----------
    inters: np.array(N-1, 2)
        A set of intervals.

    Returns
    -------
    times: np.array(N)
        A set of times.
    """
    return np.concatenate((inters.flatten()[::2], [inters[-1, -1]]), axis=0)


def get_num_frames(dur, anal):
    """Given the duration of a track and a dictionary containing analysis
    info, return the number of frames."""
    total_samples = dur * anal["sample_rate"]
    return int(total_samples / anal["hop_size"])


def get_time_frames(dur, anal):
    """Gets the time frames and puts them in a numpy array."""
    n_frames = get_num_frames(dur, anal)
    return np.linspace(0, dur, num=n_frames)


def remove_empty_segments(times, labels):
    """Removes empty segments if needed."""
    assert len(times) - 1 == len(labels)
    inters = times_to_intervals(times)
    new_inters = []
    new_labels = []
    for inter, label in zip(inters, labels):
        if inter[0] < inter[1]:
            new_inters.append(inter)
            new_labels.append(label)
    return intervals_to_times(np.asarray(new_inters)), new_labels


def sonify_clicks(audio, clicks, out_file, fs, offset=0):
    """Sonifies the estimated times into the output file.

    Parameters
    ----------
    audio: np.array
        Audio samples of the input track.
    clicks: np.array
        Click positions in seconds.
    out_file: str
        Path to the output file.
    fs: int
        Sample rate.
    offset: float
        Offset of the clicks with respect to the audio.
    """
    # Generate clicks (this should be done by mir_eval, but its
    # latest release is not compatible with latest numpy)
    times = clicks + offset
    # 1 kHz tone, 100ms
    click = np.sin(2 * np.pi * np.arange(fs * .1) * 1000 / (1. * fs))
    # Exponential decay
    click *= np.exp(-np.arange(fs * .1) / (fs * .01))
    length = int(times.max() * fs + click.shape[0] + 1)
    audio_clicks = mir_eval.sonify.clicks(times, fs, length=length)

    # Create array to store the audio plus the clicks
    out_audio = np.zeros(max(len(audio), len(audio_clicks)))

    # Assign the audio and the clicks
    out_audio[:len(audio)] = audio
    out_audio[:len(audio_clicks)] += audio_clicks

    # Write to file
    scipy.io.wavfile.write(out_file, fs, out_audio)


def synchronize_labels(new_bound_idxs, old_bound_idxs, old_labels, N):
    """Synchronizes the labels from the old_bound_idxs to the new_bound_idxs.

    Parameters
    ----------
    new_bound_idxs: np.array
        New indeces to synchronize with.
    old_bound_idxs: np.array
        Old indeces, same shape as labels + 1.
    old_labels: np.array
        Labels associated to the old_bound_idxs.
    N: int
        Total number of frames.

    Returns
    -------
    new_labels: np.array
        New labels, synchronized to the new boundary indeces.
    """
    assert len(old_bound_idxs) - 1 == len(old_labels)

    # Construct unfolded labels array
    unfold_labels = np.zeros(N)
    for i, (bound_idx, label) in enumerate(
            zip(old_bound_idxs[:-1], old_labels)):
        unfold_labels[bound_idx:old_bound_idxs[i + 1]] = label

    # Constuct new labels
    new_labels = np.zeros(len(new_bound_idxs) - 1)
    for i, bound_idx in enumerate(new_bound_idxs[:-1]):
        new_labels[i] = np.median(unfold_labels[bound_idx:new_bound_idxs[i + 1]])

    return new_labels


def process_segmentation_level(est_idxs, est_labels, N, frame_times, dur):
    """Processes a level of segmentation, and converts it into times.

    Parameters
    ----------
    est_idxs: np.array
        Estimated boundaries in frame indeces.
    est_labels: np.array
        Estimated labels.
    N: int
        Number of frames in the whole track.
    frame_times: np.array
        Time stamp for each frame.
    dur: float
        Duration of the audio track.

    Returns
    -------
    est_times: np.array
        Estimated segment boundaries in seconds.
    est_labels: np.array
        Estimated labels for each segment.
    """
    assert est_idxs[0] == 0 and est_idxs[-1] == N - 1
    assert len(est_idxs) - 1 == len(est_labels)

    # Add silences, if needed
    est_times = np.concatenate(([0], frame_times[est_idxs], [dur]))
    silence_label = np.max(est_labels) + 1
    est_labels = np.concatenate(([silence_label], est_labels, [silence_label]))

    # Remove empty segments if needed
    est_times, est_labels = remove_empty_segments(est_times, est_labels)

    # Make sure that the first and last times are 0 and duration, respectively
    assert np.allclose([est_times[0]], [0]) and \
        np.allclose([est_times[-1]], [dur])

    return est_times, est_labels


def align_end_hierarchies(hier1, hier2, thres=0.5):
    """Align the end of the hierarchies such that they end at the same exact
    second as long they have the same duration within a certain threshold.

    Parameters
    ----------
    hier1: list
        List containing hierarchical segment boundaries.
    hier2: list
        List containing hierarchical segment boundaries.
    thres: float > 0
        Threshold to decide whether two values are the same.
    """
    # Make sure we have correctly formatted hierarchies
    dur_h1 = hier1[0][-1]
    for hier in hier1:
        assert hier[-1] == dur_h1, "hier1 is not correctly formatted"
    dur_h2 = hier2[0][-1]
    for hier in hier2:
        assert hier[-1] == dur_h2, "hier2 is not correctly formatted"

    # If durations are different, do nothing
    if abs(dur_h1 - dur_h2) > thres:
        return

    # Align h1 with h2
    for hier in hier1:
        hier[-1] = dur_h2
