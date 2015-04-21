"""
Useful functions that are common in MSAF
"""

import cPickle as pickle
import copy
import mir_eval
import numba
import numpy as np
import os
import scipy.io.wavfile

import msaf
import librosa

def lognormalize_chroma(C):
    """Log-normalizes chroma such that each vector is between -80 to 0."""
    C += np.abs(C.min()) + 0.1
    C = C / C.max(axis=0)
    C = 80 * np.log10(C)  # Normalize from -80 to 0
    return C


def normalize_chroma(C):
    """Normalizes chroma such that each vector is between 0 to 1."""
    C += np.abs(C.min())
    C = C/C.max(axis=0)
    return C


def normalize_matrix(X):
    """Nomalizes a matrix such that it's maximum value is 1 and minimum is 0."""
    X += np.abs(X.min())
    X /= X.max()
    return X


def ensure_dir(directory):
    """Makes sure that the given directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def chroma_to_tonnetz(C):
    """Transforms chromagram to Tonnetz (Harte, Sandler, 2006)."""
    N = C.shape[0]
    T = np.zeros((N, 6))

    r1 = 1      # Fifths
    r2 = 1      # Minor
    r3 = 0.5    # Major

    # Generate Transformation matrix
    phi = np.zeros((6, 12))
    for i in range(6):
        for j in range(12):
            if i % 2 == 0:
                fun = np.sin
            else:
                fun = np.cos

            if i < 2:
                phi[i, j] = r1 * fun(j * 7 * np.pi / 6.)
            elif i >= 2 and i < 4:
                phi[i, j] = r2 * fun(j * 3 * np.pi / 2.)
            else:
                phi[i, j] = r3 * fun(j * 2 * np.pi / 3.)

    # Do the transform to tonnetz
    for i in range(N):
        for d in range(6):
            denom = float(C[i, :].sum())
            if denom == 0:
                T[i, d] = 0
            else:
                T[i, d] = 1 / denom * (phi[d, :] * C[i, :]).sum()

    return T


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
    return np.asarray(zip(times[:-1], times[1:]))


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
    # Generate clicks
    audio_clicks = mir_eval.sonify.clicks(clicks + offset, fs)

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
    for i, (bound_idx, label) in \
            enumerate(zip(old_bound_idxs[:-1], old_labels)):
        unfold_labels[bound_idx:old_bound_idxs[i+1]] = label

    # Constuct new labels
    new_labels = np.zeros(len(new_bound_idxs) - 1)
    for i, bound_idx in enumerate(new_bound_idxs[:-1]):
        new_labels[i] = np.median(unfold_labels[bound_idx:new_bound_idxs[i+1]])

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

    # Add silences, if needed
    est_idxs[-1] = len(frame_times) - 1  # For DTW!
    est_times = np.concatenate(([0], frame_times[est_idxs], [dur]))
    silence_label = np.max(est_labels) + 1
    est_labels = np.concatenate(([silence_label], est_labels, [silence_label]))

    # Remove empty segments if needed
    est_times, est_labels = remove_empty_segments(est_times, est_labels)

    # Make sure that the first and last times are 0 and duration, respectively
    assert np.allclose([est_times[0]], [0]) and \
        np.allclose([est_times[-1]], [dur])

    return est_times, est_labels


def align_end_hierarchies(hier1, hier2, thres=0.05):
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


def segment_labels_to_floats(segments):
    """Converts the string labels to floats.

    Parameters
    ----------
    segments: list
        List of mir_eval.segment.tree.Segment
    """
    labels = []
    for segment in segments:
        labels.append(segment.label)

    unique_labels = set(labels)
    unique_labels = list(unique_labels)

    return [unique_labels.index(label) / float(len(unique_labels))
            for label in labels]


def seconds_to_frames(seconds):
    """Converts seconds to frames based on MSAF parameters.

    Parameters
    ----------
    seconds: float
        Seconds to be converted to frames.

    Returns
    -------
    frames: int
        Seconds converted to frames
    """
    return int(seconds * msaf.Anal.sample_rate / msaf.Anal.hop_size)


def read_cqt_features(audio_file, features_dir):
    """Reads the pre-computed cqt (from the dtw project).

    Paramters
    ---------
    audio_file : str
        Path to the original audio file.
    features_dir : str
        Path to the precomputed features folder.

    Returns
    -------
    F : np.array
        Pre-computed CQT log spectrogram.
    indeces : np.array
        Indeces for the frames for each F position (sub-beats).
    """
    features_file = os.path.join(features_dir, os.path.basename(audio_file) +
                                 ".pk")
    with open(features_file, "r") as f:
        file_data = pickle.load(f)

    F = file_data["cqgram"].T
    indeces = file_data["subseg"]
    return F, indeces


def map_indeces(in_idxs, subbeats_idxs, frame_times):
    """Maps the in_index that index the subbeats_idxs into frame_times.

    Paramters
    ---------
    in_idxs: np.array
        The indeces to be mapped.
    subbeats_idxs: np.array
        The subbeats indeces to where in_idxs map to.
    frame_times: np.array
        Times for each of the final frame times that in_idxs will be mapped to.

    Returns
    -------
    out_idxs: np.array
        The new output indeces in the frame_times space.
    """
    in_frames_idxs = subbeats_idxs[in_idxs]
    in_times = np.array([idx * msaf.Anal.hop_size / float(msaf.Anal.sample_rate)
                 for idx in in_frames_idxs])
    out_idxs = np.abs(np.subtract.outer(in_times, frame_times)).argmin(axis=1)
    return np.unique(out_idxs)


def symstack(X, n_steps=5, delay=1, **kwargs):
    '''Symmetric history stacking.

    like librosa.feature.stack_memory, but IN THE FUTURE!!!
    '''
    rpad = n_steps * delay
    Xpad = np.pad(X,
                  [(0, 0), (0, rpad)],
                  **kwargs)

    Xstack = librosa.feature.stack_memory(Xpad,
                                          n_steps=2 * n_steps + 1,
                                          delay=delay,
                                          **kwargs)

    return Xstack[:, rpad:]


@numba.jit
def compute_recurrence_plot(F, model, w=5):
    """Computes the recurrence plot for the given features using the
    similarity model previously trained.

    Parameters
    ----------
    F : np.array
        Set of features: must be CQT features.
    model : object
        Scikits classifier.
    w : int
        Number of frames in front / back.

    Returns
    -------
    R : list
        R_predict : np.array
            The recurrence plot using binary prediction.
        R_proba : np.array
            The recurrence plot using predicted probabilities.
        R_mask : np.array
            The recurrence plot using the binary predictions as a mask on the
            predicted probabilities.
    """
    C = F.T
    X = symstack(C, n_steps=w, mode='edge')
    N = X.shape[1]
    R_predict = np.eye(N)
    R_proba = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            Xt = np.abs(X[:, i] - X[:, j])[np.newaxis, :]
            R_predict[i, j] = model.predict(Xt)
            R_predict[j, i] = R_predict[i, j]
            R_proba[i, j] = model.predict_proba(Xt)[0][1]
            R_proba[j, i] = R_proba[i, j]

    # Recurrence plot, mask type
    R_mask = R_proba * R_predict

    return [R_predict, R_proba, R_mask]


@numba.jit
def compute_local_plot(F, model, w=5):
    """Computes the recurrence plot for the given features using the
    similarity model previously trained.

    Parameters
    ----------
    F : np.array
        Set of features: must be CQT features.
    model : object
        Scikits classifier.
    w : int
        Number of frames in front / back.

    Returns
    -------
    R : list
        R_predict : np.array
            The recurrence plot using binary prediction.
        R_proba : np.array
            The recurrence plot using predicted probabilities.
        R_mask : np.array
            The recurrence plot using the binary predictions as a mask on the
            predicted probabilities.
    """
    C = F.T
    X = symstack(C, n_steps=w, mode='edge')
    N = X.shape[1]
    R_predict = np.eye(N)
    R_proba = np.eye(N)
    for i in range(N):
        jj = [max(i - 1, 0), min(i + 1, N - 1)]
        for j in jj:
            Xt = np.abs(X[:, i] - X[:, j])[np.newaxis, :]
            R_predict[i, j] = model.predict(Xt)
            R_predict[j, i] = R_predict[i, j]
            R_proba[i, j] = model.predict_proba(Xt)[0][1]
            R_proba[j, i] = R_proba[i, j]

    # Recurrence plot, mask type
    R_mask = R_proba * R_predict

    return [R_predict, R_proba, R_mask]


def get_recplot_file(recplots_dir, audio_file, local="", bias=""):
    """Gets the recurrence plot file.

    Parameters
    ----------
    recplots_dir : str
        Directory where to store the recurrence plots.
    audio_file : str
        Path to the audio file.
    local : str
        Whether to have local plots.

    Returns
    -------
    recplot_path : str
        Path to the recplot pk file.
    """
    return os.path.join(recplots_dir, os.path.basename(audio_file) + bias + ".pk")


def times_to_bounds(in_times, subbeats_idxs):
    """Converts the times to bounds of the given subbeats.

    Paramters
    ---------
    in_times: np.array
        The times to be converted.
    subbeats_idxs: np.array
        The subbeats indeces to where in_times map to.

    Returns
    -------
    out_idxs: np.array
        The new output indeces in the subbeats_idxs space.
    """
    frame_times = np.array(
        [idx * msaf.Anal.hop_size / float(msaf.Anal.sample_rate)
         for idx in subbeats_idxs])
    out_idxs = np.abs(np.subtract.outer(in_times, frame_times)).argmin(axis=1)
    return np.unique(out_idxs)


def get_recurrence_plot(F_dtw, recplot_file, config):
    """Either read or compute the recurrence plot given some model params.

    Returns
    -------
    R : np.array
        The recurrence plot.
    """
    if os.path.isfile(recplot_file):
        with open(recplot_file) as f:
            R = pickle.load(f)[config["recplot_type"]]
    else:
        with open(config["model"]) as f:
            model = pickle.load(f)["model"]

        R = compute_recurrence_plot(F_dtw, model, config["w"])

        R_dict = {}
        R_dict["predict"] = R[0]
        R_dict["proba"] = R[1]
        R_dict["mask"] = R[2]
        with open(recplot_file, "w") as f:
            pickle.dump(R_dict, f)
        R = R_dict[config["recplot_type"]]

    return R


def get_local_plot(F_dtw, recplot_file, config):
    """Either read or compute the local plot given some model params.

    Returns
    -------
    R : np.array
        The recurrence plot.
    """
    if os.path.isfile(recplot_file):
        with open(recplot_file) as f:
            R = pickle.load(f)[config["recplot_type"]]
    else:
        with open(config["model_local"]) as f:
            model = pickle.load(f)["model"]

        R = compute_local_plot(F_dtw, model, config["w"])

        R_dict = {}
        R_dict["predict"] = R[0]
        R_dict["proba"] = R[1]
        R_dict["mask"] = R[2]
        with open(recplot_file, "w") as f:
            pickle.dump(R_dict, f)
        R = R_dict[config["recplot_type"]]

    return R
