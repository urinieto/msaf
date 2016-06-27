Tutorial
========

This section covers the fundamentals of _msaf_, including a package overview, basic and advanced usage, and dataset exploration. We will assume basic familiarity with Python.

Overview
--------

MSAF is divided into the four different moving blocks that compose the music structural segmentation ecosystem:

	- :ref:`Features <features>`:
		Set of audio feature extraction utilities that serve
		as input to the algorithms.
	- :ref:`Algorithms <algorithms>`:
		Implementations of multiple boundary and label algorithms.
	- :ref:`Evaluations <eval>`:
		Common evaluation metrics available from _mir\_eval_ and gathered
		in this module of MSAF.
	- :ref:`Datasets <datasets>`:
		A series of human-annotated dataset to use to benchmark algorithms.
		Note: these data must be downloaded separately from here: 
		`<https://github.com/urinieto/msaf-data>`_

Quickstart
----------

.. code-block:: python
    :linenos:

    # Beat tracking example
    from __future__ import print_function
    import msaf

    # 1. Get the file path to the included audio example
    filename = librosa.util.example_audio_file()

    # 2. Load the audio as a waveform `y`
    #    Store the sampling rate as `sr`
    y, sr = librosa.load(filename)

    # 3. Run the default beat tracker
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

    # 4. Convert the frame indices of beat events into timestamps
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    print('Saving output to beat_times.csv')
    librosa.output.times_csv('beat_times.csv', beat_times)

First steps
