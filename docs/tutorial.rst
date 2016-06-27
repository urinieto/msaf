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

    # 1. Select audio file
    audio_file = "../datasets/Sargon/audio/01-Sargon-Mindless.mp3"

    # 2. Segment the file using the default MSAF parameters
    boundaries, labels = msaf.process(audio_file)

    print('Estimated boundaries:', boundaries)

    print('Saving output to beat_times.csv')
    librosa.output.times_csv('beat_times.csv', beat_times)

First steps
