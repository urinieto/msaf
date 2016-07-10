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
		A series of human-annotated datasets to benchmark algorithms.
		Note: these data must be downloaded separately from here: 
		`<https://github.com/urinieto/msaf-data>`_

.. _quickstart_example:

Quickstart
----------

Let's begin with a simple example program:

.. code-block:: python
    :linenos:

    # Simple MSAF example
    from __future__ import print_function
    import msaf

    # 1. Select audio file
    audio_file = "../datasets/Sargon/audio/01-Sargon-Mindless.mp3"

    # 2. Segment the file using the default MSAF parameters (this might take a few seconds)
    boundaries, labels = msaf.process(audio_file)
    print('Estimated boundaries:', boundaries)

    # 3. Save segments using the MIREX format
    out_file = 'segments.txt'
    print('Saving output to %s' % out_file)
    msaf.io.write_mirex(boundaries, labels, out_file)

    # 4. Evaluate the results
    evals = msaf.eval.process(audio_file)
    print(evals)

In the first step we select the appropriate audio file. 
The MSAF `datasets <https://github.com/urinieto/msaf-data>`_ contain the Sargon set, which has multiple audio files to play around. ::

    audio_file = "../datasets/Sargon/audio/01-Sargon-Mindless.mp3"

We then use the main MSAF function ``process`` to segment the given ``audio_file``.
This function will first compute the default features for the song, and after that use the default boundary and label algorithms
to segment the audio (the defaults can be easily changed, see :doc:`config`).
It returns the boundary times (in seconds), and a set of labels::

    boundaries, labels = msaf.process(audio_file)

This previous step will store the features in a temporary JSON file in the current folder, such that the next time the function ``process``
is called with the same audio file, the features will be read from the file, thuis speeding up the process.
This is especially useful when experimenting with multiple algorithms, since the features will only be computed once.

In step 3, we store the results in a file using the standard `MIREX format <http://www.music-ir.org/mirex/wiki/2016:Structural_Segmentation#Output_File_Format_.28Structural_Segmentation.29>`_
by calling the following function with a specific output file path::

    msaf.io.write_mirex(boundaries, labels, out_file)

Finally, we can evaluate the results when human references are available.
The references must be placed in ``../references/<audio_file>.jams``, following the default MSAF Dataset configuration (this path can be changed in the default configuration).
The standard structural segmentation metrics will be used when calling the following function, which returns a ``pandas`` data frame::

    results = msaf.eval.process(audio_file)

The ``eval.process`` function follows the same principle as the main ``process`` function.
If different features or algorithms want to be used instead of the defaults (see below), the same parameters can be passed in the ``eval.process`` function to evaluate the results accordingly.

Experimenting With Features
---------------------------

There are multiple features to experiment with in MSAF.
To quickly check the available features, we can access the ``features_registry`` dictionary::

    print(msaf.features_registry)

After that, we can select the desired features in the ``process`` function, using the ``feature`` parameter.
For example, if we want to use the MFCC coefficients, we can call the function as follows::

    boundaries, labels = msaf.process(audio_file, feature='mfcc')

These feature identifiers can only be the keys in the ``features_registry``.

For more information about the available features, please refer to the :doc:`features` page.


Experimenting With Algorithms
-----------------------------

Two types of algorithms are available in MSAF:

* Boundaries
* Labels

To quickly check the available boundary algorithms, we can use the following function::

    print(msaf.get_all_boundary_algorithms())

Analogously, we can do the same for the label algorithms::

    print(msaf.get_all_label_algorithms())

Once we know the desired combination of algorithms, we can run them by calling the ``process`` function with the
parameters ``boundaries_id`` and ``labels_id`` for the boundary and label algorithms, respectively.
For example, if we want to use the Checkerboard (Foote) algorithm for boundaries, and the Convex NMF for labels,
we would call ``process`` as follows::

    bounds, labels = msaf.process(audio_file, boundaries_id="foote", labels_id="cnmf")

If ``"gt"`` is passed as the ``boundaries_id``, the annotated boundaries will be used (only works if there are available annotations in a file contained in ``../references/<audio_filename>.jams``).
If ``None`` is passed as the ``labels_id``, no label algorithm is used (only silence and `-1` labels are returned).

For more information about the available algorithms, please refer to the :doc:`algorithms` page.

Experimenting With Datasets
---------------------------

So far, we have only used MSAF in `single file` mode.
We can also use MSAF to run structural segmentation algorithms across full datasets, using the `collection` mode.
Following the :ref:`quickstart_example` example, we will now run MSAF on the entire `Sargon <https://github.com/urinieto/msaf-data/tree/master/Sargon>`_ dataset:

.. code-block:: python
    :linenos:

    # MSAF on collection mode
    from __future__ import print_function
    import msaf

    # 1. Select dataset
    ds_path = "../datasets/Sargon"

    # 2. Segment all the files contained in the dataset using the default settings
    results = msaf.process(ds_path)
    print(results)

    # 3. Evaluate the results
    evals = msaf.eval.process(ds_path)
    print(evals)

In the first step we select the Sargon dataset. Note that we point to the root of the directory with the correct Dataset structure (see :doc:`datasets` for more info). ::

    ds_path = "../datasets/Sargon"

We then run MSAF on all of the files contained in this dataset, using the same ``process`` function::

    results = msaf.process(ds_path)

Note that the returned results are now a single list containing one set of results (tuple of ``(boundaries, labels)``) for each audio file in the dataset.
In this example, since the Sargon dataset has 4 audio files, the following is true ``len(results) == 4``.

Finally, in the third step, we evaluate the whole dataset, following a similar behavior as in the single file mode::

    evals = msaf.eval.process(ds_path)

The ``evals`` variable will still contain a ``pandas`` data frame, one row for each audio file.

For more information about the available datasets and their default structure, please refer to the :doc:`datasets` page.

More Examples
-------------

In the `examples <https://github.com/urinieto/msaf/tree/master/examples>`_ folder, more examples of using MSAF can be found.

Included in that folder you can find a `Jupyter Notebook <https://github.com/urinieto/msaf/blob/master/examples/Run%20MSAF.ipynb>`_ with further interactive MSAF usage.

For more information about MSAF, please refer to the original publication:
    
    Nieto, O., Bello, J. P., Systematic Exploration Of Computational Music Structure Research. Proc. of the 17th International Society for Music Information Retrieval Conference (ISMIR). New York City, NY, USA, 2016 (`PDF <http://marl.smusic.nyu.edu/nieto/publications/ISMIR2016-NietoBello.pdf>`_).
