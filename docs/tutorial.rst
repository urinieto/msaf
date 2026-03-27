Tutorial
========

This section covers the fundamentals of MSAF, including a package overview, basic and advanced usage, and dataset exploration.

Overview
--------

MSAF is divided into four moving blocks that compose the music structural segmentation ecosystem:

	- :ref:`Features <features>`:
		Set of audio feature extraction utilities that serve
		as input to the algorithms. Features are computed on the fly from audio.
	- :ref:`Algorithms <allalgorithms>`:
		Implementations of multiple boundary and label algorithms.
	- :ref:`Evaluations <eval>`:
		Common evaluation metrics available from mir_eval and gathered
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

    import msaf

    # 1. Select audio file
    audio_file = "my_song.mp3"

    # 2. Segment the file using the default MSAF parameters
    boundaries, labels = msaf.process(audio_file)
    print('Estimated boundaries:', boundaries)

    # 3. Save segments using the MIREX format
    out_file = 'segments.txt'
    print('Saving output to %s' % out_file)
    msaf.io.write_mirex(boundaries, labels, out_file)

    # 4. Evaluate the results (requires reference annotations)
    try:
        evals = msaf.eval.process(audio_file)
        print(evals)
    except msaf.exceptions.NoReferencesError:
        file_struct = msaf.input_output.FileStruct(audio_file)
        print("No references found in {}. No evaluation performed.".format(
            file_struct.ref_file))

The main MSAF function ``process`` segments the given ``audio_file``.
It computes features on the fly and uses the default boundary and label algorithms
(the defaults can be changed, see :doc:`config`).
It returns the boundary times (in seconds) and a set of labels::

    boundaries, labels = msaf.process(audio_file)

In step 3, we store the results in a file using the standard `MIREX format <http://www.music-ir.org/mirex/wiki/2016:Structural_Segmentation#Output_File_Format_.28Structural_Segmentation.29>`_::

    msaf.io.write_mirex(boundaries, labels, out_file)

Finally, we can evaluate the results when human references are available.
The references must be placed in ``../references/<audio_file>.jams``, following the default MSAF Dataset configuration (this path can be changed in the default configuration).
The standard structural segmentation metrics will be used when calling::

    results = msaf.eval.process(audio_file)

Experimenting With Features
---------------------------

There are multiple features to experiment with in MSAF.
To quickly check the available features, access the ``features_registry`` dictionary::

    print(msaf.features_registry)

Select the desired features in the ``process`` function using the ``feature`` parameter.
For example, to use MFCC coefficients::

    boundaries, labels = msaf.process(audio_file, feature='mfcc')

For more information about the available features, please refer to the :doc:`features` page.


Experimenting With Algorithms
-----------------------------

Two types of algorithms are available in MSAF:

* Boundaries
* Labels

To check the available boundary algorithms::

    print(msaf.get_all_boundary_algorithms())

Analogously, for label algorithms::

    print(msaf.get_all_label_algorithms())

Once you know the desired combination, run them with the ``boundaries_id`` and ``labels_id`` parameters.
For example, to use the Checkerboard (Foote) algorithm for boundaries and the Convex NMF for labels::

    bounds, labels = msaf.process(audio_file, boundaries_id="foote", labels_id="cnmf")

If ``"gt"`` is passed as the ``boundaries_id``, annotated boundaries will be used (requires reference annotations).
If ``None`` is passed as the ``labels_id``, no label algorithm is used.

For more information about the available algorithms, please refer to the :doc:`algorithms` page.

Experimenting With Datasets
---------------------------

So far, we have only used MSAF in *single file* mode.
We can also run structural segmentation algorithms across full datasets using *collection* mode:

.. code-block:: python
    :linenos:

    import msaf

    # 1. Select dataset
    ds_path = "../datasets/Sargon"

    # 2. Segment all the files in the dataset
    results = msaf.process(ds_path)
    print(results)

    # 3. Evaluate the results
    evals = msaf.eval.process(ds_path)
    print(evals)

Note that in collection mode a progress bar is displayed to track processing.
The returned results are a list containing one ``(boundaries, labels)`` tuple for each audio file.

For more information about the available datasets and their default structure, please refer to the :doc:`datasets` page.

More Examples
-------------

In the `examples <https://github.com/urinieto/msaf/tree/main/examples>`_ folder, more examples of using MSAF can be found.

For more information about MSAF, please refer to the original publication:

    Nieto, O., Bello, J. P., Systematic Exploration Of Computational Music Structure Research. Proc. of the 17th International Society for Music Information Retrieval Conference (ISMIR). New York City, NY, USA, 2016 (`PDF <https://ccrma.stanford.edu/~urinieto/MARL/publications/ISMIR2016-NietoBello.pdf>`_).
