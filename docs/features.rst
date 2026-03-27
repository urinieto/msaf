.. _features:

Features
========

Multiple audio features are available in MSAF, implemented using
`librosa <https://github.com/librosa/librosa>`_.

As of MSAF 1.0, features are always computed on the fly from audio. There is no
caching to disk -- with modern hardware, feature computation is fast enough that
caching is unnecessary.

Three types of feature synchronization are supported:

* **Frame-synchronous** (``framesync``): One feature vector per analysis frame.
* **Estimated beat-synchronous** (``est_beatsync``): Features synchronized to beats estimated by librosa's beat tracker.
* **Annotated beat-synchronous** (``ann_beatsync``): Features synchronized to ground-truth beat annotations (if available).


Available Features
------------------

.. automodule:: msaf.features

Adding New Features to MSAF
---------------------------

MSAF is written such that adding new features should be relatively painless.
Follow these steps:

    1. Add a new class that inherits from ``Features`` in the file `features.py <https://github.com/urinieto/msaf/blob/main/msaf/features.py>`_.
    2. Implement the following methods: ``__init__``, ``get_id``, and ``compute_features``:

        * ``__init__``: The constructor should accept the necessary parameters for the computation of the features, plus the ``file_struct`` (the audio file encapsulated in the `FileStruct` class), and ``feat_type`` (the type of features).
        * ``get_id``: Class method that returns the identifier of the new type of features.
        * ``compute_features``: The actual implementation of the features. Here the parameters of the constructor should be read.

In the `features.py <https://github.com/urinieto/msaf/blob/main/msaf/features.py>`_ file the existing features of MSAF are found, which can be used as starting points. See `custom_feature.py <https://github.com/urinieto/msaf/blob/main/examples/custom_feature.py>`_ for a complete example.
