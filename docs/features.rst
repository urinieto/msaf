.. _features:

Features
========

Multiple audio features are available in MSAF, mostly implemented using 
`librosa <https://github.com/librosa/librosa>`_.
This framework is written such that they should only be computed once for each audio file.
These features could potentially be used across all algorithms, so MSAF stores them in ``json``
files, one for each audio file, to improve computation efficiency.

Features JSON Files
-------------------

The ``json`` files are stored either in the ``features`` directory (in *collection mode*) or
in the ``.features_msaf_tmp.json`` temporary file (in *single file mode*).

The format of the ``json`` file is as follows:

.. code-block:: json

    {
        "globals": {
            "audio_file": "<path to audio file>", 
            "dur": "<duration of audio>", 
            "sample_rate": "<sample rate>", 
            "hop_length": "<hop lenght>"
        }, 
        "metadata": {
            "timestamp": "<YYYY/MM/DD hh:mm:ss>", 
            "versions": {
                "numpy": "<numpy version>", 
                "msaf": "<msaf version>", 
                "librosa": "<librosa version>"
            }
        }
        "<feature_id>": {
            "framesync": [
                [ 0.0, 0.0, "..." ],
                "..."
            ],
            "est_beatsync": [
                [ 0.0, 0.0, "..." ],
                "..."
            ],
            "ann_beatsync": [
                [ 0.0, 0.0, "..." ],
                "..."
            ],
            "params": {
                "<param_name1>": "<param_value2>", 
                "<param_name1>": "<param_value2>", 
                "..."
            } 
        }
        "est_beatsync": [ 0.0, 1.0, "..." ],
        "ann_beatsync": [ 0.0, 1.0, "..." ]
    }

A brief description for the main keys of this ``json`` file follows:

* ``globals``: contains a set of global parameters used to compute the features.
* ``metadata``: contains a set of meta-parameters that might become useful for debugging purposes.
* ``est_beats``: contains the set of estimated beats (using librosa).
* ``ann_beats``: contains the set of reference beats (only exists if reference beats are available).
* ``<feature_id>`` (e.g., ``pcp``, ``mfcc``): contains the actual features of the given audio file. Inside this key the following sub-keys can be found:

    * ``framesync``: Actual frame-wise features.
    * ``est_beatsync``: Features synchronized to the estimated beats.
    * ``ann_beatsync``: Features synchronized to the reference beats (only exists if reference beats are available).
    * ``params``: A set of parameters of the actual type of features.

Pre-computed features for the `SPAM dataset <https://github.com/urinieto/msaf-data/tree/master/SPAM>`_ can be found `here <https://ccrma.stanford.edu/%7Eurinieto/SPAM/SPAM-features.tgz>`_.


Available Features
------------------

.. automodule:: msaf.features

Adding New Features to MSAF
---------------------------

MSAF is written such that adding new features should be relatively painless.
Follow these steps:

    1. Add a new class that inherits from ``Features`` in the file `features.py <https://github.com/urinieto/msaf/blob/master/msaf/features.py>`_.
    2. Implement the following methods: ``__init``, ``get_id``, and ``compute_features``:

        * ``__init__``: The constructor should accept the necessary parameters for the computation of the features, plus the ``file_struct`` (the audio file encapsulated in the `FileStruct` class), and ``feat_type`` (the type of features).
        * ``get_id``: Class method that returns the identifier of the new type of features.
        * ``compute_features``: The actual implementation of the features. Here the parameters of the constructor should be read.

In the `features.py <https://github.com/urinieto/msaf/blob/master/msaf/features.py>`_ file the existing features of MSAF are found, which can be used as examples.
