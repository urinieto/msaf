.. _allalgorithms:

Algorithms
==========

MSAF comes with two types of algorithms: *boundary detection* and *label clustering* algorithms.
Below, the included algorithms are discussed and classified based on these two categories.
Note that some algorithms can actually approach both subproblems.

Additionally, two other algorithms published under a GPL license can be obtained from
the `msaf-gpl repo <https://github.com/urinieto/msaf-gpl>`_.

Boundary Algorithms
-------------------

.. automodule:: msaf.algorithms.foote
.. automodule:: msaf.algorithms.cnmf
.. automodule:: msaf.algorithms.olda
.. automodule:: msaf.algorithms.scluster
.. automodule:: msaf.algorithms.sf
.. automodule:: msaf.algorithms.vmo

Label Algorithms
----------------

.. automodule:: msaf.algorithms.fmc2d
.. automodule:: msaf.algorithms.cnmf
.. automodule:: msaf.algorithms.scluster
.. automodule:: msaf.algorithms.vmo

Adding A New Algorithm to MSAF
------------------------------

To include a new algorithm in MSAF, the following steps should be performed:

    1. Create new directory in the `algorithms <https://github.com/urinieto/msaf/tree/master/msaf/algorithms>`_ directory with the desired algorithm name.
    2. Create 3 new files in this new directory: ``__init__.py``, ``config.py``, and ``segmenter.py``.
    3. The ``__init__.py`` file should import everything from the new ``config.py`` and ``segmenter.py`` files.
    4. The ``config.py`` should contain the following variables:

        * ``config``: variable with the parameters of the algorithm.
        * ``algo_id``: string with the algorithm identifier.
        * ``is_boundary_type``: boolean that determines whether the algorithm allows boundary detection.
        * ``is_label_type``: boolean that determines whether the algorithm allows labeling.

    5. The ``segmenter.py`` should contain a class ``Segmenter`` that inherits from ``SegmenterInterface`` and implements the method ``processFlat``. This is where the main algorithm is implemented.

In the folder `algorithms/example <https://github.com/urinieto/msaf/tree/master/msaf/algorithms/example>`_ an example of a new algorithm is included.
The easiest way to add a new algorithm to MSAF is to copy and paste the example directory to use it as the base code of the new algorithm.
