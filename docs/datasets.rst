.. _datasets:

Datasets
========

Multiple human annotated datasets are contain within MSAF to facilitate the assessment of structural segmentation algorithms.
Due to the heavy file sizes, the ``datasets`` can be found in the following separate repo: `<https://github.com/urinieto/msaf-data>`_

Alternative, you can get this repo by getting all the submodules of the main MSAF repo, like this::

    git clone --recursive git://github.com/urinieto/msaf.git

Below we describe the format of the datasets and enumerate all the available ones.
Note that due to copyright reasons no audio is provided for most datasets.

Dataset Format
--------------

The MSAF datasets should have the following directory structure::

    my_collection/
    ├──  audio: The audio files of your collection.
    ├──  estimations: Estimations (output) by MSAF. Should be empty initially.
    ├──  features: Feature files for speeding up running time. Should be empty initially.
    └──  references: Human references for evaluation purposes. Only needed to perform evaluations.

These default directories can be changed in the MSAF config parameters (see :ref:`config`).

The references are stored using the JAMS format. Check out the official `JAMS repo <https://github.com/marl/jams/>`_ and the `original publication <http://marl.smusic.nyu.edu/papers/humphrey_jams_ismir2014.pdf>`_.



The Beatles TUT Dataset
-----------------------

Dataset corrected by the Tampere University of Technology.
It contains 174 structural annotations, one for each Beatles song.

Isophonics Dataset
------------------

A series of pop songs including, in some cases, beat and key annotations.
The total number of structural annotations is 300, one for each song.
The Beatles dataset is also included here, but the annotations slightly differ.


SALAMI Dataset
--------------
The Structural Analysis of Large Amounts of Music Information dataset (see `official website <http://ddmal.music.mcgill.ca/research/salami/annotations>`_).
The structure of 1164 tracks has been annotated, most of the times by two different music experts.


SPAM Dataset
------------
Structural Poly Annotations of Music, which contains 5 different structural annotations for each of the 50 tracks.
Pre-computed features are also available.
Please read more [here](https://github.com/urinieto/msaf-data/tree/master/SPAM).

Sargon Dataset
--------------

A small dataset of 30 minutes of heavy metal music. It includes audio.
