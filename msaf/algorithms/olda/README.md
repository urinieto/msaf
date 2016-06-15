olda
====

Music segmentation by ordinal linear discriminant analysis

Training OLDA
=============

Follow these steps:

* Create the training data using the script `make_train.py`. E.g.
    ./make_train.py ~/datasets/BeatlesTUT/ out_beatles -j 8

* Train the olda model using the script `fit_olda_model.py`. E.g.
    ./fit_olda_model.py ~/datasets/BeatlesTUT/ out_beatles/EstBeats_BeatlesTUT_data.pickle models/EstBeats_BeatlesTUT.npy

* Use the `models/EstBeats_BeatlesTUT.npy` model to estimate new data, by
    setting it up in the `config.py` file.
