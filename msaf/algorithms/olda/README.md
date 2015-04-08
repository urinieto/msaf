olda
====

Music segmentation by ordinal linear discriminant analysis

Training OLDA
=============

Follow these steps:

* Create the training data using the script `make_train.py`. E.g.
    ./make_train.py ~/datasets/Segments/ out_salami -d SALAMI -j 8

* Train the olda model using the script `fit_olda_model.py`. E.g.
    ./fit_olda_model.py ~/datasets/Segments/ out_salami/EstBeats_SALAMI-i_data.pickle models/EstBeats_SALAMI-i.npy -d SALAMI-i


* Use the `models/EstBeats_SALAMI-i.npy` model to estimate new data.
