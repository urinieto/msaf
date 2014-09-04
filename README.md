# Music Structure Analysis Framework #

## Description ##

This framework contains a set of algorithms to segment a given music audio signal. It uses [Essentia](http://mtg.upf.edu/technologies/essentia) to extract the necessary features, and is compatible with the [JAMS](https://github.com/urinieto/jams) format and [mir_eval](https://github.com/craffel/mir_eval).

## Boundary Algorithms ##

* Improved C-NMF (Nieto & Jehan 2013)
* Checkerboard-like Kernel (Foote 2000)
* Constrained Clustering (Levy & Sandler 2008) (original source code from [here](http://code.soundsoftware.ac.uk/projects/qm-dsp))
* OLDA (McFee & Ellis 2014) (original source code from [here](https://github.com/bmcfee/olda))
* Structural Features (Serrà et al. 2012)
* SI-PLCA (Weiss & Bello 2011) (original source code from [here](http://ronw.github.io/siplca-segmentation/))

## Labeling Algorithms ##

* Improved C-NMF (Nieto & Jehan 2013)
* 2D Fourier Magnitude Coefficients (Nieto & Bello 2014)
* Constrained Clustering (Levy & Sandler 2008) (original source code from [here](http://code.soundsoftware.ac.uk/projects/qm-dsp))
* SI-PLCA (Weiss & Bello 2011) (original source code from [here](http://ronw.github.io/siplca-segmentation/))

## Using MSAF ##

MSAF can be run in two different modes: **single file** and **collection** modes.

###Single File Mode###



###Collection Mode###

You can run MSAF on a collection of files by inputting the correctly formatted folder to the dataset.

####Running Collection####

The MSAF datasets should be formatted as follows:

    my_collection/
        audio: The audio files of your collection.
        estimations: Estimations (output) by MSAF. Should be empty initially.
        features: Feature files for speeding up running time. Should be empty initially.
        references: Human references for evaluation purposes.

Using this toy dataset as an example, we could run MSAF using the Foote algorithm for boundaries and using hpcp features by simply:

    ./run.py my_collection -f hpcp -bid foote

There is an example dataset included in the MSAF package, in the folder `ds_example`. 
It includes the SALAMI and Isophonics datasets (not the audio though).

Furthermore, we can spread the work across multiple processors by using the `-j` flag.
By default the number of processors is 4, this can be explicitly set by typing:

    ./run.py my_collection -f hpcp -bid foote -j 4

Additionally, we can only a subset of the collection.
For example, if you want to run on the Isophonics set, you can do:

    ./run.py my_collection -f hpcp -bid foote -d Isophonics

For more information, please type:

    ./run.py -h

####Evaluating Collection####

Once you have run the desired algorithm on a specified collection, the next thing you might probably want to do is to evaluate its results.
To do so, use the `eval.py` script, just like this (following the example above):

    ./eval.py my_collection -f hpcp -bid foote

The output contains the following evaluation metrics:

| Metric        | Description       |
| --------------|-------------------|
| D             | Information Gain  |
| DevE2R        | Median Deviation from Estimation to Reference |
| DevR2E        | Median Deviation from Reference to Estimation |



###As a Python module###

Place the ```msaf``` module in your Python Path ($PYTHONPATH).

```python
import msaf
msaf.run
```


## Requirements ##

* Python 2.7
* Numpy
* Scipy
* PyMF (for C-NMF algorithms only)
* Pandas (for evaluation only)
* joblib
* [Essentia](https://github.com/MTG/essentia)
* [mir\_eval](https://github.com/craffel/mir_eval)
* [librosa](https://github.com/bmcfee/librosa/)


## Note on Parallel Processes for OSX Users ##

By default, Numpy is compiled against the Accelerate Framework by Apple.
While this framework is extremely fast, Apple [does not want you to fork()
without exec](http://mail.scipy.org/pipermail/numpy-discussion/2012-August/063589.html), which may result in nasty crashes when using more than one thread (`-j > 1`).

The solution is to use an alternative framework, like OpenBLAS, and link it to
Numpy instead of the Accelerate Framework.
There is a nice explanation to do so [here](http://stackoverflow.com/a/14391693/777706).

## Credits ##

Created by [Oriol Nieto](https://files.nyu.edu/onc202/public/) (<oriol@nyu.edu>).
