# Music Structure Analysis Framework #

[![Build Status](https://travis-ci.org/urinieto/msaf.svg?branch=devel)](https://travis-ci.org/urinieto/msaf)

## Description ##

This framework contains a set of algorithms to segment a given music audio signal. It uses [librosa](https://github.com/bmcfee/librosa/) to extract the necessary features, and is compatible with the [JAMS](https://github.com/urinieto/jams) format and [mir_eval](https://github.com/craffel/mir_eval).

## Boundary Algorithms ##

* Improved C-NMF (Nieto & Jehan 2013)
* Checkerboard-like Kernel (Foote 2000)
* Constrained Clustering (Levy & Sandler 2008) (original source code from [here](http://code.soundsoftware.ac.uk/projects/qm-dsp))
* OLDA (McFee & Ellis 2014) (original source code from [here](https://github.com/bmcfee/olda))
* Spectral Clustering (McFee & Ellis 2014) (original source code from [here](https://github.com/bmcfee/laplacian_segmentation))
* Structural Features (Serrà et al. 2012)
* SI-PLCA (Weiss & Bello 2011) (original source code from [here](http://ronw.github.io/siplca-segmentation/))

## Labeling Algorithms ##

* Improved C-NMF (Nieto & Jehan 2013)
* 2D Fourier Magnitude Coefficients (Nieto & Bello 2014)
* Constrained Clustering (Levy & Sandler 2008) (original source code from [here](http://code.soundsoftware.ac.uk/projects/qm-dsp))
* Spectral Clustering (McFee & Ellis 2014) (original source code from [here](https://github.com/bmcfee/laplacian_segmentation))
* SI-PLCA (Weiss & Bello 2011) (original source code from [here](http://ronw.github.io/siplca-segmentation/))

## Using MSAF ##

MSAF can be run in two different modes: **single file** and **collection** modes.

###Single File Mode###

In single file mode the features will be computed on the fly (so it always takes some extra time).
To run an audio file with the Convex NMF method for boundaries and 2D-FMC for labels using HPCP as features:

    ./run.py audio_file.mp3 -bid cnmf3 -lid fmc2d -f hpcp

The input file can be of type `mp3`, `wav` or `aif`.

If you want to *sonify* the boundaries, add the `-a` flag, and a file called `out_boundaries.wav` will be created in your current folder.

If you want to plot the boundaries against the ground truth, add the `-p` (only if ground truth references are available).

For more info, type:

    ./run.py -h


###Collection Mode###

You can run MSAF on a collection of files by inputting the correctly formatted folder to the dataset.
In this mode, MSAF will precompute the features during the first run and put them in a specific folder in order to speed up the process in further runnings.
After running the collection, you can also evaluate it using the standard music segmentation evaluation metrics (as long as you have reference annotations for it).

####Running Collection####

The MSAF datasets should have the following folder structure:

    my_collection/
    ├──  audio: The audio files of your collection.
    ├──  estimations: Estimations (output) by MSAF. Should be empty initially.
    ├──  features: Feature files for speeding up running time. Should be empty initially.
    └──  references: Human references for evaluation purposes.

Using this toy dataset as an example, we could run MSAF using the Foote algorithm for boundaries and using hpcp features by simply:

    ./run.py my_collection -f hpcp -bid foote

There is an example dataset included in the MSAF package, in the folder `ds_example`. 
It includes the SALAMI and Isophonics datasets (not the audio though).

Furthermore, we can spread the work across multiple processors by using the `-j` flag.
By default the number of processors is 4, this can be explicitly set by typing:

    ./run.py my_collection -f hpcp -bid foote -j 4

Additionally, we can run only a specific subset of the collection.
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
| DevtE2R       | Median Deviation from Estimation to Reference without first and last boundaries (trimmed)|
| DevtR2E       | Median Deviation from Reference to Estimation without first and last boundaries (trimmed)|
| HitRate\_0.5F | Hit Rate F-measure using 0.5 seconds window |
| HitRate\_0.5P | Hit Rate Precision using 0.5 seconds window |
| HitRate\_0.5R | Hit Rate Recall using 0.5 seconds window |
| HitRate\_3F | Hit Rate F-measure using 3 seconds window |
| HitRate\_3P | Hit Rate Precision using 3 seconds window |
| HitRate\_3R | Hit Rate Recall using 3 seconds window |
| HitRate\_t0.5F | Hit Rate F-measure using 0.5 seconds window without first and last boundaries (trimmed)|
| HitRate\_t0.5P | Hit Rate Precision using 0.5 seconds window without first and last boundaries (trimmed)|
| HitRate\_t0.5R | Hit Rate Recall using 0.5 seconds window without first and last boundaries (trimmed)|
| HitRate\_t3F | Hit Rate F-measure using 3 seconds window without first and last boundaries (trimmed)|
| HitRate\_t3P | Hit Rate Precision using 3 seconds window without first and last boundaries (trimmed)|
| HitRate\_t3R | Hit Rate Recall using 3 seconds window without first and last boundaries (trimmed)|
| PWF           | Pairwise Frame Clustering F-measure |
| PWP           | Pairwise Frame Clustering Precision |
| PWR           | Pairwise Frame Clustering Recall |
| Sf           | Normalized Entropy Scores F-measure |
| So           | Normalized Entropy Scores Precision |
| Su           | Normalized Entropy Scores Recall |

Analogously as in `run.py`, you can evaluate only a subset of the collection, by adding the `-d` flag:

    ./eval.py my_collection -f hpcp -bid foote -d Isophonics

Please, note that before you can run the `eval.py` script on a specific feature and set of algorithms, you **must** have run the `run.py` script first.

For more information about the metrics read the segmentation metrics in the [MIREX website](http://www.music-ir.org/mirex/wiki/2014:Structural_Segmentation). Finally, you can always add the `-h` flag in `eval.py` for more options.

###As a Python module###

Place the ```msaf``` module in your Python Path ($PYTHONPATH), so that you can import it from anywhere.
The main function is `process`, which takes basically the same parameters as the command line, and it returns the estimated boundary times and labels.

```python
import msaf
est_times, est_labels = msaf.process("path/to/audio_file.mp3", feature="hpcp", boundaries_id="cnmf3", labels_id="cnmf3")
```

For more parameters, please read the function's docstring.


## Requirements ##

* Python 2.7
* Numpy
* Scipy
* PyMF (for C-NMF algorithms only)
* cvxopt (for C-NMF algorithms only)
* Pandas (for evaluation only)
* joblib
* [mir\_eval](https://github.com/craffel/mir_eval)
* [librosa](https://github.com/bmcfee/librosa/)
* BLAS and LAPACK (Linux Only, OSX will use Accelerate by default)
* ffmpeg (to read mp3 files only)


## Note on Parallel Processes for OSX Users ##

By default, Numpy is compiled against the Accelerate Framework by Apple.
While this framework is extremely fast, Apple [does not want you to fork()
without exec](http://mail.scipy.org/pipermail/numpy-discussion/2012-August/063589.html), which may result in nasty crashes when using more than one thread (`-j > 1`).

The solution is to use an alternative framework, like OpenBLAS, and link it to
Numpy instead of the Accelerate Framework.
There is a nice explanation to do so [here](http://stackoverflow.com/a/14391693/777706).

## References ##

Foote, J. (2000). Automatic Audio Segmentation Using a Measure Of Audio Novelty. In Proc. of the IEEE International Conference of Multimedia and Expo (pp. 452–455). New York City, NY, USA.

Humphrey, E. J., Salamon, J., Nieto, O., Forsyth, J., Bittner, R. M., & Bello, J. P. (2014). JAMS: A JSON Annotated Music Specification for Reproducible MIR Research. In Proc. of the 15th International Society for Music Information Retrieval Conference. Taipei, Taiwan.

Levy, M., & Sandler, M. (2008). Structural Segmentation of Musical Audio by Constrained Clustering. IEEE Transactions on Audio, Speech, and Language Processing, 16(2), 318–326. doi:10.1109/TASL.2007.910781

McFee, B., & Ellis, D. P. W. (2014). Learnign to Segment Songs With Ordinal Linear Discriminant Analysis. In Proc. of the 39th IEEE International Conference on Acoustics Speech and Signal Processing (pp. 5197–5201). Florence, Italy.

Mcfee, B., & Ellis, D. P. W. (2014). Analyzing Song Structure with Spectral Clustering. In Proc. of the 15th International Society for Music Information Retrieval Conference (pp. 405–410). Taipei, Taiwan.

Nieto, O., & Bello, J. P. (2014). Music Segment Similarity Using 2D-Fourier Magnitude Coefficients. In Proc. of the 39th IEEE International Conference on Acoustics Speech and Signal Processing (pp. 664–668). Florence, Italy.

Nieto, O., & Jehan, T. (2013). Convex Non-Negative Matrix Factorization For Automatic Music Structure Identification. In Proc. of the 38th IEEE International Conference on Acoustics Speech and Signal Processing (pp. 236–240). Vancouver, Canada.

Raffel, C., Mcfee, B., Humphrey, E. J., Salamon, J., Nieto, O., Liang, D., & Ellis, D. P. W. (2014). mir_eval: A Transparent Implementation of Common MIR Metrics. In Proc. of the 15th International Society for Music Information Retrieval Conference. Taipei, Taiwan.

Serrà, J., Müller, M., Grosche, P., & Arcos, J. L. (2014). Unsupervised Music Structure Annotation by Time Series Structure Features and Segment Similarity. IEEE Transactions on Multimedia, Special Issue on Music Data Mining, 16(5), 1229 – 1240. doi:10.1109/TMM.2014.2310701

Weiss, R., & Bello, J. P. (2011). Unsupervised Discovery of Temporal Structure in Music. IEEE Journal of Selected Topics in Signal Processing, 5(6), 1240–1251.

## Credits ##

Created by [Oriol Nieto](https://files.nyu.edu/onc202/public/) (<oriol@nyu.edu>).
