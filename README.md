# Music Structure Analysis Framework #

## Description ##

This framework contains a set of algorithms to segment a given music audio signal. It uses [Essentia](http://mtg.upf.edu/technologies/essentia) to extract the necessary features, and is compatible with the JAMS format and [mir_eval](https://github.com/craffel/mir_eval).

## Boundary Algorithms ##

* C-NMF (Nieto & Jehan 2013)
* Foote (Foote 2001)
* Levy (Levy & Sandler 2008) (original source code from [here](http://code.soundsoftware.ac.uk/projects/qm-dsp)).
* OLDA (McFee & Ellis 2014) (original source code from [here](https://github.com/bmcfee/olda)).
* Serrà (Serrà et al. 2012)
* SI-PLCA (Weiss & Bello 2011) (original source code from [here](http://ronw.github.io/siplca-segmentation/)).

## Requirements ##

* Python 2.7
* Numpy
* Scipy
* PyMF