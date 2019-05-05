Changes
=======

v0.1.8
-------

* Fixed amplitude bug for CQT (see issue #92)
* Fixed smoothing of lag matrix in SF algorithm (it was only smoothing the time domain, not the lag one) (see issue #90)
* Fixing potentially empty segments when using annotated beats

v0.1.70
-------

* Compatibility with Librosa 0.6.0 when computing CQTs.
* Added VMO algorithm.
* Modified quick-guide with clearer example for evaluation.

v0.1.61
-------

* Fixed hierarchical boundaries, which didn't work on JAMS 0.3.0.

v0.1.6
------

* Added `framesync` parameter to `compute_features.py` script example
* `joblib` was repeated in the requirements. Fixed
* Updated PyPi docs strings such that it shows that MSAF if Python 3.56 compatible
* Making `configparser` compatible with Python 2.7
* Using latest `scluster` method, from McFee's code [here](https://github.com/bmcfee/lsd_viz)
* Adopted code to JAMS 0.3.0.

v0.1.51
-------

* JAMS bug supposedly fixed in v0.1.5 was not really fixed. Fixed now. Alrighten.

v0.1.5
------

* Fixed bug tha threw a `TypeError` if multiple algorithms were run in a single JAMS file with `None` and other label_ids in it
* Added new `vmo` oracle segmentation method (by Cheng-i Wang, thanks!)
* Adapting sonify function to latest numpy
* Using KMeans from sklearn instead of scipy for 2D-FMC. Results are better
* Making sure we are never using more number of clusters than number of segments for 2D-FMC
* Added new parameter `2dfmc_offset` in the 2D-FMC method
* Using np.inf normalization for 2D-FMC now, since it seems to yield better results (at least for Beatles TUT)
* Padding beat-sync features now, seems to fix potential misalignment of boundaries. Some algorithms (2D-FMC, CNMF) seem to yield better results now
* Modified features file: two new fields may be addeed: `est_beatsync_times` and `ann_beatsync_times`.
* The member variable `_framesync_times` in the `Features` was never updated. Fixed it

v0.1.4
------

* Included Python 3.5 in the metadata
* Removed old functions from i/o module that nobody should be using
* cleanued up code for reading/writing estimations (just a tiny bit)
* Unit tested i/o module

v0.1.3
------

* Fixed bug of selecting framesync features
* OLDA and Scluster hierarchies are consistent now (first element in hierarchy in the highest in both algorithms; this was also true for Scluster before this fix)
* Warning message is displayed if jams file exists but can't be read during features computation
* If two algorithms used at the same time have the same name, and AssertionError is raised
* Fixed normalization problem: now algorithms have independent normalization parameters
* Adding `out_file` variable in main process function of the eval module
* Reporting proper weighted F-measure for the perceptual Hit Rate
* Fixed bug of annotator id not correctly passed to hierarchical evaluation function
* More unit tests
* Added script to upload to Pypi

v0.1.2
------

* Adapting code to librosa 0.5
* Improved coveralls (starting testing plots)
* Allowing computation of label algorithms for each layer of hierarchical boundaries
* Removed old notebooks that were either useless or belonged to `msaf-data`
* Removed old `ds-name` references, improving code health

v0.1.1
------

* Fixed plotting issues
* Fixed bug regarding old variable `ds_name`

v0.1.0
------

* New features: Tempogram
* Added proper configuration module (à la Theano)
* Added coveralls
* More thorough unit testing
* Proper documentation in Sphinx
* Features code more modular: easier to manage and to include new features
* Fixed a bunch of bugs


v0.0.1
------

* Initial release
