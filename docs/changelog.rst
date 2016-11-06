Changes
=======

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
