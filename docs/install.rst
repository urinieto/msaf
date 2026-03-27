Installation Instructions
=========================

Requirements
------------

MSAF requires **Python 3.10 or newer**.

Install from PyPI
-----------------

The simplest way to install MSAF is through PyPI::

    pip install msaf

For development, install with the extra dev dependencies::

    pip install msaf[dev]

Install from Source
-------------------

Clone the repository and install in editable mode::

    git clone https://github.com/urinieto/msaf.git
    cd msaf
    pip install -e .[dev]

Getting the Datasets
--------------------

The datasets of MSAF are included in a separate repo due to their size.
They can be downloaded from `<https://github.com/urinieto/msaf-data>`_.

ffmpeg
------

For broader audio format support, install *ffmpeg*:

- macOS: ``brew install ffmpeg``
- Ubuntu: ``sudo apt install ffmpeg``
- Or download from https://www.ffmpeg.org
