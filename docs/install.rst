Installation Instructions
=========================

The simplest way to install MSAF is through the Python Package Index (PyPI).  This
will ensure that all required dependencies are fulfilled.  This can be achieved by
executing the following command::

    pip install msaf

or::

    sudo pip install msaf

to install system-wide, or::

    pip install -u msaf

to install just for your own user.

If you've downloaded the archive manually from the `releases
<https://github.com/urinieto/msaf/releases/>`_ page, you can install using the
`setuptools` script::

    tar xzf msaf-VERSION.tar.gz
    cd librosa-VERSION/
    python setup.py install

Getting the Datasets
--------------------

The datasets of MSAF are included in a separate repo due to their heavy size.
They can be downloaded from `<https://github.com/urinieto/msaf-data>`_

Additional notes for OS X
-------------------------

By default, Numpy is compiled against the Accelerate Framework by Apple.
While this framework is remarkably fast, Apple `does not want you to fork()
without exec <http://mail.scipy.org/pipermail/numpy-discussion/2012-August/063589.html>`_, which may result in nasty crashes when using more than one thread (``-j > 1``).

The solution is to use an alternative framework, like OpenBLAS, and link it to
Numpy instead of the Accelerate Framework.
There is a nice explanation to do so `here <http://stackoverflow.com/a/14391693/777706>`_.

ffmpeg
------

To fuel `audioread` with more audio-decoding power, you can install *ffmpeg* which
ships with many audio decoders.

You can use *homebrew* to install the program by calling
`brew install ffmpeg` or get a binary version from their website https://www.ffmpeg.org.
