.. _config:

Configuration
=============

The ``config`` module contains many ``attributes`` that modify MSAF's behavior.
Many of these attributes are consulted during the import of the ``msaf`` module and many are assumed to be
read-only.

*As a rule, the attributes in this module should not be modified by user code.*

MSAF's code comes with default values for these attributes, but you can
override them from your ``.msafrc`` file, and override those values in turn by
the :envvar:`MSAF_FLAGS` environment variable.

The order of precedence is:

1. an assignment to ``msaf.config.<property>``
2. an assignment in :envvar:`MSAF_FLAGS`
3. an assignment in the .msafrc file (or the file indicated in :envvar:`MSAFRC`)

You can print out the current/effective configuration at any time by printing
``msaf.config``.
For example, to see a list  of all active configuration variables, type this from the command-line::

	python -c 'import msaf; print(msaf.config)' | less

Environment Variables
---------------------

.. envvar:: MSAF_FLAGS

    This is a list of comma-delimited key=value pairs that control
    MSAF's behavior.

    For example, in bash, you can override your :envvar:`MSAFRC` defaults
    for <myscript>.py by typing this:

    .. code-block:: bash

        MSAF_FLAGS='sample_rate=11025,n_fft=2048' python <myscript>.py

    If a value is defined several times in ``MSAF_FLAGS``,
    the right-most definition is used. So, for instance, if
    ``MSAF_FLAGS='sample_rate=44100, sample_rate=22050'``, then 22050 Hz will be used as the default sampling rate.

.. envvar:: MSAFRC

    The location[s] of the .msafrc file[s] in ConfigParser format.
    It defaults to ``$HOME/.msafrc``. On Windows, it defaults to
    ``$HOME/.msafrc:$HOME/.msafrc.txt`` to make Windows users' life
    easier.

    Here is the .msafrc equivalent to the MSAF_FLAGS in the example above:

    .. code-block:: cfg

        [global]
        sample_rate = 11025
        n_fft = 2048

        [cqt]
        bins = 96

    Configuration attributes that are available directly in ``config``
    (e.g. ``config.sample_rate``, ``config.hop_size``) should be defined in the
    ``[global]`` section.
    Attributes from a subsection of ``config`` (e.g. ``config.cqt.bins``,
    ``config.mfcc.n_mels``) should be defined in their corresponding
    section (e.g. ``[cqt]``, ``[mfcc]``).

    Multiple configuration files can be specified by separating them with ':'
    characters (as in $PATH).  Multiple configuration files will be merged,
    with later (right-most) files taking priority over earlier files in the
    case that multiple files specify values for a common configuration option.
    For example, to override system-wide settings with personal ones,
    set ``MSAFRC=/etc/msafrc:~/.msafrc``.

Config Attributes
-----------------

The list below describes some of the more common and important flags
that you might want to use. For the complete list (including documentation),
import MSAF and print the config variable, as in:

.. code-block:: bash

    python -c 'import msaf; print(msaf.config)' | less

.. attribute:: default_bound_id

    String value: either ``'sf'``, ``'cnmf'``, ``'foote'``, ``'olda'``,
    ``'scluster'``, ``'gt'``

    This is the identifier for the boundary algorithm to use.
    If ``'gt'`` is used the reference boundaries will be read instead of computed.
    See the :doc:`algorithms` section for more information.

.. attribute:: default_label_id

    String value: either ``None``, ``'cnmf'``, ``'fmc2d'``, ``'scluster'``

    This is the identifier for the label algorithm to use.
    If ``None`` is used, no label algorithm will be applied.
    See the :doc:`algorithms` section for more information.
