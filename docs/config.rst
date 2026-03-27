.. _config:

Configuration
=============

MSAF uses `OmegaConf <https://omegaconf.readthedocs.io/>`_ for configuration
management. The configuration is a structured object with sensible defaults
that can be overridden via a YAML file or programmatically.

Configuration Precedence
------------------------

The order of precedence is (highest to lowest):

1. Direct assignment in code: ``msaf.config.sample_rate = 44100``
2. User YAML file at ``~/.msaf.yaml`` (or the path in ``MSAF_CONFIG`` env var)
3. Built-in defaults

User YAML File
--------------

Create a ``~/.msaf.yaml`` file to override default settings. For example:

.. code-block:: yaml

    sample_rate: 44100
    hop_size: 512

    cqt:
      bins: 96

You can also set the ``MSAF_CONFIG`` environment variable to point to a
different YAML file:

.. code-block:: bash

    MSAF_CONFIG=/path/to/my_config.yaml python myscript.py

Programmatic Override
---------------------

You can override config values directly in your code:

.. code-block:: python

    import msaf
    msaf.config.sample_rate = 44100
    msaf.config.cqt.bins = 96

Config Attributes
-----------------

Global Parameters
~~~~~~~~~~~~~~~~~

.. attribute:: sample_rate

    Positive int value, default: 22050.
    The sampling rate for audio analysis. Resampling will be applied as needed.

.. attribute:: n_fft

    Positive int value, default: 4096.
    The size of the Fast Fourier Transform, in number of samples.

.. attribute:: hop_size

    Positive int value, default: 1024.
    The hop size in samples.

.. attribute:: default_bound_id

    String value, default: ``'sf'``.
    The default boundary detection algorithm.
    See the :doc:`algorithms` section for available options.

.. attribute:: default_label_id

    String or None, default: ``None``.
    The default label algorithm. If ``None``, no labels are computed.

.. attribute:: minimum_frames

    Positive int value, default: 10.
    Minimum number of frames required to run algorithms.

Feature Parameters
~~~~~~~~~~~~~~~~~~

Each feature type has its own configuration section:

- ``msaf.config.cqt``: CQT features (bins, norm, filter_scale, ref_power)
- ``msaf.config.mel``: Mel spectrogram (n_mels, f_min, f_max)
- ``msaf.config.mfcc``: MFCC features (n_mels, n_mfcc, ref_power)
- ``msaf.config.pcp``: PCP / Chroma features (bins, norm, f_min, n_octaves)
- ``msaf.config.tonnetz``: Tonnetz features (bins, norm, f_min, n_octaves)
- ``msaf.config.tempogram``: Tempogram features (win_length)

Dataset Parameters
~~~~~~~~~~~~~~~~~~

The ``msaf.config.dataset`` section controls dataset directory structure:

- ``audio_dir``: Directory containing audio files (default: ``"audio"``)
- ``estimations_dir``: Directory for estimation output (default: ``"estimations"``)
- ``references_dir``: Directory for reference annotations (default: ``"references"``)
- ``audio_exts``: Supported audio extensions (default: ``[".wav", ".mp3", ".aif"]``)

Full Default Configuration
--------------------------

.. code-block:: yaml

    sample_rate: 22050
    n_fft: 4096
    hop_size: 1024
    default_bound_id: sf
    default_label_id: null
    minimum_frames: 10
    results_dir: results
    results_ext: .csv
    out_boundaries_ext: -bounds.wav
    dataset:
      audio_dir: audio
      estimations_dir: estimations
      references_dir: references
      audio_exts: [.wav, .mp3, .aif]
      estimations_ext: .jams
      references_ext: .jams
    cqt:
      bins: 84
      norm: .inf
      filter_scale: 1.0
      ref_power: max
    mel:
      n_mels: 80
      f_min: 80.0
      f_max: 16000.0
    mfcc:
      n_mels: 128
      n_mfcc: 14
      ref_power: max
    pcp:
      bins: 84
      norm: .inf
      f_min: 27.5
      n_octaves: 6
    tonnetz:
      bins: 84
      norm: .inf
      f_min: 27.5
      n_octaves: 6
    tempogram:
      win_length: 192
