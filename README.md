# MSAF - Music Structure Analysis Framework

A Python framework for automatic music structure analysis.

[![PyPI version](https://badge.fury.io/py/msaf.svg)](https://badge.fury.io/py/msaf)
[![Python](https://img.shields.io/pypi/pyversions/msaf)](https://pypi.org/project/msaf/)
[![Build Status](https://github.com/urinieto/msaf/actions/workflows/test.yaml/badge.svg)](https://github.com/urinieto/msaf/actions/workflows/test.yaml)
[![Documentation Status](https://readthedocs.org/projects/msaf/badge/?version=latest)](https://msaf.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Quickstart

```bash
pip install msaf
```

```python
import msaf

# Segment a track using the default algorithm (SF boundaries, no labels)
boundaries, labels = msaf.process("my_song.mp3")
print(boundaries)  # Array of boundary times in seconds

# Use a specific boundary and label algorithm
boundaries, labels = msaf.process(
    "my_song.mp3",
    boundaries_id="cnmf",
    labels_id="cnmf",
    feature="pcp",
)
```

**Requirements:** Python >= 3.10

## Features

- **8 boundary detection algorithms**: Foote, SF, C-NMF, 2D-FMC, OLDA, Spectral Clustering, CBM, VMO
- **4 label algorithms**: C-NMF, 2D-FMC, Spectral Clustering, VMO
- **7 audio feature types**: CQT, Mel, Log-Mel, MFCC, PCP (Chroma), Tonnetz, Tempogram
- **Beat-synchronous and frame-synchronous** analysis
- **Hierarchical segmentation** support
- **Evaluation** against ground truth using mir_eval
- **YAML-based configuration** with OmegaConf (customizable via `~/.msaf.yaml`)
- **Progress bars** for batch processing

## Documentation

See https://msaf.readthedocs.io for the complete reference manual and tutorials.

## Configuration

MSAF uses OmegaConf for configuration. Override defaults by creating `~/.msaf.yaml`:

```yaml
sample_rate: 44100
hop_size: 512
cqt:
  bins: 96
```

Or programmatically:

```python
import msaf
msaf.config.sample_rate = 44100
```

## Citing MSAF

Nieto, O., Bello, J. P., Systematic Exploration Of Computational Music Structure Research. Proc. of the 17th International Society for Music Information Retrieval Conference (ISMIR). New York City, NY, USA, 2016 ([PDF](https://ccrma.stanford.edu/~urinieto/MARL/publications/ISMIR2016-NietoBello.pdf)).

## Credits

Created by [Oriol Nieto](https://github.com/urinieto) (<oriol@nyu.edu>).
