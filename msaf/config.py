"""MSAF Configuration using OmegaConf.

Configuration is loaded from structured defaults, optionally merged with
a user YAML file at ``~/.msaf.yaml`` or the path in the ``MSAF_CONFIG``
environment variable.
"""

import math
import os
from dataclasses import dataclass, field

from omegaconf import DictConfig, OmegaConf


@dataclass
class CQTConfig:
    bins: int = 84
    norm: float = math.inf
    filter_scale: float = 1.0
    ref_power: str = "max"


@dataclass
class MelConfig:
    n_mels: int = 80
    f_min: float = 80.0
    f_max: float = 16000.0


@dataclass
class MFCCConfig:
    n_mels: int = 128
    n_mfcc: int = 14
    ref_power: str = "max"


@dataclass
class PCPConfig:
    bins: int = 84
    norm: float = math.inf
    f_min: float = 27.5
    n_octaves: int = 6


@dataclass
class TonnetzConfig:
    bins: int = 84
    norm: float = math.inf
    f_min: float = 27.5
    n_octaves: int = 6


@dataclass
class TempogramConfig:
    win_length: int = 192


@dataclass
class DatasetConfig:
    audio_dir: str = "audio"
    estimations_dir: str = "estimations"
    references_dir: str = "references"
    audio_exts: list[str] = field(default_factory=lambda: [".wav", ".mp3", ".aif"])
    estimations_ext: str = ".jams"
    references_ext: str = ".jams"


@dataclass
class MSAFConfig:
    sample_rate: int = 22050
    n_fft: int = 4096
    hop_size: int = 1024
    default_bound_id: str = "sf"
    default_label_id: str | None = None
    minimum_frames: int = 10
    results_dir: str = "results"
    results_ext: str = ".csv"
    out_boundaries_ext: str = "-bounds.wav"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    cqt: CQTConfig = field(default_factory=CQTConfig)
    mel: MelConfig = field(default_factory=MelConfig)
    mfcc: MFCCConfig = field(default_factory=MFCCConfig)
    pcp: PCPConfig = field(default_factory=PCPConfig)
    tonnetz: TonnetzConfig = field(default_factory=TonnetzConfig)
    tempogram: TempogramConfig = field(default_factory=TempogramConfig)


def _load_config() -> DictConfig:
    """Load config from structured defaults, merged with user YAML if present."""
    cfg = OmegaConf.structured(MSAFConfig)

    # Check for user config file
    user_config_path = os.environ.get(
        "MSAF_CONFIG", os.path.expanduser("~/.msaf.yaml")
    )
    if os.path.isfile(user_config_path):
        user_cfg = OmegaConf.load(user_config_path)
        cfg = OmegaConf.merge(cfg, user_cfg)

    return cfg


config = _load_config()
