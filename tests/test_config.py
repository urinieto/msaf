import math

import msaf
from msaf.config import MSAFConfig, _load_config


def test_default_values():
    """Tests that the default config values are correct."""
    assert msaf.config.sample_rate == 22050
    assert msaf.config.n_fft == 4096
    assert msaf.config.hop_size == 1024
    assert msaf.config.default_bound_id == "sf"
    assert msaf.config.default_label_id is None
    assert msaf.config.minimum_frames == 10


def test_nested_config():
    """Tests that nested config values are accessible."""
    assert msaf.config.cqt.bins == 84
    assert math.isinf(msaf.config.cqt.norm)
    assert msaf.config.cqt.filter_scale == 1.0
    assert msaf.config.cqt.ref_power == "max"


def test_mel_config():
    """Tests mel config defaults."""
    assert msaf.config.mel.n_mels == 80
    assert msaf.config.mel.f_min == 80.0
    assert msaf.config.mel.f_max == 16000.0


def test_mfcc_config():
    """Tests mfcc config defaults."""
    assert msaf.config.mfcc.n_mels == 128
    assert msaf.config.mfcc.n_mfcc == 14
    assert msaf.config.mfcc.ref_power == "max"


def test_pcp_config():
    """Tests pcp config defaults."""
    assert msaf.config.pcp.bins == 84
    assert math.isinf(msaf.config.pcp.norm)
    assert msaf.config.pcp.f_min == 27.5
    assert msaf.config.pcp.n_octaves == 6


def test_tonnetz_config():
    """Tests tonnetz config defaults."""
    assert msaf.config.tonnetz.bins == 84
    assert msaf.config.tonnetz.n_octaves == 6


def test_tempogram_config():
    """Tests tempogram config defaults."""
    assert msaf.config.tempogram.win_length == 192


def test_dataset_config():
    """Tests dataset config defaults."""
    assert msaf.config.dataset.audio_dir == "audio"
    assert msaf.config.dataset.estimations_dir == "estimations"
    assert msaf.config.dataset.references_dir == "references"
    assert ".wav" in msaf.config.dataset.audio_exts
    assert ".mp3" in msaf.config.dataset.audio_exts


def test_results_config():
    """Tests results config defaults."""
    assert msaf.config.results_dir == "results"
    assert msaf.config.results_ext == ".csv"


def test_load_config():
    """Tests that _load_config returns a valid config."""
    cfg = _load_config()
    assert cfg.sample_rate == 22050


def test_config_dataclass():
    """Tests that MSAFConfig can be instantiated."""
    cfg = MSAFConfig()
    assert cfg.sample_rate == 22050
    assert cfg.cqt.bins == 84
