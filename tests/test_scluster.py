import numpy as np
import numpy.testing as npt
import scipy.ndimage

from msaf.algorithms.scluster import config
from msaf.algorithms.scluster.main2 import compute_laplacian, spectral_decomposition


def _synthetic_features(n=400, d=12, seed=0):
    """Beat-synchronous features with an A-B-A structure (from issue #168)."""
    rng = np.random.default_rng(seed)
    labels = np.repeat([0, 1, 0], [n * 3 // 8, n // 4, n - n * 5 // 8])
    centers = rng.normal(size=(2, d)) * 3
    noise = scipy.ndimage.uniform_filter1d(rng.normal(size=(n, d)), 5, axis=0)
    C = centers[labels] + noise
    return (C - C.min()).T


def test_spectral_decomposition_of_asymmetric_laplacian():
    """The time-lag filter makes L asymmetric, so eigh would be wrong (#168)."""
    C = _synthetic_features()
    L = compute_laplacian(C, C, config)
    assert np.abs(L - L.T).max() > 1e-3  # the case this test is about

    evals, evecs = spectral_decomposition(L)
    assert np.isrealobj(evals) and np.isrealobj(evecs)
    assert np.all(np.diff(evals) >= 0)
    npt.assert_allclose(evals[0], 0, atol=1e-10)

    # The leading eigenpairs used for clustering are true eigenpairs of L
    k = config["num_layers"]
    npt.assert_allclose(L @ evecs[:, :k], evecs[:, :k] * evals[:k], atol=1e-10)
