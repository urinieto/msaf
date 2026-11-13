import numpy as np
import numpy.testing as npt
import scipy.linalg
import scipy.ndimage

from msaf.algorithms.scluster import config
from msaf.algorithms.scluster.main2 import compute_laplacian


def _synthetic_features(n=400, d=12, seed=0):
    """Beat-synchronous features with an A-B-A structure (from issue #168)."""
    rng = np.random.default_rng(seed)
    labels = np.repeat([0, 1, 0], [n * 3 // 8, n // 4, n - n * 5 // 8])
    centers = rng.normal(size=(2, d)) * 3
    noise = scipy.ndimage.uniform_filter1d(rng.normal(size=(n, d)), 5, axis=0)
    C = centers[labels] + noise
    return (C - C.min()).T


def test_compute_laplacian_is_normalized_laplacian():
    """The median-filtered recurrence graph must stay symmetric (#168)."""
    C = _synthetic_features()
    L = compute_laplacian(C, C, config)

    npt.assert_allclose(L, L.T, atol=1e-12)
    evals = scipy.linalg.eigvalsh(L)
    # Spectrum of a normalized Laplacian lies in [0, 2]
    assert evals.min() > -1e-10
    assert evals.max() < 2 + 1e-10
