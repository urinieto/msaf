import numpy as np
import numpy.testing as npt

from msaf.algorithms.foote.segmenter import compute_gaussian_krnl


def test_compute_gaussian_krnl():
    """Gaussian-tapered checkerboard kernel (uses scipy.signal.windows)."""
    M = 8
    G = compute_gaussian_krnl(M)
    assert G.shape == (M, M)
    npt.assert_allclose(G, G.T)

    # Checkerboard sign pattern: positive diagonal blocks, negative off-diagonal
    h = M // 2
    assert np.all(G[:h, :h] > 0)
    assert np.all(G[h:, h:] > 0)
    assert np.all(G[h:, :h] < 0)
    assert np.all(G[:h, h:] < 0)
