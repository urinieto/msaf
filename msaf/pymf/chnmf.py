#!/usr/bin/python
#
# Copyright (C) Christian Thurau, 2010.
# Licensed under the GNU General Public License (GPL).
# http://www.gnu.org/licenses/gpl.txt
"""
PyMF Convex Hull Non-negative Matrix Factorization [1]

    CHNMF(NMF) : Class for Convex-hull NMF
    quickhull : Function for finding the convex hull in 2D

[1] C. Thurau, K. Kersting, and C. Bauckhage. Convex Non-Negative Matrix
Factorization in the Wild. ICDM 2009.
"""


import numpy as np

from itertools import combinations
from .dist import vq
from .pca import PCA
from .aa import AA

__all__ = ["CHNMF"]


def quickhull(sample):
    """ Find data points on the convex hull of a supplied data set

    Args:
        sample: data points as column vectors n x d
                    n - number samples
                    d - data dimension (should be two)

    Returns:
        a k x d matrix containint the convex hull data points
    """

    link = lambda a, b: np.concatenate((a, b[1:]))
    edge = lambda a, b: np.concatenate(([a], [b]))

    def dome(sample, base):
        h, t = base
        dists = np.dot(sample - h, np.dot(((0, -1), (1, 0)), (t - h)))
        outer = np.repeat(sample, dists > 0, axis=0)

        if len(outer):
            pivot = sample[np.argmax(dists)]
            return link(dome(outer, edge(h, pivot)),
                dome(outer, edge(pivot, t)))
        else:
            return base

    if len(sample) > 2:
        axis = sample[:, 0]
        base = np.take(sample, [np.argmin(axis), np.argmax(axis)], axis=0)
        return link(dome(sample, base),
            dome(sample, base[::-1]))
    else:
        return sample

class CHNMF(AA):
    """
    CHNMF(data, num_bases=4)

    Convex Hull Non-negative Matrix Factorization. Factorize a data matrix into
    two matrices s.t. F = | data - W*H | is minimal. H is restricted to convexity
    (H >=0, sum(H, axis=1) = [1 .. 1]) and W resides on actual data points.
    Factorization is solved via an alternating least squares optimization using
    the quadratic programming solver from cvxopt. The results are usually
    equivalent to Archetypal Analysis (pymf.AA) but CHNMF also works for very
    large datasets.

    Parameters
    ----------
    data : array_like, shape (_data_dimension, _num_samples)
        the input data
    num_bases: int, optional
        Number of bases to compute (column rank of W and row rank of H).
        4 (default)
    base_sel: int,
        Number of pairwise basis vector projections. Set to a value< rank(data).
        Computation time scale exponentially with this value, usually rather low
        values are sufficient (3-10).

    Attributes
    ----------
        W : "data_dimension x num_bases" matrix of basis vectors
        H : "num bases x num_samples" matrix of coefficients
        ferr : frobenius norm (after calling .factorize())

    Example
    -------
    Applying CHNMF to some rather stupid data set:

    >>> import numpy as np
    >>> from chnmf import CHNMF
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])

    Use 2 basis vectors -> W shape(data_dimension, 2).

    >>> chnmf_mdl = CHNMF(data, num_bases=2)

    And start computing the factorization.

    >>> chnmf_mdl.factorize()

    The basis vectors are now stored in chnmf_mdl.W, the coefficients in
    chnmf_mdl.H. To compute coefficients for an existing set of basis vectors
    simply copy W to chnmf_mdl.W, and set compute_w to False:

    >>> data = np.array([[1.5, 2.0], [1.2, 1.8]])
    >>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> chnmf_mdl = CHNMF(data, num_bases=2)
    >>> chnmf_mdl.W = W
    >>> chnmf_mdl.factorize(compute_w=False)

    The result is a set of coefficients chnmf_mdl.H, s.t. data = W * chnmf_mdl.H.
    """

    def __init__(self, data, num_bases=4, base_sel=3):

        # call inherited method
        AA.__init__(self, data, num_bases=num_bases)

        # base sel should never be larger than the actual data dimension
        self._base_sel = base_sel
        if base_sel > self.data.shape[0]:
            self._base_sel = self.data.shape[0]

    def init_h(self):
        self.H = np.zeros((self._num_bases, self._num_samples))

    def init_w(self):
        self.W = np.zeros((self._data_dimension, self._num_bases))

    def _map_w_to_data(self):
        """ Return data points that are most similar to basis vectors W
        """

        # assign W to the next best data sample
        self._Wmapped_index = vq(self.data, self.W)
        self.Wmapped = np.zeros(self.W.shape)

        # do not directly assign, i.e. Wdist = self.data[:,sel]
        # as self might be unsorted (in non ascending order)
        # -> sorting sel would screw the matching to W if
        # self.data is stored as a hdf5 table (see h5py)
        for i, s in enumerate(self._Wmapped_index):
            self.Wmapped[:,i] = self.data[:,s]

    def update_w(self):
        """ compute new W """
        def select_hull_points(data, n=3):
            """ select data points for pairwise projections of the first n
            dimensions """

            # iterate over all projections and select data points
            idx = np.array([])

            # iterate over some pairwise combinations of dimensions
            for i in combinations(range(n), 2):
                # sample convex hull points in 2D projection
                convex_hull_d = quickhull(data[i, :].T)

                # get indices for convex hull data points
                idx = np.append(idx, vq(data[i, :], convex_hull_d.T))
                idx = np.unique(idx)

            return np.int32(idx)

        # determine convex hull data points using either PCA or random
        # projections
        method = 'randomprojection'
        if method == 'pca':
            pcamodel = PCA(self.data)
            pcamodel.factorize(show_progress=False)
            proj = pcamodel.H
        else:
            R = np.random.randn(self._base_sel, self._data_dimension)
            proj = np.dot(R, self.data)

        self._hull_idx = select_hull_points(proj, n=self._base_sel)
        aa_mdl = AA(self.data[:, self._hull_idx], num_bases=self._num_bases)

        # determine W
        aa_mdl.factorize(niter=50, compute_h=True, compute_w=True,
                         compute_err=True, show_progress=False)

        self.W = aa_mdl.W
        self._map_w_to_data()

    def factorize(self, show_progress=False, compute_w=True, compute_h=True,
                  compute_err=True, niter=1):
        """ Factorize s.t. WH = data

            Parameters
            ----------
            show_progress : bool
                    print some extra information to stdout.
            compute_h : bool
                    iteratively update values for H.
            compute_w : bool
                    iteratively update values for W.
            compute_err : bool
                    compute Frobenius norm |data-WH| after each update and store
                    it to .ferr[k].

            Updated Values
            --------------
            .W : updated values for W.
            .H : updated values for H.
            .ferr : Frobenius norm |data-WH|.
        """

        AA.factorize(self, niter=1, show_progress=show_progress,
                  compute_w=compute_w, compute_h=compute_h,
                  compute_err=compute_err)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
