#!/usr/bin/python
#
# Copyright (C) Christian Thurau, 2010.
# Licensed under the GNU General Public License (GPL).
# http://www.gnu.org/licenses/gpl.txt
"""
PyMF K-means clustering (unary-convex matrix factorization).
"""


import numpy as np
import random

from . import dist
from .nmf import NMF

__all__ = ["Kmeans"]

class Kmeans(NMF):
    """
    Kmeans(data, num_bases=4)

    K-means clustering. Factorize a data matrix into two matrices s.t.
    F = | data - W*H | is minimal. H is restricted to unary vectors, W
    is simply the mean over the corresponding samples in "data".

    Parameters
    ----------
    data : array_like, shape (_data_dimension, _num_samples)
        the input data
    num_bases: int, optional
        Number of bases to compute (column rank of W and row rank of H).
        4 (default)

    Attributes
    ----------
    W : "data_dimension x num_bases" matrix of basis vectors
    H : "num bases x num_samples" matrix of coefficients
    ferr : frobenius norm (after calling .factorize())

    Example
    -------
    Applying K-means to some rather stupid data set:

    >>> import numpy as np
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    >>> kmeans_mdl = Kmeans(data, num_bases=2)
    >>> kmeans_mdl.factorize(niter=10)

    The basis vectors are now stored in kmeans_mdl.W, the coefficients in kmeans_mdl.H.
    To compute coefficients for an existing set of basis vectors simply    copy W
    to kmeans_mdl.W, and set compute_w to False:

    >>> data = np.array([[1.5], [1.2]])
    >>> W = [[1.0, 0.0], [0.0, 1.0]]
    >>> kmeans_mdl = Kmeans(data, num_bases=2)
    >>> kmeans_mdl.W = W
    >>> kmeans_mdl.factorize(niter=1, compute_w=False)

    The result is a set of coefficients kmeans_mdl.H, s.t. data = W * kmeans_mdl.H.
    """
    def init_h(self):
        # W has to be present for H to be initialized
        self.H = np.zeros((self._num_bases, self._num_samples))
        self.update_h()

    def init_w(self):
        # set W to some random data samples
        sel = random.sample(range(self._num_samples), self._num_bases)

        # sort indices, otherwise h5py won't work
        self.W = self.data[:, np.sort(sel)]


    def update_h(self):
        # and assign samples to the best matching centers
        self.assigned = dist.vq(self.W, self.data)
        self.H = np.zeros(self.H.shape)
        self.H[self.assigned, range(self._num_samples)] = 1.0


    def update_w(self):
        for i in range(self._num_bases):
            idx = np.where(self.assigned==i)[0]
            n = len(idx)
            if n > 1:
                self.W[:,i] = np.sum(self.data[:,idx], axis=1)/n
