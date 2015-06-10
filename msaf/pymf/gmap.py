#!/usr/bin/python
#
# Copyright (C) Christian Thurau, 2010.
# Licensed under the GNU General Public License (GPL).
# http://www.gnu.org/licenses/gpl.txt
"""
PyMF Geometric-Map

    GMAP: Class for Geometric-Map
"""


import scipy.sparse
import numpy as np

from .dist import *
from .aa import AA
from .kmeans import Kmeans

__all__ = ["GMAP"]

class GMAP(AA):
    """
    GMAP(data, num_bases=4, dist_measure='l2')


    Geometric-Map. Factorize a data matrix into two matrices s.t.
    F = | data - W*H | is minimal. G-MAP can emulate/approximate several
    standard methods including PCA, NMF, and AA.

    Parameters
    ----------
    data : array_like, shape (_data_dimension, _num_samples)
        the input data
    num_bases: int, optional
        Number of bases to compute (column rank of W and row rank of H).
        4 (default)
    method : one of 'pca' ,'nmf', 'aa', default is 'pca' which emulates
        Principal Component Analysis using the geometric map method ('nmf'
        emulates Non-negative Matrix Factorization, 'aa' emulates Archetypal
        Analysis).
    robust_map : bool, optional
        use robust_map or the standard max-val selection
        [see "On FastMap and the Convex Hull of Multivariate Data: Toward
        Fast and Robust Dimension Reduction", Ostrouchov and Samatova, PAMI
        2005]
    Attributes
    ----------
    W : "data_dimension x num_bases" matrix of basis vectors
    H : "num bases x num_samples" matrix of coefficients
    ferr : frobenius norm (after calling .factorize())

    Example
    -------
    Applying GMAP to some rather stupid data set:

    >>> import numpy as np
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    >>> gmap_mdl = GMAP(data, num_bases=2)
    >>> gmap_mdl.factorize()

    The basis vectors are now stored in gmap_mdl.W, the coefficients in gmap_mdl.H.
    To compute coefficients for an existing set of basis vectors simply copy W
    to gmap_mdl.W, and set compute_w to False:

    >>> data = np.array([[1.5, 1.3], [1.2, 0.3]])
    >>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> gmap_mdl = GMAP(data, num_bases=2)
    >>> gmap_mdl.W = W
    >>> gmap_mdl.factorize(compute_w=False)

    The result is a set of coefficients gmap_mdl.H, s.t. data = W * gmap_mdl.H.
    """

    # always overwrite the default number of iterations
    # -> any value other does not make sense.
    _NITER = 1

    def __init__(self, data, num_bases=4, method='pca', robust_map=True):

        AA.__init__(self, data, num_bases=num_bases)
        self.sub = []
        self._robust_map = robust_map
        self._method = method


    def init_h(self):
        self.H = np.zeros((self._num_bases, self._num_samples))

    def init_w(self):
        self.W = np.zeros((self._data_dimension, self._num_bases))

    def update_w(self):
        """ compute new W """

        def select_next(iterval):
            """ select the next best data sample using robust map
            or simply the max iterval ... """

            if self._robust_map:
                k = np.argsort(iterval)[::-1]
                d_sub = self.data[:,k[:self._robust_nselect]]
                self.sub.extend(k[:self._robust_nselect])

                # cluster d_sub
                kmeans_mdl = Kmeans(d_sub, num_bases=self._robust_cluster)
                kmeans_mdl.factorize(niter=10)

                # get largest cluster
                h = np.histogram(kmeans_mdl.assigned, range(self._robust_cluster+1))[0]
                largest_cluster = np.argmax(h)
                sel = pdist(kmeans_mdl.W[:, largest_cluster:largest_cluster+1], d_sub)
                sel = k[np.argmin(sel)]
            else:
                sel = np.argmax(iterval)

            return sel

        EPS = 10**-8

        if scipy.sparse.issparse(self.data):
            norm_data = np.sqrt(self.data.multiply(self.data).sum(axis=0))
            norm_data = np.array(norm_data).reshape((-1))
        else:
            norm_data = np.sqrt(np.sum(self.data**2, axis=0))


        self.select = []

        if self._method == 'pca' or self._method == 'aa':
            iterval = norm_data.copy()

        if self._method == 'nmf':
            iterval = np.sum(self.data, axis=0)/(np.sqrt(self.data.shape[0])*norm_data)
            iterval = 1.0 - iterval

        self.select.append(select_next(iterval))


        for l in range(1, self._num_bases):

            if scipy.sparse.issparse(self.data):
                c = self.data[:, self.select[-1]:self.select[-1]+1].T * self.data
                c = np.array(c.todense())
            else:
                c = np.dot(self.data[:,self.select[-1]], self.data)

            c = c/(norm_data * norm_data[self.select[-1]])

            if self._method == 'pca':
                c = 1.0 - np.abs(c)
                c = c * norm_data

            elif self._method == 'aa':
                c = (c*-1.0 + 1.0)/2.0
                c = c * norm_data

            elif self._method == 'nmf':
                c = 1.0 - np.abs(c)

            ### update the estimated volume
            iterval = c * iterval

            # detect the next best data point
            self.select.append(select_next(iterval))

            self._logger.info('cur_nodes: ' + str(self.select))

        # sort indices, otherwise h5py won't work
        self.W = self.data[:, np.sort(self.select)]

        # "unsort" it again to keep the correct order
        self.W = self.W[:, np.argsort(np.argsort(self.select))]

    def factorize(self, show_progress=False, compute_w=True, compute_h=True,
                  compute_err=True, robust_cluster=3, niter=1, robust_nselect=-1):
        """ Factorize s.t. WH = data

            Parameters
            ----------
            show_progress : bool
                    print some extra information to stdout.
                    False, default
            compute_h : bool
                    iteratively update values for H.
                    True, default
            compute_w : bool
                    iteratively update values for W.
                    default, True
            compute_err : bool
                    compute Frobenius norm |data-WH| after each update and store
                    it to .ferr[k].
            robust_cluster : int, optional
                    set the number of clusters for robust map selection.
                    3, default
            robust_nselect : int, optional
                    set the number of samples to consider for robust map
                    selection.
                    -1, default (automatically determine suitable number)

            Updated Values
            --------------
            .W : updated values for W.
            .H : updated values for H.
            .ferr : Frobenius norm |data-WH|.
        """
        self._robust_cluster = robust_cluster
        self._robust_nselect = robust_nselect

        if self._robust_nselect == -1:
            self._robust_nselect = np.round(np.log(self.data.shape[1])*2)

        AA.factorize(self, niter=1, show_progress=show_progress,
                  compute_w=compute_w, compute_h=compute_h,
                  compute_err=compute_err)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
