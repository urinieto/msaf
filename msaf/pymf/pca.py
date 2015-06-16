#!/usr/bin/python
#
# Copyright (C) Christian Thurau, 2010.
# Licensed under the GNU General Public License (GPL).
# http://www.gnu.org/licenses/gpl.txt
"""
PyMF Principal Component Analysis.

    PCA: Class for Principal Component Analysis
"""



import numpy as np

from .nmf import NMF
from .svd import SVD


__all__ = ["PCA"]

class PCA(NMF):
    """
    PCA(data, num_bases=4, center_mean=True)


    Archetypal Analysis. Factorize a data matrix into two matrices s.t.
    F = | data - W*H | is minimal. W is set to the eigenvectors of the
    data covariance.

    Parameters
    ----------
    data : array_like, shape (_data_dimension, _num_samples)
        the input data
    num_bases: int, optional
        Number of bases to compute (column rank of W and row rank of H).
        4 (default)
    center_mean: bool, True
        Make sure that the data is centred around the mean.

    Attributes
    ----------
    W : "data_dimension x num_bases" matrix of basis vectors
    H : "num bases x num_samples" matrix of coefficients
    ferr : frobenius norm (after calling .factorize())

    Example
    -------
    Applying PCA to some rather stupid data set:

    >>> import numpy as np
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    >>> pca_mdl = PCA(data, num_bases=2)
    >>> pca_mdl.factorize()

    The basis vectors are now stored in pca_mdl.W, the coefficients in pca_mdl.H.
    To compute coefficients for an existing set of basis vectors simply    copy W
    to pca_mdl.W, and set compute_w to False:

    >>> data = np.array([[1.5], [1.2]])
    >>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> pca_mdl = PCA(data, num_bases=2)
    >>> pca_mdl.W = W
    >>> pca_mdl.factorize(compute_w=False)

    The result is a set of coefficients pca_mdl.H, s.t. data = W * pca_mdl.H.
    """

    def __init__(self, data, num_bases=0, center_mean=True):

        NMF.__init__(self, data, num_bases=num_bases)

        # center the data around the mean first
        self._center_mean = center_mean

        if self._center_mean:
            # copy the data before centering it
            self._data_orig = data
            self._meanv = self._data_orig[:,:].mean(axis=1).reshape(data.shape[0],-1)
            self.data = self._data_orig -  self._meanv
        else:
            self.data = data

    def init_h(self):
        pass

    def init_w(self):
        pass

    def update_h(self):
        self.H = np.dot(self.W.T, self.data[:,:])

    def update_w(self):
        # compute eigenvectors and eigenvalues using SVD
        svd_mdl = SVD(self.data)
        svd_mdl.factorize()

        # argsort sorts in ascending order -> do reverese indexing
        # for accesing values in descending order
        S = np.diag(svd_mdl.S)
        order = np.argsort(S)[::-1]

        # select only a few eigenvectors  ...
        if self._num_bases >0:
            order = order[:self._num_bases]

        self.W = svd_mdl.U[:,order]
        self.eigenvalues =  S[order]

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

        NMF.factorize(self, niter=1, show_progress=show_progress,
                  compute_w=compute_w, compute_h=compute_h,
                  compute_err=compute_err)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
