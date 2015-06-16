#!/usr/bin/python
#
# Copyright (C) Christian Thurau, 2010.
# Licensed under the GNU General Public License (GPL).
# http://www.gnu.org/licenses/gpl.txt
"""
PyMF Semi Non-negative Matrix Factorization.

    SNMF(NMF) : Class for semi non-negative matrix factorization

[1] Ding, C., Li, T. and Jordan, M.. Convex and Semi-Nonnegative Matrix Factorizations.
IEEE Trans. on Pattern Analysis and Machine Intelligence 32(1), 45-55.
"""



import numpy as np

from .nmf import NMF

__all__ = ["SNMF"]

class SNMF(NMF):
    """
    SNMF(data, num_bases=4)

    Semi Non-negative Matrix Factorization. Factorize a data matrix into two
    matrices s.t. F = | data - W*H | is minimal.

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
    Applying Semi-NMF to some rather stupid data set:

    >>> import numpy as np
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    >>> snmf_mdl = SNMF(data, num_bases=2)
    >>> snmf_mdl.factorize(niter=10)

    The basis vectors are now stored in snmf_mdl.W, the coefficients in snmf_mdl.H.
    To compute coefficients for an existing set of basis vectors simply    copy W
    to snmf_mdl.W, and set compute_w to False:

    >>> data = np.array([[1.5], [1.2]])
    >>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> snmf_mdl = SNMF(data, num_bases=2)
    >>> snmf_mdl.W = W
    >>> snmf_mdl.factorize(niter=1, compute_w=False)

    The result is a set of coefficients snmf_mdl.H, s.t. data = W * snmf_mdl.H.
    """


    def update_w(self):
        W1 = np.dot(self.data[:,:], self.H.T)
        W2 = np.dot(self.H, self.H.T)
        self.W = np.dot(W1, np.linalg.inv(W2))

    def update_h(self):
        def separate_positive(m):
            return (np.abs(m) + m)/2.0

        def separate_negative(m):
            return (np.abs(m) - m)/2.0

        XW = np.dot(self.data[:,:].T, self.W)

        WW = np.dot(self.W.T, self.W)
        WW_pos = separate_positive(WW)
        WW_neg = separate_negative(WW)

        XW_pos = separate_positive(XW)
        H1 = (XW_pos + np.dot(self.H.T, WW_neg)).T

        XW_neg = separate_negative(XW)
        H2 = (XW_neg + np.dot(self.H.T,WW_pos)).T + 10**-9

        self.H *= np.sqrt(H1/H2)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
