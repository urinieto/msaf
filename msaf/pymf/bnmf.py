#!/usr/bin/python
#
# Copyright (C) Christian Thurau, 2010.
# Licensed under the GNU General Public License (GPL).
# http://www.gnu.org/licenses/gpl.txt
"""
PyMF Binary Matrix Factorization [1]

    BNMF(NMF) : Class for binary matrix factorization

[1]Z. Zhang, T. Li, C. H. Q. Ding, X. Zhang: Binary Matrix Factorization with
Applications. ICDM 2007
"""


import numpy as np
from .nmf import NMF

__all__ = ["BNMF"]

class BNMF(NMF):
    """
    BNMF(data, data, num_bases=4)
    Binary Matrix Factorization. Factorize a data matrix into two matrices s.t.
    F = | data - W*H | is minimal. H and W are restricted to binary values.

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
    Applying BNMF to some rather stupid data set:

    >>> import numpy as np
    >>> from bnmf import BNMF
    >>> data = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])

    Use 2 basis vectors -> W shape(data_dimension, 2).

    >>> bnmf_mdl = BNMF(data, num_bases=2)

    Set number of iterations to 5 and start computing the factorization.

    >>> bnmf_mdl.factorize(niter=5)

    The basis vectors are now stored in bnmf_mdl.W, the coefficients in bnmf_mdl.H.
    To compute coefficients for an existing set of basis vectors simply copy W
    to bnmf_mdl.W, and set compute_w to False:

    >>> data = np.array([[0.0], [1.0]])
    >>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> bnmf_mdl = BNMF(data, num_bases=2)
    >>> bnmf_mdl.W = W
    >>> bnmf_mdl.factorize(niter=10, compute_w=False)

    The result is a set of coefficients bnmf_mdl.H, s.t. data = W * bnmf_mdl.H.
    """

    # controls how fast lambda should increase:
    # this influence convergence to binary values during the update. A value
    # <1 will result in non-binary decompositions as the update rule effectively
    # is a conventional nmf update rule. Values >1 give more weight to making the
    # factorization binary with increasing iterations.
    # setting either W or H to 0 results make the resulting matrix non binary.
    _LAMB_INCREASE_W = 1.1
    _LAMB_INCREASE_H = 1.1

    def update_h(self):
        H1 = np.dot(self.W.T, self.data[:,:]) + 3.0*self._lamb_H*(self.H**2)
        H2 = np.dot(np.dot(self.W.T,self.W), self.H) + 2*self._lamb_H*(self.H**3) + self._lamb_H*self.H + 10**-9
        self.H *= H1/H2

        self._lamb_W = self._LAMB_INCREASE_W * self._lamb_W
        self._lamb_H = self._LAMB_INCREASE_H * self._lamb_H

    def update_w(self):
        W1 = np.dot(self.data[:,:], self.H.T) + 3.0*self._lamb_W*(self.W**2)
        W2 = np.dot(self.W, np.dot(self.H, self.H.T)) + 2.0*self._lamb_W*(self.W**3) + self._lamb_W*self.W  + 10**-9
        self.W *= W1/W2

    def factorize(self, niter=10, compute_w=True, compute_h=True,
                  show_progress=False, compute_err=True):
        """ Factorize s.t. WH = data

            Parameters
            ----------
            niter : int
                    number of iterations.
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
            .ferr : Frobenius norm |data-WH| for each iteration.
        """

        # init some learning parameters
        self._lamb_W = 1.0/niter
        self._lamb_H = 1.0/niter

        NMF.factorize(self, niter=niter, compute_w=compute_w,
                      compute_h=compute_h, show_progress=show_progress,
                      compute_err=compute_err)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
