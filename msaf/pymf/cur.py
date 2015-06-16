#!/usr/bin/python
#
# Copyright (C) Christian Thurau, 2010.
# Licensed under the GNU General Public License (GPL).
# http://www.gnu.org/licenses/gpl.txt
"""
PyMF CUR Decomposition [1]

    CUR(SVD) : Class for CUR Decomposition

[1] Drineas, P., Kannan, R. and Mahoney, M. (2006), 'Fast Monte Carlo Algorithms III: Computing
a Compressed Approixmate Matrix Decomposition', SIAM J. Computing 36(1), 184-206.
"""


import numpy as np
import scipy.sparse

from .svd import pinv, SVD


__all__ = ["CUR"]

class CUR(SVD):
    """
    CUR(data,  data, k=-1, rrank=0, crank=0)

    CUR Decomposition. Factorize a data matrix into three matrices s.t.
    F = | data - USV| is minimal. CUR randomly selects rows and columns from
    data for building U and V, respectively.

    Parameters
    ----------
    data : array_like [data_dimension x num_samples]
        the input data
    rrank: int, optional
        Number of rows to sample from data.
        4 (default)
    crank: int, optional
        Number of columns to sample from data.
        4 (default)
    show_progress: bool, optional
        Print some extra information
        False (default)

    Attributes
    ----------
        U,S,V : submatrices s.t. data = USV

    Example
    -------
    >>> import numpy as np
    >>> from cur import CUR
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    >>> cur_mdl = CUR(data, show_progress=False, rrank=1, crank=2)
    >>> cur_mdl.factorize()
    """

    def __init__(self, data, k=-1, rrank=0, crank=0):
        SVD.__init__(self, data,k=k,rrank=rrank, crank=rrank)

        # select all data samples for computing the error:
        # note that this might take very long, adjust self._rset and self._cset
        # for faster computations.
        self._rset = range(self._rows)
        self._cset = range(self._cols)


    def sample(self, s, probs):
        prob_rows = np.cumsum(probs.flatten())
        temp_ind = np.zeros(s, np.int32)

        for i in range(s):
            v = np.random.rand()

            try:
                tempI = np.where(prob_rows >= v)[0]
                temp_ind[i] = tempI[0]
            except:
                temp_ind[i] = len(prob_rows)

        return np.sort(temp_ind)

    def sample_probability(self):

        if scipy.sparse.issparse(self.data):
            dsquare = self.data.multiply(self.data)
        else:
            dsquare = self.data[:,:]**2

        prow = np.array(dsquare.sum(axis=1), np.float64)
        pcol = np.array(dsquare.sum(axis=0), np.float64)

        prow /= prow.sum()
        pcol /= pcol.sum()

        return (prow.reshape(-1,1), pcol.reshape(-1,1))

    def computeUCR(self):
        # the next  lines do NOT work with h5py if CUR is used -> double indices in self.cid or self.rid
        # can occur and are not supported by h5py. When using h5py data, always use CMD which ignores
        # reoccuring row/column selections.

        if scipy.sparse.issparse(self.data):
            self._C = self.data[:, self._cid] * scipy.sparse.csc_matrix(np.diag(self._ccnt**(1/2)))
            self._R = scipy.sparse.csc_matrix(np.diag(self._rcnt**(1/2))) * self.data[self._rid,:]

            self._U = pinv(self._C, self._k) * self.data[:,:] * pinv(self._R, self._k)

        else:
            self._C = np.dot(self.data[:, self._cid].reshape((self._rows, len(self._cid))), np.diag(self._ccnt**(1/2)))
            self._R = np.dot(np.diag(self._rcnt**(1/2)), self.data[self._rid,:].reshape((len(self._rid), self._cols)))

            self._U = np.dot(np.dot(pinv(self._C, self._k), self.data[:,:]),
                             pinv(self._R, self._k))

        # set some standard (with respect to SVD) variable names
        self.U = self._C
        self.S = self._U
        self.V = self._R

    def factorize(self):
        """ Factorize s.t. CUR = data

            Updated Values
            --------------
            .C : updated values for C.
            .U : updated values for U.
            .R : updated values for R.
        """
        [prow, pcol] = self.sample_probability()
        self._rid = self.sample(self._rrank, prow)
        self._cid = self.sample(self._crank, pcol)

        self._rcnt = np.ones(len(self._rid))
        self._ccnt = np.ones(len(self._cid))

        self.computeUCR()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
