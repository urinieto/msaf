#!/usr/bin/python
#
# Copyright (C) Christian Thurau, 2010.
# Licensed under the GNU General Public License (GPL).
# http://www.gnu.org/licenses/gpl.txt
"""
PyMF Compact Matrix Decomposition [1]

    CMD(CUR):  Class for Compact Matrix Decomposition

[1] Sun, J., Xie, Y., Zhang, H. and Faloutsos, C. (2007), Less is More: Compact Matrix Decomposition for Large
Sparse Graphs, in Proc. SIAM Int. Conf. on Data Mining.
"""


import numpy as np
from .cur import CUR

__all__ = ["CMD"]

class CMD(CUR):
    """
    CMD(data, rrank=0, crank=0)


    Compact Matrix Decomposition. Factorize a data matrix into three matrices s.t.
    F = | data - USV| is minimal. CMD randomly selects rows and columns from
    data for building U and V, respectively.

    Parameters
    ----------
    data : array_like [data_dimension x num_samples]
        the input data
    rrank: int, optional
        Number of rows to sample from data. Double entries are eliminiated s.t.
        the resulting rank might be lower.
        4 (default)
    crank: int, optional
        Number of columns to sample from data. Double entries are eliminiated s.t.
        the resulting rank might be lower.
        4 (default)

    Attributes
    ----------
        U,S,V : submatrices s.t. data = USV

    Example
    -------
    >>> import numpy as np
    >>> from cmd import CMD
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    >>> cmd_mdl = CMD(data, show_progress=False, rrank=1, crank=2)
    >>> cmd_mdl.factorize()
    """

    def _cmdinit(self):
        nrids = np.unique(self._rid)
        ncids = np.unique(self._cid)

        self._rcnt = np.zeros(len(nrids))
        self._ccnt = np.zeros(len(ncids))

        for i,idx in enumerate(nrids):
            self._rcnt[i] = len(np.where(self._rid == idx)[0])

        for i,idx in enumerate(ncids):
            self._ccnt[i] = len(np.where(self._cid == idx)[0])

        self._rid = np.int32(list(nrids))
        self._cid = np.int32(list(ncids))

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

        self._cmdinit()

        self.computeUCR()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
