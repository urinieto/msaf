#!/usr/bin/python2.6
#
# Copyright (C) Christian Thurau, 2010.
# Licensed under the GNU General Public License (GPL).
# http://www.gnu.org/licenses/gpl.txt
"""
PyMF Simplex Volume Maximization [1]

    SIVM_SGREEDY: class for greedy-search SiVM

[1] C. Thurau, K. Kersting, and C. Bauckhage. Yes We Can - Simplex Volume
Maximization for Descriptive Web-Scale Matrix Factorization. In Proc. Int.
Conf. on Information and Knowledge Management. ACM. 2010.
"""


import numpy as np
import time

from .dist import *
from .vol import *
from .sivm_search import SIVM_SEARCH

__all__ = ["SIVM_SGREEDY"]

class SIVM_SGREEDY(SIVM_SEARCH):
    """
    SIVM(data, num_bases=4, niter=100, show_progress=True, compW=True)


    Simplex Volume Maximization. Factorize a data matrix into two matrices s.t.
    F = | data - W*H | is minimal. H is restricted to convexity. W is iteratively
    found by maximizing the volume of the resulting simplex (see [1]). A solution
    is found by employing a simple greedy max-vol strategy.

    Parameters
    ----------
    data : array_like
        the input data
    num_bases: int, optional
        Number of bases to compute (column rank of W and row rank of H).
        4 (default)
    niter: int, optional
        Number of iterations of the alternating optimization.
        100 (default)
    show_progress: bool, optional
        Print some extra information
        False (default)
    compW: bool, optional
        Compute W (True) or only H (False). Useful for using basis vectors
        from another convexity constrained matrix factorization function
        (e.g. svmnmf) (if set to "True" niter can be set to "1")
    compH: bool, optional
        Compute H (True) or only H (False). Useful for using precomputed
        basis vectors.
    dist_measure: string, optional
        The distance measure for finding the next best candidate that
        maximizes the simplex volume ['l2','l1','cosine','sparse_graph_l2']
        'l2' (default)
    optimize_lower_bound: bool, optional
        Use the alternative selection criterion that optimizes the lower
        bound (see [1])
        False (default)

    Attributes
    ----------
        W : "data_dimension x num_bases" matrix of basis vectors
        H : "num bases x num_samples" matrix of coefficients

        ferr : frobenius norm (after applying .factoriz())

    Example
    -------
    Applying SIVM to some rather stupid data set:

    >>> import numpy as np
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    >>> sivm_mdl = SIVM_SGREEDY(data, num_bases=2, niter=10)
    >>> sivm_mdl.initialization()
    >>> sivm_mdl.factorize()

    The basis vectors are now stored in sivm_mdl.W, the coefficients in sivm_mdl.H.
    To compute coefficients for an existing set of basis vectors simply    copy W
    to sivm_mdl.W, and set compW to False:

    >>> data = np.array([[1.5, 1.3], [1.2, 0.3]])
    >>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> sivm_mdl = SIVM_SGREEDY(data, num_bases=2, niter=1, compW=False)
    >>> sivm_mdl.initialization()
    >>> sivm_mdl.W = W
    >>> sivm_mdl.factorize()

    The result is a set of coefficients sivm_mdl.H, s.t. data = W * sivm_mdl.H.
    """

    def update_w(self):
        # compute distance matrix -> requiresd for the volume
        self.init_sivm()
        next_sel = list([self.select[0]])
        self.select = []

        self._v = []
        self._t = []
        stime = time.time()

        for iter in range(self._num_bases-1):
            # add new selections to openset
            next_sel = list(np.sort(next_sel))
            D = pdist(self.data[:, next_sel], self.data[:, next_sel])
            V = np.zeros(self.data.shape[1])
            d = np.zeros((D.shape[0]+1,D.shape[1]+1))
            d[:D.shape[0], :D.shape[1]] = D[:,:]

            for i in range(self.data.shape[1]):
                # create a temp selection
                dtmp = l2_distance(self.data[:,next_sel], self.data[:,i:i+1])
                d[:-1,-1] = dtmp
                d[-1,:-1] = dtmp
                # compute volume for temp selection
                V[i] = cmdet(d)

            next_index = np.argmax(V)
            next_sel.append(next_index)
            self._v.append(np.max(V))

            self._logger.info('Iter:' + str(iter))
            self._logger.info('Current selection:' + str(next_sel))
            self._logger.info('Current volume:' + str(self._v[-1]))
            self._t.append(time.time() - stime)

        # update some values ...
        self.select = list(next_sel)
        self.W = self.data[:, self.select]



if __name__ == "__main__":
    import doctest
    doctest.testmod()
