#!/usr/bin/python
#
# Copyright (C) Christian Thurau, 2010.
# Licensed under the GNU General Public License (GPL).
# http://www.gnu.org/licenses/gpl.txt
"""
PyMF Matrix sampling methods

    SUB: apply one of the matrix factorization methods of PyMF
         on sampled data for computing W, then compute H.

Copyright (C) Christian Thurau, 2010. GNU General Public License (GPL).
"""



import numpy as np
import random
#from itertools import combinations
from .chnmf import combinations

from . import dist
from .chnmf import quickhull
from .nmf import NMF
from .pca import PCA
from .kmeans import Kmeans
from .laesa import LAESA
from .sivm import SIVM

__all__ = ["SUB"]

class SUB(NMF):
    """
    SUB(data, mfmethod, sstrategy='rand', nsub=20, show_progress=True, mapW=False,
    base_sel=2,    num_bases=3 , niterH=1, niter=100, compute_h=True, compute_w=True, )

    Evaluate a matrix factorization method "mfmethod" for a certain sampling
    strategy "sstrategy". This is particular useful for very large datasets.

    Parameters
    ----------
    todo ...

    Attributes
    ----------
    todo ....
    """

    def __init__(self, data, mfmethod, nsub=20, show_progress=True, mapW=False, base_sel=2,
                num_bases=3 , niterH=1, compute_h=True, compute_w=True, sstrategy='rand'):
        NMF.__init__(self, data, num_bases=num_bases, compute_h=compute_h, show_progress=show_progress, compute_w=compute_w)

        self._niterH = niterH
        self._nsub = nsub
        self.data = data
        self._mfmethod = mfmethod
        self._mapW = mapW
        self._sstrategy = sstrategy
        self._base_sel = base_sel

        # assign the correct distance function
        if self._sstrategy == 'cur':
            self._subfunc = self.curselect

        elif self._sstrategy == 'kmeans':
            self._subfunc = self.kmeansselect

        elif self._sstrategy == 'hull':
            self._subfunc = self.hullselect

        elif self._sstrategy == 'laesa':
            self._subfunc = self.laesaselect

        elif self._sstrategy == 'sivm':
            self._subfunc = self.sivmselect

        else:
            self._subfunc = self.randselect

    def hullselect(self):

        def selectHullPoints(data, n=20):
            """ select data points for pairwise projections of the first n
            dimensions """

            # iterate over all projections and select data points
            idx = np.array([])

            # iterate over some pairwise combinations of dimensions
            for i in combinations(range(n), 2):

                # sample convex hull points in 2D projection
                convex_hull_d = quickhull(data[i, :].T)

                # get indices for convex hull data points
                idx = np.append(idx, dist.vq(data[i, :], convex_hull_d.T))
                idx = np.unique(idx)

            return np.int32(idx)


        # determine convex hull data points only if the total
        # amount of available data is >50
        #if self.data.shape[1] > 50:
        pcamodel = PCA(self.data, show_progress=self._show_progress)
        pcamodel.factorize()

        idx = selectHullPoints(pcamodel.H, n=self._base_sel)

        # set the number of subsampled data
        self.nsub = len(idx)

        return idx

    def kmeansselect(self):
            kmeans_mdl = Kmeans(self.data, num_bases=self._nsub)
            kmeans_mdl.initialization()
            kmeans_mdl.factorize()

            # pick data samples closest to the centres
            idx = dist.vq(kmeans_mdl.data, kmeans_mdl.W)
            return idx

    def curselect(self):
        def sample_probability():
            dsquare = self.data[:,:]**2

            pcol = np.array(dsquare.sum(axis=0))
            pcol /= pcol.sum()

            return (pcol.reshape(-1,1))

        probs = sample_probability()
        prob_cols = np.cumsum(probs.flatten()) #.flatten()
        temp_ind = np.zeros(self._nsub, np.int32)

        for i in range(self._nsub):
            tempI = np.where(prob_cols >= np.random.rand())[0]
            temp_ind[i] = tempI[0]

        return np.sort(temp_ind)

    def sivmselect(self):
        sivmmdl = SIVM(self.data, num_bases=self._nsub, compute_w=True, compute_h=False, dist_measure='cosine')

        sivmmdl.initialization()
        sivmmdl.factorize()
        idx = sivmmdl.select
        return idx

    def laesaselect(self):
        laesamdl = LAESA(self.data, num_bases=self._nsub, compute_w=True, compute_h=False, dist_measure='cosine')
        laesamdl.initialization()
        laesamdl.factorize()
        idx = laesamdl.select
        return idx


    def randselect(self):
        idx = random.sample(range(self._num_samples), self._nsub)
        return np.sort(np.int32(idx))

    def update_w(self):

        idx = self._subfunc()
        idx = np.sort(np.int32(idx))


        mdl_small = self._mfmethod(self.data[:, idx],
                                num_bases=self._num_bases,
                                show_progress=self._show_progress,
                                compute_w=True)

        # initialize W, H, and beta
        mdl_small.initialization()

        # determine W
        mdl_small.factorize()


        self.mdl = self._mfmethod(self.data[:, :],
                                    num_bases=self._num_bases ,
                                    show_progress=self._show_progress,
                                    compute_w=False)


        self.mdl.initialization()

        if self._mapW:
            # compute pairwise distances
            #distance = vq(self.data, self.W)
            _Wmapped_index = dist.vq(self.mdl.data, mdl_small.W)

            # do not directly assign, i.e. Wdist = self.data[:,sel]
            # as self might be unsorted (in non ascending order)
            # -> sorting sel would screw the matching to W if
            # self.data is stored as a hdf5 table (see h5py)
            for i,s in enumerate(_Wmapped_index):
                self.mdl.W[:,i] = self.mdl.data[:,s]
        else:
            self.mdl.W = np.copy(mdl_small.W)

    def update_h(self):
        self.mdl.factorize()

    def factorize(self):
        """Do factorization s.t. data = dot(dot(data,beta),H), under the convexity constraint
            beta >=0, sum(beta)=1, H >=0, sum(H)=1
        """
        # compute new coefficients for reconstructing data points
        self.update_w()

        # for CHNMF it is sometimes useful to only compute
        # the basis vectors
        if self._compute_h:
            self.update_h()

        self.W = self.mdl.W
        self.H = self.mdl.H

        self.ferr = np.zeros(1)
        self.ferr[0] = self.mdl.frobenius_norm()
        self._print_cur_status(' Fro:' + str(self.ferr[0]))

if __name__ == "__main__":
    import doctest
    doctest.testmod()
