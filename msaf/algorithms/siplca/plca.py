# Copyright (C) 2009-2010 Ron J. Weiss (ronw@nyu.edu)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""plca: Probabilistic Latent Component Analysis

This module implements a number of variations of the PLCA algorithms
described in [2] and [3] with both Dirichlet and (approximate)
Entropic priors over the parmaters.

PLCA is a variant of non-negative matrix factorization which
decomposes a (2D) probabilitity distribution (arbitrarily normalized
non-negative matrix in the NMF case) V into the product of
distributions over the columns W = {w_k}, rows H = {h_k}, and mixing
weights Z = diag(z_k).  See [1-3] for more details.

References
----------
[1] R. J. Weiss and J. P. Bello. "Identifying Repeated Patterns in
    Music Using Sparse Convolutive Non-Negative Matrix
    Factorization". In Proc. International Conference on Music
    Information Retrieval (ISMIR), 2010.

[2] P. Smaragdis and B. Raj. "Shift-Invariant Probabilistic Latent
    Component Analysis". Technical Report TR2007-009, MERL, December
    2007.

[3] P. Smaragdis, B. Raj, and M. Shashanka. "Sparse and
    shift-invariant feature extraction from non-negative data".  In
    Proc. ICASSP, 2008.

Copyright (C) 2009-2010 Ron J. Weiss <ronw@nyu.edu>

LICENSE: This module is licensed under the GNU GPL. See COPYING for details.
"""

import functools
import logging

import numpy as np
import scipy as sp
import scipy.signal

import matplotlib.pyplot as plt
import plottools

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s %(name)s %(asctime)s '
                    '%(filename)s:%(lineno)d  %(message)s')
logger = logging.getLogger('plca')
EPS = np.finfo(np.float).eps
#EPS = 1e-100

def kldivergence(V, WZH):
    #return np.sum(V * np.log(V / WZH) - V + WZH)
    return np.sum(WZH - V * np.log(WZH))

def normalize(A, axis=None):
    Ashape = A.shape
    try:
        norm = A.sum(axis) + EPS
    except TypeError:
        norm = A.copy()
        for ax in reversed(sorted(axis)):
            norm = norm.sum(ax)
        norm += EPS
    if axis:
        nshape = np.array(Ashape)
        nshape[axis] = 1
        norm.shape = nshape
    return A / norm

def shift(a, shift, axis=None, circular=True):
    """Shift array along a given axis.

    If circular is False, zeros are inserted for elements rolled off
    the end of the array.

    See Also
    --------
    np.roll
    """
    aroll = np.roll(a, shift, axis)
    if not circular and shift != 0:
        if axis is None:
            arollflattened = aroll.flatten()
            if shift > 0:
                arollflattened[:shift] = 0
            elif shift < 0:
                arollflattened[shift:] = 0
            aroll = np.reshape(arollflattened, aroll.shape)
        else:
            index = [slice(None)] * a.ndim
            if shift > 0:
                index[axis] = slice(0, shift)
            elif shift < 0:
                index[axis] = slice(shift, None)
            aroll[index] = 0
    return aroll


class PLCA(object):
    """Probabilistic Latent Component Analysis

    Methods
    -------
    analyze
        Performs PLCA decomposition using the EM algorithm from [2].
    reconstruct(W, Z, H, norm=1.0)
        Reconstructs input matrix from the PLCA parameters W, Z, and H.
    plot(V, W, Z, H)
        Makes a pretty plot of V and the decomposition.

    initialize()
        Randomly initializes the parameters.
    do_estep(W, Z, H)
        Performs the E-step of the EM parameter estimation algorithm.
    do_mstep()
        Performs the M-step of the EM parameter estimation algorithm.

    Notes
    -----
    You probably don't want to initialize this class directly.  Most
    interactions should be through the static methods analyze,
    reconstruct, and plot.

    Subclasses that want to use a similar interface (e.g. SIPLCA)
    should also implement initialize, do_estep, and do_mstep.

    Examples
    --------
    Generate some random data:
    >>> F = 20
    >>> T = 100
    >>> rank = 3
    >>> means = [F/4.0, F/2.0, 3.0*F/4]
    >>> f = np.arange(F)
    >>> trueW = plca.normalize(np.array([np.exp(-(f - m)**2 / F)
    ...                                  for m in means]).T, 0)
    >>> trueZ = np.ones(rank) / rank
    >>> trueH = plca.normalize(np.random.rand(rank, T), 1)
    >>> V = plca.PLCA.reconstruct(trueW, trueZ, trueH)

    Perform the decomposition:
    >>> W, Z, H, norm, recon, logprob = plca.PLCA.analyze(V, rank=rank)
    INFO:plca:Iteration 0: logprob = 8.784769
    INFO:plca:Iteration 50: logprob = 8.450114
    INFO:plca:Iteration 99: final logprob = 8.449504

    Plot the parameters:
    >>> plt.figure(1)
    >>> plca.PLCA.plot(V, W, Z, H)
    >>> plt.figure(2)
    >>> plca.PLCA.plot(V, trueW, trueZ, trueH)

    W, Z, H and trueW, trueZ, trueH should be the same modulo
    permutations along the rank dimension.

    See Also
    --------
    SIPLCA : Shift-Invariant PLCA
    SIPLCA2 : 2D Shift-Invariant PLCA
    """
    def __init__(self, V, rank, alphaW=0, alphaZ=0, alphaH=0,
                 betaW=0, betaZ=0, betaH=0, nu=50.0, minpruneiter=0, **kwargs):
        """
        Parameters
        ----------
        V : array, shape (`F`, `T`)
            Matrix to analyze.
        rank : int
            Rank of the decomposition (i.e. number of columns of `W`
            and rows of `H`).
        alphaW, alphaZ, alphaH : float or appropriately shaped array
            Dirichlet prior parameters for `W`, `Z`, and `H`.
            Negative values lead to sparser distributions, positive
            values makes the distributions more uniform.  Defaults to
            0 (no prior).

            **Note** that the prior is not parametrized in the
            standard way where the uninformative prior has alpha=1.
        betaW, betaZ, betaH : non-negative float
            Entropic prior parameters for `W`, `Z`, and `H`.  Large
            values lead to sparser distributions.  Defaults to 0 (no
            prior).
        nu : float
            Approximation parameter for the Entropic prior.  It's
            probably safe to leave the default.
        """
        self.V = V.copy()
        self.rank = rank

        self.F, self.T = self.V.shape

        # Allocate the sufficient statistics here, so they don't have to be
        # reallocated at every iteration.  This becomes especially important
        # for the more sophistacted models with many hidden variables.
        self.VRW = np.empty((self.F, self.rank))
        self.VRH = np.empty((self.T, self.rank))

        self.alphaW = 1 + alphaW
        self.alphaZ = 1 + alphaZ
        self.alphaH = 1 + alphaH

        if betaW < 0 or betaZ < 0 or betaH < 0:
            raise ValueError('Entropic prior parameters beta{W,Z,H} must be '
                             'non-negative')
        self.betaW = betaW
        self.betaZ = betaZ
        self.betaH = betaH
        self.nu = nu

        self.minpruneiter = minpruneiter

    @classmethod
    def analyze(cls, V, rank, niter=100, convergence_thresh=1e-9,
                printiter=50, plotiter=None, plotfilename=None,
                initW=None, initZ=None, initH=None,
                updateW=True, updateZ=True, updateH=True, **kwargs):
        """Iteratively performs the PLCA decomposition using the EM algorithm

        Parameters
        ----------
        V : array, shape (`F`, `T`)
            Matrix to analyze.
        niter : int
            Number of iterations to perform.  Defaults to 100.
        convergence_thresh : float
        updateW, updateZ, updateH : boolean
            If False keeps the corresponding parameter fixed.
            Defaults to True.
        initW, initZ, initH : array
            Initial settings for `W`, `Z`, and `H`.  Unused by default.
        printiter : int
            Prints current log probability once every `printiter`
            iterations.  Defaults to 50.
        plotiter : int or None
            If not None, the current decomposition is plotted once
            every `plotiter` iterations.  Defaults to None.
        kwargs : dict
            Arguments to pass into the class's constructor.

        Returns
        -------
        W : array, shape (`F`, `rank`)
            Set of `rank` bases found in `V`, i.e. P(f | z).
        Z : array, shape (`rank`)
            Mixing weights over basis vector activations, i.e. P(z).
        H : array, shape (`rank`, `T`)
            Activations of each basis in time, i.e. P(t | z).
        norm : float
            Normalization constant to make `V` sum to 1.
        recon : array
            Reconstruction of `V` using `W`, `Z`, and `H`
        logprob : float
        """
        norm = V.sum()
        V /= norm

        params = cls(V, rank, **kwargs)
        iW, iZ, iH = params.initialize()

        W = iW if initW is None else initW.copy()
        Z = iZ if initZ is None else initZ.copy()
        H = iH if initH is None else initH.copy()

        params.W = W
        params.Z = Z
        params.H = H

        oldlogprob = -np.inf
        for n in xrange(niter):
            logprob, WZH = params.do_estep(W, Z, H)
            if n % printiter == 0:
                pass
                #logger.debug('Iteration %d: logprob = %f', n, logprob)
            if plotiter and n % plotiter == 0:
                params.plot(V, W, Z, H, n)
                if not plotfilename is None:
                    plt.savefig('%s_%04d.png' % (plotfilename, n))
            if logprob < oldlogprob:
                pass
                #logger.debug('Warning: logprob decreased from %f to %f at '
                             #'iteration %d!', oldlogprob, logprob, n)
                #import pdb; pdb.set_trace()
            elif n > 0 and logprob - oldlogprob < convergence_thresh:
                #logger.debug('Converged at iteration %d', n)
                break
            oldlogprob = logprob

            nW, nZ, nH = params.do_mstep(n)

            if updateW:  W = nW
            if updateZ:  Z = nZ
            if updateH:  H = nH

            params.W = W
            params.Z = Z
            params.H = H

        if plotiter:
            params.plot(V, W, Z, H, n)
            if not plotfilename is None:
                plt.savefig('%s_%04d.png' % (plotfilename, n))
        #logger.debug('Iteration %d: final logprob = %f', n, logprob)
        recon = norm * WZH
        return W, Z, H, norm, recon, logprob

    @staticmethod
    def reconstruct(W, Z, H, norm=1.0):
        """Computes the approximation to V using W, Z, and H"""
        return norm * np.dot(W * Z, H)

    @classmethod
    def plot(cls, V, W, Z, H, curriter=-1):
        WZH = cls.reconstruct(W, Z, H)
        plottools.plotall([V, WZH], subplot=(3,1), align='xy', cmap=plt.cm.hot)
        plottools.plotall(9 * [None] + [W, Z, H], subplot=(4,3), clf=False,
                          align='', cmap=plt.cm.hot, colorbar=False)
        plt.draw()

    def initialize(self):
        """Initializes the parameters

        W and H are initialized randomly.  Z is initialized to have a
        uniform distribution.
        """
        W = normalize(np.random.rand(self.F, self.rank), 0)
        Z = np.ones(self.rank) / self.rank
        H = normalize(np.random.rand(self.rank, self.T), 1)
        return W, Z, H

    def compute_logprob(self, W, Z, H, recon):
        logprob = np.sum(self.V * np.log(recon + EPS*recon))
        # Add Dirichlet and Entropic priors.
        logprob += (np.sum((self.alphaW - 1) * np.log(W + EPS*W))
                    + np.sum((self.alphaZ - 1) * np.log(Z + EPS*Z))
                    + np.sum((self.alphaH - 1) * np.log(H + EPS*H)))
        # Add Entropic priors.
        logprob += (self.betaW * np.sum(W * np.log(W + EPS*W))
                    + self.betaZ * np.sum(Z * np.log(Z + EPS*Z))
                    + self.betaH * np.sum(H * np.log(H + EPS*H)))
        return logprob

    def do_estep(self, W, Z, H):
        """Performs the E-step of the EM parameter estimation algorithm.

        Computes the posterior distribution over the hidden variables.
        """
        WZH = self.reconstruct(W, Z, H)
        logprob = self.compute_logprob(W, Z, H, WZH)

        VdivWZH = self.V / WZH
        for z in xrange(self.rank):
            tmp = np.outer(W[:,z] * Z[z], H[z,:]) * VdivWZH
            self.VRW[:,z] = tmp.sum(1)
            self.VRH[:,z] = tmp.sum(0)

        return logprob, WZH

    def do_mstep(self, curriter):
        """Performs the M-step of the EM parameter estimation algorithm.

        Computes updated estimates of W, Z, and H using the posterior
        distribution computer in the E-step.
        """
        Zevidence = self._fix_negative_values(self.VRW.sum(0) + self.alphaZ - 1)
        initialZ = normalize(Zevidence)
        Z = self._apply_entropic_prior_and_normalize(
            initialZ, Zevidence, self.betaZ, nu=self.nu)

        Wevidence = self._fix_negative_values(self.VRW + self.alphaW - 1)
        initialW = normalize(Wevidence, axis=0)
        W = self._apply_entropic_prior_and_normalize(
            initialW, Wevidence, self.betaW, nu=self.nu, axis=0)

        Hevidence = self._fix_negative_values(self.VRH.T + self.alphaH - 1)
        initialH = normalize(Hevidence, axis=1)
        H = self._apply_entropic_prior_and_normalize(
            initialH, Hevidence, self.betaH, nu=self.nu, axis=1)

        return self._prune_undeeded_bases(W, Z, H, curriter)

    @staticmethod
    def _fix_negative_values(x, fix=EPS):
        x[x <= 0] = fix
        return x

    def _prune_undeeded_bases(self, W, Z, H, curriter):
        """Discards bases which do not contribute to the decomposition"""
        threshold = 10 * EPS
        zidx = np.argwhere(Z > threshold).flatten()
        if len(zidx) < self.rank and curriter >= self.minpruneiter:
            #logger.debug('Rank decreased from %d to %d during iteration %d',
                        #self.rank, len(zidx), curriter)
            self.rank = len(zidx)
            Z = Z[zidx]
            W = W[:,zidx]
            H = H[zidx,:]
            self.VRW = self.VRW[:,zidx]
            self.VRH = self.VRH[:,zidx]
        return W, Z, H

    @staticmethod
    def _apply_entropic_prior_and_normalize(param, evidence, beta, nu=50,
                                            niter=30, convergence_thresh=1e-7,
                                            axis=None):
        """Uses the approximation to the entropic prior from Matt Hoffman."""
        for i in xrange(niter):
            lastparam = param.copy()
            alpha = normalize(param**(nu / (nu - 1.0)), axis)
            param = normalize(evidence + beta * nu * alpha, axis)
            #param = normalize(evidence + beta * nu * param**(nu / (nu - 1.0)), 1)
            if np.mean(np.abs(param - lastparam)) < convergence_thresh:
                #logger.debug('M-step finished after iteration '
                           #'%d (beta=%f)', i, beta)
                break
        return param


class SIPLCA(PLCA):
    """Sparse Shift-Invariant Probabilistic Latent Component Analysis

    Decompose V into \sum_k W_k * z_k h_k^T where * denotes
    convolution.  Each basis W_k is a matrix.  Therefore, unlike PLCA,
    `W` has shape (`F`, `win`, `rank`). This is the model used in [1].

    See Also
    --------
    PLCA : Probabilistic Latent Component Analysis
    SIPLCA2 : 2D SIPLCA
    """
    def __init__(self, V, rank, win=1, circular=False, **kwargs):
        """
        Parameters
        ----------
        V : array, shape (`F`, `T`)
            Matrix to analyze.
        rank : int
            Rank of the decomposition (i.e. number of columns of `W`
            and rows of `H`).
        win : int
            Length of each of the convolutive bases.  Defaults to 1,
            i.e. the model is identical to PLCA.
        circular : boolean
            If True, data shifted past `T` will wrap around to
            0. Defaults to False.
        alphaW, alphaZ, alphaH : float or appropriately shaped array
            Sparsity prior parameters for `W`, `Z`, and `H`.  Negative
            values lead to sparser distributions, positive values
            makes the distributions more uniform.  Defaults to 0 (no
            prior).

            **Note** that the prior is not parametrized in the
            standard way where the uninformative prior has alpha=1.
        """
        PLCA.__init__(self, V, rank, **kwargs)

        self.win = win
        self.circular = circular

        self.VRW = np.empty((self.F, self.rank, self.win))
        self.VRH = np.empty((self.T, self.rank))

    @staticmethod
    def reconstruct(W, Z, H, norm=1.0, circular=False):
        if W.ndim == 2:
            W = W[:,np.newaxis,:]
        if H.ndim == 1:
            H = H[np.newaxis,:]
        F, rank, win = W.shape
        rank, T = H.shape

        WZH = np.zeros((F, T))
        for tau in xrange(win):
            WZH += np.dot(W[:,:,tau] * Z, shift(H, tau, 1, circular))
        return norm * WZH

    def plot(self, V, W, Z, H, curriter=-1):
        rank = len(Z)
        nrows = rank + 2
        WZH = self.reconstruct(W, Z, H, circular=self.circular)
        plottools.plotall([V, WZH] + [self.reconstruct(W[:,z,:], Z[z], H[z,:],
                                                       circular=self.circular)
                                      for z in xrange(len(Z))],
                          title=['V (Iteration %d)' % curriter,
                                 'Reconstruction'] +
                          ['Basis %d reconstruction' % x
                           for x in xrange(len(Z))],
                          colorbar=False, grid=False, cmap=plt.cm.hot,
                          subplot=(nrows, 2), order='c', align='xy')
        plottools.plotall([None] + [Z], subplot=(nrows, 2), clf=False,
                          plotfun=lambda x: plt.bar(np.arange(len(x)) - 0.4, x),
                          xticks=[[], range(rank)], grid=False,
                          colorbar=False, title='Z')

        plots = [None] * (3*nrows + 2)
        titles = plots + ['W%d' % x for x in range(rank)]
        wxticks = [[]] * (3*nrows + rank + 1) + [range(0, W.shape[2], 10)]
        plots.extend(W.transpose((1, 0, 2)))
        plottools.plotall(plots, subplot=(nrows, 6), clf=False, order='c',
                          align='xy', cmap=plt.cm.hot, colorbar=False,
                          ylabel=r'$\parallel$', grid=False,
                          title=titles, yticks=[[]], xticks=wxticks)

        plots = [None] * (2*nrows + 2)
        titles=plots + ['H%d' % x for x in range(rank)]
        if np.squeeze(H).ndim < 4:
            plotH = np.squeeze(H)
        else:
            plotH = H.sum(2)
        if rank == 1:
            plotH = [plotH]
        plots.extend(plotH)
        plottools.plotall(plots, subplot=(nrows, 3), order='c', align='xy',
                          grid=False, clf=False, title=titles, yticks=[[]],
                          colorbar=False, cmap=plt.cm.hot, ylabel=r'$*$',
                          xticks=[[]]*(3*nrows-1) + [range(0, V.shape[1], 100)])
        plt.show()

    def initialize(self):
        W, Z, H = super(SIPLCA, self).initialize()
        W = np.random.rand(self.F, self.rank, self.win)
        W /= W.sum(2).sum(0)[np.newaxis,:,np.newaxis]
        return W, Z, H

    def do_estep(self, W, Z, H):
        WZH = self.reconstruct(W, Z, H, circular=self.circular)
        logprob = self.compute_logprob(W, Z, H, WZH)

        WZ = W * Z[np.newaxis,:,np.newaxis]
        VdivWZH = (self.V / (WZH + EPS))[:,:,np.newaxis]
        self.VRW[:] = 0
        self.VRH[:] = 0
        for tau in xrange(self.win):
            Ht = shift(H, tau, 1, self.circular)
            tmp = WZ[:,:,tau][:,np.newaxis,:] * Ht.T[np.newaxis,:,:] * VdivWZH
            self.VRW[:,:,tau] += tmp.sum(1)
            self.VRH += shift(tmp.sum(0), -tau, 0, self.circular)

        return logprob, WZH

    def do_mstep(self, curriter):
        Zevidence = self._fix_negative_values(self.VRW.sum(2).sum(0)
                                              + self.alphaZ - 1)
        initialZ = normalize(Zevidence)
        Z = self._apply_entropic_prior_and_normalize(
            initialZ, Zevidence, self.betaZ, nu=self.nu)

        Wevidence = self._fix_negative_values(self.VRW + self.alphaW - 1)
        initialW = normalize(Wevidence, axis=[0, 2])
        W = self._apply_entropic_prior_and_normalize(
            initialW, Wevidence, self.betaW, nu=self.nu, axis=[0, 2])

        Hevidence = self._fix_negative_values(self.VRH.T + self.alphaH - 1)
        initialH = normalize(Hevidence, axis=1)
        H = self._apply_entropic_prior_and_normalize(
            initialH, Hevidence, self.betaH, nu=self.nu, axis=1)

        return self._prune_undeeded_bases(W, Z, H, curriter)


class SIPLCA2(SIPLCA):
    """Sparse 2D Shift-Invariant Probabilistic Latent Component Analysis

    Shift invariance is over both rows and columns of `V`.  Unlike
    PLCA and SIPLCA, the activations for each basis `H_k` describes
    when the k-th basis is active in time *and* at what vertical
    (frequency) offset.  Therefore, unlike PLCA and SIPLCA, `H` has
    shape (`rank`, `win[1]`, `T`).

    Note that this is not the same as the 2D-SIPLCA decomposition
    described in Smaragdis and Raj, 2007.  `W` has the same shape as
    in SIPLCA, regardless of `win[1]`.

    See Also
    --------
    PLCA : Probabilistic Latent Component Analysis
    SIPLCA : Shift-Invariant PLCA
    """
    def __init__(self, V, rank, win=1, circular=False, **kwargs):
        """
        Parameters
        ----------
        V : array, shape (`F`, `T`)
            Matrix to analyze.
        rank : int
            Rank of the decomposition (i.e. number of columns of `W`
            and rows of `H`).
        win : int or tuple of 2 ints
            `win[0]` is the length of the convolutive bases.  `win[1]`
            is maximum frequency shift.  Defaults to (1, 1).
        circular : boolean or tuple of 2 booleans
            If `circular[0]` (`circular[1]`) is True, data shifted
            horizontally (vertically) past `T` (`F`) will wrap around
            to 0.  Defaults to (False, False).
        alphaW, alphaZ, alphaH : float or appropriately shaped array
            Sparsity prior parameters for `W`, `Z`, and `H`.  Negative
            values lead to sparser distributions, positive values
            makes the distributions more uniform.  Defaults to 0 (no
            prior).

            **Note** that the prior is not parametrized in the
            standard way where the uninformative prior has alpha=1.
        """
        PLCA.__init__(self, V, rank, **kwargs)
        self.rank = rank

        try:
            self.winF, self.winT = win
        except:
            self.winF = self.winT = win
        # Needed for compatibility with SIPLCA.
        self.win = self.winT

        try:
            self.circularF, self.circularT = circular
        except:
            self.circularF = self.circularT = circular
        # Needed for plot.
        self.circular = (self.circularF, self.circularT)

        self.VRW = np.empty((self.F, self.rank, self.winT))
        self.VRH = np.empty((self.T, self.rank, self.winF))

    @staticmethod
    def reconstruct(W, Z, H, norm=1.0, circular=False):
        if W.ndim == 2:
            W = W[:,np.newaxis,:]
        if Z.ndim == 0:
            Z = Z[np.newaxis]
        if H.ndim == 2:
            H = H[np.newaxis,:,:]
        F, rank, winT = W.shape
        rank, winF, T = H.shape

        try:
            circularF, circularT = circular
        except:
            circularF = circularT = circular

        recon = 0
        for z in xrange(rank):
            recon += sp.signal.fftconvolve(W[:,z,:] * Z[z], H[z,:,:])

        WZH = recon[:F,:T]
        if circularF:
            WZH[:winF-1,:] += recon[F:,:T]
        if circularT:
            WZH[:,:winT-1] += recon[:F,T:]
        if circularF and circularT:
            WZH[:winF-1,:winT-1] += recon[F:,T:]

        return norm * WZH

    def initialize(self):
        W, Z, H = super(SIPLCA2, self).initialize()
        W = np.random.rand(self.F, self.rank, self.winT)
        W /= W.sum(2).sum(0)[np.newaxis,:,np.newaxis]

        H = np.random.rand(self.rank, self.winF, self.T)
        H /= H.sum(2).sum(1)[:,np.newaxis,np.newaxis]
        return W, Z, H

    def do_estep(self, W, Z, H):
        WZH = self.reconstruct(W, Z, H,
                               circular=[self.circularF, self.circularT])
        logprob = self.compute_logprob(W, Z, H, WZH)

        WZ = W * Z[np.newaxis,:,np.newaxis]
        VdivWZH = (self.V / (WZH + EPS))[:,:,np.newaxis]
        self.VRW[:] = 0
        self.VRH[:] = 0
        for r in xrange(self.winF):
            WZshifted = shift(WZ, r, 0, self.circularF)
            for tau in xrange(self.winT):
                Hshifted = shift(H[:,r,:], tau, 1, self.circularT)
                tmp = ((WZshifted[:,:,tau][:,:,np.newaxis]
                        * Hshifted[np.newaxis,:,:]).transpose((0,2,1))
                       * VdivWZH)
                self.VRW[:,:,tau] += shift(tmp.sum(1), -r, 0, self.circularF)
                self.VRH[:,:,r] += shift(tmp.sum(0), -tau, 0, self.circularT)

        return logprob, WZH

    def do_mstep(self, curriter):
        Zevidence = self._fix_negative_values(self.VRW.sum(2).sum(0)
                                              + self.alphaZ - 1)
        initialZ = normalize(Zevidence)
        Z = self._apply_entropic_prior_and_normalize(
            initialZ, Zevidence, self.betaZ, nu=self.nu)

        Wevidence = self._fix_negative_values(self.VRW + self.alphaW - 1)
        initialW = normalize(Wevidence, axis=[0, 2])
        W = self._apply_entropic_prior_and_normalize(
            initialW, Wevidence, self.betaW, nu=self.nu, axis=[0, 2])

        Hevidence = self._fix_negative_values(self.VRH.transpose((1,2,0))
                                              + self.alphaH - 1)
        initialH = normalize(Hevidence, axis=[1, 2])
        H = self._apply_entropic_prior_and_normalize(
            initialH, Hevidence, self.betaH, nu=self.nu, axis=[1, 2])

        return self._prune_undeeded_bases(W, Z, H, curriter)


class FactoredSIPLCA2(SIPLCA2):
    """Sparse 2D Shift-Invariant PLCA with factored `W`

    This class performs the same decomposition as SIPLCA2, except W is
    factored into two independent terms:
      W = P(f, \tau | k) = P(f | \tau, k) P(\tau | k)
    and H is also factored into two independent terms:
      H = P(t, r | k) = P(t | k) P(r | t, k)

    This enables priors to be enforced *independently* over the rows
    and columns of W_k.  The `alphaW` and `betaW` arguments now
    control sparsity in each column of W_k and `alphaT` and `betaT`
    control sparsity in the rows.

    See Also
    --------
    SIPLCA2 : 2D Shift-Invariant PLCA
    """
    def __init__(self, V, rank, alphaT=0, betaT=0, alphaR=0, betaR=0, **kwargs):
        SIPLCA2.__init__(self, V, rank, **kwargs)
        self.alphaT = 1 + alphaT
        self.betaT = betaT
        self.alphaR = 1 + alphaR
        self.betaR = betaR

    def do_mstep(self, curriter):
        Zevidence = self._fix_negative_values(self.VRW.sum(2).sum(0)
                                              + self.alphaZ - 1)
        initialZ = normalize(Zevidence)
        Z = self._apply_entropic_prior_and_normalize(
            initialZ, Zevidence, self.betaZ, nu=self.nu)

        # Factored W = P(f, \tau | k) = P(f | \tau, k) P(\tau | k)
        # P(f | \tau, k)
        Pf_evidence = self._fix_negative_values(self.VRW + self.alphaW - 1)
        initialPf = normalize(Pf_evidence, 0)
        Pf = self._apply_entropic_prior_and_normalize(
            initialPf, Pf_evidence, self.betaW, nu=self.nu, axis=0)

        # P(\tau | k)
        Ptau_evidence = self._fix_negative_values(self.VRW.sum(0)
                                                  + self.alphaT - 1)
        initialPtau = normalize(Ptau_evidence, 1)
        Ptau = self._apply_entropic_prior_and_normalize(
            initialPtau, Ptau_evidence, self.betaT, nu=self.nu, axis=1)

        # W = P(f, \tau | k)
        W = Pf * Ptau[np.newaxis,:,:]

        # Factored H = P(t, r | k) = P(t | k) P(r | t, k)
        # P(t | k)
        Pt_evidence = self._fix_negative_values(self.VRH.sum(2).T
                                                + self.alphaH - 1)
        initialPt = normalize(Pt_evidence, 1)
        Pt = self._apply_entropic_prior_and_normalize(
            initialPt, Pt_evidence, self.betaH, nu=self.nu, axis=1)

        # P(r | t, k)
        Pr_evidence = self._fix_negative_values(self.VRH.transpose((1,2,0))
                                                + self.alphaR - 1)
        initialPr = normalize(Pr_evidence, 1)
        Pr = self._apply_entropic_prior_and_normalize(
            initialPr, Pr_evidence, self.betaR, nu=self.nu, axis=1)

        # H = P(r, t | k)
        H = Pt[:,np.newaxis,:] * Pr

        #Hevidence = self._fix_negative_values(self.VRH.transpose((1,2,0))
        #                             + self.alphaH - 1)
        #initialH = normalize(Hevidence, axis=[1, 2])
        #H = self._apply_entropic_prior_and_normalize(
        #    initialH, Hevidence, self.betaH, nu=self.nu, axis=[1, 2])

        return self._prune_undeeded_bases(W, Z, H, curriter)


class DiscreteWSIPLCA2(FactoredSIPLCA2):
    """Sparse (Time) Warp and 2D Shift-Invariant PLCA

    See Also
    --------
    PLCA : Probabilistic Latent Component Analysis
    SIPLCA2 : 2D SIPLCA
    """
    def __init__(self, V, rank, warpfactors=[1], **kwargs):
        FactoredSIPLCA2.__init__(self, V, rank, **kwargs)

        self.warpfactors = np.array(warpfactors, dtype=np.float)
        self.nwarp = len(self.warpfactors)
        self.VRH = np.empty((self.T, self.rank, self.winF, self.nwarp))

        # Need to weigh each path by the number of repetitions of each
        # tau.  Keep track of it here.
        self.taus = []
        self.tauproportions = []
        for n, warp in enumerate(self.warpfactors):
            currtaus = np.floor(warp * np.arange(self.win/warp))

            currtauproportions = np.empty(len(currtaus))
            for m,tau in enumerate(currtaus):
                currtauproportions[m] = 1.0 / np.sum(currtaus == tau)

            self.taus.append([int(x) for x in currtaus])
            self.tauproportions.append(currtauproportions)

        #print self.taus
        #print self.tauproportions
        #print [x.sum() for x in self.tauproportions]

    def reconstruct(self, W, Z, H, norm=1.0, circular=False):
        if W.ndim == 2:
            W = W[:,np.newaxis,:]
        if H.ndim == 3:
            H = H[np.newaxis,:,:,:]
        F, rank, winT = W.shape
        rank, winF, nwarp, T = H.shape

        try:
            circularF, circularT = circular
        except:
            circularF = circularT = circular

        recon = np.zeros((F, T))
        for r in xrange(self.winF):
            Wshifted = shift(W, r, 0, circularF)
            for n, warp in enumerate(self.warpfactors):
                for delay, tau in enumerate(self.taus[n]):
                    recon += np.dot(Wshifted[:,:,tau] * Z,
                                    shift(H[:,r,n,:], delay, 1, circularT)
                                    * self.tauproportions[n][delay])
        return norm * recon

    def initialize(self):
        W, Z, H = super(DiscreteWSIPLCA2, self).initialize()
        H = normalize(np.random.rand(self.rank, self.winF, self.nwarp, self.T),
                      axis=[1, 2, 3])
        return W, Z, H

    def do_estep(self, W, Z, H):
        WZH = self.reconstruct(W, Z, H,
                               circular=[self.circularF, self.circularT])
        logprob = self.compute_logprob(W, Z, H, WZH)

        WZ = W * Z[np.newaxis,:,np.newaxis]
        VdivWZH = (self.V / (WZH + EPS))[:,:,np.newaxis]
        self.VRW[:] = 0
        self.VRH[:] = 0
        for r in xrange(self.winF):
            WZshifted = shift(WZ, r, 0, self.circularF)
            for n, warp in enumerate(self.warpfactors):
                for delay, tau in enumerate(self.taus[n]):
                    Hshifted = (shift(H[:,r,n,:], delay, 1, self.circularT)
                                * self.tauproportions[n][delay])# / warp) # FIXME
                    tmp = ((WZshifted[:,:,tau][:,:,np.newaxis]
                            * Hshifted[np.newaxis,:,:]).transpose((0,2,1))
                           * VdivWZH)
                    self.VRW[:,:,tau] += shift(tmp.sum(1), -r, 0,
                                               self.circularF)
                    self.VRH[:,:,r,n] += shift(tmp.sum(0), -delay, 0,
                                               self.circularT)

        return logprob, WZH

    def do_mstep(self, curriter):
        Zevidence = self._fix_negative_values(self.VRW.sum(2).sum(0)
                                              + self.alphaZ - 1)
        initialZ = normalize(Zevidence)
        Z = self._apply_entropic_prior_and_normalize(
            initialZ, Zevidence, self.betaZ, nu=self.nu)

        # Factored W = P(f, \tau | k) = P(f | \tau, k) P(\tau | k)
        # P(f | \tau, k)
        Pf_evidence = self._fix_negative_values(self.VRW + self.alphaW - 1)
        initialPf = normalize(Pf_evidence, 0)
        Pf = self._apply_entropic_prior_and_normalize(
            initialPf, Pf_evidence, self.betaW, nu=self.nu, axis=0)

        # P(\tau | k)
        Ptau_evidence = self._fix_negative_values(self.VRW.sum(0)
                                                  + self.alphaT - 1)
        initialPtau = normalize(Ptau_evidence, 1)
        Ptau = self._apply_entropic_prior_and_normalize(
            initialPtau, Ptau_evidence, self.betaT, nu=self.nu, axis=1)

        # W = P(f, \tau | k)
        W = Pf * Ptau[np.newaxis,:,:]

        # Factored H = P(t, r, w | k) = P(t | k) P(r, n | t, k)
        # P(t | k)
        Pt_evidence = self._fix_negative_values(self.VRH.sum(3).sum(2).T
                                                + self.alphaH - 1)
        initialPt = normalize(Pt_evidence, 1)
        Pt = self._apply_entropic_prior_and_normalize(
            initialPt, Pt_evidence, self.betaH, nu=self.nu, axis=1)

        # P(r, n | t, k)
        Prn_evidence = self._fix_negative_values(self.VRH.transpose((1,2,3,0))
                                                + self.alphaR - 1)
        initialPrn = normalize(Prn_evidence, [1, 2])
        Prn = self._apply_entropic_prior_and_normalize(
            initialPrn, Prn_evidence, self.betaR, nu=self.nu, axis=[1, 2])

        # H = P(r, n, t | k)
        H = Pt[:,np.newaxis,np.newaxis,:] * Prn

        return self._prune_undeeded_bases(W, Z, H, curriter)
