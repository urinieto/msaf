#!/usr/bin/env python
"""Class that implements X-means."""

import argparse
import numpy as np
import time
import pylab as plt
import scipy.cluster.vq as vq
from scipy.spatial import distance


class XMeans:
    def __init__(self, X, init_K=2, plot=False):
        self.X = X
        self.init_K = init_K
        self.plot = plot

    def estimate_K_xmeans(self, th=0.2, maxK = 10):
        """Estimates K running X-means algorithm (Pelleg & Moore, 2000)."""

        # Run initial K-means
        means, labels = self.run_kmeans(self.X, self.init_K)

        # Run X-means algorithm
        stop = False
        curr_K = self.init_K
        while not stop:
            stop = True
            final_means = []
            for k in range(curr_K):
                # Find the data that corresponds to the k-th cluster
                D = self.get_clustered_data(self.X, labels, k)
                if len(D) == 0 or D.shape[0] == 1:
                    continue

                # Whiten and find whitened mean
                stdD = np.std(D, axis=0)
                #D = vq.whiten(D)
                D /= float(stdD)  # Same as line above
                mean = D.mean(axis=0)

                # Cluster this subspace by half (K=2)
                half_means, half_labels = self.run_kmeans(D, K=2)

                # Compute BICs
                bic1 = self.compute_bic(D, [mean], K=1,
                                        labels=np.zeros(D.shape[0]),
                                        R=D.shape[0])
                bic2 = self.compute_bic(D, half_means, K=2,
                                        labels=half_labels, R=D.shape[0])

                # Split or not
                max_bic = np.max([np.abs(bic1), np.abs(bic2)])
                norm_bic1 = bic1 / float(max_bic)
                norm_bic2 = bic2 / float(max_bic)
                diff_bic = np.abs(norm_bic1 - norm_bic2)

                # Split!
                #print "diff_bic", diff_bic
                if diff_bic > th:
                    final_means.append(half_means[0] * stdD)
                    final_means.append(half_means[1] * stdD)
                    curr_K += 1
                    stop = False
                # Don't split
                else:
                    final_means.append(mean * stdD)

            final_means = np.asarray(final_means)

            #print "Estimated K: ", curr_K
            if self.plot:
                plt.scatter(self.X[:, 0], self.X[:, 1])
                plt.scatter(final_means[:, 0], final_means[:, 1], color="y")
                plt.show()

            if curr_K >= maxK or self.X.shape[-1] != final_means.shape[-1]:
                stop = True
            else:
                labels, dist = vq.vq(self.X, final_means)

        return curr_K

    def estimate_K_knee(self, th=.015, maxK=12):
        """Estimates the K using K-means and BIC, by sweeping various K and
            choosing the optimal BIC."""
        # Sweep K-means
        if self.X.shape[0] < maxK:
            maxK = self.X.shape[0]
        if maxK < 2:
            maxK = 2
        K = np.arange(1, maxK)
        bics = []
        for k in K:
            means, labels = self.run_kmeans(self.X, k)
            bic = self.compute_bic(self.X, means, labels, K=k,
                                   R=self.X.shape[0])
            bics.append(bic)
        diff_bics = np.diff(bics)
        finalK = K[-1]

        if len(bics) == 1:
            finalK = 2
        else:
            # Normalize
            bics = np.asarray(bics)
            bics -= bics.min()
            #bics /= bics.max()
            diff_bics -= diff_bics.min()
            #diff_bics /= diff_bics.max()

            #print bics, diff_bics

            # Find optimum K
            for i in range(len(K[:-1])):
                #if bics[i] > diff_bics[i]:
                if diff_bics[i] < th and K[i] != 1:
                    finalK = K[i]
                    break

        #print "Estimated K: ", finalK
        if self.plot:
            plt.subplot(2, 1, 1)
            plt.plot(K, bics, label="BIC")
            plt.plot(K[:-1], diff_bics, label="BIC diff")
            plt.legend(loc=2)
            plt.subplot(2, 1, 2)
            plt.scatter(self.X[:, 0], self.X[:, 1])
            plt.show()

        return finalK

    def get_clustered_data(self, X, labels, label_index):
        """Returns the data with a specific label_index, using the previously
         learned labels."""
        D = X[np.argwhere(labels == label_index)]
        return D.reshape((D.shape[0], D.shape[-1]))

    def run_kmeans(self, X, K):
        """Runs k-means and returns the labels assigned to the data."""
        wX = vq.whiten(X)
        means, dist = vq.kmeans(wX, K, iter=100)
        labels, dist = vq.vq(wX, means)
        return means, labels

    def compute_bic(self, D, means, labels, K, R):
        """Computes the Bayesian Information Criterion."""
        D = vq.whiten(D)
        Rn = D.shape[0]
        M = D.shape[1]

        if R == K:
            return 1

        # Maximum likelihood estimate (MLE)
        mle_var = 0
        for k in range(len(means)):
            X = D[np.argwhere(labels == k)]
            X = X.reshape((X.shape[0], X.shape[-1]))
            for x in X:
                mle_var += distance.euclidean(x, means[k])
                #print x, means[k], mle_var
        mle_var /= float(R - K)

        # Log-likelihood of the data
        l_D = - Rn/2. * np.log(2*np.pi) - (Rn * M)/2. * np.log(mle_var) - \
            (Rn - K) / 2. + Rn * np.log(Rn) - Rn * np.log(R)

        # Params of BIC
        p = (K-1) + M * K + mle_var

        #print "BIC:", l_D, p, R, K

        # Return the bic
        return l_D - p / 2. * np.log(R)

    @classmethod
    def generate_2d_data(self, N=100, K=5):
        """Generates N*K 2D data points with K means and N data points
            for each mean."""
        # Seed the random
        np.random.seed(seed=int(time.time()))

        # Amount of spread of the centroids
        spread = 30

        # Generate random data
        X = np.empty((0, 2))
        for i in range(K):
            mean = np.array([np.random.random()*spread,
                             np.random.random()*spread])
            x = np.random.normal(0.0, scale=1.0, size=(N, 2)) + mean
            X = np.append(X, x, axis=0)

        return X


def test_kmeans(K=5):
    """Test k-means with the synthetic data."""
    X = XMeans.generate_2d_data(K=4)
    wX = vq.whiten(X)
    dic, dist = vq.kmeans(wX, K, iter=100)

    plt.scatter(wX[:, 0], wX[:, 1])
    plt.scatter(dic[:, 0], dic[:, 1], color="m")
    plt.show()


def main(args):
    #test_kmeans(6)
    X = XMeans.generate_2d_data(K=args.k)
    xmeans = XMeans(X, init_K=2, plot=args.plot)
    est_K = xmeans.estimate_K_xmeans()
    est_K_knee = xmeans.estimate_K_knee()
    #print "Estimated x-means K:", est_K
    #print "Estimated Knee Point Detection K:", est_K_knee

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Runs x-means")
    parser.add_argument("k",
                        metavar="k", type=int,
                        help="Number of clusters to estimate.")
    parser.add_argument("-p", action="store_true", default=False,
                        dest="plot", help="Plot the results")
    main(parser.parse_args())
