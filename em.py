#!/usr/bin/env python3
"""Sample Expectation-Maximization Algorithm for n-dimensional multivariate normals
Written by Han, A.I. System Research
"""
from itertools import count
from typing import List, Sequence, Tuple, Optional

import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def make_data(ndim: int, ngrp: int, ns: Sequence[int], means: Optional[np.ndarray] = None, covs: Optional[np.ndarray] = None) -> np.ndarray:
    """Generate multivariate gaussian data of R^ndim, with group count ngrp
    
    Keyword Arguments:
        ndim {int} -- number of dimension of data
        ngrp {int} -- number of gaussian distribution
        ns {Sequence[int]} -- number of data to draw
    
    Returns:
        np.ndarray -- samples that are drawn
    """
    if means is None:
        means = [np.ones(ndim) * 2 * i for i in range(ngrp)]

    if covs is None:
        # Generate a positive semi-definite matrix, as covariance matrix
        vs = np.random.multivariate_normal(np.zeros(ndim), np.identity(ndim), size=(ngrp, ndim))
        vTs = np.swapaxes(vs, 1, 2)
        covs = vTs @ vs

    print("Ground truth parameters" + "=" * 50)
    for mean, cov, r in zip(means, covs, [n / sum(ns) for n in ns]):
        print("R:", r)
        print("Mean: ", mean)
        print("Covariance Mat: \n", cov)
        print()

    # sample data
    samples = []
    for i, (m, n) in enumerate(zip(means, ns)):
        samples_i = np.random.multivariate_normal(m, covs[i], size=n)
        samples.append(samples_i)
    data = np.concatenate(samples)
    return data.T


def make_prior(ndim: int, ngrp: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Makes a prior
    
    Arguments:
        ndim {int} -- [description]
        ngrp {int} -- [description]
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray] -- mean, covariance matrix, r
            mean has shape of ngrp x ndim
            covariance matrix has shape of ngrp x ndim x ndim
            r has shape of ngrp
    """

    prior_means = np.arange(ngrp)[:, None] @ np.ones(ndim)[None, :]
    prior_covs = np.stack([np.identity(ndim),] * ngrp, axis=0)
    prior_r = np.ones(ngrp) / ngrp
    return prior_means, prior_covs, prior_r


def em_step(data: np.ndarray, prior_means: np.ndarray, prior_covs: np.ndarray, prior_r: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """EM main step.
    Note that this function returns likelihood is the likelihood of prior, due to calculation duplication.

    Arguments:
        data {np.ndarray} -- Data, of shape ndim x ndata
        prior_means {np.ndarray} -- Means, of shape ngrp x ndim
        prior_covs {np.ndarray} -- Covariance matrix, of shape ngrp x ndim x ndim
        prior_r {np.ndarray} -- Ratio of each group, of shape ngrp
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, float] --
            post_means {np.ndarray} -- Means, of shape ngrp x ndim
            post_covs {np.ndarray} -- Covariance matrix, of shape ngrp x ndim x ndim
            post_rs {np.ndarray} -- Ratio of each group, of shape ngrp
            prior_likelihood -- side product of em
    """
    # constants
    ngrp1, ndim1 = prior_means.shape
    ngrp2, ndim2, ndim3 = prior_covs.shape
    ngrp3, = prior_r.shape
    ndim4, ndata = data.shape

    assert ngrp1 == ngrp2 == ngrp3, "Failed test : {} == {} == {}".format(ngrp1, ngrp2, ngrp3)
    assert ndim1 == ndim2 == ndim3 == ndim4, "Failed test : {} == {} == {} == {}".format(ndim1, ndim2, ndim3, ndim4)
    ndim = ndim1
    ngrp = ngrp1

    distribs = [multivariate_normal(mean, cov) for mean, cov in zip(prior_means, prior_covs)]

    # P( x | C_i ), probability of the data given that it follows cluster with C_i
    p_x_ci = np.array([d.pdf(data.T) for d in distribs])
    prior_likelihood = np.log((prior_r[:, None] * p_x_ci).sum(axis=-1)).sum()

    # probability of the data x belonging to cluster i
    r = p_x_ci * prior_r[:, None]
    r_xwise = r / r.sum(axis=0)[None, :]  # normalize along C_i

    # expected number of data belonging to cluster c_i
    ns = r_xwise.sum(axis=1)

    # updating params
    ## mean
    ## index denotes i_grp, i_dim
    post_means = (r_xwise[:, None, :] * data[None, ...]).sum(axis=-1) / ns[:, None]

    ## covariance matrix
    ## devs[i] denotes deviation on i'th cluster's mean
    ## dev[i, j, k] denotes deviation of j'th axis component of k'th data, with respect to i'th cluster's mean
    devs = (data[None, ...] - post_means[..., None])
    post_covs = (r_xwise[:, None, :] * devs) @ (np.swapaxes(devs, 1, 2)) / ns[:, None, None]

    return post_means, post_covs, (ns / ns.sum()), prior_likelihood


def draw(data: np.ndarray, means: np.ndarray, cov: np.ndarray, rs: np.ndarray) -> None:
    if data.shape[0] == 1:
        _draw1d(data, means, cov, rs)
    elif data.shape[0] == 2:
        _draw2d(data, means, cov, rs)
    else:
        raise NotImplementedError("Visualization not supported for ndim >= 3")
    plt.grid(True)
    plt.draw()

def _draw1d(data: np.ndarray, means: np.ndarray, covs: np.ndarray, rs: np.ndarray) -> None:
    """Draw 1 dimensional multivariate normal distribution"""
    ndata = data.shape[-1]
    nx = 1000
    min_x, max_x = data.min(), data.max()
    for mean, stdev, r in zip(means, covs.flatten(), rs):
        x = np.linspace(min_x, max_x, nx)
        plt.plot(x, r * ndata * multivariate_normal.pdf(x, mean, stdev))
    plt.hist(data[0], bins='auto')


def _draw2d(data: np.ndarray, means: np.ndarray, covs: np.ndarray, rs: np.ndarray) -> None:
    """Draw 2 dimensional multivariate normal distribution"""
    ndata = data.shape[-1]
    plt.scatter(data[0, :], data[1, :], marker='.')

    nx = 1000
    ny = 1000
    x = np.linspace(data[0].min(), data[0].max(), nx)
    y = np.linspace(data[1].min(), data[1].max(), ny)
    xv, yv = np.meshgrid(x, y, sparse=False)
    for i_grp, (mean, cov) in enumerate(zip(means, covs)):
        distrib = multivariate_normal(mean, cov)
        xys = np.dstack((xv, yv))
        z = distrib.pdf(xys)
        CS = plt.contour(xv, yv, z, 6,
            colors='k',  # negative contours will be dashed by default
            levels=[1e-2,]
        )


def main(ndim: int, ngrp: int) -> None:
    """Main entry of program"""
    # Prepare sample data
    data = make_data(ndim, ngrp, [100 * (i+1) for i in range(ngrp)])

    # Prepare priors
    prior_means, prior_covs, rs = make_prior(ndim, ngrp)

    # Thresholding setup
    lh_delta = 1000
    lh_last = - np.inf

    # EM loop
    for i in count():
        post_means, post_covs, post_rs, prior_lh = em_step(data, prior_means, prior_covs, rs)
        prior_means = post_means
        prior_covs = post_covs
        lh_delta = prior_lh - lh_last
        lh_last = prior_lh

        print("loop #{:0>2}, lh:{:.6f}".format(i, lh_last))
        if i >= 3 and lh_delta < 1e-6:
            break
    draw(data, post_means, post_covs, rs)
    plt.show(block=True)


if __name__ == "__main__":
    main(ndim=1, ngrp=4)
    main(ndim=2, ngrp=4)
