"""Approximate gaussian mixture for ellipse"""
import numpy as np
from ellipse_em import sample_xy_by_arc, angle2mat
from ellipse_em.probability import pdf_torch
from ellipse_em.integration import integrate
import scipy.stats
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch

device = 'cpu'
# device = 'cuda'
_ = None


def sample_gt(PARAM, INITIAL_STD, PAD_X, PAD_Y, N_X, N_Y, N_CLUSTER):
    xs = np.linspace(- PARAM[1][0] - PAD_X, PARAM[1][0] + PAD_X, N_X)
    ys = np.linspace(- PARAM[1][1] - PAD_Y, PARAM[1][1] + PAD_Y, N_Y)

    rho = np.ones(N_CLUSTER)
    rho /= rho.sum()
    xy_c = sample_xy_by_arc(PARAM, (0, 2 *np.pi), len(rho))
    pxs = scipy.stats.norm.pdf(xy_c[:, _, 0] - xs[_, :], 0, INITIAL_STD)
    pys = scipy.stats.norm.pdf(xy_c[:, _, 1] - ys[_, :], 0, INITIAL_STD)
    pxys = pxs[:, _, :] * pys[:, :, _]
    # Equivalent to the following code:
    # xys = np.meshgrid(xs, ys)
    # xys = np.dstack(xys)
    # pys = scipy.stats.norm.pdf(devs[..., 1], 0, INITIAL_STD)
    # pxys = scipy.stats.multivariate_normal.pdf(devs, (0, 0), cov=INITIAL_STD ** 2)
    # np.testing.assert_almost_equal(pxs * pys, pxys)
    probs_gt = (rho[:, None, None] * pxys).sum(axis=0)
    return xs, ys, probs_gt


def approximation_by_angle(PARAM, INITIAL_STD, N_MIXTURE: int):
    p0p1s = np.lib.stride_tricks.sliding_window_view(
        2 * np.pi * (.25 + np.arange(0, N_MIXTURE + 1)) / N_MIXTURE, 2,)

    int_r, int_rr = integrate(p0p1s)
    dp = p0p1s[:, 1] - p0p1s[:, 0]
    r_c = int_r / dp[:, _]

    # put derivative into consideration
    rho = (
        np.sqrt((r_c[:, _, :] @ np.diag(PARAM[1][::-1] ** 2) @ r_c[:, :, _]).squeeze((-1, -2)))
        / np.linalg.norm(r_c, axis=-1)
    )
    rho /= rho.sum()

    rotmat = angle2mat(PARAM[2])
    scale = np.diag(PARAM[1])
    xy_c = PARAM[0] + (rotmat @ scale @ r_c[:, :, _]).squeeze(-1)
    covs = (2 * np.pi * dp[:, _, _]) * rotmat.T @ scale @ int_rr @ scale @ rotmat
    covs += np.diag((INITIAL_STD, INITIAL_STD)) ** 2
    return rho, xy_c, covs


def torch_learn_distribution():
    PARAM = (np.array((0, 0)), np.array((5, 1)), 0)

    xs, ys, pxy_gt = sample_gt(PARAM, 0.5, 1, 5, 500, 500, 300)
    pxy_gt = torch.from_numpy(pxy_gt.reshape(-1))
    pxy_gt /= pxy_gt.sum()
    xys = np.meshgrid(xs, ys)
    xys = np.dstack(xys)
    xys = xys.reshape(-1, 2)
    xys = torch.from_numpy(xys)

    N_CLUSTER = 10
    N_LEVELS = 20

    prior_r, means, covs = approximation_by_angle(PARAM, 0.5, N_CLUSTER)

    # Prepare variables
    means = torch.from_numpy(means) + 0.01
    means.requires_grad = True
    means = means.to(device)

    # covs = np.repeat([np.diag((1, 1))], repeats=N_CLUSTER, axis=0).astype(float)
    covs = torch.from_numpy(covs)
    covs.requires_grad = True
    covs = covs.to(device)

    # prior_r = torch.ones(N_CLUSTER) / N_CLUSTER
    prior_r = torch.from_numpy(prior_r)
    prior_r.requires_grad = True
    prior_r = prior_r.to(device)
    # Prepare variables done

    optim = torch.optim.SGD([means, covs, prior_r], lr=1)
    for i in tqdm(range(1000)):
        pixy = torch.moveaxis(pdf_torch(xys, means, covs), 0, 1)
        prior_r_ = prior_r / prior_r.sum()
        pxy_pr = (prior_r_[:, _] * pixy).sum(axis=0)

        # visualize
        if i % 100 == 0:
            with torch.no_grad():
                pxy = (pxy_pr * prior_r[:, _]).sum(axis=0).reshape(500, 500)
                fig, axes = plt.subplots(1, 2, figsize=(14, 7))
                axes[0].contour(xs, ys, pxy_gt.reshape(500, 500), levels=np.linspace(pxy_gt.min(), pxy_gt.max(), N_LEVELS))
                axes[1].scatter(*means.T)
                # axes[1].contour(xs, ys, (ps * prior_r[:, _])[0].reshape(500, 500), levels=np.linspace(ps.min(), ps.max(), N_LEVELS))
                axes[1].contour(xs, ys, pxy, levels=np.linspace(pxy.min(), pxy.max(), N_LEVELS))
                axes[0].set_aspect('equal', adjustable='box')
                axes[1].set_aspect('equal', adjustable='box')
                plt.savefig(f"exp2/{i:0>4}.png")

        # Expectation
        # P( x | C_i ), probability of the data given that it follows cluster with C_i
        pxy_pr_ = pxy_pr / pxy_pr.sum()
        # loss = torch.nn.functional.cross_entropy(pxy_pr_, pxy_gt)
        loss = ((pxy_pr_ - pxy_gt) ** 2).sum()
        print(f"{float(loss):.4E}")
        optim.zero_grad()
        loss.backward()
        optim.step()

        # # Maximization
        # # probability of the data x belonging to cluster i
        # membership = ps * prior_r[:, _]
        # membership = membership / membership.sum(axis=0)[_]
        # ns = (weights * membership).sum(axis=1)

        # means_new = (weights[..., _] * membership[..., _] * xys[_]).sum(axis=1) / ns[:, _]
        # means = means + epsilon * (means_new - means)

        # # (weights[..., _] * membership[..., _] * xys[_]).sum(axis=1)
        # devs = xys[_] - means[:, _]
        # covs_new  = (
        #     weights[..., _, _] * membership[..., _, _]* devs[..., :, _] * devs[..., _, :]
        # ).sum(axis=1) / ns[:, _, _]
        # covs = covs + epsilon * (covs_new - covs)


if __name__ == '__main__':
    torch_learn_distribution()
