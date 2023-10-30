"""Change of variable in continuous gaussian mixture"""
from matplotlib import pyplot as plt
import scipy.stats
import numpy as np
from ellipse_em.integration import integrate
from ellipse_em import sample_xy_by_arc, angle2mat, sample_angles_by_arc
from ellipse_em.probability import pdf
from ellipse_em.approximation import sample_gt, approximation_by_angle

N_PT = 10000
N_MIXTURE = 15
RANGE = (-.5, 1.5)


def draw1d(x, mean, std, rho, ax_cum, ax_den, show_yticks: bool):
    x2y = np.zeros(len(x) - 1)
    step_size = (RANGE[1] - RANGE[0]) / N_PT
    x_c = .5 * (x[:-1] + x[1:])
    x_s = np.diff(x)
    for _m, _std, _r in zip(mean, std, rho):
        y = scipy.stats.norm.pdf(x_c, loc=_m, scale=_std)
        ax_cum.plot(x_c, np.cumsum(y * x_s), c='black', lw=0.5)
        ax_den.plot(x_c, y,                  c='black', lw=0.5)
        x2y += _r * y
        # np.testing.assert_almost_equal(y.sum() * step_size, 1., decimal=3)
    # np.testing.assert_almost_equal(x2y.sum() * step_size, 1., decimal=3)

    if not show_yticks:
        ax_cum.get_yaxis().set_visible(False)
        ax_den.get_yaxis().set_visible(False)
    ax_cum.vlines(mean, -100, 100, linestyles='dashed', lw=0.5)
    ax_den.vlines(mean, -100, 100, linestyles='dashed', lw=0.5)
    ax_cum.set_xlim(*RANGE)
    ax_den.set_xlim(*RANGE)
    ax_cum.set_ylim(-.1, 1.1)
    ax_den.set_ylim(-.1, 5.1)
    ax_cum.plot(x_c, np.cumsum(x2y * step_size), c='r', lw=0.5)
    ax_den.plot(x_c, x2y, c='r', lw=0.5)


def main1d():
    INITIAL_STD = 0.1
    fs = [
        lambda x: x,
        np.sqrt,
        lambda x: np.sqrt(x+1),
        lambda x: x ** 1.5,
        lambda x: (x+1) ** 1.5,
        lambda x: x**2 - 1,
        lambda x: (x+1)**2 - 1,
        lambda x: (x+1)**3 - 1,
    ]
    n_cols = len(fs)
    fig, axes = plt.subplots(2, n_cols)

    # before transform
    x = np.linspace(*RANGE, N_PT)
    mean = (.5 + np.arange(N_MIXTURE)) / N_MIXTURE
    std = INITIAL_STD * np.ones(N_MIXTURE)
    rho = np.ones(N_MIXTURE) / N_MIXTURE  # mixture coefficient

    # after transform
    # f = lambda x: ((x+1)**2 - 1) / 3
    for i, _f in enumerate(fs, start=0):
        f = lambda x: ((_f(x) - _f(0)) / (_f(1) - _f(0)))
        np.testing.assert_almost_equal(f(1), 1)
        np.testing.assert_almost_equal(f(0), 0)

        mean_new = f(mean)
        rho_new = np.diff(f(rho.cumsum()), prepend=0)
        std_new = std * np.sqrt(rho_new / rho)
        draw1d(x, mean_new, std_new, rho_new, axes[0][i], axes[1][i], show_yticks=i == 0)
    plt.show()


def main2d():
    INITIAL_STD = 0.5
    N_LEVELS = 30
    _ = None

    PARAM = (np.array((0, 0)), np.array((5, 1)), 0)
    xs = np.linspace(- PARAM[1][0] - .5, PARAM[1][0] + .5, 500)
    ys = np.linspace(- PARAM[1][1] - .5, PARAM[1][1] + .5, 500)
    xys = np.meshgrid(xs, ys)
    xys = np.dstack(xys)

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle(f"Distribution on elliptic arc with {N_MIXTURE} Mixtures")


    def setup(ax):
        ax.set_xlim(- PARAM[1][0] - .5, PARAM[1][0] + .5)
        ax.set_ylim(- PARAM[1][1] - .5, PARAM[1][1] + .5)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect('equal', adjustable='box')

    # plot 1; sample by arc
    xs, ys, probs_gt = sample_gt(PARAM, INITIAL_STD, 0.5, 0.5, 500, 500, 300)
    print(f"max probs: {probs_gt.max()}")
    axes[0, 0].set_title("Ground truth")
    setup(axes[0, 0])
    # axes[0, 0].scatter(*xy_c.T)
    cs = axes[0, 0].contour(xs, ys, probs_gt, np.linspace(0, probs_gt.max(), N_LEVELS))

    # plot 2; sample by arc
    rho = np.ones(N_MIXTURE) / N_MIXTURE
    angles = np.sort((sample_angles_by_arc(PARAM, (0, 2*np.pi), N_MIXTURE)[::-1] - np.pi / 2) % (2 * np.pi))
    rotmat = angle2mat(PARAM[2])
    scale = np.diag(PARAM[1])

    # method 1
    # This method is approximates better when eccentricity is low
    p0p1s = np.lib.stride_tricks.sliding_window_view(
        np.concatenate(([angles[-1] - 2 * np.pi], angles)), 2,)
    int_r, int_rr = integrate(p0p1s)
    dp = p0p1s[:, 1] - p0p1s[:, 0]
    r_c = int_r / dp[:, _]
    xy_c = PARAM[0] + (rotmat @ scale @ r_c[:, :, _]).squeeze(-1)
    covs = (2 * np.pi * dp[:, _, _]) * rotmat.T @ scale @ int_rr @ scale @ rotmat
    covs += np.diag((INITIAL_STD, INITIAL_STD)) ** 2

    # method 2
    # This method is approximates better when eccentricity is very high
    # xy_c = PARAM[0] + (rotmat @ scale @ np.stack((np.cos(angles), np.sin(angles)))).T
    # covs = np.diag((INITIAL_STD, INITIAL_STD)) ** 2

    probs1 = (rho[:, None, None] * np.rollaxis(pdf(xys, xy_c, covs), -1, 0)).sum(axis=0)

    print(f"max probs: {probs1.max()}")
    axes[1, 0].set_title("Estimation with arclength")
    setup(axes[1, 0])
    axes[1, 0].scatter(*xy_c.T)
    cs = axes[1, 0].contour(xs, ys, probs1, np.linspace(0, probs1.max(), N_LEVELS))
    # plt.clabel(cs, inline=True, fontsize=10)

    # plot 3; sample by angle
    rho, xy_c, covs = approximation_by_angle(PARAM, INITIAL_STD, N_MIXTURE)

    probs = np.rollaxis(pdf(xys, xy_c, covs), -1, 0)
    probs2 = (rho[:, None, None] * probs).sum(axis=0)

    print(f"max probs: {probs2.max()}")
    axes[2, 0].set_title("Estimation with angle")
    axes[2, 0].scatter(*xy_c.T)
    setup(axes[2, 0])
    cs = axes[2, 0].contour(xs, ys, probs2, np.linspace(0, probs2.max(), N_LEVELS),)
    # cs = plt.contour(xs, ys, probs, np.linspace(0, 0.025, 20),)
    # plt.clabel(cs, inline=True, fontsize=10)


    # what do we put here?
    setup(axes[0, 1])

    diff = probs1 - probs_gt
    print(f"GT vs ellipse: {(diff ** 2).sum()}")
    minmax = diff.min(), diff.max()
    axes[1, 1].set_title(f"GT vs ellipse ({minmax[0]:.4f} ~ {minmax[1]:.4f})")
    setup(axes[1, 1])
    axes[1, 1].contour(xs, ys, diff, np.linspace(diff.min(), diff.max(), N_LEVELS))


    # diff 
    diff = probs2 - probs_gt
    print(f"GT vs angle: {(diff ** 2).sum()}")
    minmax = diff.min(), diff.max()
    axes[2, 1].set_title(f"GT vs angle ({minmax[0]:.4f} ~ {minmax[1]:.4f})")
    setup(axes[2, 1])
    axes[2, 1].contour(xs, ys, diff, np.linspace(*minmax, N_LEVELS))

    fig.tight_layout()
    plt.savefig(f"{N_MIXTURE}.png")
    print(f"[*] Saved to `{N_MIXTURE}.png`")


# main1d()
main2d()
