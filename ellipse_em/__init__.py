from typing import Sized, Tuple

import numpy as np
from pynverse import inversefunc
from scipy.special import ellipeinc as _ellipeinc
import functools
from matplotlib import pyplot as plt


EllipseParametersAffine = Tuple[
    Tuple[float, float],  # center coord
    Tuple[float, float],  # axes length
    float,                # tilt angle
]



def ell(angle: float, eccentricity2: float) -> float:
    """Get elliptic arc length from 0 to `angle`, given eccentricity squared `eccentricity2`."""
    return _ellipeinc(angle, eccentricity2)


def ell_inv(length: float, eccentricity2: float) -> float:
    """Inverse of `ell`, fixed `eccentricity`."""
    func = functools.partial(ell, eccentricity2=eccentricity2)
    if isinstance(length, Sized):
        return inversefunc(func, y_values=length, domain=[- 2 * np.pi - 1e-7, 2 * np.pi + 1e-7], accuracy=7)
    return inversefunc(func, y_values=[length], domain=[-2 * np.pi - 1e-7, 2 * np.pi + 1e-7], accuracy=7)[0]


def angle2mat(angle):
    return np.array((
        (np.cos(angle), -np.sin(angle)),
        (np.sin(angle),  np.cos(angle)),
    ))


def angle2vec(angle):
    return np.array((np.cos(angle), np.sin(angle),)).T


def sample_xy_by_arc(
    params: EllipseParametersAffine,
    angle_range: Tuple[float, float],
    n: int,
):
    """Sample points from angle range by equal distance"""
    (cx, cy), (major, minor), tilt = params
    angles = sample_angles_by_arc(params, angle_range, n)
    pts = angle2mat(tilt) @ np.diag((major, minor)) @ np.stack((np.sin(angles), np.cos(angles)))
    return pts.T + (cx, cy)  # N x 2


def sample_angles_by_arc(params, angle_range, n):
    _, (major, minor), _ = params
    angle_start, angle_end = angle_range
    assert major >= minor
    focus        = np.sqrt(major * major - minor * minor)
    ecc          = focus / major
    ecc2         = ecc   * ecc
    start_length = ell(angle_start, ecc2)
    end_length   = ell(angle_end,   ecc2)
    l = (np.arange(n) + 0.5) / n
    angles = ell_inv(start_length + (end_length - start_length) * l, ecc2)
    return angles


ROT90 = angle2mat(np.pi / 2)
_ = np.newaxis


def main():
    n_mix = 32
    n_obs = 256

    # prepare samples
    param_gt = (np.array((0, 0)), np.array((4, 3)), np.pi / 4)
    d2xy0 = sample_xy_by_arc(param_gt, (0, 2 * np.pi), n_obs)
    d2xy1 = np.random.normal(size=(d2xy0.shape))
    sigma_bef = 0.1
    d2xy = d2xy0 + sigma_bef * d2xy1

    # prepare clusters
    param_bef = (np.array((0, 0)), np.array((1, 1)), 0)
    c2r, c2xy = sample_by_angle(param_bef, n_mix)

    # Initialize sigma values
    c2v = np.linalg.norm(np.diag(param_bef[1])[_, ...] @ ROT90 @ c2r[:, :, _], axis=(1, 2))
    major, minor = param_bef[1]
    focus        = np.sqrt(major * major - minor * minor)
    ecc          = focus / major
    ecc2         = ecc   * ecc
    circumf      = major * ell(2 * np.pi, ecc2)
    del ecc, ecc2, focus, major, minor

    c2v *= 10 * circumf / c2v.sum()
    sigma_bef = c2v
    del c2v, circumf

    fig, ax = plt.subplots(1, 1)
    ax.set_xlim(xmin=-5, xmax=5)
    ax.set_ylim(ymin=-5, ymax=5)
    ax.scatter(*c2xy.T, s=10 * sigma_bef, c='b')
    ax.scatter(*d2xy.T, s=1, c='r')
    ax.set_aspect('equal', adjustable='box')
    fig.show()


    # P( x | C_i ), probability of the data given that it follows cluster with C_i
    c2p_bef   = np.ones(n_mix, dtype=float) / n_mix

    import scipy.stats
    normal = scipy.stats.norm()

    # Cluster, Data, xy
    c2d2dxy = d2xy[None, :, :] - c2xy[:, None, :]
    c2d2pxy = normal.pdf(c2d2dxy / sigma_bef[:, _, _])
    c2d2p_bef = c2d2pxy[:, :, 0] * c2d2pxy[:, :, 1]

    # prior_likelihood = np.log((c2p_bef[:, None] * c2d2p_bef).sum(axis=-1)).sum()
    _c2d2p_aft = c2d2p_bef * c2p_bef[:, _]
    c2d2p = _c2d2p_aft / _c2d2p_aft.sum(axis=0)[_, :]  # normalize along C_i

    c2p_aft = c2d2p.sum(axis=1) / n_obs
    np.testing.assert_almost_equal(c2p_aft.sum(0), 1)

    # Maximization
    # jacobian and hessian of sigma^2
    c2s = c2r[:, _, :] @ ROT90.T[_] @ np.diag(param_bef[1])[_, :, :] @ ROT90[_] @ c2r[:, :, _]
    c2s = c2s.squeeze((1, 2))
    _0 = np.zeros((n_mix,))
    v2c2ds = np.array((
        param_bef[1][0] * c2r[:, 1] ** 2,
        param_bef[1][1] * c2r[:, 0] ** 2,
        _0, _0, _0))
    v2v2c2dds = np.zeros((5, 5, n_mix,))
    v2v2c2dds[0, 0, :] = 2 * c2r[:, 1] ** 2
    v2v2c2dds[1, 1, :] = 2 * c2r[:, 0] ** 2

    # jacobian and hessian of points
    v2c2dx, v2v2c2ddxy = jacob_hess(param_bef, n_mix)

    # jacobian and hessian of loss
    c2d2dev = c2xy[:, _, :] - d2xy[_, :, :]
    c2d2sq = c2d2p * (c2d2dev[..., _, :] @ c2d2dev[..., :, _]).squeeze((-1, -2))
    c2f = c2d2sq.sum(axis=-1)
    f = (c2f / c2s).sum(axis=-1)

    c2dev = (c2d2p[..., _] * c2d2dev).sum(axis=-2)
    v2df = (
        (
            (c2d2p / c2s[:, _])[_, ..., _]  # add v, xy
            * (c2d2dev[_] @ v2c2dx[..., _])
        ).squeeze(-1).sum(axis=(-1, -2))
        - (c2f[_] * v2c2ds / (c2s ** 2)).sum(axis=-1)
    )
    import code
    code.interact(local={**globals(), **locals()})
    v2v2ddf = (
        (
            2 / c2s[_, _, ...] * (c2dev[_, _, ..., _, :] @ v2v2c2ddxy[..., _]).squeeze((-1, -2))
        ).sum(axis=-1) +
        (v2v2c2dds * c2f[_, _] / (c2s ** 2)[_, _]).sum(axis=-1) +
        c2dev[_, _, _, :] @ (
            v2c2ds[_, :, :, _, _] *
            v2c2dx[:, _, :, :, _] +
            v2c2ds[:, _, :, _, _] *
            v2c2dx[_, :, :, :, _]
        ).squeeze(axis=-1)
    )
    


    v2ddfdvdv = (
        (
            (c2d2p[..., None] * (c2xy[:, None, :] - d2xy[None, :, :]))[None, None, :, :, :]
            @ v2v2c2ddxy[:, :, :, :, None]
        ).squeeze(-1).sum(axis=(-1, -2))
        +
        (
            c2d2p.sum(axis=1)[None, None, :] *
            (v2c2dx[:, None, :, None, :] @ v2c2dx[None, :, :, :, None]).squeeze((-1, -2))
        ).sum(axis=-1)
    )
    - np.linalg.pinv(v2ddfdvdv) @ v2dfdv


def jacob_hess(param: EllipseParametersAffine, n_mix: int):
    c2r, c2xy = sample_by_angle(param, n_mix)
    rotmat = angle2mat(param[2])

    vec_M, vec_m = rotmat[:, 0], rotmat[:, 1]
    c2dxydM  = (c2r[:, 0, _, _] * vec_M[_, :, _]).squeeze(2)
    c2dxydm  = (c2r[:, 1, _, _] * vec_m[_, :, _]).squeeze(2)
    c2dxydt   = ((ROT90 @ rotmat @ np.diag(param[1]))[_, :, :] @ c2r[:, :, _]).squeeze(2)
    c2dxydcx = np.repeat(((1, 0),), n_mix, axis=0).reshape(-1, 2)
    c2dxydcy = np.repeat(((0, 1),), n_mix, axis=0).reshape(-1, 2)
    v2c2dxydv = np.array((c2dxydM, c2dxydm, c2dxydt, c2dxydcx, c2dxydcy))

    c2ddxy_dMdt = (ROT90[None] @ c2dxydM[:, :, None]).squeeze(2)
    c2ddxy_dmdt = (ROT90[None] @ c2dxydm[:, :, None]).squeeze(2)
    c2ddxy_dtdt = - (c2xy - param[0])
    _0 = np.zeros((n_mix, 2))
    v2v2c2ddxy = np.array((
        (_0,          _0,          c2ddxy_dMdt, _0, _0,),
        (_0,          _0,          c2ddxy_dmdt, _0, _0,),
        (c2ddxy_dMdt, c2ddxy_dmdt, c2ddxy_dtdt, _0, _0,),
        (_0,          _0,          _0,          _0, _0,),
        (_0,          _0,          _0,          _0, _0,),
    ))
    return v2c2dxydv, v2v2c2ddxy


def sample_by_angle(param: EllipseParametersAffine, n_mix: int):
    c2phi = 2 * np.pi * (0.5 + np.arange(n_mix)) / n_mix
    c2r = angle2vec(c2phi)
    rotmat = angle2mat(param[2])
    c2xy = np.array(param[0])[_, :, _] + rotmat[_, ...] @ np.diag(param[1])[_, ...] @ c2r[:, :, _]
    c2xy = c2xy.squeeze(2)
    return c2r, c2xy


if __name__ == '__main__':
    main()
