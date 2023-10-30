import numpy as np

_ = None

def integrate(p0p1):
    assert p0p1.ndim >= 2
    p0, p1 = p0p1[..., 0], p0p1[..., 1]
    c0, s0 = np.cos(p0), np.sin(p0)
    c1, s1 = np.cos(p1), np.sin(p1)
    dp = p1 - p0
    dcs = c1 * s1 - c0 * s0
    dss = s1 * s1 - s0 * s0
    dc = c1 - c0
    ds = s1 - s0

    int_r = np.stack((ds, -dc), axis=-1)
    int_rr = np.stack((
        np.stack((.5 * (dp + dcs), dss / 2), axis=-1),
        np.stack((dss / 2, .5 * (dp - dcs)), axis=-1),
    ), axis=-1)
    return int_r, int_rr + (1 / dp - 2 / dp)[..., _, _] * int_r[..., :, _] @ int_r[..., _, :]


def test():
    # Test by comparing numerical integration to analytic calculation
    N = 100000
    p1, p0 = np.pi / 4, np.pi / 6
    p      = np.linspace(p0, p1, N)
    c,  s  = np.cos(p),  np.sin(p)
    dp     = p1 - p0
    r      = np.array((c, s)).T

    int_r, int_rr = integrate(np.array((p0, p1)))
    mean = int_r / dp
    np.testing.assert_almost_equal(r.mean(axis=0), mean)

    outer = lambda x, y: (x[:, :, _] @ y[:, _, :])
    np.testing.assert_almost_equal(outer(r - mean, r - mean).mean(axis=0), int_rr / dp)

    ps = 2 * np.pi * np.arange(0, 101) / 100
    p0p1s = np.lib.stride_tricks.sliding_window_view(ps, 2)

    integrate(p0p1s)


def test_1():
    N = 100000
    p1, p0 = - np.pi / 50, np.pi / 50
    p      = np.linspace(p0, p1, N)
    c,  s  = np.cos(p),  np.sin(p)
    dp     = p1 - p0
    r      = np.array((c, s)).T

    int_r, int_rr = integrate(np.array((p0, p1)))
    mean = int_r / dp
    np.testing.assert_almost_equal(r.mean(axis=0), mean, decimal=5)

    outer = lambda x, y: (x[:, :, _] @ y[:, _, :])
    np.testing.assert_almost_equal(outer(r - mean, r - mean).mean(axis=0), int_rr / dp, decimal=5)

    ps = 2 * np.pi * np.arange(0, 101) / 100
    p0p1s = np.lib.stride_tricks.sliding_window_view(ps, 2)

    integrate(p0p1s)



if __name__ == '__main__':
    test()
    test_1()

