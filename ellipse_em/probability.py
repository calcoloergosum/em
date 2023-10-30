import numpy as np
import torch
import scipy.stats

_ = None

def pdf(data, means, covs):
    """Given M data and N means/covs, return MxN probabilities"""
    assert data.shape[-1] == means.shape[-1] == covs.shape[-1]
    k = data.shape[-1]

    means = means[(_,) * (data.ndim - 1)]
    covs = covs[(_,) * (data.ndim - 1)]
    data = data[..., _, :]
    
    vs = (data - means)[..., _, :] @ np.linalg.inv(covs) @ (data - means)[..., :, _]
    vs = vs.squeeze((-1, -2))

    return np.exp(-.5 * vs) / np.sqrt(np.linalg.det(covs) * (2 * np.pi) ** k)


def pdf_torch(data, means, covs):
    assert data.shape[-1] == means.shape[-1] == covs.shape[-1]
    k = data.shape[-1]

    means = means[(_,) * (data.ndim - 1)]
    covs = covs[(_,) * (data.ndim - 1)]
    data = data[..., _, :]
    
    vs = (data - means)[..., _, :] @ torch.linalg.inv(covs) @ (data - means)[..., :, _]
    vs = vs.squeeze((-1, -2))

    return torch.exp(-.5 * vs) / torch.sqrt(torch.linalg.det(covs) * (2 * np.pi) ** k)


def test():
    covs = np.array((np.diag((1, 1)), np.diag((4, 4))))
    means = np.array(((0, 0), (3, 3)))
    data = np.array(((1, 1), (2, 2)))
    probs = pdf(data, means, covs)

    for i, (m, c) in enumerate(zip(means, covs)):
        np.testing.assert_almost_equal(
            probs[:, i],
            scipy.stats.multivariate_normal.pdf(data, mean=m, cov=c)
        )

if __name__ == '__main__':
    test()
