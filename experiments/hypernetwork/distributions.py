import numpy as np
from scipy.special import logsumexp


def generate_covariance_matrix(rng: np.random.Generator, d: int, correlation: float = None):
    stds = rng.uniform(0.1, 1.0, size=d)
    corr_matrix = np.full((d, d), correlation if correlation is not None else 0.0)
    np.fill_diagonal(corr_matrix, 1.0)

    try:
        np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
        eigenvalues[eigenvalues < 0] = 0
        corr_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    S = np.diag(stds)
    cov_matrix = S @ corr_matrix @ S
    return cov_matrix


def gaussian_logpdf(
    x: np.ndarray,
    mu: np.ndarray,
    cov_inv: np.ndarray,
    log_norm_const: float,
):
    diff = x - mu.reshape(-1, 1)
    quad = np.einsum('ib,ij,jb->b', diff, cov_inv, diff)
    return log_norm_const - 0.5 * quad


def mixture_logpdf(
        x: np.ndarray,
        weights,
        means,
        covs_inv,
        log_norms
):
    M = len(weights)
    d, B = x.shape
    logs = np.empty((M, B), dtype=np.float64)
    for m in range(M):
        logs[m] = np.log(weights[m]) + gaussian_logpdf(x, means[m], covs_inv[m], log_norms[m])
    return logsumexp(logs, axis=0)


def sample_mixture_params(rng, d, n_components=2):
    raw_w = rng.uniform(0.5, 1.5, size=n_components)
    weights = raw_w / raw_w.sum()
    means = rng.uniform(-0.5, 0.5, size=(n_components, d))
    covs, invs, log_norms = [], [], []
    for _ in range(n_components):
        if d == 1:
            var = float(rng.uniform(0.05, 0.5))
            cov = np.array([[var]], dtype=np.float64)
        else:
            cov = generate_covariance_matrix(rng, d)
        covs.append(cov)
        inv = np.linalg.inv(cov)
        invs.append(inv)
        sign, logdet = np.linalg.slogdet(cov)
        log_norm = -0.5 * (d * np.log(2 * np.pi) + logdet)
        log_norms.append(log_norm)
    return weights.astype(np.float64), means.astype(np.float64), covs, invs, np.array(log_norms, dtype=np.float64)

