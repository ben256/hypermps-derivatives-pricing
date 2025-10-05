import math
import torch

from utils import generate_covariance_matrix, gaussian_conditioning_param_len


def gaussian_logpdf(
    x: torch.Tensor,
    mu: torch.Tensor,
    cov_inv: torch.Tensor,
    log_normalisation_const: float,
):
    """
    x: data points, shape [d, S]
    mu: mean vector, shape [d, 1]
    cov_inv: inverse of covariance matrix, shape [d, d]
    log_normalisation_const: pre-computed for efficiency (doesn't depend on x)
    """
    diff = x - mu
    quad = torch.einsum('ib,ij,jb->b', diff, cov_inv, diff)
    return log_normalisation_const - 0.5 * quad


def sample_gaussian_params(
        rng: torch.Generator,
        d: int,
):
    """
    rng: torch random number generator
    d: dimensionality of the Gaussian

    Also returns conditioning params for model.
    """
    means = torch.empty(d, 1).uniform_(-0.25, 0.25, generator=rng)
    cov = generate_covariance_matrix(rng, d)

    cov_inv = torch.linalg.inv(cov)

    _, cov_log_det = torch.linalg.slogdet(cov)
    log_normalisation_const = -0.5 * (d * math.log(2 * math.pi) + cov_log_det)

    cov_triu = cov[torch.triu_indices(d, d)[0], torch.triu_indices(d, d)[1]]
    conditioning_params = torch.cat((means.flatten(), cov_triu))

    param_list = {
        "means": means,
        "cov_inv": cov_inv,
        "log_normalisation_const": log_normalisation_const
    }

    return conditioning_params, param_list


def sample_gaussian_params_batch(
        batch_size: int,
        rng: torch.Generator,
        d: int
):
    """
    Batched version of sample_gaussian_params.
    """
    conditioning_dim = gaussian_conditioning_param_len(d)
    conditioning_params = torch.zeros(batch_size, conditioning_dim)

    params_list = []

    for i in range (batch_size):
        cond_params, param_dict = sample_gaussian_params(rng, d)
        conditioning_params[i] = cond_params
        params_list.append(param_dict)

    return conditioning_params, params_list


def gaussian_analytical(
        indices: torch.Tensor,
        grid: torch.Tensor,
        params_list: list,
):
    device = indices.device
    batch_size, n_samples, d = indices.shape
    outputs = []
    
    for batch in range(batch_size):
        X_b = torch.stack([grid[indices[batch, :, j]] for j in range(d)])

        p = params_list[batch]
        logp = gaussian_logpdf(
            X_b,
            p["means"].to(device),
            p["cov_inv"].to(device),
            p["log_normalisation_const"]
        )

        outputs.append(logp)

    return torch.stack(outputs, dim=0)
