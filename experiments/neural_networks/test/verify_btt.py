import matplotlib.pyplot as plt
import numpy as np
import torch
import tntorch as tn


def generate_covariance_matrix(
        rng: np.random.Generator,
        d: int,
        correlation: float = None,
):
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


def target_function(
        x: np.ndarray,
        A: float,
        c: np.ndarray,
        cov_matrix: np.ndarray
):
    c = c.reshape(-1, 1)
    cov_inv = np.linalg.inv(cov_matrix)

    diff = x - c
    exponent_term = -0.5 * np.einsum('ib,ij,jb->b', diff, cov_inv, diff)

    return torch.from_numpy(A * np.exp(exponent_term))


def function_wrapper(*ix, A, c, cov_matrix, N, format, device):
    if format == 'TT':
        d = len(ix)
        x_vector = []
        for i in range(d):
            indices = ix[i].cpu().numpy() if isinstance(ix[i], torch.Tensor) else ix[i]
            indices = indices.astype(int)
            x_vector.append(np.take(np.linspace(-1, 1, N), indices))
        out = target_function(np.stack(x_vector), A, c, cov_matrix)

    elif format == 'BTT':  # Binary base
        d = len(c)
        k = int(np.log2(N))
        x_vector = []
        for i in range(d):
            bits = ix[i * k : (i + 1) * k]
            bits_arr = [
                b.cpu().numpy().astype(int) if isinstance(b, torch.Tensor)
                else np.array(b, dtype=int)
                for b in bits
            ]
            idx = np.zeros_like(bits_arr[0], dtype=int)
            for j, bit in enumerate(bits_arr):
                idx += bit << (k - 1 - j)
            x_vector.append(np.linspace(-1, 1, N)[idx])
        out = target_function(np.stack(x_vector), A, c, cov_matrix)

    else:
        raise ValueError(f"Unsupported type: {format}")

    return out.to(device)


def eval_tt(tt_cores, x_indices):
    """Evaluates a TT at given indices."""
    v = tt_cores[0][:, :, x_indices[0], :]
    for i in range(1, len(tt_cores)):
        v = v @ tt_cores[i][:, :, x_indices[i], :]
    return v


def eval_btt(btt_cores, x_indices, N):
    """Evaluates a BTT at given indices."""
    k = int(np.log2(N))
    binary_indices = []
    for idx in x_indices:
        binary_indices.extend([int(b) for b in bin(idx)[2:].zfill(k)])

    v = btt_cores[0][:, :, binary_indices[0], :]
    for i in range(1, len(btt_cores)):
        v = v @ btt_cores[i][:, :, binary_indices[i], :]
    return v


def main():
    d = 4
    N_tt = 128
    N_btt = 64
    correlation = 0.3
    max_rank = 20
    device = 'cpu'
    seed = 42
    rng = np.random.default_rng(seed)

    A = rng.uniform(low=0.2, high=1.0)
    c = rng.uniform(low=-0.5, high=0.5, size=d)
    cov_matrix = generate_covariance_matrix(rng, d, correlation=correlation)

    print("Running TT-Cross for standard TT...")
    domain_tt = [torch.arange(N_tt, device=device) for _ in range(d)]
    ranks_tt = [max_rank] * (d - 1)
    tt_tensor = tn.cross(
        function=lambda *ix: function_wrapper(*ix, A=A, c=c, cov_matrix=cov_matrix, N=N_tt, format='TT', device=device),
        domain=domain_tt,
        ranks_tt=ranks_tt,
        verbose=False,
        device=device
    )
    tt_tensor.cores = [c.unsqueeze(0) for c in tt_tensor.cores]
    print("TT-Cross for TT finished.")

    print("Running TT-Cross for BTT...")
    k = int(np.log2(N_btt))
    domain_btt = [torch.arange(2, device=device) for _ in range(d * k)]
    ranks_btt = [1]
    for i in range(d * k):
        if len(ranks_btt) < (d * k) // 2:
            ranks_btt.append(min(ranks_btt[-1]*2, max_rank))
        else:
            ranks_btt.append(min(ranks_btt[-1]*2, max_rank))
            break
    ranks_btt.extend(ranks_btt[::-1][1:])
    ranks_btt = ranks_btt[1:-1]

    btt_tensor = tn.cross(
        function=lambda *ix: function_wrapper(*ix, A=A, c=c, cov_matrix=cov_matrix, N=N_btt, format='BTT', device=device),
        domain=domain_btt,
        ranks_tt=ranks_btt,
        verbose=False,
        device=device
    )
    print(btt_tensor)
    btt_tensor.cores = [c.unsqueeze(0) for c in btt_tensor.cores]
    print("TT-Cross for BTT finished.")

    grid = np.linspace(-1, 1, 100)
    tt_grid = np.linspace(-1, 1, N_tt)
    btt_grid = np.linspace(-1, 1, N_btt)

    original_values = target_function(grid.reshape(1, -1), A, c, cov_matrix).numpy()
    original_values_tt = target_function(tt_grid.reshape(1, -1), A, c, cov_matrix).numpy()
    original_values_btt = target_function(btt_grid.reshape(1, -1), A, c, cov_matrix).numpy()

    tt_values = np.array([eval_tt(tt_tensor.cores, [i]*d) for i in range(N_tt)]).squeeze()
    btt_values = np.array([eval_btt(btt_tensor.cores, [i]*d, N_btt) for i in range(N_btt)]).squeeze()

    tt_distances = np.linalg.norm(tt_values - original_values_tt)
    btt_distances = np.linalg.norm(btt_values - original_values_btt)
    print(f'TT-Cross (TT) distances: {tt_distances}')
    print(f'BTT-Cross (BTT) distances: {btt_distances}')

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(grid, original_values, label='Original Function', color='black', linewidth=2)
    plt.plot(tt_grid, tt_values, label='TT-Cross (TT)', color='red', linestyle='--', marker='o', markersize=4)
    plt.plot(btt_grid, btt_values, label='TT-Cross (BTT)', color='blue', linestyle=':', marker='x', markersize=4)
    plt.title('Comparison of TT and BTT Representations')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
