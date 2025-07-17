import numpy as np
import tntorch as tn
import torch
from tqdm import tqdm


def target_function(
        x: np.ndarray,
        mu: np.ndarray,
):
    """
    Target function:
    $f(x | \mu) = \sum^{K}_{k=1} A_k \exp \left(- \frac{(x-c_k)^2}{2 \sigma^2_k} \right)$
    -> sum of K Gaussian functions, parameterised by $\mu$
    where $\mu = (A, c, \sigma)$ is a matrix shape (K, 3) with K being the number of components.
    """
    A, c, sigma = mu.T
    A = A.reshape(-1, 1)
    c = c.reshape(-1, 1)
    sigma = sigma.reshape(-1, 1)

    return np.sum(A * np.exp(-((x - c) ** 2) / (2 * sigma ** 2)), axis=0)


def function_wrapper(*ix, mu):
    x_idx = 0
    for i in range(len(ix)):
        x_idx += ix[i] * (2**i)

    N = 2**len(ix)
    x_coord = np.take(np.linspace(-1, 1, N), x_idx)

    return torch.from_numpy(target_function(x_coord, mu))

initial_seed = 42
k = 5
A_range = [0.2, 1.0]
c_range = [-0.8, 0.8]
sigma_range = [0.05, 0.3]
low_bounds = [A_range[0], c_range[0], sigma_range[0]]
high_bounds = [A_range[1], c_range[1], sigma_range[1]]

n_samples = 10000
d = 5
N = 32
domain = [2] * d
x = np.linspace(-1, 1, N)

train_size, val_size, test_size = 0.8, 0.1, 0.1

data = []

for n in tqdm(range(n_samples)):
    seed = initial_seed + n
    rng = np.random.default_rng(seed)
    mu = rng.uniform(low=low_bounds, high=high_bounds, size=(k, 3))

    tt_tensor= tn.cross(
        function=lambda *ix: function_wrapper(*ix, mu=mu),
        domain=domain,
        eps=1e-14,
        ranks_tt=[2, 4, 4, 2],
        max_iter=50,
        early_stopping_patience=5,
        early_stopping_tolerance=1e-7,
        verbose=False,
    )

    mu = np.concatenate(mu)
    mu = torch.from_numpy(mu).to(torch.float32)

    test = np.array(tt_tensor.cores)

    data.append((
        mu,
        tt_tensor.cores,
    ))

np.random.seed(initial_seed)
np.random.shuffle(data)

train_end = int(train_size * len(data))
val_end = train_end + int(val_size * len(data))
train_data = data[:train_end]
val_data = data[train_end:val_end]
test_data = data[val_end:]

torch.save(train_data, '../data/datasets/train.pt')
torch.save(val_data, '../data/datasets/val.pt')
torch.save(test_data, '../data/datasets/test.pt')
