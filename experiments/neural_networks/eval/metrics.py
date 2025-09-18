import numpy as np


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Compute error metrics from arrays of shape (num_samples, N)."""
    residuals = (y_pred - y_true).ravel()
    abs_err = np.abs(residuals)

    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(abs_err))
    max_abs = float(np.max(abs_err))
    p95 = float(np.percentile(abs_err, 95))
    p99 = float(np.percentile(abs_err, 99))

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MaxAbsError': max_abs,
        'P95AbsError': p95,
        'P99AbsError': p99,
    }, residuals, abs_err


def gaussian_kde(xs: np.ndarray, points: int = 256):
    xs = xs.astype(np.float64)
    n = xs.size
    if n == 0:
        return np.array([0.0]), np.array([0.0])
    std = np.std(xs)
    if std <= 0 or not np.isfinite(std):
        grid = np.linspace(xs.min() - 1.0, xs.max() + 1.0, points)
        dens = np.zeros_like(grid)
        return grid, dens
    h = 1.06 * std * (n ** (-1 / 5))  # Silverman's rule
    xmin, xmax = np.quantile(xs, 0.005), np.quantile(xs, 0.995)
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
        xmin, xmax = xs.min(), xs.max()
        if xmin == xmax:
            xmin, xmax = xmin - 1.0, xmax + 1.0
    grid = np.linspace(xmin, xmax, points)
    # Compute density
    diffs = (grid[:, None] - xs[None, :]) / h
    coeff = 1.0 / (np.sqrt(2.0 * np.pi) * h)
    dens = coeff * np.exp(-0.5 * diffs ** 2).mean(axis=1)
    return grid, dens
