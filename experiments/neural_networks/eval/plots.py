import numpy as np
from matplotlib import pyplot as plt

from metrics import gaussian_kde


def plot_slices(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        d: int,
        N: int,
        output_path: str,
):

    indices = np.random.choice(y_true.shape[0], size=4, replace=False)
    y_true = y_true[indices]
    y_pred = y_pred[indices]

    x = np.linspace(0, 1, y_true.shape[1])

    fig, ax = plt.subplots(2, 2, figsize=(14, 10), tight_layout=True)

    for values in zip(y_true, y_pred, ax.ravel()):
        true, pred, axis = values
        axis.plot(x, true, label='True')
        axis.plot(x, pred, label='Predicted')
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.legend()
        axis.grid(visible=True, which='major')
        axis.grid(visible=True, which='minor', linestyle='-', alpha=0.3)
        axis.minorticks_on()

    fig.suptitle(f'True vs Predicted ({d}D, N={N})')

    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_parity(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        d: int,
        N: int,
        output_path: str
):
    vmin = min(y_true.min(), y_pred.min())
    vmax = max(y_true.max(), y_pred.max())

    fig, ax = plt.subplots(figsize=(7, 7), tight_layout=True)

    ax.scatter(y_true, y_pred, s=6, alpha=0.3, edgecolors='none')
    ax.plot([vmin, vmax], [vmin, vmax], 'r-', linewidth=1.5, label='y=x')

    ax.set_title(f'Parity Plot ({d}D, N={N})')
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')

    ax.set_yscale('log')
    ax.set_xscale('log')

    plt.grid(visible=True, which='major')
    plt.legend()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_residuals(
        residuals: np.ndarray,
        abs_err: np.ndarray,
        metrics: dict,
        d: int,
        N: int,
        output_path: str
):

    fig, ax = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)

    gx, gd = gaussian_kde(residuals)
    ax[0].hist(residuals, bins=60, density=True, alpha=0.5, label='Residuals')
    ax[0].plot(gx, gd, 'r-', label='KDE')

    ax[0].set_title('Residuals (pred - true)')
    ax[0].set_xlabel('Residual')
    ax[0].set_ylabel('Density')
    ax[0].legend()
    ax[0].grid(visible=True, which='major')
    ax[0].grid(visible=True, which='minor', linestyle='-', alpha=0.3)
    ax[0].minorticks_on()

    ax[1].hist(abs_err, bins=60, density=True, alpha=0.5, label='|Error|')
    ax[1].axvline(metrics['P95AbsError'], color='k', linestyle='--', linewidth=1.2, label='P95')
    ax[1].axvline(metrics['P99AbsError'], color='k', linestyle='-.', linewidth=1.2, label='P99')

    ax[1].set_title('Absolute Error with P95/P99')
    ax[1].set_xlabel('|Error|')
    ax[1].set_ylabel('Density')
    ax[1].legend()
    ax[1].grid(visible=True, which='major')
    ax[1].grid(visible=True, which='minor', linestyle='-', alpha=0.3)
    ax[1].minorticks_on()

    fig.suptitle(f'Residual analysis ({d}D, N={N})')

    plt.savefig(output_path, dpi=300)
    plt.close()

