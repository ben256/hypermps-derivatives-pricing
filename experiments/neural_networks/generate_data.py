import numpy as np
import tntorch as tn
import torch


def target_function(x, y):
    return x**2 + y**2


def function_wrapper(*ix):
    x_idx = 0
    for i in range(5):
        x_idx += ix[i] * (2**i)

    y_idx = 0
    for i in range(5):
        y_idx += ix[i+5] * (2**i)

    x_coord = np.take(np.linspace(-1, 1, 32), x_idx)
    y_coord = np.take(np.linspace(-1, 1, 32), y_idx)

    return torch.from_numpy(target_function(x_coord, y_coord))


d = 2
N = 32
domain = [2] * 10

x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
xx, yy = np.meshgrid(x, y)
func_matrix = target_function(xx, yy)

qtt_tensor_full = func_matrix.reshape(domain)

tt_tensor, tt_info = tn.cross(
    function=function_wrapper,
    domain=domain,
    eps=1e-5,
    rmax=10,
    max_iter=50,
    kickrank=1,
    early_stopping_patience=5,
    early_stopping_tolerance=1e-7,
    return_info=True
)

torch.save(tt_tensor, './output/ground_truth_tt.pt')
