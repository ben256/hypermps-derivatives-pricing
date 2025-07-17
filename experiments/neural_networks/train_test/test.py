import numpy as np
import torch

# from shared_decoder_mps import NeuralMPS
from model.cnn_heads_architecture import NeuralMPS

data = torch.load('../output/mini_tt_tensor.pt', weights_only=False)  # same as output/tt_tensor.pt from create dataset, just truncated

ranks = [1, 2, 4, 4, 2, 1]

model = NeuralMPS(ranks)
test_mu = np.stack([x[0] for x in data])

test_mu = test_mu.reshape(test_mu.shape[0], -1)
test_mu = torch.from_numpy(test_mu).to(torch.float32)

test_output = model.forward(test_mu)
print('test')

