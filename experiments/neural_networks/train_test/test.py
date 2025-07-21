import torch

# from shared_decoder_mps import NeuralMPS
from model.neural_mps import NeuralMPS

data = torch.load('../data/datasets/test.pt')  # same as output/tt_tensor.pt from create dataset, just truncated

ranks = [1, 2, 4, 4, 2, 1]

model = NeuralMPS(ranks, decoder_type='shared')
test_mu = torch.stack([x[0] for x in data])

test_output = model.forward(test_mu)
print('test')

