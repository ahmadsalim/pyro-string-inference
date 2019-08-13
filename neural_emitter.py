import torch
import torch.nn


class NeuralEmitter(torch.nn.Module):
    def __init__(self, gru_hidden_size: int, alphabet_size: int, *, smoothing: float, intermediate_size: int = 100):
        super(NeuralEmitter, self).__init__()
        self.gru_hidden_size = gru_hidden_size
        self.alphabet_size = alphabet_size
        self.intermediate_size = intermediate_size
        self.smoothing = smoothing
        self.network = torch.nn.Sequential(
            torch.nn.Linear(gru_hidden_size, intermediate_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(intermediate_size, alphabet_size)
        )

    def forward(self, *input):
        nout = self.network(*input)
        return torch.exp(nout) + self.smoothing
