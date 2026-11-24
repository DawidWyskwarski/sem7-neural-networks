from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, architecture: nn.Sequential, flatten_input: bool = True):
        super().__init__()
        self.flatten = nn.Flatten() if flatten_input else nn.Identity()
        self.architecture = architecture

    def forward(self, x):
        return self.architecture(self.flatten(x))
