from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, architecture: nn.Sequential):
        super().__init__()

        self.flatten = nn.Flatten()
        self.architecture = architecture

    def forward(self, x):
        x = self.flatten(x)
        logits = self.architecture(x)
        return logits
