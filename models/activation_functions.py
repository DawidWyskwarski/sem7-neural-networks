import numpy as np

class ActivationFunction:

    def apply(self, input):
        pass

    def derivative(self, input):
        pass

class ReLu(ActivationFunction):

    def apply(self, input):
        return np.maximum(0, input)

    def derivative(self, input):
        return np.where(input > 0, 1.0, 0.0)

class Sigmoid(ActivationFunction):

    def apply(self, input):
        return 1.0 / (1.0 + np.exp(-input))

    def derivative(self, input):
        s = self.apply(input)
        return s * (1.0 - s)