from abc import ABC, abstractmethod
import numpy as np


class LossFunction(ABC):
    @abstractmethod
    def apply(self, y, predicted):
        pass

    @abstractmethod
    def derivative(self, y, predicted):
        pass


class LogLoss(LossFunction):
    def apply(self, y, predicted):
        predicted = np.clip(predicted, 1e-12, 1 - 1e-12)
        return (-y * np.log(predicted)) - ((1 - y) * np.log(1 - predicted))

    def derivative(self, y, predicted):
        predicted = np.clip(predicted, 1e-12, 1 - 1e-12)
        return (-y / predicted) + ((1 - y) / (1 - predicted))


class PyTorchLogLoss(LossFunction):
    def apply(self, y, predicted):
        with np.errstate(divide="ignore"):
            log_p   = np.maximum(np.log(predicted), -100.0)
            log_1mp = np.maximum(np.log(1 - predicted), -100.0)
        return -(y * log_p + (1 - y) * log_1mp)

    def derivative(self, y, predicted):
        return (predicted - y) / np.maximum(predicted * (1 - predicted), 1e-12)
