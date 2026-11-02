from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    @abstractmethod
    def step(self, grad: np.ndarray) -> np.ndarray:
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def step(self, grad):
        return self.learning_rate * grad


class Adam(Optimizer):
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self._m = None
        self._v = None
        self._t = 0

    def step(self, grad):
        if self._m is None:
            self._m = np.zeros_like(grad)
            self._v = np.zeros_like(grad)

        self._t += 1
        self._m = self.beta1 * self._m + (1 - self.beta1) * grad
        self._v = self.beta2 * self._v + (1 - self.beta2) * grad**2

        m_hat = self._m / (1 - self.beta1**self._t)
        v_hat = self._v / (1 - self.beta2**self._t)

        return self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
