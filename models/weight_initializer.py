from abc import ABC, abstractmethod
import numpy as np


class WeightInitializer(ABC):
    @abstractmethod
    def initialize(self, input_dim: tuple):
        pass


class ZeroInitializer(WeightInitializer):
    def initialize(self, input_dim: tuple):
        return np.zeros(input_dim)


class HeInitializer(WeightInitializer):
    def initialize(self, input_dim: tuple):
        return np.random.normal(0, np.sqrt(2 / input_dim), input_dim)


class NormalInitializer(WeightInitializer):
    def __init__(self, deviation: float):
        self.deviation = deviation

    def initialize(self, input_dim: tuple):
        return np.random.normal(0, self.deviation, input_dim)
