from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class WeightInitializer(ABC):
    @abstractmethod
    def initialize(self, input_dim: tuple, rng: Optional[np.random.Generator] = None):
        pass


class ZeroInitializer(WeightInitializer):
    def initialize(self, input_dim: tuple, rng: Optional[np.random.Generator] = None):
        return np.zeros(input_dim)


class HeInitializer(WeightInitializer):
    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state

    def initialize(self, input_dim: tuple, rng: Optional[np.random.Generator] = None):
        rng = rng if rng is not None else np.random.default_rng(self.random_state)
        # weight shape is (out_features, in_features); He/Kaiming fan_in uses in_features
        fan_in = input_dim[1]
        std = np.sqrt(2.0 / fan_in)
        return rng.normal(0, std, size=input_dim)


class NormalInitializer(WeightInitializer):
    def __init__(self, deviation: float, random_state: Optional[int] = None):
        self.deviation = deviation
        self.random_state = random_state

    def initialize(self, input_dim: tuple, rng: Optional[np.random.Generator] = None):
        rng = rng if rng is not None else np.random.default_rng(self.random_state)
        return rng.normal(0, self.deviation, input_dim)
