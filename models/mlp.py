from typing import Callable, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from models.activation_functions import ActivationFunction, Sigmoid
from models.loss_functions import LossFunction
from models.weight_initializer import WeightInitializer


class Neuron:
    def __init__(
        self,
        activation_function: ActivationFunction,
        weight_initializer: WeightInitializer,
        input_dim: int,
        learning_rate: float,
    ):
        self.activation_function = activation_function

        self.learning_rate = learning_rate

        self.weights_ = weight_initializer.initialize(input_dim)
        self.bias = 0

        self.cache_input = None
        self.cache_transformed = None

    def forward(self, input):
        self.cache_input = input
        self.cache_transformed = (input @ self.weights_) + self.bias

        return self.activation_function.apply(self.cache_transformed)

    def backward(self, derivative):
        base_derivative = derivative * self.activation_function.derivative(
            self.cache_transformed
        )

        new_derivative = base_derivative * self.weights_

        dw = base_derivative * self.cache_input
        db = base_derivative

        self.weights_ -= self.learning_rate * dw.T
        self.bias -= self.learning_rate * db

        return new_derivative


class Layer:
    def __init__(
        self,
        size: int,
        activation_function: ActivationFunction,
        weight_initializer: WeightInitializer,
        input_dim: int,
        learning_rate: float,
    ):
        self.neurons = [
            Neuron(
                activation_function=activation_function,
                weight_initializer=weight_initializer,
                input_dim=input_dim,
                learning_rate=learning_rate,
            )
            for _ in range(size)
        ]

    def forward(self, X):
        return np.array([neuron.forward(X) for neuron in self.neurons])

    def backward(self, derivative):
        grad_input = np.zeros_like(self.neurons[0].weights_)

        for neuron, d in zip(self.neurons, derivative):
            grad_input += neuron.backward(d)

        return grad_input


class MultiLayerPerceptron(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        hidden_layer_sizes: list[int],
        activation_function: ActivationFunction,
        weight_initializer: WeightInitializer,
        loss_function: LossFunction,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        stop_criterion: Optional[Callable[[dict], bool]] = None,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.loss_function = loss_function

        self.activation_function = activation_function
        self.weight_initializer = weight_initializer
        self.learning_rate = learning_rate

        self.max_iter = max_iter
        self.stop_criterion = stop_criterion

        self.layers: list[Layer] = []

        self.history: dict = {
            "loss": [],
            "f1_score": [],
            "precision": [],
            "recall": [],
            "accuracy": [],
        }

    def forward(self, input):
        if self.layers == []:
            raise RuntimeError("Layers not initialized")

        tmp = input

        for layer in self.layers:
            tmp = layer.forward(tmp)

        return tmp

    def backward(self, y, predicted):
        if self.layers == []:
            raise RuntimeError("Layers not initialized")

        gradient = self.loss_function.derivative(y, predicted)

        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)

        self._initialize_layers(n_features)
        self.history = {
            "loss": [],
            "f1_score": [],
            "precision": [],
            "recall": [],
            "accuracy": [],
        }

        for _ in range(self.max_iter):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for i in indices:
                y_predicted = self.forward(X[i])
                self.backward(y[i], y_predicted)

            self._update_history(X, y)

            if self.stop_criterion is not None and self.stop_criterion(self.history):
                break

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([int(self.forward(x)[0] >= 0.5) for x in X])

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        p1 = np.array([self.forward(x)[0] for x in X])
        return np.column_stack([1 - p1, p1])

    def _initialize_layers(self, input_size):
        self.layers = []

        previous_size = input_size

        for layer_size in self.hidden_layer_sizes:
            self.layers.append(
                Layer(
                    size=layer_size,
                    input_dim=previous_size,
                    weight_initializer=self.weight_initializer,
                    learning_rate=self.learning_rate,
                    activation_function=self.activation_function,
                )
            )

            previous_size = layer_size

        self.layers.append(
            Layer(
                size=1,
                input_dim=previous_size,
                weight_initializer=self.weight_initializer,
                learning_rate=self.learning_rate,
                activation_function=Sigmoid(),
            )
        )

    def _update_history(self, X, y):
        p1 = self.predict_proba(X)[:, 1]
        y_pred = (p1 >= 0.5).astype(int)

        epoch_loss = float(np.mean(self.loss_function.apply(y, p1)))
        epoch_acc = accuracy_score(y, y_pred)
        epoch_prec = precision_score(y, y_pred, zero_division=0)
        epoch_rec = recall_score(y, y_pred, zero_division=0)
        epoch_f1 = f1_score(y, y_pred, zero_division=0)

        self.history["loss"].append(epoch_loss)
        self.history["accuracy"].append(epoch_acc)
        self.history["precision"].append(epoch_prec)
        self.history["recall"].append(epoch_rec)
        self.history["f1_score"].append(epoch_f1)
