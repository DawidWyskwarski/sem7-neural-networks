from typing import Callable, Optional
import copy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models.activation_functions import ActivationFunction, Sigmoid
from models.loss_functions import LossFunction
from models.weight_initializer import WeightInitializer, NormalInitializer
from models.optimizer import Optimizer


class Layer:
    def __init__(
        self,
        size: int,
        input_dim: int,
        activation_function: ActivationFunction,
        weight_initializer: WeightInitializer,
        optimizer: Optimizer,
        rng: Optional[np.random.Generator] = None,
    ):
        self.activation_function = activation_function
        self.optimizer = optimizer

        self.weights_ = weight_initializer.initialize((size, input_dim), rng=rng)
        self.biases_ = np.zeros((1, size))

        self.weight_optimizer = copy.deepcopy(optimizer)
        self.bias_optimizer = copy.deepcopy(optimizer)

        self.cache_input = None
        self.cache_transformed = None

    def forward(self, X):
        self.cache_input = X
        self.cache_transformed = X @ self.weights_.T + self.biases_

        return self.activation_function.apply(self.cache_transformed)

    def backward(self, derivative):
        base_derivative = derivative * self.activation_function.derivative(
            self.cache_transformed
        )

        new_derivative = base_derivative @ self.weights_

        dw = base_derivative.T @ self.cache_input
        db = base_derivative.sum(axis=0, keepdims=True)

        self.weights_ -= self.weight_optimizer.step(dw)
        self.biases_ -= self.bias_optimizer.step(db)

        return new_derivative


class MultiLayerPerceptron(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        hidden_layer_sizes: list[int],
        loss_function: LossFunction,
        activation_function: ActivationFunction,
        weight_initializer: WeightInitializer,
        optimizer: Optimizer,
        batch_size: int = 32,
        max_iter: int = 1000,
        stop_criterion: Optional[Callable[[dict], bool]] = None,
        random_state: Optional[int] = None,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.loss_function = loss_function

        self.activation_function = activation_function
        self.weight_initializer = weight_initializer
        self.optimizer = optimizer

        self.batch_size = batch_size
        self.max_iter = max_iter
        self.stop_criterion = stop_criterion
        self.random_state = random_state

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

        y = np.asarray(y, dtype=float).reshape(predicted.shape)
        gradient = self.loss_function.derivative(y, predicted) / predicted.shape[0]

        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)

        rng = np.random.default_rng(self.random_state)
        self._initialize_layers(n_features, rng)
        self.history = {
            "loss": [],
            "f1_score": [],
            "precision": [],
            "recall": [],
            "accuracy": [],
        }

        for _ in range(self.max_iter):
            indices = np.arange(n_samples)
            rng.shuffle(indices)

            for batch_start_idx in range(0, n_samples, self.batch_size):
                batch_indices = indices[
                    batch_start_idx : batch_start_idx + self.batch_size
                ]

                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                y_batch_predicted = self.forward(X_batch)
                self.backward(y_batch, y_batch_predicted)

            self._update_history(X, y)

            if self.stop_criterion is not None and self.stop_criterion(self.history):
                break

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return (self.forward(X).ravel() >= 0.5).astype(int)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        p1 = self.forward(X).ravel()
        return np.column_stack([1 - p1, p1])

    def _initialize_layers(self, input_size, rng: Optional[np.random.Generator] = None):
        self.layers = []

        previous_size = input_size

        for layer_size in self.hidden_layer_sizes:
            self.layers.append(
                Layer(
                    size=layer_size,
                    input_dim=previous_size,
                    weight_initializer=self.weight_initializer,
                    optimizer=self.optimizer,
                    activation_function=self.activation_function,
                    rng=rng,
                )
            )

            previous_size = layer_size

        self.layers.append(
            Layer(
                size=1,
                input_dim=previous_size,
                weight_initializer=NormalInitializer(
                    0.01, random_state=self.random_state
                ),
                optimizer=self.optimizer,
                activation_function=Sigmoid(),
                rng=rng,
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

    def plot_history(self, figsize=(14, 8), ncols: int = 3):
        metrics = {
            name: values for name, values in self.history.items() if len(values) > 0
        }

        n_plots = len(metrics)
        ncols = min(ncols, n_plots)
        nrows = int(np.ceil(n_plots / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        axes_flat = axes.ravel()

        for ax, (name, values) in zip(axes_flat, metrics.items()):
            ax.plot(range(1, len(values) + 1), values, marker="o", markersize=3)
            ax.set_title(name.replace("_", " ").title())
            ax.set_xlabel("Epoch")
            ax.set_ylabel(name.replace("_", " "))
            ax.grid(True, alpha=0.3)

        for ax in axes_flat[n_plots:]:
            ax.set_visible(False)

        fig.suptitle(
            "MLP Hyperparameters: "
            f"hidden_layer_sizes={self.hidden_layer_sizes}, "
            f"loss={self.loss_function.__class__.__name__}, "
            f"activation={self.activation_function.__class__.__name__}, "
            f"initializer={self.weight_initializer.__class__.__name__}, "
            f"optimizer={self.optimizer.__class__.__name__}, "
            f"batch_size={self.batch_size}, max_iter={self.max_iter}"
        )
        fig.tight_layout()

        return fig, axes
