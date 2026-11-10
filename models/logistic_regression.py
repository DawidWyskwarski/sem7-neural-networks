import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)

from helpers.training_history import TrainingHistory


class LogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        learning_rate: float = 0.01,
        epoches: int = 100,
        batch_size: int = 32,
        pred_threshold: float = 0.5,
        random_weights: bool = True,
    ) -> None:
        super().__init__()

        self.learning_rate = learning_rate
        self.epoches = epoches
        self.pred_threshold = pred_threshold
        self.batch_size = batch_size

        self.history = TrainingHistory()

        self.random_weights = random_weights

    def fit(self, X, y):

        n_samples = X.shape[0]

        X_bias = np.c_[X, np.ones(n_samples)]

        self.history = TrainingHistory()

        n_features = X_bias.shape[1]

        if self.random_weights:
            self.weights_ = np.random.random_sample(n_features)
        else:
            self.weights_ = np.zeros(n_features)

        for _ in range(self.epoches):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for batch_start_idx in range(0, n_samples, self.batch_size):
                batch_indices = indices[
                    batch_start_idx : batch_start_idx + self.batch_size
                ]

                X_batch = X_bias[batch_indices]
                y_batch = y[batch_indices]

                y_batch_predicted = self._sigmoid(X_batch @ self.weights_)

                dw = ((y_batch - y_batch_predicted).T @ X_batch) / (-len(X_batch))

                self.weights_ -= self.learning_rate * dw.T

            self._update_history(X, y)

        return self

    def predict(self, X):
        X_bias = np.c_[X, np.ones(X.shape[0])]
        prob_positive = self._sigmoid(X_bias @ self.weights_)

        return prob_positive >= self.pred_threshold

    def predict_proba(self, X):
        X_bias = np.c_[X, np.ones(X.shape[0])]
        prob_positive = self._sigmoid(X_bias @ self.weights_)

        return np.vstack((1 - prob_positive, prob_positive))

    def _sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def _update_history(self, X, y):
        y_pred = self.predict(X)
        probs = self.predict_proba(X)

        self.history.append(
            loss=log_loss(y, probs.T),
            accuracy=accuracy_score(y, y_pred),
            precision=precision_score(y, y_pred, zero_division=0),
            recall=recall_score(y, y_pred, zero_division=0),
            f1_score=f1_score(y, y_pred, zero_division=0),
        )

    def plot_history(self, figsize=(14, 8), ncols: int = 3):
        return self.history.plot(
            figsize=figsize,
            ncols=ncols,
            title=(
                f"Training history epoches: {self.epoches}, "
                f"batch_size: {self.batch_size}, "
                f"learning_rate: {self.learning_rate}, "
                f"random_weights: {self.random_weights}"
            ),
        )
