import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

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

        self.history: dict = {
            "loss": [],
            "f1_score": [],
            "precision": [],
            "recall": [],
            "accuracy": [],
        }
        
        self.random_weights = random_weights

    def fit(self, X, y):
        
        n_samples= X.shape[0]

        X_bias = np.c_[X, np.ones(n_samples)]

        self.history = {key: [] for key in self.history}

        n_features = X_bias.shape[1]

        if self.random_weights:
            self.weights_ = np.random.random_sample(n_features)
        else:
            self.weights_ = np.zeros(n_features)

        for _ in range(self.epoches):
            
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            
            for batch_start_idx in range(0, n_samples, self.batch_size):

                batch_indices = indices[batch_start_idx : batch_start_idx + self.batch_size]
                
                X_batch = X_bias[batch_indices]
                y_batch = y[batch_indices]

                y_batch_predicted = self._sigmoid( X_batch @ self.weights_ )

                dw = ((y_batch - y_batch_predicted).T @ X_batch)/(-len(X_batch))

                self.weights_ -= self.learning_rate * dw.T

            self._update_history(X, y)

        return self


    def predict(self, X):
        X_bias = np.c_[X, np.ones(X.shape[0])]
        prob_positive = self._sigmoid( X_bias @ self.weights_ )
        
        return prob_positive >= self.pred_threshold


    def predict_proba(self, X):
        X_bias = np.c_[X, np.ones(X.shape[0])]
        prob_positive = self._sigmoid( X_bias @ self.weights_ )

        return np.vstack((1 - prob_positive, prob_positive))


    def _sigmoid(self, X):
        return 1 / (1 + np.exp(-X))


    def _update_history(self, X, y):
        y_pred = self.predict(X)
        probs = self.predict_proba(X)
    
        epoch_loss = log_loss(y, probs.T)
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
        metrics = {name: values for name, values in self.history.items() if len(values) > 0}

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

        fig.suptitle(f"Training history epoches: {self.epoches}, batch_size: {self.batch_size}, learning_rate: {self.learning_rate}, random_weights: {self.random_weights}")
        fig.tight_layout()

        return fig, axes
