from abc import ABC, abstractmethod
from collections import defaultdict

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.nn import Module
from torch.nn.modules.loss import _WeightedLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class _TorchClassifier(ABC):
    def __init__(
        self,
        model: Module,
        loss_fn: _WeightedLoss,
        optimizer: Optimizer,
        device: str = "cpu",
    ):
        self.device = device
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.history: dict[str, list[float]] = defaultdict(list)

    def _run_epoch(self, dataloader: DataLoader) -> None:
        self.model.train()

        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)

            loss = self.loss_fn(self.model(X), y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    @abstractmethod
    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        pass

    def train(
        self,
        dataloader: DataLoader,
        max_iter: int,
        verbose: bool = False
    ) -> None:
        for e in range(max_iter):
            if verbose:
                print(f"Epoch: {e}")

            self._run_epoch(dataloader)
            self._log(self.evaluate(dataloader))

    def _log(self, metrics: dict[str, float]) -> None:
        for name, value in metrics.items():
            self.history[name].append(value)

    @staticmethod
    def _compute_metrics(
        total_loss: float,
        n_samples: int,
        y_true,
        y_pred,
        average: str,
        labels: list[int] | None = None,
    ) -> dict[str, float]:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=average, zero_division=0
        )
        return {
            "loss": total_loss / n_samples,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


class TorchBinaryClassifier(_TorchClassifier):
    def __init__(self, *args, threshold: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        self.model.eval()

        preds, targets = [], []
        total_loss = 0.0

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)

                logits = self.model(X)
                total_loss += self.loss_fn(logits, y).item() * len(X)

                probs = torch.sigmoid(logits.squeeze(1))
                preds.append((probs > self.threshold).long().cpu())
                targets.append(y.squeeze(1).long().cpu())

        return self._compute_metrics(
            total_loss,
            len(dataloader.dataset),
            torch.cat(targets).numpy(),
            torch.cat(preds).numpy(),
            average="binary",
        )


class TorchMulticlassClassifier(_TorchClassifier):
    def __init__(self, *args, num_classes: int, average: str = "macro", **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.average = average

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        self.model.eval()

        preds, targets = [], []
        total_loss = 0.0

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)

                logits = self.model(X)
                total_loss += self.loss_fn(logits, y).item() * len(X)

                preds.append(logits.argmax(dim=1).cpu())
                targets.append(y.cpu())

        return self._compute_metrics(
            total_loss,
            len(dataloader.dataset),
            torch.cat(targets).numpy(),
            torch.cat(preds).numpy(),
            average=self.average,
            labels=list(range(self.num_classes)),
        )
