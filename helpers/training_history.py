import math

import matplotlib.pyplot as plt


class TrainingHistory:
    def __init__(self):
        self.loss = []
        self.accuracy = []
        self.f1_score = []
        self.precision = []
        self.recall = []

    def append(
        self,
        loss: float,
        accuracy: float,
        f1_score: float,
        precision: float,
        recall: float,
    ) -> None:
        self.loss.append(loss)
        self.accuracy.append(accuracy)
        self.f1_score.append(f1_score)
        self.precision.append(precision)
        self.recall.append(recall)

    def __getitem__(self, key: str) -> list[float]:
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def plot(self, figsize=(14, 8), ncols: int = 3, title: str = "Training History"):
        metrics = {
            name: values
            for name, values in {
                "loss": self.loss,
                "accuracy": self.accuracy,
                "f1_score": self.f1_score,
                "precision": self.precision,
                "recall": self.recall,
            }.items()
            if len(values) > 0
        }

        n_plots = len(metrics)
        if n_plots == 0:
            raise ValueError("No metrics to plot")

        ncols = min(ncols, n_plots)
        nrows = math.ceil(n_plots / ncols)

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

        fig.suptitle(title)
        fig.tight_layout()

        return fig, axes


def plot_compare(
    *histories: tuple[str, TrainingHistory],
    figsize=(14, 8),
    ncols: int = 3,
    title: str = "Training History Comparison",
):
    if len(histories) == 0:
        raise ValueError("At least one (name, TrainingHistory) pair is required")

    metric_names = ("loss", "accuracy", "f1_score", "precision", "recall")
    metrics = [
        name
        for name in metric_names
        if any(len(history[name]) > 0 for _, history in histories)
    ]

    n_plots = len(metrics)
    if n_plots == 0:
        raise ValueError("No metrics to plot")

    ncols = min(ncols, n_plots)
    nrows = math.ceil(n_plots / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    for ax, metric in zip(axes_flat, metrics):
        for name, history in histories:
            values = history[metric]
            if len(values) == 0:
                continue
            ax.plot(
                range(1, len(values) + 1),
                values,
                marker="o",
                markersize=3,
                label=name,
            )
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax in axes_flat[n_plots:]:
        ax.set_visible(False)

    fig.suptitle(title)
    fig.tight_layout()

    return fig, axes
