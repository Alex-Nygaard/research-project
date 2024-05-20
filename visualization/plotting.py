import scienceplots
import matplotlib.pyplot as plt

from visualization.data import History, Metric
from logger.logger import get_logger

plt.style.use(["science", "ieee"])

logger = get_logger("visualization.plotting")


def plot_metrics(
    metrics: list[Metric],
    title: str = "Metrics per Round",
    x_label="Round number",
    y_label="Value",
    save_dir: str = None,
):
    plt.figure(figsize=(10, 6))

    for metric in metrics:
        plot_single_metric(metric)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    if save_dir:
        logger.info(f"Saving figure to {save_dir}.")
        plt.savefig(save_dir)
    else:
        plt.show()


def plot_single_metric(metric: Metric):
    plt.plot(
        metric.x,
        metric.y,
        label=metric.label,
        color=metric.style.color,
        marker=metric.style.marker,
        linestyle=metric.style.linestyle,
    )
