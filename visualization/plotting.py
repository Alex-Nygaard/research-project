import scienceplots
import matplotlib.pyplot as plt
from visualization.data import Metric, RunData
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


def create():

    # new version run_3739099 (dep), run_3739100, run_3739101, run_3739102

    # deployment = RunData.build("run_3739099")
    simulations = RunData.build_many(["run_3739100", "run_3739101", "run_3739102"])

    runs = simulations  # [deployment] +

    for key in ["accuracy"]:  # , "loss"]:
        plot_metrics(
            [run.get_metric(key) for run in runs],
            title=f"{key.capitalize()} per Round",
            x_label="Round number",
            y_label=key.capitalize(),
            save_dir=f"for-report/{key}.png",
        )

    RunData.many_to_csv(simulations, "for-report", "results.csv")
