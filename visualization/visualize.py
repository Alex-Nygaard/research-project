from typing import List


def create():
    from visualization.data import RunData
    from visualization.plotting import plot_metrics

    deployment = RunData.build("run_3711520")
    simulations = RunData.build_many(["run_3711645", "run_3711513", "run_3711520"])

    runs = [deployment] + simulations

    plot_metrics(
        [run.get_metric("accuracy") for run in runs],
        title="Accuracy per Round",
        x_label="Round number",
        y_label="Value",
        save_dir="accuracy.png",
    )
