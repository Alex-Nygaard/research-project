from typing import List


def create():
    from visualization.data import RunData
    from visualization.plotting import plot_metrics

    deployment = RunData.build("run_3711520")
    simulations = RunData.build_many(["run_3711645", "run_3711513", "run_3711520"])

    runs = [deployment] + simulations

    for key in ["accuracy", "loss"]:
        plot_metrics(
            [run.get_metric(key) for run in runs],
            title=f"{key.capitalize()} per Round",
            x_label="Round number",
            y_label="Value",
            save_dir=f"from-delftblue/logs/{key}.png",
        )
