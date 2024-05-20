from typing import List


def create():
    from visualization.data import RunData
    from visualization.plotting import plot_metrics

    deployment = RunData.build("run_3727381")
    simulations = RunData.build_many(
        ["run_3727382", "run_3727383", "run_3727384", "run_3727385", "run_3727386"]
    )

    runs = [deployment] + simulations

    for key in ["accuracy"]:  # , "loss"]:
        plot_metrics(
            [run.get_metric(key) for run in runs],
            title=f"{key.capitalize()} per Round",
            x_label="Round number",
            y_label="Value",
            save_dir=f"for-report/{key}.png",
        )
