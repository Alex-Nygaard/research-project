import os

import scienceplots
import pandas as pd
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

    # new version run_3740021 (dep), run_3740022, run_3740023, run_3740024

    # runs = RunData.build_many(
    #     ["run_3739595", "run_3739596", "run_3739569", "run_3739568"]
    # )

    deployment = ["run_3740021"]
    simulation_base = ["run_3740023"]

    simulation_concentration = ["run_3740022", "run_3740024"]
    simulation_resources = ["run_3740245", "run_3740246"]
    simulation_variability = ["run_3740247", "run_3740248"]

    run_combinations = {
        "concentration": deployment + simulation_base + simulation_concentration,
        "resources": deployment + simulation_base + simulation_resources,
        "variability": deployment + simulation_base + simulation_variability,
    }
    run_combinations = {
        key: RunData.build_many(runs) for key, runs in run_combinations.items()
    }

    all_csv_files = []
    output_path = "for-report"

    for category, runs in run_combinations.items():
        for key in ["accuracy", "loss"]:
            plot_metrics(
                [run.get_metric(key) for run in runs],
                title=f"{key.capitalize()} per Round",
                x_label="Round number",
                y_label=key.capitalize(),
                save_dir=f"for-report/{category}-{key}.png",
            )

        filename = f"{category}-results.csv"
        RunData.many_to_csv(runs, output_path, filename)
        logger.info(f"Saved {filename} to {output_path}.")
        all_csv_files.append(os.path.join(output_path, filename))

    combine_csv_files(all_csv_files, output_path, "all-results.csv")


def combine_csv_files(csv_files: list[str], output_path: str, filename: str):
    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not dfs:
        raise ValueError("No DataFrames to concatenate.")

    custom_order = [
        "dep_R-mid_C-mid_V-mid_Q-mid",
        "sim_R-mid_C-mid_V-mid_Q-mid",
        "sim_R-low_C-mid_V-mid_Q-mid",
        "sim_R-high_C-mid_V-mid_Q-mid",
        "sim_R-mid_C-low_V-mid_Q-mid",
        "sim_R-mid_C-high_V-mid_Q-mid",
        "sim_R-mid_C-mid_V-low_Q-mid",
        "sim_R-mid_C-mid_V-high_Q-mid",
    ]

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset="code")
    combined_df = combined_df.set_index("code")
    combined_df = combined_df.reindex(custom_order).reset_index()
    combined_df.to_csv(os.path.join(output_path, filename), index=False)
    logger.info(f"Saved {filename} to {output_path}.")
