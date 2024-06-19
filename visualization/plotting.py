import os
import itertools

import scienceplots
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from visualization.data import Metric, RunData
from logger.logger import get_logger
import numpy as np

from visualization.utils import (
    calculate_mae,
    calculate_mse,
    calculate_dtw,
    calculate_pearson_correlation,
)

plt.style.use(["science", "ieee"])
logger = get_logger("visualization.plotting")


def plot_metrics(
    metrics: list[Metric],
    title: str = "Metrics per Round",
    x_label="Round number",
    y_label="Value",
    save_dir: str = None,
):
    plt.figure(figsize=(505 / 72 / 1.5, 505 / 72 / 3))
    rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
    rc("text", usetex=True)

    for metric in metrics:
        plot_single_metric(metric)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(fontsize="6")
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

    # 50 ROUNDS
    # deployment = ["run_3740021"]
    # simulation_base = ["run_3740023"]
    #
    # simulation_concentration = ["run_3740022", "run_3740024"]
    # simulation_resources = ["run_3740245", "run_3740246"]
    # simulation_variability = ["run_3740247", "run_3740248"]

    # 100 ROUNDS
    # deployment = ["run_3749632"]
    # simulation_base = ["run_3749633"]
    #
    # simulation_resources = ["run_3749634", "run_3749635"]
    # simulation_variability = ["run_3749636", "run_3749637"]
    # simulation_concentration = ["run_3749638", "run_3749639"]

    # run_combinations = {
    #     "concentration": deployment + simulation_base + simulation_concentration,
    #     "resources": deployment + simulation_base + simulation_resources,
    #     "variability": deployment + simulation_base + simulation_variability,
    # }
    # run_combinations = {
    #     key: RunData.build_many(runs) for key, runs in run_combinations.items()
    # }

    run_combinations = {
        # "blind": ["run_100", "run_102"],
        # "batch_size": ["run_100", "run_103"],
        # "local_epochs": ["run_100", "run_104"],
        # "data_volume": ["run_100", "run_105"],
        # "data_labels": ["run_100", "run_106"],
        # "real": ["run_100", "run_107"],
        # "all": [
        #     "run_100",
        #     "run_102",
        #     "run_103",
        #     "run_104",
        #     "run_105",
        #     "run_106",
        #     "run_107",
        # ],
        # "all-50-100": ["run_"],
        "all-20-100": [
            "run_120",
            "run_121",
            "run_122",
            "run_123",
            "run_124",
            "run_125",
            "run_126",
        ],
    }

    run_combinations = {
        key: RunData.build_many(runs, base_path="logs")
        for key, runs in run_combinations.items()
    }

    all_csv_files = []
    output_path = "for-report/20"
    os.makedirs(output_path, exist_ok=True)

    for category, runs in run_combinations.items():
        for key in ["accuracy", "loss"]:
            plot_metrics(
                [run.get_metric(key) for run in runs],
                title=f"{key.capitalize()} per Round",
                x_label="Round number",
                y_label=key.capitalize(),
                save_dir=os.path.join(output_path, f"{category}-{key}.pdf"),
            )

        filename = f"{category}-results.csv"
        RunData.many_to_csv(runs, output_path, filename)
        logger.info(f"Saved {filename} to {output_path}.")
        all_csv_files.append(os.path.join(output_path, filename))
        create_correlelogram(
            runs, os.path.join(output_path, f"{category}-correlogram.pdf")
        )

    combine_csv_files(all_csv_files, output_path, "all-results.csv")


def create_correlelogram(runs: list["RunData"], output_path: str):
    accuracies = [100 * np.array(run.get_metric("accuracy").y) for run in runs]
    num_arrays = len(runs)
    matrix = np.zeros((num_arrays, num_arrays))

    for i, j in itertools.combinations(range(num_arrays), 2):
        mse = calculate_mse(accuracies[i], accuracies[j])
        matrix[i, j] = mse
        matrix[j, i] = mse

    fig, ax = plt.subplots()
    cax = ax.matshow(matrix, cmap="gist_gray", interpolation="none")
    fig.colorbar(cax)

    avg = np.average(matrix)

    for (i, j), z in np.ndenumerate(matrix):
        ax.text(
            j,
            i,
            "{:0.1f}".format(z),
            ha="center",
            va="center",
            color="white" if z < avg / 1.5 else "black",
        )

    # Set ticks and labels
    ax.set_xticks(range(num_arrays))
    ax.set_yticks(range(num_arrays))
    ax.set_xticklabels([f"{run.run_config.get_label(short=True)}" for run in runs])
    ax.set_yticklabels([f"{run.run_config.get_label(short=True)}" for run in runs])

    ax.tick_params(axis="x", labelrotation=80)
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    plt.title("MSE Correlogram")
    plt.savefig(output_path)


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
        "dep_B-noniid_E-noniid_V-noniid_L-noniid",
        "sim_B-noniid_E-noniid_V-noniid_L-noniid",
        "sim_B-noniid_E-iid_V-iid_L-iid",
        "sim_B-iid_E-noniid_V-iid_L-iid",
        "sim_B-iid_E-iid_V-noniid_L-iid",
        "sim_B-iid_E-iid_V-iid_L-noniid",
        "sim_B-iid_E-iid_V-iid_L-iid",
    ]

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset="code")
    combined_df = combined_df.set_index("code")
    combined_df = combined_df.reindex(custom_order).reset_index()
    combined_df.to_csv(os.path.join(output_path, filename), index=False)
    logger.info(f"Saved {filename} to {output_path}.")
