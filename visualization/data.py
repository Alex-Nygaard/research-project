import json
import os
import numpy as np
from typing import List, Tuple, Dict
import scienceplots
import matplotlib.pyplot as plt
from config.run_config import RunConfig
from client.client import FlowerClient
from visualization.utils import (
    calculate_mae,
    calculate_mse,
    calculate_dtw,
    calculate_pearson_correlation,
)


class Style:
    def __init__(self, color="b", marker="", linestyle="-"):
        self.color = color
        self.marker = marker
        self.linestyle = linestyle

    def modify(self, color=None, marker=None, linestyle=None):
        self.color = color if color else self.color
        self.marker = marker if marker else self.marker
        self.linestyle = linestyle if linestyle else self.linestyle
        return self


class Metric:
    styles = {
        "accuracy": {
            "batch_niid": Style(color="#577590", marker=""),
            "epoch_niid": Style(color="#31E981", marker=""),
            "volume_niid": Style(color="#F3722C", marker=""),
            "label_niid": Style(color="#F9C74F", marker=""),
            "blind": Style(color="#17BEBB", marker=""),
            "real": Style(color="#8A1C7C", marker=""),
            "deployment": Style(color="#EF3E36", marker=""),
        },
        "loss": {
            "batch_niid": Style(color="#577590", marker=""),
            "epoch_niid": Style(color="#31E981", marker=""),
            "volume_niid": Style(color="#F3722C", marker=""),
            "label_niid": Style(color="#F9C74F", marker=""),
            "blind": Style(color="#17BEBB", marker=""),
            "real": Style(color="#8A1C7C", marker=""),
            "deployment": Style(color="#EF3E36", marker=""),
        },
    }

    def __init__(
        self, data: List[Tuple[int, float]], label: str, key: str, scenario: str
    ):
        self.x, self.y = zip(*data)
        self.label = label
        self.key = key
        self.scenario = scenario
        self.style = self.styles[key][scenario]

    @classmethod
    def average_many(cls, metrics: List["Metric"]) -> "Metric":
        num_metrics = len(metrics)
        assert num_metrics > 0, "No metrics to average"

        key = metrics[0].key
        scenario = metrics[0].scenario

        assert [metric.key for metric in metrics].count(
            key
        ) == num_metrics, "Metrics must have the same key"
        assert [metric.scenario for metric in metrics].count(
            scenario
        ) == num_metrics, "Metrics must have the same scenario"

        data = []
        for i in range(len(metrics[0].x)):
            round_values = [metric.y[i] for metric in metrics]
            data.append((i, sum(round_values) / num_metrics))
        return cls(data, f"Average {key} of {num_metrics} runs", key, scenario)


class History:
    def __init__(
        self,
        metrics_distributed: Dict[str, List[Tuple[int, float]]],
        metrics_centralized: Dict[str, List[Tuple[int, float]]],
    ):
        self.metrics_distributed = metrics_distributed
        self.metrics_centralized = metrics_centralized

    def get_centralized_metrics(self, run_config: RunConfig) -> Dict[str, Metric]:
        metrics = {}
        for key, data in self.metrics_centralized.items():
            if key in Metric.styles:
                metrics[key] = Metric(
                    data, run_config.get_label(), key, run_config.get_scenario()
                )
        return metrics

    @staticmethod
    def average_many(histories: List["History"]) -> "History":
        num = len(histories)
        assert num > 0, "Length of 'histories' must be greater than 0"

        metrics_distributed = {}
        metrics_centralized = {}

        for key in histories[0].metrics_distributed.keys():
            round_nums = [
                round_num for round_num, _ in histories[0].metrics_distributed[key]
            ]
            values = np.zeros(len(round_nums))
            for history in histories:
                values += np.array(
                    [value for _, value in history.metrics_distributed[key]]
                )
            values /= num
            metrics_distributed[key] = zip(round_nums, values)

        for key in histories[0].metrics_centralized.keys():
            round_nums = [
                round_num for round_num, _ in histories[0].metrics_centralized[key]
            ]
            values = np.zeros(len(round_nums))
            for history in histories:
                values += np.array(
                    [value for _, value in history.metrics_centralized[key]]
                )
            values /= num
            metrics_centralized[key] = zip(round_nums, values)

        return History(
            metrics_distributed=metrics_distributed,
            metrics_centralized=metrics_centralized,
        )

    @staticmethod
    def read(filepath):
        with open(filepath, "r") as file:
            data = json.load(file)
        return History(
            metrics_distributed=data.get("metrics_distributed", {}),
            metrics_centralized=data.get("metrics_centralized", {}),
        )

    @staticmethod
    def write(history, filepath):
        with open(filepath, "w") as file:
            json.dump(history.__dict__, file, indent=4)

    def __repr__(self):
        return json.dumps(self.__dict__, indent=4)


class RunData:
    def __init__(
        self,
        run_id: str,
        history: History,
        run_config: RunConfig,
        clients: list[dict],
    ):
        self.run_id = run_id
        self.history = history
        self.run_config = run_config
        self.clients = clients

        self.metrics = self.history.get_centralized_metrics(self.run_config)

    def get_metric(self, target_key: str) -> Metric:
        return next(
            (self.metrics[key] for key in self.metrics.keys() if key == target_key),
            None,
        )

    @staticmethod
    def many_to_csv(runs: List["RunData"], path: str, filename: str):
        deployment = next(
            (run for run in runs if run.run_config.get_scenario() == "deployment"), None
        )
        if deployment is None:
            raise ValueError("No deployment run found in list of runs")

        with open(os.path.join(path, filename), "w") as file:
            columns = [
                "code",
                "max_acc",
                "max_acc_round",
                "acc_conv(80%)",
                "acc_conv(90%)",
                "min_loss",
                "min_loss_round",
                "loss_conv(80%)",
                "loss_conv(90%)",
                "mse",
                "mae",
                "dtw",
                "pearson",
            ]
            file.write(",".join(columns) + "\n")
            for run in runs:
                accuracies = run.metrics.get("accuracy")
                max_acc, max_acc_round = -1, -1
                acc_conv_80, acc_conv_90 = -1, -1
                if accuracies:
                    max_acc_idx = np.argmax(accuracies.y)
                    max_acc = accuracies.y[max_acc_idx]
                    max_acc_round = accuracies.x[max_acc_idx]

                    for i, acc in enumerate(accuracies.y):
                        if acc > 0.8 * max_acc and acc_conv_80 == -1:
                            acc_conv_80 = i
                        if acc > 0.9 * max_acc and acc_conv_90 == -1:
                            acc_conv_90 = i
                        if acc_conv_80 != -1 and acc_conv_90 != -1:
                            break

                losses = run.get_metric("loss")
                min_loss, min_loss_round = -1, -1
                loss_conv_80, loss_conv_90 = -1, -1
                if losses:
                    min_loss_idx = np.argmin(losses.y)
                    min_loss = losses.y[min_loss_idx]
                    min_loss_round = losses.x[min_loss_idx]

                    for i, loss in enumerate(losses.y):
                        if loss < 1.2 * min_loss and loss_conv_80 == -1:
                            loss_conv_80 = i
                        if loss < 1.1 * min_loss and loss_conv_90 == -1:
                            loss_conv_90 = i
                        if loss_conv_80 != -1 and loss_conv_90 != -1:
                            break
                data = [
                    run.run_config.code,
                    max_acc,
                    max_acc_round,
                    acc_conv_80,
                    acc_conv_90,
                    min_loss,
                    min_loss_round,
                    loss_conv_80,
                    loss_conv_90,
                    calculate_mse(
                        run.get_metric("loss").y, deployment.get_metric("loss").y
                    ),
                    calculate_mae(
                        run.get_metric("loss").y, deployment.get_metric("loss").y
                    ),
                    calculate_dtw(
                        run.get_metric("loss").y, deployment.get_metric("loss").y
                    ),
                    calculate_pearson_correlation(
                        run.get_metric("loss").y, deployment.get_metric("loss").y
                    ),
                ]
                file.write(",".join([str(e) for e in data]) + "\n")
            file.flush()

        mses = []
        labels = []
        for run in runs:
            mses.append(run.get_metric("loss").y)
            labels.append(run.run_config.code)
        num_arrays = len(mses)
        mse_matrix = np.zeros((num_arrays, num_arrays))
        for i in range(num_arrays):
            for j in range(num_arrays):
                mse_matrix[i, j] = calculate_mse(mses[i], mses[j])
        plt.figure(figsize=(8, 6))
        plt.imshow(
            mse_matrix,
            cmap="viridis",
            interpolation="nearest",
        )
        plt.colorbar(label="MSE")
        plt.title("MSE Heatmap")
        plt.xlabel("Array Index")
        plt.ylabel("Array Index")

        for i in range(num_arrays):
            for j in range(num_arrays):
                plt.text(
                    j,
                    i,
                    f"{mse_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                )

        plt.savefig(os.path.join(path, "mse.pdf"))

    @staticmethod
    def build(run_id: str, base_path: str = "from-delftblue/logs"):
        log_dir = os.path.join(base_path, run_id)
        history = History.read(os.path.join(log_dir, "history-data.json"))
        run_config = RunConfig.read_from_json(log_dir, "run_config.json")
        clients = FlowerClient.read_from_json(os.path.join(log_dir, "clients.json"))
        return RunData(run_id, history, run_config, clients)

    @classmethod
    def build_many(cls, run_ids: list, base_path: str = "from-delftblue/logs"):
        runs = []
        for run_id in run_ids:
            runs.append(cls.build(run_id, base_path))

        return runs

    @classmethod
    def average_many(cls, runs: List["RunData"]) -> "RunData":
        assert len(runs) > 0, "Must have more than 0 runs"
        history = History.average_many([run.history for run in runs])
        run_config = RunConfig.combine([run.run_config for run in runs])
        clients = runs[0].clients
        return RunData("average", history, run_config, clients)
