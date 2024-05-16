import json
import os
import numpy as np
from typing import List, Tuple, Dict
import scienceplots
import matplotlib.pyplot as plt
from config.run_config import RunConfig
from client.client import FlowerClient


class Style:
    def __init__(self, color="b", marker="o", linestyle="-"):
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
            "low": Style(color="#FFFAE6", marker="o"),
            "mid": Style(color="#FF9F66", marker="s"),
            "high": Style(color="#FF5F00", marker="x"),
            "deployment": Style(color="#002379", marker="^"),
        },
        "loss": {
            "low": Style(color="#D9EDBF", marker="o"),
            "mid": Style(color="#90D26D", marker="s"),
            "high": Style(color="#2C7865", marker="x"),
            "deployment": Style(color="#FF9800", marker="^"),
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
        # losses_distributed: List[Tuple[int, float]],
        # losses_centralized: List[Tuple[int, float]],
        metrics_distributed: Dict[str, List[Tuple[int, float]]],
        metrics_centralized: Dict[str, List[Tuple[int, float]]],
    ):
        # self.losses_distributed = losses_distributed
        # self.losses_centralized = losses_centralized
        self.metrics_distributed = metrics_distributed
        self.metrics_centralized = metrics_centralized

    def get_centralized_metrics(self, run_config: RunConfig) -> List[Metric]:
        metrics = []
        for key, data in self.metrics_centralized.items():
            label = f"{key} - {run_config.code}"
            if key in Metric.styles:
                metrics.append(Metric(data, label, key, run_config.get_scenario()))
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
            # losses_distributed=data.get("losses_distributed", []),
            # losses_centralized=data.get("losses_centralized", []),
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
        clients: list[FlowerClient],
    ):
        self.run_id = run_id
        self.history = history
        self.run_config = run_config
        self.clients = clients

        self.metrics = self.history.get_centralized_metrics(self.run_config)

    def get_metric(self, key: str) -> Metric:
        return next((metric for metric in self.metrics if metric.key == key), None)

    @staticmethod
    def build(run_id: str, base_path: str = "from-delftblue/logs"):
        log_dir = os.path.join(base_path, run_id)
        history = History.read(os.path.join(log_dir, "history-data.json"))
        run_config = RunConfig.read_from_json(log_dir, "run_config.json")
        clients = FlowerClient.read_from_csv(log_dir, "clients.csv")
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