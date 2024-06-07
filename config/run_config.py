import json
import os
from typing import List
from logger.logger import get_logger

log = get_logger("config.run_config")


class RunConfig:
    def __init__(
        self,
        run_id: int,
        num_clients: int,
        option: str,
        trace_file: str = "",
        batch_size: str = "iid",
        local_epochs: str = "iid",
        data_volume: str = "iid",
        data_labels: str = "iid",
    ):
        self.run_id = run_id
        self.num_clients = num_clients
        self.option = option
        self.trace_file = trace_file
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.data_volume = data_volume
        self.data_labels = data_labels
        self.code = f"{option[:3]}_R-{batch_size}_C-{local_epochs}_V-{data_volume}_D-{data_labels}"

    def get_scenario(self):
        if self.option == "deployment":
            scenario = "deployment"
        elif "low" in [
            self.batch_size,
            self.local_epochs,
            self.data_volume,
            self.data_labels,
        ]:
            scenario = "low"
        elif "high" in [
            self.batch_size,
            self.local_epochs,
            self.data_volume,
            self.data_labels,
        ]:
            scenario = "high"
        else:
            scenario = "mid"
        return scenario

    def get_label(self):
        if self.batch_size != "mid":
            tag = f"Batch size = {self.batch_size.upper()}"
        elif self.local_epochs != "mid":
            tag = f"Concentration = {self.local_epochs.upper()}"
        elif self.data_volume != "mid":
            tag = f"Variability = {self.data_volume.upper()}"
        elif self.data_labels != "mid":
            tag = f"data_labels = {self.data_labels.upper()}"
        else:
            tag = "Base"
        return f"{self.option.capitalize()} - {tag}"

    def write_to_json(self, path: str, filename: str):
        full_path = os.path.join(path, filename)
        log.info(f"Writing RunConfig to {full_path}.")
        with open(full_path, "w") as json_file:
            json.dump(self.__dict__, json_file)

    @staticmethod
    def read_from_json(path: str, filename: str):
        with open(os.path.join(path, filename), "r") as file:
            data = json.load(file)
        return RunConfig(
            run_id=data.get("run_id", -1),
            num_clients=data.get("num_clients", -1),
            option=data.get("option", "INVALID_OPT"),
            trace_file=data.get("trace_file", "INVALID_TRACE"),
            batch_size=data.get("batch_size", "INVALID_BATCH"),
            local_epochs=data.get("local_epochs", "INVALID_EPOCH"),
            data_volume=data.get("data_volume", "INVALID_VOL"),
            data_labels=data.get("data_labels", "INVALID_LAB"),
        )

    @staticmethod
    def combine(runs: List["RunConfig"]) -> "RunConfig":
        num = len(runs)
        assert num > 0, "[RunConfig] Need atleast 1 run configs to combine"

        assert [run.code for run in runs].count(
            runs[0].code
        ) == num, "[RunConfig] All 'code's must be the same"

        return runs[0]

    def __repr__(self):
        return f"RunConfig(run_id={self.run_id}, code={self.code})"
