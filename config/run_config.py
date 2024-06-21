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
        client_config_file: str = "",
        server_address: str = "",
        batch_size: str = "iid",
        local_epochs: str = "iid",
        data_volume: str = "iid",
        data_labels: str = "iid",
    ):
        self.run_id = run_id
        self.num_clients = num_clients
        self.option = option
        self.trace_file = trace_file
        self.client_config_file = client_config_file
        self.server_address = server_address
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.data_volume = data_volume
        self.data_labels = data_labels
        self.code = f"{option[:3]}_B-{batch_size}_E-{local_epochs}_V-{data_volume}_L-{data_labels}"

    def get_scenario(self):
        combined = [
            self.batch_size,
            self.local_epochs,
            self.data_volume,
            self.data_labels,
        ]
        if self.option == "deployment":
            scenario = "deployment"
        elif all(["iid" == val for val in combined]):
            scenario = "blind"
        elif all(["noniid" == val for val in combined]):
            scenario = "real"
        elif self.batch_size == "noniid":
            scenario = "batch_niid"
        elif self.local_epochs == "noniid":
            scenario = "epoch_niid"
        elif self.data_volume == "noniid":
            scenario = "volume_niid"
        elif self.data_labels == "noniid":
            scenario = "label_niid"
        else:
            scenario = "invalid"
        return scenario

    def get_label(self, short: bool = False):
        combined = [
            self.batch_size,
            self.local_epochs,
            self.data_volume,
            self.data_labels,
        ]

        if self.option == "deployment":
            return "Deployment"  # if not short else "Dep."

        tag = ""
        if all(["iid" == val for val in combined]):
            tag = "Blind"
        elif all(["noniid" == val for val in combined]):
            tag = "Real"
        elif self.batch_size == "noniid":
            tag = "Batch size Non-IID" if not short else "BS"
        elif self.local_epochs == "noniid":
            tag = "Local Epochs Non-IID" if not short else "LE"
        elif self.data_volume == "noniid":
            tag = "Data volume Non-IID" if not short else "DV"
        elif self.data_labels == "noniid":
            tag = "Data labels Non-IID" if not short else "DL"
        else:
            tag = "Invalid"

        option = self.option.capitalize() if not short else "Sim."
        return f"{option} {tag}"

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
            client_config_file=data.get("client_config_file", "INVALID_CLI_CONF"),
            server_address=data.get("server_address", "INVALID_SERV"),
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
