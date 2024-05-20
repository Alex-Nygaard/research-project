import json
import os
from typing import List
from logger.logger import get_logger

log = get_logger("config.run_config")


class RunConfig:
    def __init__(
        self, run_id: int, option: str, client_variation: str, data_variation: str
    ):
        self.run_id = run_id
        self.option = option
        self.client_variation = client_variation
        self.data_variation = data_variation
        self.code = f"{option[:3]}_C-{client_variation}_D-{data_variation}"

    def get_scenario(self):
        scenario = ""
        if self.option == "deployment":
            scenario = "deployment"
        elif self.client_variation == "low" or self.data_variation == "low":
            scenario = "low"
        elif self.client_variation == "high" or self.data_variation == "high":
            scenario = "high"
        else:
            scenario = "mid"
        return scenario

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
            option=data.get("option", "INVALID_OPT"),
            client_variation=data.get("client_variation", "INVALID_CLI_VAR"),
            data_variation=data.get("data_variation", "INVALID_DAT_VAR"),
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
        return f"RunConfig(run_id={self.run_id}, option={self.option}, client_variation={self.client_variation}, data_variation={self.data_variation})"
