import json
import os
from typing import List
from logger.logger import get_logger

log = get_logger("config.run_config")


class RunConfig:
    def __init__(
        self,
        run_id: int,
        option: str,
        resources: str,
        concentration: str,
        variability: str,
        quality: str,
    ):
        self.run_id = run_id
        self.option = option
        self.resources = resources
        self.concentration = concentration
        self.variability = variability
        self.quality = quality
        self.code = (
            f"{option[:3]}_R-{resources}_C-{concentration}_V-{variability}_Q-{quality}"
        )

    def get_scenario(self):
        if self.option == "deployment":
            scenario = "deployment"
        elif "low" in [
            self.resources,
            self.concentration,
            self.variability,
            self.quality,
        ]:
            scenario = "low"
        elif "high" in [
            self.resources,
            self.concentration,
            self.variability,
            self.quality,
        ]:
            scenario = "high"
        else:
            scenario = "mid"
        return scenario

    def get_label(self):
        if self.resources != "mid":
            tag = f"Resources = {self.resources.upper()}"
        elif self.concentration != "mid":
            tag = f"Concentration = {self.concentration.upper()}"
        elif self.variability != "mid":
            tag = f"Variability = {self.variability.upper()}"
        elif self.quality != "mid":
            tag = f"Quality = {self.quality.upper()}"
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
            option=data.get("option", "INVALID_OPT"),
            resources=data.get("resources", "INVALID_RES"),
            concentration=data.get("concentration", "INVALID_CONC"),
            variability=data.get("variability", "INVALID_VAR"),
            quality=data.get("quality", "INVALID_QUAL"),
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
