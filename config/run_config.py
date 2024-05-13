import json
import os


class RunConfig:
    def __init__(
        self, run_id: int, option: str, client_variation: str, data_variation: str
    ):
        self.run_id = run_id
        self.option = option
        self.client_variation = client_variation
        self.data_variation = data_variation

    def write_to_json(self, path: str, filename: str):
        with open(os.path.join(path, filename), "w") as json_file:
            json.dump(self.__dict__, json_file)

    def __repr__(self):
        return f"RunConfig(run_id={self.run_id}, option={self.option}, client_variation={self.client_variation}, data_variation={self.data_variation})"
