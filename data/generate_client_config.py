from typing import List
from datasets import DatasetInfo

class ClientConfig:
    def __init__(self, id: int, num_data_points: int, missing_classes: List[str]):
        self.id = id
        self.num_data_points = num_data_points
        self.missing_classes = missing_classes

    @staticmethod
    def of(id: int, num_data_points: int, missing_classes: List[str]):
        return ClientConfig(id, num_data_points, missing_classes)

