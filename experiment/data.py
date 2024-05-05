import os
from typing import Tuple

import torchvision
import datasets
from datasets import Dataset
from torchvision.transforms import ToTensor, Normalize, Compose

DATASET = "cifar100"
NUM_CLIENTS = 50
DOWNLOAD_DIR = "data"
def download_data(dir: str = None) -> Dataset:
    dir = dir or DOWNLOAD_DIR
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Download CIFAR-100 dataset
    ds = datasets.load_dataset(path=dir, name=DATASET)
    return ds

def apply_transforms(batch):
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch

def split_data(ds: Dataset, test_size: float = 0.2) -> Tuple[Dataset, Dataset]:
    splits = ds.train_test_split(test_size=test_size, seed=1010)
    return splits["train"], splits["test"]

