import os

import datasets
from datasets import Dataset

from config.constants import DATA_SAVE_PATH, DATASET
from torchvision.transforms import ToTensor, Normalize, Compose


try:
    dataset = datasets.load_from_disk(DATA_SAVE_PATH, keep_in_memory=True)
except FileNotFoundError:
    print("Dataset not found, downloading...")
    dataset = datasets.load_dataset(path=DATASET, keep_in_memory=True)
    dataset.save_to_disk(DATA_SAVE_PATH)

train_set = dataset["train"]
centralized_test_set = dataset["test"]


def apply_transforms(batch):
    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch["img"] = [transform(img) for img in batch["img"]]
    return batch


train_set = train_set.shuffle(seed=1010).with_transform(apply_transforms)
centralized_test_set = centralized_test_set.shuffle(seed=1010).with_transform(
    apply_transforms
)


def get_data_for_client(
    cid: int,
    num_data_points: int,
) -> Dataset:
    return train_set[cid * num_data_points : (cid + 1) * num_data_points]
