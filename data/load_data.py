import os

import datasets
from config.constants import NUM_CLIENTS
from torchvision.transforms import ToTensor, Normalize, Compose


DATASET = "cifar10"
SAVEPATH = "data/storage"

try:
    dataset = datasets.load_from_disk(SAVEPATH, keep_in_memory=True)
except FileNotFoundError:
    print("Dataset not found, downloading...")
    dataset = datasets.load_dataset(path=DATASET, keep_in_memory=True)
    if not os.path.exists(SAVEPATH):
        os.makedirs(SAVEPATH)
    dataset.save_to_disk(SAVEPATH)

train_set = dataset["train"]
print("[DATA] Client training examples: ", len(train_set))
centralized_test_set = dataset["test"]
print("[DATA] Centralized testset examples: ", len(centralized_test_set))


def apply_transforms(batch):
    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch["img"] = [transform(img) for img in batch["img"]]
    return batch


train_set = train_set.shuffle(seed=1010).with_transform(apply_transforms)
centralized_test_set = centralized_test_set.shuffle(seed=1010).with_transform(
    apply_transforms
)


def get_data_for_client(cid: int):
    return train_set.shard(num_shards=NUM_CLIENTS, index=cid)
