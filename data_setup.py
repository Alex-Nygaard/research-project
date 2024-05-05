from flwr_datasets import FederatedDataset

fds = FederatedDataset(dataset="cifar10", partitioners={"train": 10})

FederatedDataset()

ds = fds.load_full("train")

print("Supervised keys: ", ds.supervised_keys)
print("Arrow schema: ", ds.features.arrow_schema)
print("Num examples: ", len(ds))
print("Info: ", ds.info)
