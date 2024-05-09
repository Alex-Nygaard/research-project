import torch


torch_device = "cpu"
if torch.cuda.is_available():
    torch_device = "cuda"
elif torch.backends.mps.is_available():
    torch_device = "mps"
DEVICE = torch.device(torch_device)

NUM_ROUNDS = 10

NUM_CLIENTS = 2
PERC_DROPOUT = 0.5
BATCH_SIZE = 32
LOCAL_EPOCHS = 3
NUM_DATAPOINTS = 200
PERC_NEW_DATA = 0.2
PERC_MISSING_LABELS = 0.2
