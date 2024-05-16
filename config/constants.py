import torch
import os
from utils.run_counter import read_run_counter

RUN_ID = os.environ.get("SLURM_JOB_ID", None) or read_run_counter()
LOG_DIR = os.path.join("logs", f"run_{RUN_ID}")

torch_device = "cpu"
if torch.cuda.is_available():
    torch_device = "cuda"
elif torch.backends.mps.is_available():
    torch_device = "mps"
DEVICE = torch.device(torch_device)

DATASET = "cifar10"
DATA_SAVE_PATH = "data/storage"

NUM_ROUNDS = 30

NUM_CLIENTS = 20
PERC_DROPOUT = 0.5
BATCH_SIZE = 32
LOCAL_EPOCHS = 5
NUM_DATAPOINTS = 200
PERC_NEW_DATA = 0.2
PERC_MISSING_LABELS = 0.2
