import torch
import os
from utils.run_counter import read_run_counter

RUN_ID = os.environ.get("SLURM_JOB_ID", None) or read_run_counter()
LOG_DIR = os.path.join("logs", f"run_{RUN_ID}")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

HARDWARE_TYPE = "cpu"
if torch.cuda.is_available():
    HARDWARE_TYPE = "cuda"
elif torch.backends.mps.is_available():
    HARDWARE_TYPE = "mps"
DEVICE = torch.device(HARDWARE_TYPE)

DATASET = "cifar10"
DATA_SAVE_PATH = "data/storage"

NUM_ROUNDS = 50
