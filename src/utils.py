import os
import yaml
import torch
import random
import numpy as np


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config(path, config):
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")
    return torch.device("cuda")
