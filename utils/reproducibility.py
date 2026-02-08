import torch
import numpy as np
import random
import os
import json
import logging

def apply_reproducibility_manifest(config_path='repro_config.json'):
    """
    Applies the frozen clinical validation manifest to the current process.
    Ensures bit-for-bit reproducibility and safety protocol pinning.
    """
    if not os.path.exists(config_path):
        logging.warning(f"Reproducibility manifest {config_path} not found. Operating in UNVERIFIED mode.")
        return False

    with open(config_path, 'r') as f:
        config = json.load(f)

    # 1. Pin Protocol & Mode
    protocol = config.get('protocol_version', 'UNKNOWN')
    mode = config.get('validation_mode', 'development')
    logging.info(f"PROTOCOL PINNED: {protocol}")
    logging.info(f"VALIDATION MODE: {mode.upper()}")

    # 2. Apply Deterministic Seeds
    repro = config.get('reproducibility', {})
    seed = repro.get('random_seed', 42)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if repro.get('cuda_deterministic', True):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logging.info("CUDA DETECTORS: Bit-for-bit deterministic mode ACTIVE.")

    # 3. Verify Frozen State (Placeholder for actual hash checks)
    logging.info(f"Frozen Architecture: {'LOCKED' if config['frozen_state']['architecture_locked'] else 'WARNING: UNLOCKED'}")
    logging.info(f"Safety Layer: {config['frozen_state']['deterministic_safety_layer']}")
    
    return True

def get_clinical_hyperparams(config_path='repro_config.json'):
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config.get('hyperparameters', {})
