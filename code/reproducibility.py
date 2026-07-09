import json
import os
import random
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def save_config(model_name, hyperparams, metrics=None, output_dir=None):
    if output_dir is None:
        output_dir = PROJECT_ROOT / 'results'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        'model': model_name,
        'hyperparameters': hyperparams,
        'metrics': metrics or {},
        'seed': 42,
    }
    path = output_dir / f'{model_name}_config.json'
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {path}")
    return config
