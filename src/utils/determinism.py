import os
import random
import numpy as np
import torch

DEFAULT_SEED = 42

def seed_everything(seed: int = DEFAULT_SEED, deterministic: bool = True, cudnn_benchmark: bool = False):
    """
    Set all relevant seeds and Torch deterministic settings in one place.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

def get_current_seed() -> int:
    return int(os.environ.get("PYTHONHASHSEED", DEFAULT_SEED))
