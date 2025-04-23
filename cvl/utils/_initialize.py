import torch
import random
import numpy as np


def set_seed(seed):
    if seed is not None:
        random.seed(a=seed)
        np.random.seed(seed=seed)
        torch.manual_seed(seed=seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True