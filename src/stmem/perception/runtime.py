"""Explicit model-process seeding. The caller selects CUDA_VISIBLE_DEVICES."""

import os
import random
import sys

import numpy as np


def seed_everything(seed=42):
    current = sys.modules.get("torch")
    cuda = getattr(current, "cuda", None)
    if cuda is not None and cuda.is_initialized():
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError("Seed models before initializing CUDA")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
