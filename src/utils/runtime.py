"""Runtime helpers shared by every training entry point.

Centralises the three pieces of boilerplate that used to live in every
train_*.py script: device detection, global RNG seeding, and timestamped
run-dir defaulting.
"""

from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def detect_device(prefer: str | None = None) -> torch.device:
    """Return a torch device. If *prefer* is given, honour it; else auto-detect."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int | None) -> None:
    """Seed Python, NumPy, and PyTorch from one value. No-op if *seed* is None."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_run_dir(prefix: str, base: str | Path = "runs") -> Path:
    """Return ``<base>/<prefix>_YYYYMMDD_HHMMSS`` for use as a fresh run directory."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(base) / f"{prefix}_{stamp}"
