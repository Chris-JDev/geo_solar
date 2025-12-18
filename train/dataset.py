"""Dataset loader for synthetic PV layout samples."""

from __future__ import annotations

import glob
import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class SolarLayoutDataset(Dataset):
    """Loads samples from data/synthetic/sample_*/

    Expected files per sample dir:
      - cond.npy   float32 (2,256,256)
      - target.npy float32 (1,256,256)
    """

    def __init__(self, root_dir: str, limit: Optional[int] = None):
        self.root_dir = root_dir
        sample_dirs = sorted(glob.glob(os.path.join(root_dir, "sample_*")))
        if limit is not None:
            sample_dirs = sample_dirs[: int(limit)]
        if not sample_dirs:
            raise FileNotFoundError(f"No sample_* directories found under: {root_dir}")
        self.sample_dirs = sample_dirs

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        d = self.sample_dirs[idx]
        cond_path = os.path.join(d, "cond.npy")
        target_path = os.path.join(d, "target.npy")
        cond = np.load(cond_path).astype(np.float32)
        target = np.load(target_path).astype(np.float32)

        # Simple sanity checks (helpful for beginners).
        if cond.ndim != 3 or cond.shape[0] != 2:
            raise ValueError(f"Bad cond shape in {cond_path}: {cond.shape}")
        if target.ndim != 3 or target.shape[0] != 1:
            raise ValueError(f"Bad target shape in {target_path}: {target.shape}")
        if cond.shape[1:] != target.shape[1:]:
            raise ValueError(f"cond/target spatial mismatch: {cond.shape} vs {target.shape}")

        cond_t = torch.from_numpy(cond)  # (2,H,W)
        target_t = torch.from_numpy(target)  # (1,H,W)
        return cond_t, target_t
