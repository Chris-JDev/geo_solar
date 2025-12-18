"""Utilities for loading a trained cVAE checkpoint and sampling candidates.

This keeps Streamlit code simple: the app just calls `load_cvae(...)` and
`generate_candidates(...)`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from src.gen.cvae import CVAEConfig, ConditionalVAE


def _resolve_device(device: str) -> str:
    device = (device or "cpu").lower().strip()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def load_cvae(model_path: str = "models/layout_cvae.pt", device: str = "cpu") -> Tuple[ConditionalVAE, CVAEConfig]:
    """Load a trained cVAE checkpoint.

    The training script saves a dict with:
      - state_dict
      - config

    Returns:
        (model, config)
    """

    resolved_device = _resolve_device(device)
    ckpt_path = Path(model_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=resolved_device)
    cfg_dict = ckpt.get("config", {})
    cfg = CVAEConfig(**cfg_dict)

    model = ConditionalVAE(cfg)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(resolved_device)
    model.eval()
    return model, cfg


@torch.no_grad()
def generate_candidates(
    model: ConditionalVAE,
    cond_np: np.ndarray,
    n: int = 8,
    thr: float = 0.5,
    device: str = "cpu",
) -> np.ndarray:
    """Generate N candidate binary masks from a conditioning tensor.

    Args:
        cond_np: float32 array of shape (2,256,256) with values in {0,1}

    Returns:
        masks: uint8 array of shape (n,256,256) with values in {0,1}
    """

    if not isinstance(cond_np, np.ndarray):
        raise TypeError("cond_np must be a numpy array")
    if cond_np.shape != (2, 256, 256):
        raise ValueError(f"cond_np must have shape (2,256,256), got {cond_np.shape}")

    resolved_device = _resolve_device(device)

    cond_t = torch.from_numpy(cond_np.astype(np.float32, copy=False)).unsqueeze(0)  # (1,2,H,W)
    cond_t = cond_t.to(resolved_device)

    # model.sample returns probabilities in [0,1]
    probs = model.sample(cond_t, n=int(n))  # (n,1,H,W)
    probs = probs[:, 0]  # (n,H,W)

    masks = (probs > float(thr)).to(torch.uint8).cpu().numpy()  # {0,1}
    return masks
