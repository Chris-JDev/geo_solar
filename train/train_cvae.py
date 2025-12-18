"""Train a conditional VAE on the synthetic dataset.

This script is CPU-friendly by default but will use CUDA if available.
Outputs:
  - models/layout_cvae.pt (state_dict + config)
  - data/runs/epoch_##.png previews
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
try:
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError as e:
    if e.name != "torch":
        raise
    _repo_root = Path(__file__).resolve().parents[1]
    _venv_py = _repo_root / ".venv" / "Scripts" / "python.exe"
    raise SystemExit(
        "PyTorch (torch) is not installed for this Python interpreter.\n\n"
        "Fix: run training using the project virtualenv Python:\n"
        f"  {_venv_py} -m train.train_cvae --epochs 10 --batch_size 16\n\n"
        "Or activate the venv first:\n"
        "  .\\.venv\\Scripts\\Activate.ps1\n"
        "  python -m train.train_cvae --epochs 10 --batch_size 16\n"
    )

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.dataset import SolarLayoutDataset
from src.gen.cvae import CVAEConfig, ConditionalVAE


def _auto_device(device_arg: str) -> str:
    if device_arg and device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def _save_epoch_preview(model: ConditionalVAE, cond: torch.Tensor, out_path: Path, thr: float) -> None:
    """Save a 3-panel preview (boundary/keepout/generated) for up to 4 samples."""
    model.eval()
    device = next(model.parameters()).device
    cond = cond.to(device)

    n_show = min(4, int(cond.shape[0]))
    fig, axes = plt.subplots(n_show, 3, figsize=(8, 2.6 * n_show))
    if n_show == 1:
        axes = np.expand_dims(axes, axis=0)

    for row in range(n_show):
        c = cond[row]  # (2,H,W)
        boundary = c[0].detach().cpu().numpy()
        keepout = c[1].detach().cpu().numpy()

        probs = model.sample(c, n=1)[0, 0]  # (H,W) in [0,1]
        mask = (probs > float(thr)).float().detach().cpu().numpy()

        axes[row, 0].imshow(boundary, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title("boundary")
        axes[row, 1].imshow(keepout, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("keepout")
        axes[row, 2].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title(f"generated (thr={thr:.2f})")
        for col in range(3):
            axes[row, col].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/synthetic")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--beta_max", type=float, default=0.1)
    parser.add_argument("--warmup_frac", type=float, default=0.4)
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=str, default="models/layout_cvae.pt")
    parser.add_argument("--log_every", type=int, default=25)
    args = parser.parse_args()

    if args.beta is not None:
        args.beta_max = float(args.beta)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = _auto_device(args.device)
    print(f"Using device: {device}")

    ds = SolarLayoutDataset(args.data_dir, limit=args.limit)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Infer image size from first sample.
    cond0, target0 = ds[0]
    img_size = int(cond0.shape[-1])

    cfg = CVAEConfig(img_size=img_size, latent_dim=args.latent_dim)
    model = ConditionalVAE(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training schedule.
    total_steps = int(args.epochs) * max(1, len(dl))
    warmup_steps = int(math.ceil(float(args.warmup_frac) * float(total_steps)))
    warmup_steps = max(1, warmup_steps)

    criterion = None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    runs_dir = Path("data") / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    first_batch_done = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        for i, (cond, target) in enumerate(dl, start=1):
            cond = cond.to(device)
            target = target.to(device)

            if not first_batch_done:
                # A) Dataset sanity checks (first batch)
                t = target.detach().float()
                print(
                    "First batch sanity checks: "
                    f"target min/max/mean={t.min().item():.4f}/{t.max().item():.4f}/{t.mean().item():.4f}"
                )
                c = cond.detach().float()
                for ch in range(int(c.shape[1])):
                    cc = c[:, ch]
                    print(
                        f"  cond[ch={ch}] min/max/mean="
                        f"{cc.min().item():.4f}/{cc.max().item():.4f}/{cc.mean().item():.4f}"
                    )

                # E) Class imbalance handling via pos_weight from this first batch.
                pos = float(t.mean().item())
                neg = float(1.0 - pos)
                pos_weight = neg / (pos + 1e-6)
                pos_weight_t = torch.tensor([pos_weight], device=device, dtype=torch.float32)
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_t, reduction="mean")
                print(f"  pos={pos:.6f} neg={neg:.6f} pos_weight={pos_weight:.6f}")
                first_batch_done = True

            opt.zero_grad(set_to_none=True)
            recon_logits, mu, logvar = model(cond, target)

            # D) KL warm-up schedule.
            step_index = global_step + 1
            beta = float(args.beta_max) * min(1.0, float(step_index) / float(warmup_steps))

            # BCE with logits + beta * KL
            if criterion is None:
                criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
            bce = criterion(recon_logits, target)
            kld = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
            loss = bce + beta * kld
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            global_step += 1
            if global_step % args.log_every == 0:
                print(
                    f"epoch={epoch} step={global_step} "
                    f"loss={loss.item():.4f} bce={bce.item():.4f} kld={kld.item():.4f} beta={beta:.4f}"
                )

        # Save preview and checkpoint each epoch.
        _save_epoch_preview(model, cond=cond, out_path=runs_dir / f"epoch_{epoch:02d}.png", thr=args.thr)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": {
                    "img_size": cfg.img_size,
                    "cond_channels": cfg.cond_channels,
                    "target_channels": cfg.target_channels,
                    "latent_dim": cfg.latent_dim,
                    "base_channels": cfg.base_channels,
                },
            },
            str(out_path),
        )
        print(f"Saved: {out_path} and preview {runs_dir / f'epoch_{epoch:02d}.png'}")

    print("Training complete")


if __name__ == "__main__":
    main()
