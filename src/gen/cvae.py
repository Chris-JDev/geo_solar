"""Conditional VAE (cVAE) for binary layout masks.

Goal:
- Condition on (boundary_mask, keepout_mask) -> 2 channels
- Generate a target layout mask (rows) -> 1 channel

Training trick:
- Encoder sees both cond + target (3 channels total) and predicts latent distribution.
- Decoder sees cond + a sampled latent z and predicts target logits.

This is a small model intended to train quickly on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CVAEConfig:
    img_size: int = 256
    cond_channels: int = 2
    target_channels: int = 1
    latent_dim: int = 64
    base_channels: int = 32


def _conv(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
        nn.GroupNorm(num_groups=8, num_channels=out_ch),
        nn.ReLU(inplace=True),
    )


class ConditionalVAE(nn.Module):
    def __init__(self, cfg: CVAEConfig):
        super().__init__()
        self.cfg = cfg

        # Encoder input: concat(target, cond) -> (1+2)=3 channels
        enc_in = cfg.target_channels + cfg.cond_channels
        ch = cfg.base_channels

        self.enc = nn.Sequential(
            _conv(enc_in, ch, stride=1),
            _conv(ch, ch * 2, stride=2),  # 128
            _conv(ch * 2, ch * 4, stride=2),  # 64
            _conv(ch * 4, ch * 4, stride=2),  # 32
            _conv(ch * 4, ch * 8, stride=2),  # 16
        )

        # Compute encoder feature size based on img_size.
        # With 4 downsamples by factor 2: img_size / 16.
        feat_hw = cfg.img_size // 16
        feat_dim = (ch * 8) * feat_hw * feat_hw

        self.fc_mu = nn.Linear(feat_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(feat_dim, cfg.latent_dim)

        # Decoder: project z -> feature map, then fuse with downsampled cond at bottleneck.
        self.fc_z = nn.Linear(cfg.latent_dim, (ch * 8) * feat_hw * feat_hw)

        # We concatenate z_feature (ch*8) with cond_down (cond_channels).
        dec_in = (ch * 8) + cfg.cond_channels
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(dec_in, ch * 8, kernel_size=4, stride=2, padding=1),  # 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ch * 8, ch * 4, kernel_size=4, stride=2, padding=1),  # 64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ch * 4, ch * 2, kernel_size=4, stride=2, padding=1),  # 128
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ch * 2, ch, kernel_size=4, stride=2, padding=1),  # 256
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, cfg.target_channels, kernel_size=3, padding=1),
        )

    def encode(self, cond: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return mu, logvar."""
        x = torch.cat([target, cond], dim=1)
        h = self.enc(x)
        h = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        # Keep variance in a reasonable range for stable training.
        # exp(-6)≈0.0025, exp(2)≈7.39
        logvar = torch.clamp(logvar, min=-6.0, max=2.0)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, cond: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Return logits for target mask (no sigmoid)."""
        b = z.shape[0]
        feat_hw = self.cfg.img_size // 16
        z_feat = self.fc_z(z).view(b, -1, feat_hw, feat_hw)

        # Downsample cond to match bottleneck.
        cond_down = F.interpolate(cond, size=(feat_hw, feat_hw), mode="bilinear", align_corners=False)
        h = torch.cat([z_feat, cond_down], dim=1)
        logits = self.dec(h)
        return logits

    def forward(self, cond: torch.Tensor, target: torch.Tensor):
        """Training forward.

        Args:
            cond: (B,2,256,256)
            target: (B,1,256,256)

        Returns:
            recon_logits: (B,1,256,256)
            mu, logvar: (B,latent_dim)
        """
        mu, logvar = self.encode(cond, target)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decode(cond, z)
        return recon_logits, mu, logvar

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, n: int = 8) -> torch.Tensor:
        """Sample n masks conditioned on a single cond.

        Args:
            cond: (2,256,256) or (1,2,256,256)

        Returns:
            probs: (n,1,256,256) in [0,1]
        """
        self.eval()
        if cond.ndim == 3:
            # Single conditioning tensor: (2,H,W)
            cond_b = cond.unsqueeze(0).repeat(n, 1, 1, 1)
        elif cond.ndim == 4:
            # Batch provided: (B,2,H,W). If B != n, sample n using the first element.
            if int(cond.shape[0]) == n:
                cond_b = cond
            else:
                cond_b = cond[:1].repeat(n, 1, 1, 1)
        else:
            raise ValueError(f"cond must have shape (2,H,W) or (B,2,H,W), got {tuple(cond.shape)}")

        z = torch.randn((n, self.cfg.latent_dim), device=cond_b.device)
        logits = self.decode(cond_b, z)  # logits
        probs = torch.sigmoid(logits)
        return probs


def cvae_loss(
    recon_logits: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """BCE-with-logits reconstruction + beta*KL."""

    bce = F.binary_cross_entropy_with_logits(recon_logits, target, reduction="mean")
    kld = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    loss = bce + float(beta) * kld
    return loss, {"bce": float(bce.item()), "kld": float(kld.item()), "loss": float(loss.item())}
