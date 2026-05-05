"""Photometric loss: Kerbl et al. 2023, L = (1 - lam) * L1 + lam * (1 - SSIM).

Lambda = 0.2 in the paper. We compute L1 over [0, 1]-normalized images and
SSIM via pytorch_msssim (pure-Python, no CUDA build). fused_ssim is a faster
drop-in if installed.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from fused_ssim import fused_ssim as _ssim_fn

    def _ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # fused_ssim expects (B, C, H, W); both inputs already in that layout.
        return _ssim_fn(pred, target)

except ImportError:
    from pytorch_msssim import ssim as _msssim

    def _ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return _msssim(pred, target, data_range=1.0, size_average=True)


def _to_bchw(img: torch.Tensor) -> torch.Tensor:
    """(H, W, 3) or (B, H, W, 3) -> (B, 3, H, W)."""
    if img.dim() == 3:
        img = img.unsqueeze(0)
    return img.permute(0, 3, 1, 2).contiguous()


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lam: float = 0.2,
) -> tuple[torch.Tensor, dict]:
    """Returns (loss, metrics_dict).

    pred, target: (H, W, 3) or (B, H, W, 3) in [0, 1].
    """
    pred_b = _to_bchw(pred)
    target_b = _to_bchw(target)

    l1 = F.l1_loss(pred_b, target_b)
    ssim_val = _ssim(pred_b, target_b)
    loss = (1.0 - lam) * l1 + lam * (1.0 - ssim_val)
    return loss, {"l1": l1.detach(), "ssim": ssim_val.detach()}


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Peak signal-to-noise ratio in dB."""
    pred_b = _to_bchw(pred)
    target_b = _to_bchw(target)
    mse = F.mse_loss(pred_b, target_b)
    return 10.0 * torch.log10((max_val * max_val) / mse.clamp_min(1e-12))
