"""Training loop for 3DGS.

Lifecycle (matches the plan walkthrough):

  for step in 0..max_steps-1:
    1. sample camera + load image
    2. promote SH degree if step is a multiple of `sh_promote_every`
    3. forward: gsplat.rasterization(...)
    4. loss = (1 - lam) * L1 + lam * (1 - SSIM)
    5. strategy.step_pre_backward()
    6. loss.backward()
    7. for each Adam: step + zero_grad
    8. update means LR (exponential decay)
    9. strategy.step_post_backward()  (mutates params + optimizer state)
    10. periodic eval / checkpoint

The hyperparameters live in `TrainConfig` so train_custom.py can override them
from CLI without rewriting the loop.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from gsplat import rasterization
from gsplat.strategy import DefaultStrategy

from .dataset import ColmapDataset
from .losses import combined_loss, psnr
from .model import GaussianModel


@dataclass
class TrainConfig:
    max_steps: int = 7000
    sh_promote_every: int = 1000
    max_sh_degree: int = 3

    ssim_lambda: float = 0.2

    # Means LR is multiplied by scene_extent (set at build time).
    means_lr_init_factor: float = 1.6e-4
    means_lr_final_factor: float = 1.6e-6
    scales_lr: float = 5e-3
    quats_lr: float = 1e-3
    opacities_lr: float = 5e-2
    sh_dc_lr: float = 2.5e-3
    sh_rest_lr: float = 2.5e-3 / 20.0

    # DefaultStrategy knobs (gsplat defaults; copied here for traceability).
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    refine_every: int = 100
    reset_every: int = 3000
    grow_grad2d: float = 2e-4
    grow_scale3d: float = 0.01
    prune_opa: float = 5e-3
    prune_scale3d: float = 0.1

    # Logging / checkpoint cadence.
    eval_every: int = 1000
    log_every: int = 100
    ckpt_every: int = 5000

    seed: int = 42


def build_optimizers(
    model: GaussianModel,
    scene_extent: float,
    cfg: TrainConfig,
) -> dict[str, torch.optim.Adam]:
    """One Adam per parameter; required by DefaultStrategy.check_sanity."""
    p = model.params
    lrs = {
        "means": cfg.means_lr_init_factor * scene_extent,
        "scales": cfg.scales_lr,
        "quats": cfg.quats_lr,
        "opacities": cfg.opacities_lr,
        "sh_dc": cfg.sh_dc_lr,
        "sh_rest": cfg.sh_rest_lr,
    }
    return {
        name: torch.optim.Adam([p[name]], lr=lr, eps=1e-15)
        for name, lr in lrs.items()
    }


def update_means_lr(opt: torch.optim.Adam, step: int, cfg: TrainConfig, scene_extent: float) -> float:
    """Log-linear decay from init -> final over max_steps."""
    init = cfg.means_lr_init_factor * scene_extent
    final = cfg.means_lr_final_factor * scene_extent
    t = min(max(step / max(1, cfg.max_steps), 0.0), 1.0)
    lr = init * (final / init) ** t
    opt.param_groups[0]["lr"] = lr
    return lr


def _render_one(model: GaussianModel, viewmat: torch.Tensor, K: torch.Tensor,
                width: int, height: int, sh_degree: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Render a single view; returns (image[H,W,3], alpha[H,W,1], info)."""
    renders, alphas, info = rasterization(
        means=model.params["means"],
        quats=model.params["quats"],
        scales=torch.exp(model.params["scales"]),
        opacities=torch.sigmoid(model.params["opacities"]),
        colors=model.sh_coeffs(),
        viewmats=viewmat[None],
        Ks=K[None],
        width=width,
        height=height,
        sh_degree=sh_degree,
        near_plane=0.01,
        far_plane=1e10,
        packed=True,
        rasterize_mode="classic",
        render_mode="RGB",
    )
    return renders[0], alphas[0], info


@torch.no_grad()
def evaluate(model: GaussianModel, dataset: ColmapDataset, sh_degree: int, device: str) -> dict:
    """Compute mean PSNR / SSIM / L1 on the held-out set."""
    if not dataset.test_indices:
        return {}
    psnrs, ssims, l1s = [], [], []
    for idx in dataset.test_indices:
        sample = dataset[idx]
        target = sample["image"].to(device)
        viewmat = sample["viewmat"].to(device)
        K = sample["K"].to(device)
        pred, _, _ = _render_one(
            model, viewmat, K, sample["width"], sample["height"], sh_degree
        )
        pred = pred.clamp(0.0, 1.0)
        _, m = combined_loss(pred, target)
        psnrs.append(psnr(pred, target).item())
        ssims.append(m["ssim"].item())
        l1s.append(m["l1"].item())
    return {
        "psnr": sum(psnrs) / len(psnrs),
        "ssim": sum(ssims) / len(ssims),
        "l1": sum(l1s) / len(l1s),
        "n_test": len(psnrs),
    }


def train(
    model: GaussianModel,
    dataset: ColmapDataset,
    scene_extent: float,
    result_dir: Path,
    cfg: TrainConfig,
    device: str = "cuda",
) -> None:
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    ply_dir = result_dir / "ply"
    ply_dir.mkdir(exist_ok=True)
    tb = SummaryWriter(log_dir=str(result_dir / "tb"))

    rng = random.Random(cfg.seed)

    optimizers = build_optimizers(model, scene_extent, cfg)
    strategy = DefaultStrategy(
        prune_opa=cfg.prune_opa,
        grow_grad2d=cfg.grow_grad2d,
        grow_scale3d=cfg.grow_scale3d,
        prune_scale3d=cfg.prune_scale3d,
        refine_start_iter=cfg.refine_start_iter,
        refine_stop_iter=cfg.refine_stop_iter,
        reset_every=cfg.reset_every,
        refine_every=cfg.refine_every,
        verbose=False,
    )
    strategy.check_sanity(model.params, optimizers)
    strat_state = strategy.initialize_state(scene_scale=scene_extent)

    active_sh_degree = 0
    train_indices = dataset.train_indices
    if not train_indices:
        raise RuntimeError("Dataset has no training images.")

    t_start = time.time()
    pbar = trange(cfg.max_steps, desc="train")
    for step in pbar:
        # SH warm-up.
        if step > 0 and step % cfg.sh_promote_every == 0 and active_sh_degree < cfg.max_sh_degree:
            active_sh_degree += 1

        # Sample one camera.
        idx = rng.choice(train_indices)
        sample = dataset[idx]
        target = sample["image"].to(device)
        viewmat = sample["viewmat"].to(device)
        K = sample["K"].to(device)

        # Forward.
        pred, _, info = _render_one(
            model, viewmat, K, sample["width"], sample["height"], active_sh_degree
        )
        loss, metrics = combined_loss(pred, target, lam=cfg.ssim_lambda)

        # ADC pre-backward (calls retain_grad on means2d).
        strategy.step_pre_backward(model.params, optimizers, strat_state, step, info)

        loss.backward()

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)

        means_lr = update_means_lr(optimizers["means"], step, cfg, scene_extent)

        # ADC post-backward (split / clone / prune / opacity reset).
        strategy.step_post_backward(
            model.params, optimizers, strat_state, step, info, packed=True
        )

        # Logging.
        if step % cfg.log_every == 0:
            tb.add_scalar("train/loss", loss.item(), step)
            tb.add_scalar("train/l1", metrics["l1"].item(), step)
            tb.add_scalar("train/ssim", metrics["ssim"].item(), step)
            tb.add_scalar("train/num_gaussians", model.num_gaussians, step)
            tb.add_scalar("train/active_sh_degree", active_sh_degree, step)
            tb.add_scalar("train/means_lr", means_lr, step)
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                n=model.num_gaussians,
                sh=active_sh_degree,
            )

        if step > 0 and step % cfg.eval_every == 0:
            eval_metrics = evaluate(model, dataset, active_sh_degree, device)
            for k, v in eval_metrics.items():
                if isinstance(v, (int, float)):
                    tb.add_scalar(f"eval/{k}", v, step)

        if step > 0 and (step % cfg.ckpt_every == 0 or step == cfg.max_steps - 1):
            ply_path = ply_dir / f"iter_{step:06d}.ply"
            model.export_ply(ply_path)

    # Final checkpoint, in case max_steps - 1 was reached without hitting the
    # if-branch above (e.g., max_steps < ckpt_every).
    final_path = ply_dir / f"iter_{cfg.max_steps - 1:06d}.ply"
    if not final_path.exists():
        model.export_ply(final_path)

    final_eval = evaluate(model, dataset, active_sh_degree, device)
    if final_eval:
        for k, v in final_eval.items():
            if isinstance(v, (int, float)):
                tb.add_scalar(f"eval/{k}", v, cfg.max_steps)

    tb.close()
    elapsed = time.time() - t_start
    print(f"Training finished in {elapsed/60:.1f} min; final ply at {final_path}")
    if final_eval:
        print(
            f"Final eval (n={final_eval.get('n_test', 0)}): "
            f"PSNR={final_eval.get('psnr', math.nan):.2f} dB  "
            f"SSIM={final_eval.get('ssim', math.nan):.4f}  "
            f"L1={final_eval.get('l1', math.nan):.4f}"
        )
