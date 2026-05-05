#!/usr/bin/env python3
"""From-scratch 3D Gaussian Splatting training driver.

Replaces the gsplat.examples.simple_trainer wrapper in train.py. Wires
together the building blocks under scripts/gs/.

Usage:
    python scripts/train_custom.py \
        --data_dir data/colmap \
        --frames_dir data/frames \
        --result_dir results/my_route \
        --max_steps 7000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

# Make `gs/` importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gs.colmap import load_colmap                       # noqa: E402
from gs.dataset import ColmapDataset                    # noqa: E402
from gs.model import GaussianModel                      # noqa: E402
from gs.trainer import TrainConfig, train               # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a 3D Gaussian Splatting model from a COLMAP workspace."
    )
    p.add_argument("--data_dir", type=Path, required=True,
                   help="COLMAP workspace (output of preprocess.sh, contains sparse/0/).")
    p.add_argument("--frames_dir", type=Path, default=None,
                   help="Directory of frames referenced by COLMAP. "
                        "Defaults to <data_dir>/../frames if not set.")
    p.add_argument("--result_dir", type=Path, default=Path("results/default"))
    p.add_argument("--data_factor", type=int, default=1,
                   help="Image downsample factor.")
    p.add_argument("--max_steps", type=int, default=30000)
    p.add_argument("--max_sh_degree", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Reproducibility: fix RNGs before any torch CUDA allocation.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Resolve frames_dir: COLMAP image filenames are relative to the image_path
    # passed to `colmap automatic_reconstructor`, which preprocess.sh sets to
    # `data/frames` -- a sibling of `data/colmap`.
    frames_dir = args.frames_dir or (args.data_dir.parent / "frames")
    if not frames_dir.is_dir():
        raise FileNotFoundError(
            f"frames_dir not found: {frames_dir}. "
            f"Pass --frames_dir explicitly if your layout differs."
        )

    print(f"==> Loading COLMAP scene from {args.data_dir}")
    scene = load_colmap(args.data_dir)
    print(f"    cameras: {len(scene.cameras)}  images: {len(scene.images)}  "
          f"points: {scene.points.xyz.shape[0]}  scene_extent: {scene.scene_extent:.3f}")

    dataset = ColmapDataset(
        scene=scene,
        frames_dir=frames_dir,
        data_factor=args.data_factor,
        holdout_every=8,
    )
    print(f"    train: {len(dataset.train_indices)}  test: {len(dataset.test_indices)}  "
          f"data_factor: {args.data_factor}")

    print(f"==> Initializing Gaussians from {scene.points.xyz.shape[0]} SfM points")
    model = GaussianModel.from_points(
        xyz=scene.points.xyz,
        rgb=scene.points.rgb,
        max_sh_degree=args.max_sh_degree,
        device=args.device,
    )

    cfg = TrainConfig(
        max_steps=args.max_steps,
        max_sh_degree=args.max_sh_degree,
        seed=args.seed,
    )

    args.result_dir.mkdir(parents=True, exist_ok=True)
    config_dump = {"args": {k: str(v) for k, v in vars(args).items()}, "train": asdict(cfg)}
    (args.result_dir / "config.json").write_text(json.dumps(config_dump, indent=2))

    print(f"==> Training for {cfg.max_steps} steps")
    train(
        model=model,
        dataset=dataset,
        scene_extent=scene.scene_extent,
        result_dir=args.result_dir,
        cfg=cfg,
        device=args.device,
    )


if __name__ == "__main__":
    main()
