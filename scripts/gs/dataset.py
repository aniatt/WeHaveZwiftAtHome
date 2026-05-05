"""Torch Dataset over a parsed COLMAP scene.

Each item yields the ground-truth image, the world->cam viewmat, and the
intrinsics K, all as torch tensors on CPU. The caller moves them to device.

Train/test split: every Nth image (by sorted filename) is held out. The split
is deterministic so quantitative results are reproducible across runs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .colmap import ColmapScene


class ColmapDataset(Dataset):
    def __init__(
        self,
        scene: ColmapScene,
        frames_dir: Path,
        data_factor: int = 4,
        holdout_every: int = 8,
    ):
        self.scene = scene
        self.frames_dir = Path(frames_dir)
        self.data_factor = max(1, int(data_factor))
        self.holdout_every = max(0, int(holdout_every))

        # Validate every image file resolves on disk before we kick off training.
        for im in scene.images:
            if not (self.frames_dir / im.name).is_file():
                raise FileNotFoundError(
                    f"Image '{im.name}' referenced by COLMAP not found in "
                    f"{self.frames_dir}"
                )

        n = len(scene.images)
        if self.holdout_every > 0:
            self.test_indices = list(range(0, n, self.holdout_every))
            test_set = set(self.test_indices)
            self.train_indices = [i for i in range(n) if i not in test_set]
        else:
            self.test_indices = []
            self.train_indices = list(range(n))

    def __len__(self) -> int:
        return len(self.scene.images)

    def __getitem__(self, idx: int) -> dict:
        im = self.scene.images[idx]
        cam = self.scene.cameras[im.camera_id]

        path = self.frames_dir / im.name
        with Image.open(path) as pil:
            pil = pil.convert("RGB")
            if self.data_factor > 1:
                w = pil.width // self.data_factor
                h = pil.height // self.data_factor
                pil = pil.resize((w, h), Image.BILINEAR)
            arr = np.asarray(pil, dtype=np.float32) / 255.0  # (H, W, 3)

        H, W = arr.shape[:2]
        sx = W / cam.width
        sy = H / cam.height
        K = np.array(
            [
                [cam.fx * sx, 0.0, cam.cx * sx],
                [0.0, cam.fy * sy, cam.cy * sy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        return {
            "image": torch.from_numpy(arr),                     # (H, W, 3) float32
            "viewmat": torch.from_numpy(im.viewmat).float(),    # (4, 4)
            "K": torch.from_numpy(K),                           # (3, 3)
            "image_id": im.image_id,
            "name": im.name,
            "height": H,
            "width": W,
        }
