"""GaussianModel: the six learnable per-Gaussian tensors.

Parameter names match the convention `gsplat.strategy.DefaultStrategy` expects
(check_sanity asserts the dict contains exactly these keys), and store the
*un-activated* values:

  means       (N, 3)            xyz in world coords
  scales      (N, 3)            log-space; rendered as exp(scales)
  quats       (N, 4)            wxyz; gsplat normalizes internally
  opacities   (N,)              pre-sigmoid logit; rendered as sigmoid(opacities)
  sh_dc       (N, 1, 3)         DC band of spherical harmonics
  sh_rest     (N, K-1, 3)       higher SH bands; K = (max_sh_degree + 1)^2

The INRIA `.ply` format also stores un-activated values; the viewer applies
exp/sigmoid at render time, and so does the densification strategy when it
splits Gaussians. So we never store activated values anywhere.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from gsplat.exporter import export_splats

# Value of the degree-0 spherical harmonic basis: Y_0^0 = 1 / (2 * sqrt(pi)).
# Constant in direction. Used to invert the convention `color = SH0 * c_DC + 0.5`
# when seeding sh_dc from an SfM point's RGB color.
SH0 = 0.28209479177387814


def _knn_mean_distance(xyz: np.ndarray, k: int = 3) -> np.ndarray:
    """Mean distance from each point to its k nearest neighbours.

    Used to set initial (isotropic) Gaussian scales. We use a brute-force
    pairwise distance because point clouds from COLMAP are typically <1M
    points and this runs once per training job.
    """
    n = xyz.shape[0]
    if n <= 1:
        return np.full((n,), 0.01, dtype=np.float32)
    # For very large clouds, chunk to bound memory.
    chunk = 4096
    means = np.empty(n, dtype=np.float32)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sub = xyz[start:end]                                         # (m, 3)
        d2 = np.sum((sub[:, None, :] - xyz[None, :, :]) ** 2, axis=2)
        # Self-distance is 0; mask it by setting to +inf before partition.
        np.fill_diagonal(d2[:, start:end], np.inf)
        kth = np.partition(d2, kth=k, axis=1)[:, :k]
        kth = np.sqrt(np.maximum(kth, 1e-12))
        means[start:end] = kth.mean(axis=1).astype(np.float32)
    return np.maximum(means, 1e-6)


class GaussianModel(nn.Module):
    def __init__(
        self,
        means: torch.Tensor,
        scales: torch.Tensor,
        quats: torch.Tensor,
        opacities: torch.Tensor,
        sh_dc: torch.Tensor,
        sh_rest: torch.Tensor,
        max_sh_degree: int,
    ):
        super().__init__()
        self.max_sh_degree = int(max_sh_degree)
        self.params = nn.ParameterDict(
            {
                "means": nn.Parameter(means),
                "scales": nn.Parameter(scales),
                "quats": nn.Parameter(quats),
                "opacities": nn.Parameter(opacities),
                "sh_dc": nn.Parameter(sh_dc),
                "sh_rest": nn.Parameter(sh_rest),
            }
        )

    @classmethod
    def from_points(
        cls,
        xyz: np.ndarray,
        rgb: np.ndarray,
        max_sh_degree: int = 3,
        initial_opacity: float = 0.1,
        device: str | torch.device = "cuda",
    ) -> "GaussianModel":
        """Seed Gaussians from a sparse SfM point cloud.

        - means      <- xyz
        - scales     <- log(mean kNN distance), repeated 3x (isotropic, log-space)
        - quats      <- [1, 0, 0, 0] (identity)
        - opacities  <- inverse_sigmoid(initial_opacity)  (default alpha=0.1)
        - sh_dc      <- (rgb - 0.5) / SH0      (so render reproduces input rgb)
        - sh_rest    <- zeros
        """
        n = xyz.shape[0]
        rgb = rgb.astype(np.float32) / 255.0  # (N, 3) in [0, 1]

        means = torch.from_numpy(xyz.astype(np.float32))

        # Scales: isotropic, log-space. One value per point, repeated across xyz axes.
        knn = _knn_mean_distance(xyz.astype(np.float32), k=3)  # (N,)
        scales = torch.from_numpy(np.log(knn))[:, None].repeat(1, 3)

        # Identity rotation (wxyz).
        quats = torch.zeros((n, 4), dtype=torch.float32)
        quats[:, 0] = 1.0

        # Inverse sigmoid: log(p / (1 - p)).
        p = float(initial_opacity)
        opa_logit = float(np.log(p / (1.0 - p)))
        opacities = torch.full((n,), opa_logit, dtype=torch.float32)

        # SH degree 0 basis = SH0 (constant). Pick coeff so SH0 * coeff + 0.5
        # equals the input RGB. (gsplat / INRIA convention: shift by 0.5.)
        sh_dc = ((torch.from_numpy(rgb) - 0.5) / SH0)[:, None, :]  # (N, 1, 3)

        k_rest = (max_sh_degree + 1) ** 2 - 1
        sh_rest = torch.zeros((n, k_rest, 3), dtype=torch.float32)

        return cls(
            means=means.to(device),
            scales=scales.to(device),
            quats=quats.to(device),
            opacities=opacities.to(device),
            sh_dc=sh_dc.to(device),
            sh_rest=sh_rest.to(device),
            max_sh_degree=max_sh_degree,
        )

    @property
    def num_gaussians(self) -> int:
        return int(self.params["means"].shape[0])

    def sh_coeffs(self) -> torch.Tensor:
        """Concatenate DC + rest into a single (N, K, 3) tensor for rasterization."""
        return torch.cat([self.params["sh_dc"], self.params["sh_rest"]], dim=1)

    def export_ply(self, path: Path | str) -> None:
        """Write an INRIA-format .ply consumable by the web viewer.

        export_splats stores the un-activated values directly (log-space
        scales, pre-sigmoid opacity logits); the viewer applies exp/sigmoid
        at render time.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            export_splats(
                means=self.params["means"].detach(),
                scales=self.params["scales"].detach(),
                quats=self.params["quats"].detach(),
                opacities=self.params["opacities"].detach(),
                sh0=self.params["sh_dc"].detach(),
                shN=self.params["sh_rest"].detach(),
                format="ply",
                save_to=str(path),
            )
