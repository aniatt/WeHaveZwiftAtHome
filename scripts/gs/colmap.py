"""Parse COLMAP sparse reconstructions into typed dataclasses.

COLMAP's `automatic_reconstructor` writes binary files to `<workspace>/sparse/0/`:

  cameras.bin     -> intrinsics per camera_id
  images.bin      -> per-image extrinsics (cam_from_world) + filename + camera_id
  points3D.bin    -> sparse SfM point cloud (xyz + RGB)

We ingest them via `pycolmap.Reconstruction`, convert poses to 4x4 world->cam
viewmats (the convention `gsplat.rasterization` expects), and compute a
`scene_extent` value used to scale the means learning rate.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap

# Camera models we accept. SIMPLE_PINHOLE/PINHOLE are exact; SIMPLE_RADIAL and
# RADIAL get their distortion coefficient(s) silently dropped (treated as
# pinhole). `colmap automatic_reconstructor` defaults to SIMPLE_RADIAL, so
# rejecting it would break the existing preprocess pipeline.
_PINHOLE_MODELS = {"SIMPLE_PINHOLE", "PINHOLE"}
_APPROX_PINHOLE_MODELS = {"SIMPLE_RADIAL", "RADIAL"}


@dataclass(frozen=True)
class ColmapCamera:
    camera_id: int
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    name: str
    camera_id: int
    viewmat: np.ndarray  # (4, 4) world -> cam (float64)


@dataclass(frozen=True)
class ColmapPoints:
    xyz: np.ndarray  # (P, 3) float64
    rgb: np.ndarray  # (P, 3) uint8


@dataclass(frozen=True)
class ColmapScene:
    cameras: dict[int, ColmapCamera]
    images: list[ColmapImage]
    points: ColmapPoints
    scene_extent: float


def _camera_from_pycolmap(cam: pycolmap.Camera) -> ColmapCamera:
    name = cam.model_name
    if name in _PINHOLE_MODELS or name in _APPROX_PINHOLE_MODELS:
        fx = float(cam.focal_length_x)
        fy = float(cam.focal_length_y)
        cx = float(cam.principal_point_x)
        cy = float(cam.principal_point_y)
        return ColmapCamera(
            camera_id=cam.camera_id,
            width=int(cam.width),
            height=int(cam.height),
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )
    raise ValueError(
        f"Camera model {name!r} is not supported. "
        f"Re-run COLMAP with a pinhole-family model "
        f"(SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, or RADIAL)."
    )


def _viewmat_from_pycolmap(image: pycolmap.Image) -> np.ndarray:
    """Return a (4, 4) world->cam matrix from a pycolmap.Image."""
    cfw = image.cam_from_world
    if callable(cfw):
        cfw = cfw()
    rt = np.asarray(cfw.matrix(), dtype=np.float64)  # (3, 4)
    viewmat = np.eye(4, dtype=np.float64)
    viewmat[:3, :4] = rt
    return viewmat


def _compute_scene_extent(viewmats: list[np.ndarray]) -> float:
    """Radius around the camera centroid that contains all cameras, x1.1.

    Camera centers in world space are c = -R^T t.
    """
    centers = []
    for vm in viewmats:
        R = vm[:3, :3]
        t = vm[:3, 3]
        centers.append(-R.T @ t)
    centers = np.stack(centers, axis=0)  # (N, 3)
    centroid = centers.mean(axis=0)
    max_dist = float(np.linalg.norm(centers - centroid, axis=1).max())
    return max_dist * 1.1


def load_colmap(workspace: Path) -> ColmapScene:
    """Load a COLMAP reconstruction from `<workspace>/sparse/0/`."""
    workspace = Path(workspace)
    sparse_dir = workspace / "sparse" / "0"
    if not sparse_dir.is_dir():
        raise FileNotFoundError(
            f"Expected COLMAP sparse model at {sparse_dir}; "
            f"did `scripts/preprocess.sh` complete successfully?"
        )

    recon = pycolmap.Reconstruction(str(sparse_dir))

    cameras: dict[int, ColmapCamera] = {
        cam_id: _camera_from_pycolmap(cam) for cam_id, cam in recon.cameras.items()
    }

    # Stable order by image filename so train/test split is deterministic.
    images_sorted = sorted(recon.images.values(), key=lambda im: im.name)
    images: list[ColmapImage] = [
        ColmapImage(
            image_id=int(im.image_id),
            name=str(im.name),
            camera_id=int(im.camera_id),
            viewmat=_viewmat_from_pycolmap(im),
        )
        for im in images_sorted
    ]

    if not images:
        raise RuntimeError(f"No registered images found in {sparse_dir}")

    points_iter = recon.points3D.values()
    xyz = np.array([p.xyz for p in points_iter], dtype=np.float64)
    points_iter = recon.points3D.values()
    rgb = np.array([p.color for p in points_iter], dtype=np.uint8)
    if xyz.size == 0:
        raise RuntimeError(
            f"No 3D points in reconstruction at {sparse_dir}; "
            f"COLMAP may have failed to triangulate."
        )

    scene_extent = _compute_scene_extent([im.viewmat for im in images])

    return ColmapScene(
        cameras=cameras,
        images=images,
        points=ColmapPoints(xyz=xyz, rgb=rgb),
        scene_extent=scene_extent,
    )
