#!/usr/bin/env python3
"""Sanity-check a COLMAP reconstruction before spending GPU hours training on it.

Structure-from-Motion fails *quietly*. It will happily return a full set of
poses that look plausible while the recovered camera path bears no
resemblance to how the camera actually moved.

The specific failure this catches is monocular scale collapse. SfM infers how
far the camera travelled from parallax on *nearby* geometry. When a shot opens
out into a wide vista and everything in frame is distant, moving two metres and
moving twenty centimetres look nearly identical. Scale silently drifts toward zero, 
and the reconstructed camera decelerates to a standstill while the video keeps moving.

Run this straight after preprocess.sh:

    python scripts/check_reconstruction.py --data_dir data/colmap

Exits non-zero if any check FAILs, so it can gate a pipeline script.

Note: the look-vs-travel check assumes forward-facing POV footage (a bar- or
helmet-mounted camera). It will report a false failure on deliberately
sideways-facing or tracking-shot footage; ignore that one line if so.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gs.colmap import load_colmap  # noqa: E402

# Thresholds. Chosen so a steady, well-reconstructed ride passes comfortably
# while the scale-collapse signature fails clearly.
STEP_CV_PASS, STEP_CV_WARN = 0.50, 0.80
ARC_CORR_PASS, ARC_CORR_WARN = 0.98, 0.94
LOOK_ANGLE_PASS, LOOK_ANGLE_WARN = 25.0, 45.0
SCALE_RATIO_LO, SCALE_RATIO_HI = 0.40, 2.50
TRI_ANGLE_PASS, TRI_ANGLE_WARN = 5.0, 3.0
REPROJ_PASS, REPROJ_WARN = 1.50, 2.50

FAILED: list[str] = []


def report(name: str, value: str, status: str, detail: str = "") -> None:
    """Print one check line and record failures for the exit code."""
    mark = {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL", "INFO": "    "}[status]
    print(f"  [{mark}] {name:<34} {value:>14}   {detail}")
    if status == "FAIL":
        FAILED.append(name)


def grade(value: float, pass_at: float, warn_at: float, higher_is_better: bool) -> str:
    if higher_is_better:
        return "PASS" if value >= pass_at else ("WARN" if value >= warn_at else "FAIL")
    return "PASS" if value <= pass_at else ("WARN" if value <= warn_at else "FAIL")


def image_motion(frames_dir: Path, n_expected: int) -> np.ndarray | None:
    """Mean abs difference between consecutive frames -- a proxy for true camera speed.

    Deliberately crude: downsampled greyscale, no optical flow. We only need to
    know whether apparent motion is roughly steady, to compare against what the
    reconstruction claims. Returns None if frames aren't usable.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    paths = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    if len(paths) != n_expected:
        # Frame count must line up with registered images or the comparison is
        # meaningless (e.g. COLMAP dropped frames).
        return None
    small = [
        np.asarray(Image.open(p).convert("L").resize((320, 180)), dtype=np.float32)
        for p in paths
    ]
    return np.array([np.abs(small[i + 1] - small[i]).mean() for i in range(len(small) - 1)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", type=Path, default=Path("data/colmap"),
                    help="COLMAP workspace containing sparse/0/.")
    ap.add_argument("--frames_dir", type=Path, default=None,
                    help="Frames referenced by COLMAP. Defaults to <data_dir>/../frames. "
                         "Enables the scale-collapse check.")
    args = ap.parse_args()

    frames_dir = args.frames_dir or (args.data_dir.parent / "frames")

    scene = load_colmap(args.data_dir)
    n = len(scene.images)
    print(f"\nCOLMAP workspace : {args.data_dir}")
    print(f"registered images: {n}   points: {scene.points.xyz.shape[0]:,}   "
          f"scene_extent: {scene.scene_extent:.2f}\n")
    if n < 3:
        print("Too few registered images to assess.")
        sys.exit(1)

    # Camera centres in world space: c = -R^T t
    C = np.array([-im.viewmat[:3, :3].T @ im.viewmat[:3, 3] for im in scene.images])
    step = np.linalg.norm(np.diff(C, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(step)])

    print("TRAJECTORY")

    # 1. Steadiness. A rider's speed varies, but not by orders of magnitude.
    #    Scale collapse shows up here first: early steps metres long, later
    #    steps centimetres.
    cv = float(step.std() / step.mean()) if step.mean() > 0 else float("inf")
    report("step-size variation (CV)", f"{cv:.2f}",
           grade(cv, STEP_CV_PASS, STEP_CV_WARN, higher_is_better=False),
           f"want <{STEP_CV_PASS}; steady ride ~0.1-0.3")

    # 2. Does distance travelled advance steadily with frame number?
    corr = float(np.corrcoef(np.arange(n), arc)[0, 1])
    report("corr(frame index, arc length)", f"{corr:.4f}",
           grade(corr, ARC_CORR_PASS, ARC_CORR_WARN, higher_is_better=True),
           f"want >{ARC_CORR_PASS}")

    back = int((np.diff(arc) <= 0).sum())
    report("backward steps", f"{back}/{n - 1}",
           "PASS" if back <= max(1, n // 20) else "WARN", "camera should advance")

    # 3. Is the camera looking where it is going? Compared against a *smoothed*
    #    local travel direction so a curving road or a hairpin doesn't trip it.
    look = np.array([im.viewmat[:3, :3].T @ np.array([0.0, 0.0, 1.0])
                     for im in scene.images])
    w = 2
    travel = np.array([C[min(i + w, n - 1)] - C[max(i - w, 0)] for i in range(n)])
    norms = np.linalg.norm(travel, axis=1)
    ok = norms > 1e-9
    cos = (look[ok] * (travel[ok] / norms[ok, None])).sum(1)
    look_ang = float(np.median(np.degrees(np.arccos(np.clip(cos, -1, 1)))))
    report("angle(look, travel) median", f"{look_ang:.1f} deg",
           grade(look_ang, LOOK_ANGLE_PASS, LOOK_ANGLE_WARN, higher_is_better=False),
           f"want <{LOOK_ANGLE_PASS} deg for POV footage")

    # 4. Scale collapse: compare how much the reconstruction slows down against
    #    how much the *image* slows down. These should agree. When they diverge
    #    the reconstruction is losing scale, not the rider losing speed.
    print("\nSCALE CONSISTENCY")
    motion = image_motion(frames_dir, n)
    if motion is None:
        report("recon vs image motion", "skipped", "INFO",
               f"needs {n} frames in {frames_dir}")
    else:
        third = max(1, (n - 1) // 3)
        r_rec = step[:third].mean() / max(step[-third:].mean(), 1e-9)
        r_img = motion[:third].mean() / max(motion[-third:].mean(), 1e-9)
        disc = r_rec / max(r_img, 1e-9)
        status = "PASS" if SCALE_RATIO_LO <= disc <= SCALE_RATIO_HI else "FAIL"
        report("recon/image slowdown ratio", f"{disc:.1f}x", status,
               f"want {SCALE_RATIO_LO}-{SCALE_RATIO_HI}x "
               f"(recon {r_rec:.1f}x vs image {r_img:.1f}x)")

    # 5. Structure quality. Informational unless clearly bad -- these degrade
    #    gracefully, whereas the trajectory checks above are pass/fail.
    print("\nSTRUCTURE")
    try:
        import pycolmap
        recon = pycolmap.Reconstruction(str(args.data_dir / "sparse" / "0"))
        pts = list(recon.points3D.values())
        tl = np.array([len(p.track.elements) for p in pts])
        err = np.array([p.error for p in pts])
        report("mean reprojection error", f"{err.mean():.2f} px",
               grade(float(err.mean()), REPROJ_PASS, REPROJ_WARN, higher_is_better=False),
               f"want <{REPROJ_PASS} px")
        report("median track length", f"{np.median(tl):.0f}",
               "PASS" if np.median(tl) >= 4 else "WARN", "images observing each point")

        # Triangulation angle governs how well depth is constrained. Sampled,
        # since this is O(track^2) per point.
        cmap = {im.image_id: -im.viewmat[:3, :3].T @ im.viewmat[:3, 3]
                for im in scene.images}
        rng = np.random.default_rng(0)
        idx = rng.choice(len(pts), size=min(3000, len(pts)), replace=False)
        angs = []
        for i in idx:
            p = pts[i]
            obs = [cmap[e.image_id] for e in p.track.elements if e.image_id in cmap]
            if len(obs) < 2:
                continue
            V = np.array(obs) - p.xyz
            V /= np.linalg.norm(V, axis=1, keepdims=True)
            angs.append(np.degrees(np.arccos(np.clip(V @ V.T, -1, 1))).max())
        if angs:
            med = float(np.median(angs))
            report("median triangulation angle", f"{med:.2f} deg",
                   grade(med, TRI_ANGLE_PASS, TRI_ANGLE_WARN, higher_is_better=True),
                   f"want >{TRI_ANGLE_PASS} deg; needs nearby geometry")
    except ImportError:
        report("structure stats", "skipped", "INFO", "pycolmap not available")

    print()
    if FAILED:
        print(f"RESULT: FAIL ({len(FAILED)} check(s): {', '.join(FAILED)})")
        print("Do not train on this reconstruction -- the poses are wrong.")
        print("Try: more frames (higher fps), footage with nearer geometry, or GLOMAP.")
        sys.exit(1)
    print("RESULT: OK -- reconstruction looks trainable.")
    sys.exit(0)


if __name__ == "__main__":
    main()
