#!/usr/bin/env python3
"""Wrapper around gsplat's simple_trainer for 3DGS model training."""
# TODO(anirudha): MVP. We will likely need to replace this with a custom training script using gsplat library functions.
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Train a 3D Gaussian Splatting model using gsplat."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to COLMAP workspace (output of preprocess.sh).",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="results/default",
        help="Directory to write trained model output.",
    )
    parser.add_argument(
        "--data_factor",
        type=int,
        default=4,
        help="Image downsample factor (default: 4).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=7000,
        help="Maximum training iterations (default: 7000).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    result_dir = Path(args.result_dir)

    if not data_dir.exists():
        print(f"Error: data directory not found: {data_dir}")
        sys.exit(1)

    result_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "gsplat.examples.simple_trainer",
        "default",
        f"--data_dir={data_dir}",
        f"--data_factor={args.data_factor}",
        f"--result_dir={result_dir}",
        f"--max_steps={args.max_steps}",
    ]

    print(f"==> Training 3DGS model")
    print(f"    Data:       {data_dir}")
    print(f"    Output:     {result_dir}")
    print(f"    Factor:     {args.data_factor}x downsample")
    print(f"    Max steps:  {args.max_steps}")
    print()

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("Error: gsplat is not installed. Run: pip install gsplat")
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"Error: training exited with code {exc.returncode}")
        sys.exit(exc.returncode)

    ply_files = sorted(result_dir.rglob("*.ply"))
    if ply_files:
        print(f"\n==> Training complete. Model saved to: {ply_files[-1]}")
        print(f"    Load it in the viewer:  cd viewer && npm run dev")
    else:
        print(f"\n==> Training finished but no .ply file found in {result_dir}.")
        print("    Check the gsplat output above for details.")


if __name__ == "__main__":
    main()
