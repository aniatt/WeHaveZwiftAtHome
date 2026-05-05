#!/usr/bin/env bash
set -euo pipefail

# COLMAP links against Qt and tries to initialize a GUI app even in CLI mode.
# Force the offscreen platform plugin so it works headless (over SSH, in a
# container, with no X display set).
export QT_QPA_PLATFORM=offscreen

usage() {
    echo "Usage: $0 <video_file> [workspace_dir]"
    echo
    echo "  video_file     Path to the input video (e.g. data/raw/climb.mp4)"
    echo "  workspace_dir  Output directory (default: data/)"
    echo
    echo "This script extracts frames with ffmpeg and runs COLMAP SfM."
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

VIDEO="$1"
WORKSPACE="${2:-data}"

if [[ ! -f "$VIDEO" ]]; then
    echo "Error: video file not found: $VIDEO"
    exit 1
fi

for cmd in ffmpeg colmap; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "Error: '$cmd' is not installed or not on PATH."
        echo "Install it before running this script."
        exit 1
    fi
done

FRAMES_DIR="$WORKSPACE/frames"
COLMAP_DIR="$WORKSPACE/colmap"

# Clean any prior output so the run starts fresh. COLMAP's automatic_reconstructor
# can leave incompatible state if re-run on top of an existing workspace.
if [[ -d "$FRAMES_DIR" ]] || [[ -d "$COLMAP_DIR" ]]; then
    echo "==> Cleaning previous output: $FRAMES_DIR  $COLMAP_DIR"
    rm -rf "$FRAMES_DIR" "$COLMAP_DIR"
fi

mkdir -p "$FRAMES_DIR" "$COLMAP_DIR"

echo "==> Extracting frames from $VIDEO ..."
ffmpeg -i "$VIDEO" -vf "fps=2" "$FRAMES_DIR/frame_%05d.jpg" -y
FRAME_COUNT=$(ls "$FRAMES_DIR"/frame_*.jpg 2>/dev/null | wc -l)
echo "    Extracted $FRAME_COUNT frames to $FRAMES_DIR"

if [[ "$FRAME_COUNT" -lt 10 ]]; then
    echo "Warning: only $FRAME_COUNT frames extracted. Consider a longer video or higher fps."
fi

echo "==> Running COLMAP automatic_reconstructor ..."
# --use_gpu 0 forces CPU SIFT extraction + matching. Slower than SiftGPU but
# works headless (SiftGPU needs an OpenGL context, which requires a display
# server). Flip back to 1 being run with display and speedup is desired.
colmap automatic_reconstructor \
    --workspace_path "$COLMAP_DIR" \
    --image_path "$FRAMES_DIR" \
    --single_camera 1 \
    --use_gpu 0

SPARSE_DIR="$COLMAP_DIR/sparse"
if [[ -d "$SPARSE_DIR" ]] && [[ -n "$(ls -A "$SPARSE_DIR" 2>/dev/null)" ]]; then
    echo "==> COLMAP reconstruction complete."
    echo "    Sparse model: $SPARSE_DIR"
else
    echo "Error: COLMAP did not produce a sparse model in $SPARSE_DIR"
    echo "Check the logs above for reconstruction failures."
    exit 1
fi

echo
echo "Done. Next step: train the 3DGS model with:"
echo "  python scripts/train.py --data_dir $COLMAP_DIR --result_dir results/my_route"
