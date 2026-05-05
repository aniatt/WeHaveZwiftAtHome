#!/usr/bin/env bash
# Source this (don't execute) before running CUDA-dependent code:
#
#   source scripts/activate_cuda.sh
#
# Configures CUDA 12.1 toolkit for the current shell only -- does not modify
# ~/.bashrc, so other shells / other projects are untouched. Re-source after
# opening a new terminal.

if [[ ! -x /usr/local/cuda-12.1/bin/nvcc ]]; then
    echo "error: /usr/local/cuda-12.1/bin/nvcc not found." >&2
    echo "install with: sudo apt install -y cuda-toolkit-12-1" >&2
    return 1 2>/dev/null || exit 1
fi

export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
echo "CUDA 12.1 active (CUDA_HOME=$CUDA_HOME)"
