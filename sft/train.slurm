#!/bin/bash
#SBATCH --job-name=vts-sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

# Run from the root of your LLaMA-Factory checkout, e.g. ~/LLaMA-Factory.
# Symlink (or copy) this repo's `sft/` folder into LLaMA-Factory as `vts_sft/`
# first — see VTS/README.md for the full setup.

set -e
mkdir -p slurm_logs

# Activate the env you created for LLaMA-Factory.
source .venv/bin/activate
which python

# Pick a config:
#   sft_qwen3.yaml      — Qwen3-VL-8B-Instruct (default)
#   sft_qwen2.5.yaml    — Qwen2.5-VL-7B-Instruct
python -m llamafactory.cli train vts_sft/sft_qwen3.yaml
