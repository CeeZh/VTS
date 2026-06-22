#!/bin/bash
# Rollout server for GRPO training. Run this on its own GPU before
# launching grpo_all60.sh — the trainer connects to this server over
# the port below (default 8100).
#
# Set MODEL to the policy you want to sample from. After SFT, this is
# typically the SFT checkpoint (converted to a transformers-4.x layout
# via scripts/convert_checkpoint.py if it came from LLaMA-Factory). You
# can also start from the bare base model.

# MODEL=Qwen/Qwen3-VL-8B-Instruct
MODEL=${MODEL:-/path/to/sft_checkpoint-converted}

mkdir -p slurm_logs

sbatch \
  --cpus-per-task=48 \
  --gpus=1 \
  -p h100 \
  -o slurm_logs/output_%j.log \
  -J vts_rollout \
  --wrap="export DECORD_EOF_RETRY_MAX=20480 \
FPS_MAX_FRAMES=64 \
CLIP_MAX_FRAMES=64 \
IMAGE_MAX_TOKEN_NUM=64 \
NUM_FRAMES_PER_SEGMENT=64 \
MAX_TURNS=10 \
ACTION_MODE=free \
OUTPUT_FORMAT=sft \
USE_HF=1; \
swift rollout \
    --model $MODEL \
    --model_type qwen3_vl \
    --vllm_use_async_engine true \
    --external_plugins rl/video_crop_plugin.py \
    --multi_turn_scheduler tree_search_scheduler \
    --vllm_max_model_len 65536 \
    --vllm_gpu_memory_utilization 0.85 \
    --vllm_mm_processor_cache_gb 0 \
    --max_turns 20 \
    --vllm_limit_mm_per_prompt '{\"image\": 64, \"video\": 0}' \
    --vllm_max_num_seqs 8 \
    --vllm_enforce_eager true \
    --vllm_enable_prefix_caching false \
    --port 8100
  "
