#!/bin/bash
# Rollout (vLLM) server for GRPO — serves completions on $PORT (default 8100).
# Run this on 1 GPU before launching grpo.sh (the trainer connects over $PORT).
# Run from the VTS repo root; the scheduler reads trees/frames from data/, so no
# CLIP/segmentation server is needed.

set -e

MODEL=${MODEL:-output/vts_qwen3_vl_8b-converted}
PORT=${PORT:-8100}

export CUDA_VISIBLE_DEVICES=${ROLLOUT_GPUS:-0}

export DECORD_EOF_RETRY_MAX=20480 \
  FPS_MAX_FRAMES=64 \
  CLIP_MAX_FRAMES=64 \
  IMAGE_MAX_TOKEN_NUM=64 \
  NUM_FRAMES_PER_SEGMENT=64 \
  MAX_TURNS=10 \
  ACTION_MODE=free \
  OUTPUT_FORMAT=sft \
  USE_HF=1

swift rollout \
    --model "$MODEL" \
    --model_type qwen3_vl \
    --vllm_use_async_engine true \
    --external_plugins rl/vts_plugin.py \
    --multi_turn_scheduler tree_search_scheduler \
    --vllm_max_model_len 65536 \
    --vllm_gpu_memory_utilization 0.85 \
    --vllm_mm_processor_cache_gb 0 \
    --max_turns 20 \
    --vllm_limit_mm_per_prompt '{"image": 64, "video": 0}' \
    --vllm_max_num_seqs 8 \
    --vllm_enforce_eager true \
    --vllm_enable_prefix_caching false \
    --port "$PORT"
