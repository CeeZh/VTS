#!/bin/bash
# GRPO trainer — connects to the rollout server at $VLLM_HOST:$VLLM_PORT.
# Run rollout.sh first on 1 GPU, then run this on the remaining 3 GPUs.
# Run from the VTS repo root so the dataset's data/ paths resolve.
#
# Reward stack (see rl/vts_plugin.py):
#   tree_acc_reward          — answer correctness
#   tree_format_reward       — <think>...</think> + JSON action format
#   tree_iou_reward          — evidence-interval IoU vs GT
#   tree_gt_distance_reward  — couples IoU payoff to tree navigation
#   tree_debug_logger        — passthrough logger (weight 0)
#
# --padding_side left is REQUIRED: GRPO assumes left-padded (right-aligned)
# completions; right padding crashes the trainer at step 1 in selective_log_softmax.

set -e

MODEL=${MODEL:-output/vts_qwen3_vl_8b-converted}
DATASET=${DATASET:-data/rl/rl.json}
OUTPUT_DIR=${OUTPUT_DIR:-output/vts_qwen3_vl_8b-grpo}

VLLM_HOST=${VLLM_HOST:-127.0.0.1}
VLLM_PORT=${VLLM_PORT:-8100}

export CUDA_VISIBLE_DEVICES=${TRAIN_GPUS:-1,2,3}
NPROC=${NPROC:-3}

mkdir -p "$OUTPUT_DIR"

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
  DECORD_EOF_RETRY_MAX=20480 \
  FPS_MAX_FRAMES=64 \
  CLIP_MAX_FRAMES=64 \
  IMAGE_MAX_TOKEN_NUM=64 \
  NUM_FRAMES_PER_SEGMENT=64 \
  MAX_TURNS=10 \
  DEBUG_TRAJECTORIES=1 \
  DEBUG_OUTPUT_DIR="$OUTPUT_DIR/debug_trajectories" \
  DEBUG_LOG_INTERVAL=5 \
  ACTION_MODE=free \
  OUTPUT_FORMAT=sft \
  USE_HF=1

NPROC_PER_NODE=$NPROC swift rlhf \
    --model "$MODEL" \
    --model_type qwen3_vl \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host "$VLLM_HOST" \
    --vllm_server_port "$VLLM_PORT" \
    --vllm_server_pass_dataset true \
    --rlhf_type grpo \
    --train_type lora \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --padding_side left \
    --freeze_vit true \
    --freeze_aligner false \
    --freeze_llm false \
    --external_plugins rl/vts_plugin.py \
    --reward_funcs tree_acc_reward tree_format_reward tree_iou_reward tree_gt_distance_reward tree_debug_logger \
    --reward_weights 0.5 0.5 1.0 0.0 0.0 \
    --dataset "$DATASET" \
    --split_dataset_ratio 0 \
    --max_completion_length ${MAX_COMPLETION_LENGTH:-2048} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${PER_DEVICE:-1} \
    --learning_rate 5e-5 \
    --lr_scheduler_type constant_with_warmup \
    --gradient_accumulation_steps ${GRAD_ACCUM:-32} \
    --save_only_model false \
    --save_strategy 'steps' \
    --save_steps ${SAVE_STEPS:-20} \
    --save_total_limit 8 \
    --logging_steps 1 \
    --warmup_ratio 0 \
    --dataloader_num_workers 64 \
    --dataset_num_proc 64 \
    --num_generations ${NUM_GENERATIONS:-8} \
    --temperature 1.0 \
    --log_completions true \
    --log_entropy true \
    --steps_per_generation ${STEPS_PER_GENERATION:-8} \
    --beta 0.04 \
    --num_iterations 1 \
    --attn_impl flash_attn \
    --deepspeed zero2 \
    --gradient_checkpointing true \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --report_to tensorboard \
    --output_dir "$OUTPUT_DIR" \
    ${EXTRA_GRPO_ARGS:-}
