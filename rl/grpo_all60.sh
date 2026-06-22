#!/bin/bash
# GRPO trainer launcher.
#
# Reward stack (see rl/video_crop_plugin.py):
#   tree_acc_reward          — answer correctness
#   tree_format_reward       — <think>...</think> + JSON action format
#   tree_iou_reward          — evidence-interval IoU vs GT
#   tree_gt_distance_reward  — couples IoU payoff to tree navigation
#   tree_debug_logger        — passthrough logger (weight 0)
#
# Run rollout.sh first on a separate GPU to serve completions on port 8100.

# Set MODEL to the same SFT checkpoint as rollout.sh.
# MODEL=Qwen/Qwen3-VL-8B-Instruct
MODEL=${MODEL:-/path/to/sft_checkpoint-converted}

# JSONL produced by rl/data_scripts/*.py (one record per QA sample).
DATASET=${DATASET:-rl/data/merged_cg60_ego60_yt40.jsonl}

OUTPUT_DIR=${OUTPUT_DIR:-rl/ckpt/grpo_lr5e-5_turn10_beta0.04_acc0.5+iou1+fmt0.5}

mkdir -p slurm_logs

sbatch \
  --cpus-per-task=48 \
  --gpus=3 \
  -p h100 \
  -o slurm_logs/output_%j.log \
  -J vts_grpo \
  --wrap="export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
DECORD_EOF_RETRY_MAX=20480 \
FPS_MAX_FRAMES=64 \
CLIP_MAX_FRAMES=64 \
IMAGE_MAX_TOKEN_NUM=64 \
NUM_FRAMES_PER_SEGMENT=64 \
MAX_TURNS=10 \
NPROC_PER_NODE=3 \
DEBUG_TRAJECTORIES=1 \
DEBUG_OUTPUT_DIR=$OUTPUT_DIR/debug_trajectories \
DEBUG_LOG_INTERVAL=5 \
ACTION_MODE=free \
OUTPUT_FORMAT=sft \
USE_HF=1; \
swift rlhf \
    --model $MODEL \
    --model_type qwen3_vl \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 0.0.0.0 \
    --vllm_server_port 8100 \
    --vllm_server_pass_dataset true \
    --rlhf_type grpo \
    --train_type lora \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --freeze_vit true \
    --freeze_aligner false \
    --freeze_llm false \
    --external_plugins rl/video_crop_plugin.py \
    --reward_funcs tree_acc_reward tree_format_reward tree_iou_reward tree_gt_distance_reward tree_debug_logger \
    --reward_weights 0.5 0.5 1.0 0.0 0.0 \
    --dataset $DATASET \
    --split_dataset_ratio 0 \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 5e-5 \
    --lr_scheduler_type constant_with_warmup \
    --gradient_accumulation_steps 32 \
    --save_only_model false \
    --save_strategy 'steps' \
    --save_steps 20 \
    --save_total_limit 8 \
    --logging_steps 1 \
    --warmup_ratio 0 \
    --dataloader_num_workers 64 \
    --dataset_num_proc 64 \
    --num_generations 8 \
    --temperature 1.0 \
    --log_completions true \
    --log_entropy true \
    --steps_per_generation 8 \
    --beta 0.04 \
    --num_iterations 1 \
    --attn_impl flash_attn \
    --deepspeed zero2 \
    --gradient_checkpointing true \
    --gradient_checkpointing_kwargs '{\"use_reentrant\": false}' \
    --report_to tensorboard \
    --output_dir $OUTPUT_DIR
  "
