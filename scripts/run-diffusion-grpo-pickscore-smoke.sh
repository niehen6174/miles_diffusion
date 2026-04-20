#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

# Use two colocated train/rollout GPUs plus one dedicated PickScore reward GPU.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_NAME="diffusion_grpo_pickscore_smoke_$(date +%Y%m%d_%H%M%S)"

WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_ARGS+=(
    --use-wandb
    --wandb-project miles-diffusion-grpo
    --wandb-group "${RUN_NAME}"
    --wandb-key "${WANDB_API_KEY}"
    --diffusion-log-images 4
    --diffusion-log-image-interval 1
    --disable-wandb-random-suffix
  )
fi

"${PYTHON_BIN}" "${ROOT_DIR}/tools/prepare_ocr_jsonl.py"

"${PYTHON_BIN}" -u "${ROOT_DIR}/train_diffusion.py" \
  --train-backend fsdp \
  --diffusion-train \
  --rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout \
  --hf-checkpoint gpt2 \
  --prompt-data "${ROOT_DIR}/data/ocr/train.jsonl" \
  --input-key input \
  --rollout-batch-size 1 \
  --n-samples-per-prompt 2 \
  --num-rollout 1 \
  --diffusion-timestep-batch 10 \
  --gradient-checkpointing \
  --actor-num-gpus-per-node 2 \
  --rollout-num-gpus 2 \
  --rollout-num-gpus-per-engine 1 \
  --num-gpus-per-node 3 \
  --colocate \
  --no-offload-rollout \
  --use-lora \
  --lora-rank 64 \
  --use-miles-router \
  --sglang-server-concurrency 2 \
  --diffusion-model Qwen/Qwen-Image \
  --diffusion-reward pickscore:1.0 \
  --advantage-estimator grpo \
  --globalize-reward-norm \
  --rm-type pickscore \
  --pickscore-num-workers 1 \
  --pickscore-num-gpus-per-worker 1.0 \
  --pickscore-batch-size 2 \
  --pickscore-dtype fp32 \
  --pickscore-processor-path laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
  --pickscore-model-path yuvalkirstain/PickScore_v1 \
  --diffusion-dtype bf16 \
  --diffusion-num-steps 10 \
  --diffusion-num-batches-per-epoch 1 \
  --diffusion-guidance-scale 4.0 \
  --diffusion-true-cfg-scale 4.0 \
  --diffusion-rollout-noise-level 0.7 \
  --diffusion-height 256 \
  --diffusion-width 256 \
  --global-batch-size 2 \
  --diffusion-ignore-last 1 \
  --diffusion-rollout-debug-mode \
  --debug-skip-optimizer-step \
  "${WANDB_ARGS[@]}"
