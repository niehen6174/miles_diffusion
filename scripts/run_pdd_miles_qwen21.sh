#!/usr/bin/env bash
set -euo pipefail
# First integration: two FSDP ranks, 8 prompts and 8 optimizer updates per rollout.
# Pass --load "$OUTPUT/ckpt" only to resume a Miles checkpoint; init is weights-only.
: "${MODEL:?Set MODEL to the local Qwen-Image 2.1 snapshot}"
: "${CACHE:?Set CACHE to the verified sharded 1024 prompt cache}"
: "${PROMPTS:?Set PROMPTS to its train.jsonl}"
: "${OUTPUT:?Set OUTPUT to a new experiment directory}"
init_args=()
if [[ -n "${PDD_INIT:-}" ]]; then
  init_args=(--pdd-init-checkpoint "$PDD_INIT")
fi
PYTHON=${PYTHON:-python}
exec "$PYTHON" train_sft.py \
  --loss-type pdd --train-only --train-backend fsdp \
  --hf-checkpoint "$MODEL" "${init_args[@]}" \
  --diffusion-model-family qwen_image21 --diffusion-task t2i \
  --pdd-train-cache "$CACHE" --prompt-data "$PROMPTS" --input-key input \
  --actor-num-gpus-per-node 2 --num-gpus-per-node 2 \
  --rollout-batch-size 8 --n-samples-per-prompt 1 --num-steps-per-rollout 8 \
  --num-rollout 2 --micro-batch-size 1 --gradient-checkpointing \
  --fsdp-master-dtype fp32 --fsdp-reduce-dtype fp32 --diffusion-forward-dtype bf16 \
  --diffusion-height 1024 --diffusion-width 1024 \
  --diffusion-guidance-scale 1 --diffusion-true-cfg-scale 1 \
  --lr 5e-7 --weight-decay 0 --adam-beta2 0.999 --seed 42 \
  --save "$OUTPUT/ckpt" --save-interval 1 --skip-eval-before-train "$@"
