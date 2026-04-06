# ps -ef | grep train.py | grep -v grep
#WANDB_API_KEY=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 nohup bash /data/zhiheng/miles/scripts/run-diffusion-grpo-ocr.sh > /data/zhiheng/miles/logs/diffusion_grpo_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# nohup bash /data/zhiheng/miles/scripts/run-diffusion-grpo-ocr.sh > /data/zhiheng/miles/logs/diffusion_grpo_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# pkill -f "/data/zhiheng/miles/train.py"
# rollout needs 1 gpu for now, or there's going to be precision issue.
# parameter rollout-num-gpus and --rollout-num-gpus-per-engine  only makes sense in sglang diffusion case.
#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=1,2,3,4
# WandB: enable if WANDB_API_KEY is present.
RUN_NAME="diffusion_grpo_$(date +%Y%m%d_%H%M%S)"
WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_ARGS+=(
    --use-wandb
    --wandb-project miles-diffusion-grpo
    --wandb-group "${RUN_NAME}"
    --wandb-key "${WANDB_API_KEY}"
    --diffusion-log-images 8
    --diffusion-log-image-interval 10
    --disable-wandb-random-suffix
  )
fi
# Prepare OCR prompts into JSONL expected by Miles data loader.
python "${ROOT_DIR}/tools/prepare_ocr_jsonl.py"

# Minimal diffusion GRPO run, aligned with flow_grpo single-node settings.

#hf-checkpoint can be any text generation model from HuggingFace, used to generate initial prompts for diffusion model.
python -u "${ROOT_DIR}/train.py" \
  --train-backend fsdp \
  --diffusion-train \
  --rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout \
  --hf-checkpoint gpt2 \
  --prompt-data "${ROOT_DIR}/data/ocr/train.jsonl" \
  --input-key input \
  --rollout-batch-size 32 \
  --n-samples-per-prompt 16 \
  --num-rollout 100000 \
  --diffusion-train-batch-size 2 \
  --actor-num-gpus-per-node 4 \
  --rollout-num-gpus 4 \
  --rollout-num-gpus-per-engine 1 \
  --num-gpus-per-node 4 \
  --colocate \
  --diffusion-model Qwen/Qwen-Image \
  --diffusion-reward ocr:1.0 \
  --reward-type ocr \
  --reward-key ocr \
  --diffusion-dtype fp32 \
  --diffusion-num-steps 10 \
  --diffusion-num-batches-per-epoch 8 \
  --diffusion-guidance-scale 4.5 \
  --diffusion-rollout-noise-level 0.7 \
  --diffusion-height 512 \
  --diffusion-width 512 \
  --sglang-mem-fraction-static 0.7 \
  --sglang-cuda-graph-max-bs 16 \
  --global-batch-size 128 \
  "${WANDB_ARGS[@]}"
