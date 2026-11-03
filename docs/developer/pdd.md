# PDD training in Miles

This implementation adds Parallel Decoding Distillation (PDD) for Qwen-Image 2.1
to the existing `train_sft.py` lifecycle. It reuses the Ray actors, FSDP2 loader,
prepare/loss hooks, optimizer, dataset cursor and distributed checkpoints.
It does not add a separate training loop or modify SGLang inference.

## Supported configuration

- Qwen-Image 2.1 text-to-image, 1024 × 1024, CFG = 1.
- 32 output heads, four intervals per block: eight student forwards per image.
- Full student weights, FP32 parameters/reduction and BF16 autocast.
- Frozen BF16 teacher loaded from the same base model.
- FSDP with SP = 1, microbatch = 1, eight optimizer updates per rollout.
- No LoRA, EMA, RL log probabilities or reference-policy loss.

The student output projection is replicated before FSDP wrapping and optimizer
creation. Each block randomly selects one interval for midpoint-teacher velocity
supervision. Teacher inputs and the state carried to the next block are detached;
the carried state uses the prediction made before that block's optimizer update.
Data-parallel splitting happens before trajectory expansion, keeping each lane
on the same rank for all eight updates.

The PDD loader preserves the Diffusers computation and precision used to train
the student. The existing post-load transformation for SGLang alignment is not
applied in this mode. Other loss modes retain their existing behavior.

## Prepare data

Use the repository's supported CUDA environment with Qwen-Image 2.1 support in
Diffusers. These examples run from the repository root with `PYTHONPATH=.`.
The base model must be downloaded locally. Use a versioned model directory:
the cache records its directory name and must be reused with the same model.

The JSONL contains unique IDs and unique prompt text; all rows are training data:

```json
{"id":"train-0001","split":"train","input":"A red ceramic teapot on a wooden table."}
```

```bash
export PYTHONPATH=.
export MODEL=/path/to/qwen-image-2.1-snapshot
export PROMPTS=/path/to/train.jsonl
export CACHE=/scratch/pdd/cache
export OUTPUT=/scratch/pdd/run
python scripts/prepare_pdd_cache.py \
  --model "$MODEL" --prompts "$PROMPTS" --output "$CACHE"
```

Encoding uses one GPU and native single-prompt BF16 text encoding. It stores
bounded-size shards and publishes the manifest only on completion. The output
must be a new directory; this minimal encoder does not resume partial caches.
Training validates the input-file hash, row IDs, shard hashes and tensor shapes.
Existing compatible format-v1 training caches can be reused.

## Train and resume

Use the existing training entry point directly:

```bash
python train_sft.py \
  --loss-type pdd --train-only --train-backend fsdp \
  --hf-checkpoint "$MODEL" \
  --diffusion-model-family qwen_image21 --diffusion-task t2i \
  --pdd-train-cache "$CACHE" --prompt-data "$PROMPTS" --input-key input \
  --actor-num-gpus-per-node 2 --num-gpus-per-node 2 \
  --rollout-batch-size 8 --n-samples-per-prompt 1 --num-steps-per-rollout 8 \
  --num-rollout 128 --micro-batch-size 1 --gradient-checkpointing \
  --fsdp-master-dtype fp32 --fsdp-reduce-dtype fp32 --diffusion-forward-dtype bf16 \
  --diffusion-height 1024 --diffusion-width 1024 \
  --diffusion-guidance-scale 1 --diffusion-true-cfg-scale 1 \
  --lr 5e-7 --weight-decay 0 --adam-beta2 0.999 --seed 42 \
  --save "$OUTPUT/ckpt" --save-interval 1 --skip-eval-before-train
```

This recipe uses two GPUs and eight trajectories per rollout. By default, all
student heads start from the base model's output projection. An existing
compatible PDD weights file can be passed with `--pdd-init-checkpoint /path/to/export.pt`
for weights-only initialization with a fresh optimizer. Export tooling is outside
this training change.

For single-GPU CPU offload, change both GPU-count arguments to `1` and add
`--fsdp-cpu-offload`. To resume, rerun the command with `--load "$OUTPUT/ckpt"`
and increase `--num-rollout` to the desired total (for example, `256`). Keep the
model, cache and training configuration unchanged.

Checkpoints contain model, optimizer, scheduler, per-rank Torch/CUDA RNG, protocol
signature and dataset cursor. Saves occur only at complete trajectory boundaries.
Resume rejects missing state or a changed protocol/data/topology. An interrupted
trajectory is replayed from the preceding complete checkpoint.

## Verification and limits

```bash
python -m pytest tests/fast/distillation
python -m pytest tests/fast-gpu/backends/fsdp_utils/test_pdd.py
```

Tests cover detached teacher supervision, selected-head gradients, signed
nonuniform grids, trajectory carry, prompt-cache integrity and unsupported
configurations. The GPU suite compares tiny-model FSDP updates with an unsharded
reference and verifies checkpoint restoration and the next update on one and two
ranks. The two-rank case skips when fewer than two GPUs are available. A separate
case checks CPU-offloaded accumulated gradients and AdamW updates against eager
execution. All GPU cases skip when CUDA is unavailable; pytest manages temporary
checkpoint directories.

These are correctness checks, not image-quality claims. This change does not
include SGLang serving, online weight synchronization, automatic image metrics,
historical datasets/reports or experiment orchestration.
