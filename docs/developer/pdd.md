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

## Train, resume and export

```bash
bash scripts/run_pdd_miles_qwen21.sh --num-rollout 128
```

The recipe defaults to two GPUs and eight trajectories per rollout. Without
`PDD_INIT`, all student heads start from the base model's output projection.
Set `PDD_INIT=/path/to/export.pt` to initialize from exported PDD weights with a
fresh optimizer. Single-GPU CPU offload can be selected with
`--actor-num-gpus-per-node 1 --num-gpus-per-node 1 --fsdp-cpu-offload`.

Resume with the same model, cache and training configuration:

```bash
bash scripts/run_pdd_miles_qwen21.sh --num-rollout 256 --load "$OUTPUT/ckpt"
```

Checkpoints contain model, optimizer, scheduler, per-rank Torch/CUDA RNG, protocol
signature and dataset cursor. Saves occur only at complete trajectory boundaries.
Resume rejects missing state or a changed protocol/data/topology. An interrupted
trajectory is replayed from the preceding complete checkpoint.

```bash
python scripts/export_pdd_miles_checkpoint.py \
  --checkpoint "$OUTPUT/ckpt/iter_0000128" --output "$OUTPUT/student.pt"
```

Export merges only student weights on CPU and refuses to overwrite an existing
output. Allow host RAM and temporary disk space for the full model. The exported
weights, grid and configuration can be loaded with `QwenImage21PDDStudent` and
sampled with `sample_pdd`; text conditioning and VAE decoding remain the caller's
responsibility. Exported weights are not a full training-resume checkpoint.

## Verification and limits

```bash
python -m pytest tests/fast/distillation
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  scripts/verify_pdd_miles_fsdp.py --output /scratch/pdd/tiny-fsdp-check
python scripts/verify_pdd_cpu_offload.py
```

Tests cover detached teacher supervision, selected-head gradients, signed
nonuniform grids, trajectory carry, prompt-cache integrity and unsupported
configurations. The tiny FSDP verifier compares updates with an unsharded
reference and verifies checkpoint restoration and the next update. The offload
verifier checks accumulated gradients and AdamW updates against eager execution.

These are correctness checks, not image-quality claims. This change does not
include SGLang serving, online weight synchronization, automatic image metrics,
historical datasets/reports or experiment orchestration.
