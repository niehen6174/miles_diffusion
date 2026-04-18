#!/usr/bin/env bash
# One-click installer for miles-diffusion on a clean Linux GPU box.
# Idempotent: re-running skips steps that are already done.
#
# Overrides (env vars):
#   ENV_NAME    conda env name (default: miles-diffusion)
#   PY_VER      python version (default: 3.11)
#   SGLANG_DIR  where to clone sglang (default: ../sglang)
#   SGLANG_PR   sglang PR number (default: 20464)
#   CUDA_VER    torch cuda tag (default: 12.4 -> cu124)

set -euo pipefail

ENV_NAME="${ENV_NAME:-miles-diffusion}"
PY_VER="${PY_VER:-3.11}"
CUDA_VER="${CUDA_VER:-12.4}"
SGLANG_PR="${SGLANG_PR:-20464}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SGLANG_DIR="${SGLANG_DIR:-$(dirname "$REPO_DIR")/sglang}"

log()  { printf "\033[1;34m[install]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[warn]\033[0m %s\n" "$*"; }
die()  { printf "\033[1;31m[error]\033[0m %s\n" "$*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || die "missing required tool: $1"; }

# ------------------------------------------------------------------ preflight
log "repo:   $REPO_DIR"
log "env:    $ENV_NAME (python $PY_VER)"
log "sglang: $SGLANG_DIR (PR #$SGLANG_PR)"
log "cuda:   $CUDA_VER"

need git
if command -v mamba >/dev/null 2>&1; then
  CONDA_BIN=mamba
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN=conda
else
  die "conda/mamba not found. Install miniforge: https://github.com/conda-forge/miniforge"
fi
log "using: $CONDA_BIN"

# ---------------------------------------------------------------- apt deps
if command -v apt-get >/dev/null 2>&1; then
  log "apt: libglib2.0-0 libgl1"
  SUDO=""
  [[ $EUID -ne 0 ]] && command -v sudo >/dev/null 2>&1 && SUDO="sudo"
  $SUDO apt-get update -qq || warn "apt-get update failed; continuing"
  $SUDO apt-get install -y libglib2.0-0 libgl1 || warn "apt install failed; continuing (check libGL/libglib presence manually)"
else
  warn "apt-get not available; skip system lib install"
fi

# ---------------------------------------------------------------- conda env
source "$($CONDA_BIN info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  log "conda env '$ENV_NAME' exists; reusing"
else
  log "creating conda env '$ENV_NAME'"
  $CONDA_BIN create -y -n "$ENV_NAME" "python=$PY_VER"
fi
conda activate "$ENV_NAME"
log "python: $(python --version)"

python -m pip install --upgrade pip wheel setuptools

# ---------------------------------------------------------------- pytorch
CU_TAG="cu$(echo "$CUDA_VER" | tr -d .)"
if python -c "import torch" 2>/dev/null; then
  log "torch: $(python -c 'import torch; print(torch.__version__, torch.version.cuda)') (already installed)"
else
  log "installing torch ($CU_TAG)"
  pip install torch --index-url "https://download.pytorch.org/whl/$CU_TAG"
fi

# ---------------------------------------------------------------- sglang-diffusion
if [[ ! -d "$SGLANG_DIR" ]]; then
  log "cloning sglang -> $SGLANG_DIR"
  git clone https://github.com/sgl-project/sglang "$SGLANG_DIR"
fi

pushd "$SGLANG_DIR" >/dev/null
if ! git rev-parse --verify --quiet "pr-$SGLANG_PR" >/dev/null; then
  log "fetching sglang PR #$SGLANG_PR"
  git fetch origin "pull/$SGLANG_PR/head:pr-$SGLANG_PR"
fi
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$CURRENT_BRANCH" != "pr-$SGLANG_PR" ]]; then
  log "checkout sglang pr-$SGLANG_PR"
  git checkout "pr-$SGLANG_PR"
fi

if python -c "import sglang.multimodal_gen" 2>/dev/null; then
  log "sglang.multimodal_gen already importable; skip pip install"
else
  log "pip install sglang (editable, all extras)"
  pip install -e "python[all]"
fi
popd >/dev/null

# ---------------------------------------------------------------- miles
cd "$REPO_DIR"
log "pip install -r requirements.txt"
pip install -r requirements.txt
log "pip install -e . (miles)"
pip install -e . --no-deps

# ---------------------------------------------------------------- flow_grpo deps
if [[ -f "$REPO_DIR/flow_grpo/setup.sh" ]]; then
  log "installing flow_grpo OCR deps (paddleocr, peft, diffusers, ...)"
  pushd "$REPO_DIR/flow_grpo" >/dev/null
  # Skip the `pip install -e .` line inside setup.sh — flow_grpo is a sibling
  # tree we reference, not a package to install into miles' env. The rest of
  # the file is pinned --no-deps pip installs plus apt calls we already did.
  grep -v '^pip install -e . --no-deps$' setup.sh | \
    grep -v '^apt-get install ' | \
    bash
  popd >/dev/null
else
  warn "flow_grpo/setup.sh not found; skipping OCR reward deps"
fi

# ---------------------------------------------------------------- optional
if ! python -c "import torch_memory_saver" 2>/dev/null; then
  log "installing torch_memory_saver (optional)"
  pip install torch_memory_saver || warn "torch_memory_saver install failed; continuing without it"
fi

# ---------------------------------------------------------------- smoke test
log "smoke test: nvidia-smi"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || warn "nvidia-smi returned non-zero"
else
  warn "nvidia-smi not found; GPU presence unknown"
fi

log "smoke test: python import train_diffusion"
cd "$REPO_DIR"
python -c "
import train_diffusion  # noqa
from miles.utils.arguments import parse_args  # noqa
from miles.backends.fsdp_utils import FSDPTrainRayActor  # noqa
import sglang.multimodal_gen  # noqa
print('miles-diffusion import OK')
"

log ""
log "=========================================="
log "  install done."
log "  next:"
log "    conda activate $ENV_NAME"
log "    export WANDB_API_KEY=...  # optional"
log "    bash scripts/run-diffusion-grpo-ocr.sh"
log "=========================================="
