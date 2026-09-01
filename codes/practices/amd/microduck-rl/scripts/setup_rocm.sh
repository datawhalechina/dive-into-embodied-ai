#!/usr/bin/env bash
set -euo pipefail

# Build the experimental AMD/ROCm stack in an isolated environment. Run from
# any directory; pass a gfx target as the first argument to override automatic
# detection (for example, gfx1201).

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROCM_ROOT="${ROCM_PATH:-/opt/rocm}"
ENV_DIR="$PROJECT_DIR/.venv-rocm"
SOURCE_DIR="$PROJECT_DIR/.rocm-src"
WARP_DIR="$SOURCE_DIR/warp"
MJWARP_DIR="$SOURCE_DIR/mujoco_warp"

WARP_COMMIT="6530951390fa905321011dedd074572e4cdce00b"
MJWARP_COMMIT="9229bb9d1a698c9464df862a915b46899720338c"
BAM_COMMIT="62bd8ce12154340be97e06f7f41a0ca8f116d967"

TORCH_WHEEL="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl"
VISION_WHEEL="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchvision-0.24.0%2Brocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl"
TRITON_WHEEL="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl"

for command_name in uv git rocminfo; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "error: required command not found: $command_name" >&2
        exit 1
    fi
done
if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "error: the pinned AMD wheels require x86_64 Linux" >&2
    exit 1
fi
if [[ ! -x "$ROCM_ROOT/bin/hipcc" ]]; then
    echo "error: hipcc not found under ROCM_PATH=$ROCM_ROOT" >&2
    exit 1
fi

GPU_ARCH="${1:-}"
if [[ -z "$GPU_ARCH" ]]; then
    GPU_ARCH="$(rocminfo | awk '/Name:[[:space:]]+gfx[0-9]+/ {print $2; exit}')"
fi
if [[ ! "$GPU_ARCH" =~ ^gfx[0-9]+$ ]]; then
    echo "error: could not detect an AMD gfx target; pass one explicitly" >&2
    echo "example: scripts/setup_rocm.sh gfx1201" >&2
    exit 1
fi

prepare_repo() {
    local url="$1"
    local directory="$2"
    local commit="$3"

    if [[ ! -d "$directory/.git" ]]; then
        git clone --filter=blob:none "$url" "$directory"
    fi
    if ! git -C "$directory" diff --quiet || \
       ! git -C "$directory" diff --cached --quiet; then
        echo "error: managed source tree has local changes: $directory" >&2
        exit 1
    fi
    git -C "$directory" fetch --depth 1 origin "$commit"
    git -C "$directory" switch --detach "$commit"
}

mkdir -p "$SOURCE_DIR"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    uv venv "$ENV_DIR" --python 3.12
fi
PYTHON="$ENV_DIR/bin/python"

echo "Installing AMD PyTorch 2.9.1 / ROCm 7.2.1 wheels..."
uv pip install --python "$PYTHON" \
    "$TORCH_WHEEL" "$VISION_WHEEL" "$TRITON_WHEEL" "numpy==1.26.4"

prepare_repo "https://github.com/ROCm/warp.git" "$WARP_DIR" "$WARP_COMMIT"
echo "Building ROCm Warp for $GPU_ARCH..."
(
    cd "$WARP_DIR"
    PATH="$ROCM_ROOT/bin:$PATH" ROCM_PATH="$ROCM_ROOT" \
        "$PYTHON" build_lib.py \
        --rocm-path "$ROCM_ROOT" \
        --hip-arch "$GPU_ARCH" \
        --hipcc-options=-O0
)
uv pip install --python "$PYTHON" --no-deps -e "$WARP_DIR"

prepare_repo \
    "https://github.com/ROCm/mujoco_warp.git" \
    "$MJWARP_DIR" \
    "$MJWARP_COMMIT"
uv pip install --python "$PYTHON" --no-deps -e "$MJWARP_DIR"
uv pip install --python "$PYTHON" \
    absl-py "etils[epath]" "mujoco==3.10.0" "numpy==1.26.4"

# Keep the AMD wheels explicit in this transaction. Otherwise the resolver can
# replace the working ROCm build with the newest CUDA-oriented PyPI torch.
uv pip install --python "$PYTHON" \
    "mjlab==1.3.0" \
    "$TORCH_WHEEL" "$VISION_WHEEL" "$TRITON_WHEEL" "numpy==1.26.4"

uv pip install --python "$PYTHON" --no-deps \
    -e "$PROJECT_DIR" \
    "better-actuator-models @ git+https://github.com/Rhoban/bam.git@$BAM_COMMIT"
uv pip install --python "$PYTHON" \
    "onnxruntime>=1.24.4" \
    "rustypot>=1.4.2" \
    "huggingface-hub>=0.27.0" \
    "matplotlib>=3.10.9" \
    "imageio>=2.37" \
    "imageio-ffmpeg>=0.6" \
    "scipy>=1.16" \
    "protobuf>=4,<7" \
    "onnx>=1.20.1" \
    "numpy==1.26.4" \
    pytest \
    ruff \
    colorama

echo "Verifying the ROCm training stack..."
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" "$PYTHON" - <<'PY'
import torch
import warp as wp

import mjlab_microduck  # noqa: F401 -- installs the mjlab/ROCm compatibility hooks

assert torch.cuda.is_available(), "PyTorch cannot see an AMD GPU through ROCm"
assert torch.version.hip is not None, "installed torch is not a ROCm build"
wp.init()
device = wp.get_device("cuda:0")
assert device.is_hip, f"Warp device is not HIP: {device}"
print(f"torch={torch.__version__} hip={torch.version.hip}")
print(f"torch_device={torch.cuda.get_device_name(0)}")
print(f"warp={wp.__version__} warp_device={device} arch={device.arch_str}")
PY

# mjlab's ``list-envs`` command currently returns the number of registered
# environments as its process status (45 here), even though listing succeeds.
# Capture its output without treating that non-zero status as an install error.
ENV_LIST="$(HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
    "$ENV_DIR/bin/list-envs" || true)"
grep "Mjlab-Velocity-Flat-MicroDuck" <<<"$ENV_LIST"

HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
    "$PYTHON" "$PROJECT_DIR/scripts/check_rocm_contacts.py" --device cuda:0

echo
echo "ROCm environment is ready: $ENV_DIR"
echo "Run training with:"
echo "  HIP_VISIBLE_DEVICES=0 $ENV_DIR/bin/train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max-iterations 5"
