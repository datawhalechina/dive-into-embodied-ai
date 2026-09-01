#!/usr/bin/env bash
# Lab 6 可选进阶：MJX/brax 训练环境一键安装（自动识别加速后端）。
#
# 平台无关的核心依赖（jax / mujoco-mjx / brax 的互容版本）固化在
# requirements-mjx.txt；本脚本只负责在其之上挑对**加速器插件**，并把环境装进
# 一个独立 venv（默认 exercises/.venv-mjx），不与主线 .venv 混装。
#
# 用法（在仓库任意位置执行都可以，路径按脚本自身位置解析）：
#
#   bash lab_6_rl_pupper/setup_mjx_env.sh                  # 自动识别
#   bash lab_6_rl_pupper/setup_mjx_env.sh --backend rocm   # 强制指定
#   bash lab_6_rl_pupper/setup_mjx_env.sh --venv .venv-mjx-rocm --python 3.12
#
# 后端取值：auto | rocm | cuda12 | cuda13 | cpu
#
# 装完直接训练：
#   .venv-mjx/bin/python lab_6_rl_pupper/train_brax_ppo.py --output portfolio/pupper_mjx

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXERCISES_DIR="$(dirname "$SCRIPT_DIR")"
REQ_FILE="$SCRIPT_DIR/requirements-mjx.txt"

BACKEND="auto"
VENV_PATH="$EXERCISES_DIR/.venv-mjx"
PYTHON_VERSION="3.12"   # brax 要求 >=3.11

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="${2:?--backend 需要一个值}"; shift 2 ;;
    --venv)    VENV_PATH="${2:?--venv 需要一个路径}"; shift 2 ;;
    --python)  PYTHON_VERSION="${2:?--python 需要一个版本}"; shift 2 ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "未知参数: $1（可用: --backend / --venv / --python / --help）" >&2; exit 2 ;;
  esac
done

case "$BACKEND" in
  auto|rocm|cuda12|cuda13|cpu) ;;
  *) echo "--backend 只能是 auto | rocm | cuda12 | cuda13 | cpu，收到: $BACKEND" >&2; exit 2 ;;
esac

command -v uv >/dev/null 2>&1 || {
  echo "找不到 uv，请先安装：https://docs.astral.sh/uv/" >&2; exit 1
}
[[ -f "$REQ_FILE" ]] || { echo "找不到 $REQ_FILE" >&2; exit 1; }

# 插件版本必须和核心 jax 完全一致，所以直接从 requirements-mjx.txt 里读，避免两处漂移。
JAX_VERSION="$(sed -n 's/^jax==\([0-9][^ ]*\).*/\1/p' "$REQ_FILE" | head -n1)"
[[ -n "$JAX_VERSION" ]] || { echo "无法从 $REQ_FILE 解析 jax 版本" >&2; exit 1; }

# ---------------------------------------------------------------- 后端识别

detect_backend() {
  # AMD：ROCm 的 KFD 设备节点存在，且系统装了 ROCm 运行库
  #（jax-rocm7-plugin 只有 ~24 MB，不自带运行库，依赖系统 /opt/rocm，
  #  这点和 jax[cuda12] 用 pip 拉齐整套 nvidia-* 运行库不同）
  if [[ -e /dev/kfd ]] && { [[ -d /opt/rocm ]] || command -v rocminfo >/dev/null 2>&1; }; then
    echo "rocm"; return
  fi

  # NVIDIA：按驱动支持的 CUDA 大版本选 cuda12 / cuda13
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    local cuda_major
    cuda_major="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]\+\).*/\1/p' | head -n1)"
    if [[ "${cuda_major:-0}" -ge 13 ]]; then echo "cuda13"; else echo "cuda12"; fi
    return
  fi

  echo "cpu"
}

describe_backend() {
  case "$1" in
    rocm)
      local gfx
      gfx="$(rocminfo 2>/dev/null | sed -n 's/.*Name: *\(gfx[0-9a-z]*\).*/\1/p' | head -n1)"
      echo "ROCm${gfx:+ / $gfx}" ;;
    cuda12) echo "CUDA 12 (NVIDIA)" ;;
    cuda13) echo "CUDA 13 (NVIDIA)" ;;
    cpu)    echo "纯 CPU（无加速器插件）" ;;
  esac
}

if [[ "$BACKEND" == "auto" ]]; then
  BACKEND="$(detect_backend)"
  echo "  detected: $(describe_backend "$BACKEND")  -> $BACKEND"
else
  echo "  forced:   $BACKEND ($(describe_backend "$BACKEND"))"
fi

# 每个后端要额外装的加速器插件；版本一律跟随 JAX_VERSION。
PLUGIN_PKGS=()
case "$BACKEND" in
  rocm)
    # 注意：jax 没有 `rocm` extra（写 jax[rocm] 只会被静默忽略并告警），
    # 只能显式装 pjrt + plugin 两个包。这点和 CUDA 的 jax[cuda12] 不对称。
    PLUGIN_PKGS=("jax-rocm7-pjrt==$JAX_VERSION"
                 "jax-rocm7-plugin==$JAX_VERSION")
    if [[ ! -d /opt/rocm ]]; then
      echo "  ! 警告：没找到 /opt/rocm。jax 的 ROCm 插件依赖系统 ROCm 运行库（7.x），" >&2
      echo "    缺失时 jax 会静默回退到 CPU。" >&2
    fi
    ;;
  cuda12)
    PLUGIN_PKGS=("jax[cuda12]==$JAX_VERSION"
                 "jax-cuda12-pjrt==$JAX_VERSION"
                 "jax-cuda12-plugin==$JAX_VERSION") ;;
  cuda13)
    PLUGIN_PKGS=("jax[cuda13]==$JAX_VERSION"
                 "jax-cuda13-pjrt==$JAX_VERSION"
                 "jax-cuda13-plugin==$JAX_VERSION") ;;
  cpu)
    PLUGIN_PKGS=() ;;
esac

# ---------------------------------------------------------------- 安装

echo "  creating $VENV_PATH (py$PYTHON_VERSION)"
uv venv "$VENV_PATH" --python "$PYTHON_VERSION"

VENV_PY="$VENV_PATH/bin/python"
uv pip install --python "$VENV_PY" -r "$REQ_FILE" ${PLUGIN_PKGS[@]+"${PLUGIN_PKGS[@]}"}

# ---------------------------------------------------------------- 自检

echo "  self-check:"
# mjx 探测不到可选的 warp 后端时会打一行无害告警，这里滤掉以免喧宾夺主。
BACKEND="$BACKEND" "$VENV_PY" - <<'PY'
import os
import jax

expected = os.environ["BACKEND"]
actual = jax.default_backend()
devices = jax.devices()
print(f"    jax.default_backend() = {actual}  [{', '.join(map(str, devices))}]")

if expected == "cpu":
    raise SystemExit(0)
if actual != "gpu":
    raise SystemExit(
        f"    ! 期望 {expected} 加速，实际回退到 {actual}。\n"
        "      常见原因：系统驱动/运行库与插件的大版本不匹配（ROCm 需系统 7.x；\n"
        "      CUDA 13 插件需驱动 >=580）。可先用 --backend cpu 跑通流程。"
    )
PY

cat <<EOF

  完成。接下来（在 $EXERCISES_DIR 下执行）：

    $VENV_PATH/bin/python lab_6_rl_pupper/train_brax_ppo.py --output portfolio/pupper_mjx

EOF
