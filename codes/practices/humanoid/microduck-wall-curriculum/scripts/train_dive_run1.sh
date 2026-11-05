#!/bin/bash
# Dive-roll (前扑滚翻) run 1: warm-start from the roulade kip-up policy
# (model_9999, 100% roll+rise) and learn the dive entry: élan spawns carry
# 0.1–0.3 m/s forward, roulade_dive_launch pays the airborne launch phase.
# Obs/action spaces identical to the base roulade task, so the checkpoint
# loads cleanly; the new dive_launch term is the only reward delta.
#
# 用法: nohup bash scripts/train_dive_run1.sh > logs/train_dive_run1.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline

LOAD_RUN=warmstart_run6_model9999   # copy of run6/model_9999.pt under the dive experiment dir
LOAD_CKPT=model_9999.pt
MAX_ITER=6000   # 实测语义: 目标迭代数 = 载入迭代(9999) + 本值 => 15999, 即新训 6000 轮

for ENVS in 4096 2048; do
  echo "=== [dive-run1] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDive-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [dive-run1 rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 2048 (OOM guard)"
done
echo "=== dive-run1 结束 @ $(date) ==="
