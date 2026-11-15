# Curriculum v2 stage 22 前扑跳: hop credit + pitch ramp 15-60deg; vx_full raised 0.8->1.2 (run faster, jump farther).
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline
export 
export DIVEWALL_STAGE=22

LOAD_RUN=warmstart_curric21_model168748
LOAD_CKPT=model_168748.pt
MAX_ITER=6000

for ENVS in 2048 1024; do
  echo "=== [divewall-run15-stage21 DIVE-POUNCE] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [divewall-run15-stage21 rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 1024 (OOM guard)"
done
echo "=== divewall-run15-stage21 结束 @ $(date) ==="
