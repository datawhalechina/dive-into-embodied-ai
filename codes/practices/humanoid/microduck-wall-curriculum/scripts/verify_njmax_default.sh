# Profiling run: B24 config, scratch, 30 iters — 用于拆解 sim/obs-reward/PPO/IO 时间占比
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline
export DIVEWALL_STAGE=4
export DIVEWALL_WALL_SINK=0.070
export DIVEWALL_WALL_DIST=0.40,0.65
export DIVEWALL_POSTDIVE_PROB=0.0
export DIVEWALL_PHASE_B1=0
export DIVEWALL_PHASE_B3=1
export DIVEWALL_PHASE_B17=0
export DIVEWALL_PHASE_B18=1
export DIVEWALL_B18_MIDROLL_FRAC=0.22
export DIVEWALL_PHASE_B21=1

LOG=logs/verify_njmax_default_5iter_2048_30iter.log
echo "=== [profile] envs=2048 iters=30 scratch START @ $(date +%T) ==="
.venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs 2048 \
    --agent.max-iterations 5 \
    --agent.algorithm.entropy-coef 0.004 \
    --agent.resume False
RC=$?
echo "=== [profile rc=$RC] END @ $(date +%T) ==="
