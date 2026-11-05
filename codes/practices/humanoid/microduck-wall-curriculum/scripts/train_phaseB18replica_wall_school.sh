#!/bin/bash
# B18-replica (2026-10-29 大人拍板): 严格复刻原版 B18 配方, 排除复原链引入的
# 4 个变量 —— 原版 (2026-10-21) 实际跑的是: envs=2048, njmax=基类默认1500,
# 无 B18B/B18C 补丁, takeoff w6/10-45°, rise gate 150°, airborne 180°。
# warm = 复原链 B14 product model_176474 (reward 27.9 ≈ 原版 B14 27.4)。
# 唯一已知谱系差: 原版 B14 前有 B12b 12k 加练轮 (本链跳过)。
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline
export DIVEWALL_STAGE=4
export DIVEWALL_WALL_SINK=0.100
export DIVEWALL_WALL_DIST=0.40,0.65
export DIVEWALL_POSTDIVE_PROB=0.0
export DIVEWALL_PHASE_B1=0
export DIVEWALL_PHASE_B3=1
export DIVEWALL_PHASE_B17=0
export DIVEWALL_PHASE_B18=1
export DIVEWALL_B18_MIDROLL_FRAC=0.4
# 严格复刻: 不设 B18B/B18C —— 奖励回到原版形态

LOAD_RUN="${LOAD_RUN:-2026-10-28_13-23-09_microduck_roulade_divewall_run15}"  # 复原 B14
LOAD_CKPT="${LOAD_CKPT:-model_176474.pt}"
MAX_ITER="${MAX_ITER:-8000}"

for ENVS in 2048; do  # 原版就是 2048, 不用 4096 提速档
  echo "=== [Phase-B v18 REPLICA original-recipe] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --env.sim.njmax 1500 \
    --agent.max-iterations "$MAX_ITER" \
    --agent.algorithm.entropy-coef 0.004 \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [Phase-B v18 REPLICA rc=$RC] envs=$ENVS @ $(date) ==="
done
echo "=== Phase-B v18 REPLICA 训练结束 @ $(date) ==="

# 自动出 GIF
RUN_DIR=$(ls -dt logs/rsl_rl/microduck_roulade_dive/2026-*_run15 | head -1)
CKPT=$(ls -t "$RUN_DIR"/model_*.pt | head -1)
echo "=== [gif Phase-B v18 replica] ckpt=$CKPT @ $(date) ==="
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python scripts/video_rollout.py \
  Mjlab-RouladeDiveWall-Flat-MicroDuck \
  --checkpoint-file "$CKPT" \
  --camera trunk_base --elevation 3 --azimuth 95 --distance 2.0 \
  --video-width 1280 --video-height 720 --video-length 900 --num-envs 1 \
  --speed-overlay \
  > logs/gif_phaseB18replica_wall_final.log 2>&1
MP4=$(ls -t "$RUN_DIR"/videos/play/*.mp4 2>/dev/null | head -1)
[ -n "$MP4" ] && ffmpeg -y -i "$MP4" -vf "fps=12,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" logs/gifs/phaseB18replica_wall_final.gif > /dev/null 2>&1 && echo "=== [gif done] logs/gifs/phaseB18replica_wall_final.gif ==="
echo "=== ALL DONE @ $(date) ==="
