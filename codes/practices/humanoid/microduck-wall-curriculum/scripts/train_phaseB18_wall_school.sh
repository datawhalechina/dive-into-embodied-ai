#!/bin/bash
# Phase-B v18 (2026-10-21, 静态物理扫描后的双杠杆: midroll spawn + 过顶续速).
# B17 证伪 gate 假设后, 静态扫描 (static_roll_sweep.csv) 证明滚通物理可行:
# 门速度≈3 rad/s, 210° 趴地是最难窗口, 165-270° 后半圈可滚通。B18 = B14
# 全配置 + ① DIVEWALL_B18_MIDROLL_FRAC=0.4 的 env 直接 spawn 在后半圈
# (165-270°, 2-5 rad/s 前滚初速, 复用 postdive_spawn); ② 新 reward
# roulade_overhead_drive (w 1.5): 180-240° 窗口接地保持前滚角速度。
# warm 接 B14 最好 ckpt model_188473。
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

LOAD_RUN="${LOAD_RUN:-2026-10-19_07-56-52_microduck_roulade_divewall_run15}"  # 可被课程链主脚本覆盖
LOAD_CKPT="${LOAD_CKPT:-model_188473.pt}"  # 可被课程链主脚本覆盖
MAX_ITER="${MAX_ITER:-8000}"

LOG=logs/train_phaseB18_wall.log
for ENVS in ${ENVS_LIST:-4096 2048}; do  # perf: 2026-10-27 起 4096 档 (+31%), OOM 回落 2048
  echo "=== [Phase-B v18 midroll+overhead] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --env.sim.njmax 160 \
    --agent.max-iterations "$MAX_ITER" \
    --agent.algorithm.entropy-coef 0.004 \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [Phase-B v18 midroll+overhead rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 1024 (OOM guard)"
done
echo "=== Phase-B v18 训练结束 @ $(date) ==="

# 自动出 GIF — 低机位贴墙视角
RUN_DIR=$(ls -dt logs/rsl_rl/microduck_roulade_dive/2026-*_run15 | head -1)
CKPT=$(ls -t "$RUN_DIR"/model_*.pt | head -1)
echo "=== [gif Phase-B v18] ckpt=$CKPT @ $(date) ==="
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python scripts/video_rollout.py \
  Mjlab-RouladeDiveWall-Flat-MicroDuck \
  --checkpoint-file "$CKPT" \
  --camera trunk_base --elevation 3 --azimuth 95 --distance 2.0 \
  --video-width 1280 --video-height 720 --video-length 900 --num-envs 1 \
  --speed-overlay \
  > logs/gif_phaseB18_wall_final.log 2>&1
MP4=$(ls -t "$RUN_DIR"/videos/play/*.mp4 2>/dev/null | head -1)
[ -n "$MP4" ] && ffmpeg -y -i "$MP4" -vf "fps=12,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" logs/gifs/phaseB18_wall_final.gif > /dev/null 2>&1 && echo "=== [gif done] logs/gifs/phaseB18_wall_final.gif ==="
echo "=== ALL DONE @ $(date) ==="

# 课程链: 训练完自动出 GIF (供 Obsidian/blog 留档)
bash scripts/make_stage_gif.sh "${GIF_NAME:-lean_b18_midroll}"
