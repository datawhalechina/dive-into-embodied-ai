#!/bin/bash
# Phase-A v3 跳跃学校 (user 2026-10-14 09:38 拍板, v1 GIF verdict: 跑步偏离
# 墙、没有跳):
#   ① 跳的收益压过跑: hop_forward 6→15, track 2→1, vx 命令带宽 (0.25, 0.6)
#   ② 航向锚定: track_angular_velocity (ang_vel_z 命令=0) — 治跑偏
#   ③ air_time 高度门控: air_time_height (腾空且躯干 z>0.13 才计分)
#   其余复用 DIVEWALL_STAGE=2 配置 (墙远 4.5-5.5m, flop guard)。
#   warm-start 自 Phase-A v2 收官 model_74491。
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline
export DIVEWALL_STAGE=2
export DIVEWALL_WALL_SINK=0.115
export DIVEWALL_POSTDIVE_PROB=0.0
export DIVEWALL_PHASE_A3=1

LOAD_RUN="${LOAD_RUN:-2026-10-14_09-41-01_microduck_roulade_divewall_run15}"  # 可被课程链主脚本覆盖
LOAD_CKPT="${LOAD_CKPT:-model_74490.pt}"  # 可被课程链主脚本覆盖
MAX_ITER="${MAX_ITER:-6000}"

LOG=logs/train_phaseA3_jump.log
for ENVS in ${ENVS_LIST:-4096 2048}; do  # perf: 2026-10-27 起 4096 档 (+31%), OOM 回落 2048
  echo "=== [Phase-A v3 jump school] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.algorithm.entropy-coef 0.002 \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [Phase-A v3 jump school rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 1024 (OOM guard)"
done
echo "=== Phase-A v3 jump school 结束 @ $(date) ==="

# 自动出 GIF (标准机位)
RUN_DIR=$(ls -dt logs/rsl_rl/microduck_roulade_dive/2026-*_run15 | head -1)
CKPT=$(ls -t "$RUN_DIR"/model_*.pt | head -1)
echo "=== [gif Phase-A v3] ckpt=$CKPT @ $(date) ==="
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python scripts/video_rollout.py \
  Mjlab-RouladeDiveWall-Flat-MicroDuck \
  --checkpoint-file "$CKPT" \
  --camera trunk_base --elevation 8 --azimuth 100 --distance 1.6 \
  --video-width 1280 --video-height 720 --video-length 600 --num-envs 1 \
  > logs/gif_phaseA3_jump_final.log 2>&1
MP4=$(ls -t "$RUN_DIR"/videos/play/*.mp4 2>/dev/null | head -1)
[ -n "$MP4" ] && ffmpeg -y -i "$MP4" -vf "fps=15,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" logs/gifs/phaseA3_jump_final.gif > /dev/null 2>&1 && echo "=== [gif done] logs/gifs/phaseA3_jump_final.gif ==="
echo "=== ALL DONE @ $(date) ==="

# 课程链: 训练完自动出 GIF (供 Obsidian/blog 留档)
bash scripts/make_stage_gif.sh "${GIF_NAME:-lean_a3_jumpschool}"
