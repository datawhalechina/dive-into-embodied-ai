#!/bin/bash
# Phase-B v5 1cm-critical 近墙学校 (user 2026-10-16 10:46: 起点太远 0.85-1.2m, 全速助跑
# 撞墙挂住)。出生拉近 0.40-0.65m, 其余同 v1 (墙高 0.5cm / stage-4 v3.8 /
# heading_hold / takeoff 60° 门)。warm 接 Phase-B v1 (model_104485)。
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline
export DIVEWALL_STAGE=4
export DIVEWALL_WALL_SINK=0.110
export DIVEWALL_WALL_DIST=0.40,0.65
export DIVEWALL_POSTDIVE_PROB=0.0
export DIVEWALL_PHASE_B1=0
export DIVEWALL_PHASE_B3=1

LOAD_RUN="${LOAD_RUN:-2026-10-16_10-47-44_microduck_roulade_divewall_run15}"  # 可被课程链主脚本覆盖
LOAD_CKPT="${LOAD_CKPT:-model_110484.pt}"  # 可被课程链主脚本覆盖
MAX_ITER="${MAX_ITER:-6000}"

LOG=logs/train_phaseB5_wall.log
for ENVS in ${ENVS_LIST:-4096 2048}; do  # perf: 2026-10-27 起 4096 档 (+31%), OOM 回落 2048
  echo "=== [Phase-B v5 1cm-critical low-wall pounce] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.algorithm.entropy-coef 0.004 \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [Phase-B v5 1cm-critical low-wall pounce rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 1024 (OOM guard)"
done
echo "=== Phase-B v5 1cm-critical low-wall pounce 结束 @ $(date) ==="

# 自动出 GIF — 广角机位, 墙全程可见
RUN_DIR=$(ls -dt logs/rsl_rl/microduck_roulade_dive/2026-*_run15 | head -1)
CKPT=$(ls -t "$RUN_DIR"/model_*.pt | head -1)
echo "=== [gif Phase-B v5 1cm-critical] ckpt=$CKPT @ $(date) ==="
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python scripts/video_rollout.py \
  Mjlab-RouladeDiveWall-Flat-MicroDuck \
  --checkpoint-file "$CKPT" \
  --camera trunk_base --elevation 18 --azimuth 100 --distance 5.0 \
  --video-width 1280 --video-height 720 --video-length 900 --num-envs 1 \
  --speed-overlay \
  > logs/gif_phaseB5_wall_final.log 2>&1
MP4=$(ls -t "$RUN_DIR"/videos/play/*.mp4 2>/dev/null | head -1)
[ -n "$MP4" ] && ffmpeg -y -i "$MP4" -vf "fps=12,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" logs/gifs/phaseB5_wall_final.gif > /dev/null 2>&1 && echo "=== [gif done] logs/gifs/phaseB5_wall_final.gif ==="
echo "=== ALL DONE @ $(date) ==="
