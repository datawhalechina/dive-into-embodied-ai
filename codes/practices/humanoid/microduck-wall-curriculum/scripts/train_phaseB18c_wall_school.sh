#!/bin/bash
# B18c (2026-10-29): B18b 两连败复盘 — "头顶地+脚蹬地划过墙, 无翻滚"。
# rise/airborne 260° 门已生效, 但过墙三件套 (takeoff w12 + 飞行 jackpot +
# 过墙年金) 不看翻滚, 作弊路径依旧全收。B18c 双补丁:
#   ① 飞行 jackpot 延迟支付 (roll_gate 150→260° + head latch, mdp 侧
#      wall_flight_clear_bonus 新参数) — 划过墙不给钱, 滚完才付;
#   ② roulade_headstand_penalty (w 2.0): 头触地+倒置+frontier<100° 每步罚。
# warm 从 B14 model_176474 (干净谱系, 不接 B18/B18b 作弊惯性)。
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
export DIVEWALL_PHASE_B18B=1
export DIVEWALL_PHASE_B18C=1
export DIVEWALL_B18_MIDROLL_FRAC=0.4

LOAD_RUN="${LOAD_RUN:-2026-10-28_13-23-09_microduck_roulade_divewall_run15}"  # B14
LOAD_CKPT="${LOAD_CKPT:-model_176474.pt}"
MAX_ITER="${MAX_ITER:-8000}"

LOG=logs/restore_B18c.log
for ENVS in ${ENVS_LIST:-4096 2048}; do
  echo "=== [Phase-B v18c roll-gated flight + headstand tax] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --env.sim.njmax 192 \
    --agent.max-iterations "$MAX_ITER" \
    --agent.algorithm.entropy-coef 0.004 \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [Phase-B v18c rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back (OOM guard)"
done
echo "=== Phase-B v18c 训练结束 @ $(date) ==="

# 自动出 GIF — 低机位贴墙视角
RUN_DIR=$(ls -dt logs/rsl_rl/microduck_roulade_dive/2026-*_run15 | head -1)
CKPT=$(ls -t "$RUN_DIR"/model_*.pt | head -1)
echo "=== [gif Phase-B v18c] ckpt=$CKPT @ $(date) ==="
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python scripts/video_rollout.py \
  Mjlab-RouladeDiveWall-Flat-MicroDuck \
  --checkpoint-file "$CKPT" \
  --camera trunk_base --elevation 3 --azimuth 95 --distance 2.0 \
  --video-width 1280 --video-height 720 --video-length 900 --num-envs 1 \
  --speed-overlay \
  > logs/gif_phaseB18c_wall_final.log 2>&1
MP4=$(ls -t "$RUN_DIR"/videos/play/*.mp4 2>/dev/null | head -1)
[ -n "$MP4" ] && ffmpeg -y -i "$MP4" -vf "fps=12,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" logs/gifs/phaseB18c_wall_final.gif > /dev/null 2>&1 && echo "=== [gif done] logs/gifs/phaseB18c_wall_final.gif ==="
echo "=== ALL DONE @ $(date) ==="

bash scripts/make_stage_gif.sh "${GIF_NAME:-lean_b18c_midroll}"
