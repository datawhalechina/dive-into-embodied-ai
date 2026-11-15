# Phase-B v22 (2026-10-22 晚, B20 验收"可以, 但有时顶在墙体上翻滚").
# B21 = B20 全配置, 唯一变量: WALL_SINK 0.100→0.085 (有效墙高 2→3.5cm, 大人 21:14 定稿 "到3.5cm就行").
# 目的: 堵贴墙滚套利, 逼真跳+前扑过墙. warm 接 B20 final model_212470.
#
#      把样本还给 takeoff/flight 前扑段;
#   ② DIVEWALL_PHASE_B21=1: 新 reward roulade_flight_pitch (w 1.5) —
#      空中 + 墙前-0.40m~墙后+0.45m + vx≥0.5 门, 只奖前倾姿态 (15°→60° ramp),
#      无旋转因子 (避 B15 空中刷旋转套利).
# warm 接 B18 final ckpt model_212470 (run 2026-10-22_15-12-59).
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline
export DIVEWALL_STAGE=4
export DIVEWALL_WALL_SINK=0.085
export DIVEWALL_WALL_DIST=0.90,1.30
export DIVEWALL_POSTDIVE_PROB=0.0
export DIVEWALL_PHASE_B1=0
export DIVEWALL_PHASE_B3=1
export DIVEWALL_PHASE_B17=0
export DIVEWALL_PHASE_B18=1
export DIVEWALL_B18_MIDROLL_FRAC=0.22
export DIVEWALL_PHASE_B21=1
export DIVEWALL_PHASE_B22=1

LOAD_RUN=2026-10-22_21-15-59_microduck_roulade_divewall_run15
LOAD_CKPT=model_220469.pt
MAX_ITER=8000

LOG=logs/train_phaseB22_wall.log
for ENVS in 2048 1024; do
  echo "=== [Phase-B v22 midroll0.22+flight_pitch] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.algorithm.entropy-coef 0.004 \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [Phase-B v22 midroll0.22+flight_pitch rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 1024 (OOM guard)"
done
echo "=== Phase-B v22 训练结束 @ $(date) ==="

# 自动出 GIF — 低机位贴墙视角
RUN_DIR=$(ls -dt logs/rsl_rl/microduck_roulade_dive/2026-*_run15 | head -1)
CKPT=$(ls -t "$RUN_DIR"/model_*.pt | head -1)
echo "=== [gif Phase-B v22] ckpt=$CKPT @ $(date) ==="
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python scripts/video_rollout.py \
  Mjlab-RouladeDiveWall-Flat-MicroDuck \
  --checkpoint-file "$CKPT" \
  --camera trunk_base --elevation 3 --azimuth 95 --distance 2.0 \
  --video-width 1280 --video-height 720 --video-length 900 --num-envs 1 \
  --speed-overlay \
  > logs/gif_phaseB22_wall_final.log 2>&1
MP4=$(ls -t "$RUN_DIR"/videos/play/*.mp4 2>/dev/null | head -1)
[ -n "$MP4" ] && ffmpeg -y -i "$MP4" -vf "fps=12,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" logs/gifs/phaseB22_wall_final.gif > /dev/null 2>&1 && echo "=== [gif done] logs/gifs/phaseB22_wall_final.gif ==="
echo "=== ALL DONE @ $(date) ==="
