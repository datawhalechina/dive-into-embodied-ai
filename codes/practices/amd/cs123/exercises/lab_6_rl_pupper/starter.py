"""Lab 6 filled starter：你的第一只 RL Pupper。

作者侧先写 `starter_todo.py`，再把三处 TODO 填成这份 `starter.py`。
交付学生版时抽走本文件，并把 `starter_todo.py` 改名为 `starter.py`。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw

EXERCISES_DIR = Path(__file__).resolve().parents[1]
if str(EXERCISES_DIR) not in sys.path:
    sys.path.insert(0, str(EXERCISES_DIR))

from shared.viz import gif_utils  # noqa: E402
from shared.viz.training_plot import plot_training_curves  # noqa: E402

LAB_DIR = Path(__file__).resolve().parent
PORTFOLIO_DIR = LAB_DIR / "portfolio"
TB_DIR = LAB_DIR / "tb"

sys.path.insert(0, str(LAB_DIR))
from envs.pupper_env import (  # noqa: E402
    DEFAULT_POSE,
    DEFAULT_XML,
    FOOT_BODIES,
    FrameStackedPupperEnv,
    JOINT_NAMES,
    KD,
    KP,
    MAX_STEPS,
    N_STACK,
    REWARD_WEIGHTS,
    PupperEnv,
)

GIF_FPS = 12
GIF_WIDTH = 1280
GIF_HEIGHT = 480
CHECKPOINT_NAME = "pupper_ppo"

CMD_SCRIPT = [
    ((0.0, 0.0, 0.0), 2.0),
    ((0.5, 0.0, 0.0), 3.0),
    ((0.0, 0.0, 0.5), 2.5),
    ((-0.3, 0.0, 0.0), 2.5),
    ((0.0, 0.0, 0.0), 2.0),
]

CMD_LABELS = {
    (0.0, 0.0, 0.0): "stand",
    (0.5, 0.0, 0.0): "forward 0.5 m/s",
    (0.0, 0.0, 0.5): "turn left 0.5 rad/s",
    (-0.3, 0.0, 0.0): "backward 0.3 m/s",
}


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def train_ppo(
    seed: int = 0,
    total_timesteps: int = 150_000_000,
    n_envs: int = 16,
    warm_start: Path | None = None,
) -> Path:
    """SB3 PPO 训练入口，超参与 brax-PPO 配方等价换算：
    - lr=3e-4, ent_coef=0.01, gamma=0.97, gae_lambda=0.95, clip_range=0.2（直接对应）
    - n_epochs=4（brax `num_updates_per_batch`）
    - batch_size=256（brax minibatch）
    - n_steps=2048：单 rollout 收集 n_envs × n_steps = 32k 样本，与 brax
      `unroll_length × num_envs / num_minibatches` 数量级匹配
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
    from callbacks import RewardComponentCallback

    def make_env(rank: int):
        def _thunk():
            env = FrameStackedPupperEnv(n_stack=N_STACK)
            env = Monitor(env)
            env.reset(seed=seed + rank)
            return env
        return _thunk

    vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(5_000_000 // n_envs, 1),
        save_path=str(PORTFOLIO_DIR),
        name_prefix=CHECKPOINT_NAME,
    )
    reward_cb = RewardComponentCallback()
    callbacks_list = [checkpoint_cb, reward_cb]

    if warm_start is not None and Path(warm_start).exists():
        print(f"warm-starting from {warm_start}")
        model = PPO.load(str(warm_start), env=vec_env)
        model.tensorboard_log = str(TB_DIR)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            n_steps=2048,
            batch_size=256,
            n_epochs=4,
            learning_rate=3e-4,
            gamma=0.97,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            tensorboard_log=str(TB_DIR),
            verbose=1,
            seed=seed,
        )
    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callbacks_list),
    )

    final_path = PORTFOLIO_DIR / f"{CHECKPOINT_NAME}.zip"
    model.save(str(final_path))
    vec_env.close()

    _cleanup_intermediate_checkpoints()
    return final_path


def _cleanup_intermediate_checkpoints():
    """删除中间 checkpoint，只留最终 zip。"""
    final = PORTFOLIO_DIR / f"{CHECKPOINT_NAME}.zip"
    for f in PORTFOLIO_DIR.glob(f"{CHECKPOINT_NAME}_*_steps.zip"):
        if f != final:
            f.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 评估 / 渲染
# ---------------------------------------------------------------------------

def _get_cmd_at_time(t: float) -> tuple[tuple[float, float, float], str]:
    elapsed = 0.0
    for cmd, duration in CMD_SCRIPT:
        if t < elapsed + duration:
            label = CMD_LABELS.get(cmd, f"cmd {cmd}")
            return cmd, label
        elapsed += duration
    last_cmd = CMD_SCRIPT[-1][0]
    return last_cmd, CMD_LABELS.get(last_cmd, "stand")


def _caption_frame(frame: np.ndarray, cmd_label: str, t: float) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = gif_utils.load_font(22)
    small = gif_utils.load_font(16)
    draw.rectangle((0, 0, image.width, 44), fill=(255, 255, 255, 220))
    draw.text((14, 10), cmd_label, fill=(18, 30, 44, 255), font=font)
    draw.rectangle((0, image.height - 32, 160, image.height), fill=(0, 0, 0, 160))
    draw.text((10, image.height - 28), f"t = {t:.1f} s", fill=(230, 230, 230, 240), font=small)
    return np.asarray(image)


def render_command_demo(checkpoint_path: str | Path | None = None, caption_prefix: str = "") -> list[np.ndarray]:
    """加载 checkpoint，按 CMD_SCRIPT 播放命令序列，返回帧列表。"""
    from stable_baselines3 import PPO

    if checkpoint_path is None:
        checkpoint_path = PORTFOLIO_DIR / f"{CHECKPOINT_NAME}.zip"
    checkpoint_path = Path(checkpoint_path)

    env = FrameStackedPupperEnv(n_stack=N_STACK)
    model = PPO.load(str(checkpoint_path), env=env)

    total_seconds = sum(d for _, d in CMD_SCRIPT)
    total_steps = int(total_seconds / env.dt)

    half_w = GIF_WIDTH // 2
    env.model.vis.global_.offwidth = max(env.model.vis.global_.offwidth, half_w)
    env.model.vis.global_.offheight = max(env.model.vis.global_.offheight, GIF_HEIGHT)
    renderer = mujoco.Renderer(env.model, height=GIF_HEIGHT, width=half_w)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.model, camera)
    camera.distance = 1.4
    camera.elevation = -20.0
    camera.azimuth = 90.0

    obs, _ = env.reset(seed=42)
    frames: list[np.ndarray] = []
    next_frame_time = 0.0

    try:
        for step_i in range(total_steps):
            t = step_i * env.dt
            cmd_tuple, cmd_label = _get_cmd_at_time(t)
            env.cmd = np.array(cmd_tuple, dtype=np.float32)

            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)

            if t + 0.5 * env.dt >= next_frame_time:
                camera.lookat[:] = env.data.xpos[env._base_id]
                renderer.update_scene(env.data, camera=camera)
                frame = renderer.render()
                frames.append(_caption_frame(frame, f"{caption_prefix}{cmd_label}", t))
                next_frame_time += 1.0 / GIF_FPS

            if terminated or truncated:
                obs, _ = env.reset(seed=42)
    finally:
        renderer.close()

    return frames


def render_velocity_tracking(
    checkpoint_path: str | Path | None = None,
    vx_cmds: tuple[float, ...] = (0.2, 0.4, 0.6),
    seconds: float = 8.0,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """在多档 vx_cmd 下跑策略，返回 {vx_cmd: (time, vx_actual)} 字典。"""
    from stable_baselines3 import PPO

    if checkpoint_path is None:
        checkpoint_path = PORTFOLIO_DIR / f"{CHECKPOINT_NAME}.zip"

    env = FrameStackedPupperEnv(n_stack=N_STACK)
    model = PPO.load(str(checkpoint_path), env=env)
    results = {}

    for vx_cmd in vx_cmds:
        obs, _ = env.reset(seed=42)
        env.cmd = np.array([vx_cmd, 0.0, 0.0], dtype=np.float32)
        ts, vxs = [], []
        n_steps = int(seconds / env.dt)
        for i in range(n_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            ts.append(i * env.dt)
            vxs.append(float(env.data.qvel[0]))
            if terminated or truncated:
                obs, _ = env.reset(seed=42)
                env.cmd = np.array([vx_cmd, 0.0, 0.0], dtype=np.float32)
        results[vx_cmd] = (np.array(ts), np.array(vxs))

    return results


# ---------------------------------------------------------------------------
# 上游 policy 对比渲染
# ---------------------------------------------------------------------------

def _load_upstream_policy():
    """加载 shared/rl/policies/test_policy.json 的上游 RTNeural policy。"""
    from shared.upstream.lab_5_mujoco import (
        RTNeuralPolicy, build_obs_frame, roll_obs_history,
        quat_inv_rotate, OBS_TOTAL,
    )
    policy_path = EXERCISES_DIR / "shared" / "rl" / "policies" / "test_policy.json"
    if not policy_path.exists():
        raise FileNotFoundError(
            f"{policy_path} not found. Run shared/rl/fetch_policies.sh first."
        )
    return RTNeuralPolicy(policy_path)


def render_upstream_demo() -> list[np.ndarray]:
    """用上游 test_policy.json 跑同一段 CMD_SCRIPT，返回帧列表。"""
    from shared.upstream.lab_5_mujoco import (
        build_obs_frame, roll_obs_history, quat_inv_rotate, OBS_TOTAL,
    )

    policy = _load_upstream_policy()
    model = mujoco.MjModel.from_xml_path(DEFAULT_XML)
    model.opt.timestep = 0.002
    model.actuator_gainprm[:, 0] = policy.kp
    model.actuator_biasprm[:, 1] = -policy.kp
    model.actuator_biasprm[:, 2] = -policy.kd
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, 0.22]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:19] = policy.default_joint_pos
    data.ctrl[:] = policy.default_joint_pos
    mujoco.mj_forward(model, data)

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    half_w = GIF_WIDTH // 2
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, half_w)
    model.vis.global_.offheight = max(model.vis.global_.offheight, GIF_HEIGHT)
    renderer = mujoco.Renderer(model, height=GIF_HEIGHT, width=half_w)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.distance = 1.4
    camera.elevation = -20.0
    camera.azimuth = 90.0

    control_dt = 0.02
    phys_dt = model.opt.timestep
    control_ratio = int(round(control_dt / phys_dt))

    total_seconds = sum(d for _, d in CMD_SCRIPT)
    total_phys_steps = int(total_seconds / phys_dt)

    settle_seconds = 0.5
    obs_history = np.zeros(OBS_TOTAL, dtype=np.float32)
    last_action = np.zeros(12, dtype=np.float32)

    frames: list[np.ndarray] = []
    next_frame_time = 0.0

    total_phys_steps = int((total_seconds + settle_seconds) / phys_dt)

    try:
        for step_i in range(total_phys_steps):
            sim_t = step_i * phys_dt
            in_settle = sim_t < settle_seconds
            t_policy = 0.0 if in_settle else (sim_t - settle_seconds)
            cmd_tuple, cmd_label = _get_cmd_at_time(t_policy)
            cmd = np.array(cmd_tuple if not in_settle else (0.0, 0.0, 0.0),
                           dtype=np.float32)

            if step_i % control_ratio == 0:
                quat = data.xquat[base_id].copy()
                world_ang = data.cvel[base_id, 0:3].copy()
                local_ang = quat_inv_rotate(quat, world_ang)
                gravity_b = quat_inv_rotate(quat, np.array([0.0, 0.0, -1.0]))
                n = np.linalg.norm(gravity_b)
                if n > 1e-6:
                    gravity_b = gravity_b / n
                joint_pos = data.qpos[7:19].copy()
                desired_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)

                frame_obs = build_obs_frame(
                    local_ang_vel=local_ang,
                    gravity_in_body=gravity_b,
                    command=cmd,
                    desired_world_z_in_body=desired_z,
                    joint_pos=joint_pos,
                    last_action=last_action,
                    default_pose=policy.default_joint_pos.astype(np.float64),
                )
                obs_history = roll_obs_history(obs_history, frame_obs)
                if in_settle:
                    action = np.zeros(12, dtype=np.float32)
                else:
                    action = policy.forward(obs_history)
                last_action = action.copy()
                target = policy.default_joint_pos + policy.action_scale * action
                data.ctrl[:] = target

            mujoco.mj_step(model, data)

            if not in_settle and t_policy + 0.5 * phys_dt >= next_frame_time:
                camera.lookat[:] = data.xpos[base_id]
                renderer.update_scene(data, camera=camera)
                raw = renderer.render()
                frames.append(_caption_frame(
                    np.array(Image.fromarray(raw).resize((half_w, GIF_HEIGHT), Image.Resampling.BILINEAR)),
                    f"[upstream] {cmd_label}", t_policy,
                ))
                next_frame_time += 1.0 / GIF_FPS
    finally:
        renderer.close()

    return frames


def render_comparison_gif(
    checkpoint_path: str | Path | None = None,
    out_path: Path | None = None,
) -> None:
    """渲染 side-by-side 对比 GIF：左=我们训的 PPO，右=上游 test_policy。"""
    if out_path is None:
        out_path = PORTFOLIO_DIR / "comparison.gif"

    print("Rendering our PPO policy...")
    our_frames = render_command_demo(checkpoint_path, caption_prefix="[ours] ")
    print("Rendering upstream test_policy...")
    upstream_frames = render_upstream_demo()

    n = min(len(our_frames), len(upstream_frames))
    combined = []
    for i in range(n):
        combined.append(np.concatenate([our_frames[i], upstream_frames[i]], axis=1))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    gif_utils.write_gif(combined, out_path, fps=GIF_FPS, max_frames=120, width=840)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Comparison GIF: {out_path} ({size_mb:.2f} MB, {n} frames)")


# ---------------------------------------------------------------------------
# 画图
# ---------------------------------------------------------------------------

def save_reward_curve(out_path: Path | None = None) -> None:
    if out_path is None:
        out_path = PORTFOLIO_DIR / "reward_curve.png"
    tb_runs = sorted(TB_DIR.glob("PPO_*")) if TB_DIR.exists() else []
    if not tb_runs:
        print("未找到 tensorboard 日志，跳过 reward_curve。")
        return
    plot_training_curves(tb_runs[-1], out_path)


def save_velocity_tracking(
    results: dict[float, tuple[np.ndarray, np.ndarray]] | None = None,
    out_path: Path | None = None,
) -> None:
    if out_path is None:
        out_path = PORTFOLIO_DIR / "velocity_tracking.png"
    if results is None:
        results = render_velocity_tracking()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (vx_cmd, (ts, vxs)) in zip(axes, sorted(results.items())):
        ax.axhline(vx_cmd, color="tab:red", linestyle="--", linewidth=1.5, label=f"cmd = {vx_cmd}")
        ax.plot(ts, vxs, color="tab:blue", linewidth=1.0, label="actual vx")
        ax.set_xlabel("时间 [s]")
        ax.set_title(f"vx_cmd = {vx_cmd} m/s")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("vx [m/s]")
    fig.suptitle("速度跟踪")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> None:
    print("Lab 6 starter.py — 直接运行会执行一次快速环境检查。")
    env = PupperEnv()
    obs, _ = env.reset(seed=0)
    print(f"obs shape: {obs.shape}, action space: {env.action_space}")
    total_r = 0.0
    for _ in range(100):
        obs, r, term, trunc, _ = env.step(np.zeros(12, dtype=np.float32))
        total_r += r
        if term or trunc:
            break
    print(f"100 步零动作: total reward = {total_r:.2f}, terminated = {term}")


if __name__ == "__main__":
    main()
