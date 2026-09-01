"""使用 Stable-Baselines3 PPO 训练 Pupper 速度跟踪策略。"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from pupper_env import PupperEnv


LAB_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = LAB_DIR / "configs" / "train.yaml"

REQUIRED_CONFIG_KEYS = {
    "run": (
        "seed", "total_timesteps", "n_envs", "device", "tensorboard",
        "output_dir", "resume_from", "transfer_from",
    ),
    "environment": ("max_steps",),
    "gait": ("enabled", "types", "contact_reward_weight", "switch_steps"),
    "ppo": (
        "policy", "n_steps", "batch_size", "n_epochs", "learning_rate",
        "gamma", "gae_lambda", "clip_range", "ent_coef", "ent_coef_final",
        "net_arch", "verbose",
    ),
    "checkpoint": ("every_steps", "name_prefix", "final_name"),
}


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """加载并检查训练 YAML。"""
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"配置文件顶层必须是 mapping：{config_path}")

    for section, keys in REQUIRED_CONFIG_KEYS.items():
        values = config.get(section)
        if not isinstance(values, dict):
            raise ValueError(f"配置缺少 mapping：{section}")
        missing = [key for key in keys if key not in values]
        if missing:
            raise ValueError(f"配置 {section} 缺少字段：{', '.join(missing)}")

    if int(config["run"]["n_envs"]) < 1:
        raise ValueError("run.n_envs 必须大于 0")
    if int(config["run"]["total_timesteps"]) < 1:
        raise ValueError("run.total_timesteps 必须大于 0")
    if int(config["environment"]["max_steps"]) < 1:
        raise ValueError("environment.max_steps 必须大于 0")
    if config["run"].get("resume_from") and config["run"].get("transfer_from"):
        raise ValueError("resume_from 和 transfer_from 不能同时设置")
    if not isinstance(config["gait"]["types"], list) or not config["gait"]["types"]:
        raise ValueError("gait.types 必须是非空列表")
    return config


def _lab_relative_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else LAB_DIR / path


class EntropyCoefDecayCallback(BaseCallback):
    """把 ent_coef 随训练进度从初始值线性衰减到目标值。

    早期较高的熵系数帮助策略跳出"冻结不动"的局部最优，
    后期低熵让动作分布收窄、步态平滑。
    """

    def __init__(self, initial: float, final: float):
        super().__init__()
        self.initial = float(initial)
        self.final = float(final)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        progress = 1.0 - float(self.model._current_progress_remaining)
        self.model.ent_coef = (
            self.initial + (self.final - self.initial) * progress
        )
        self.logger.record("train/ent_coef", self.model.ent_coef)


def make_env(
    seed: int,
    rank: int,
    max_steps: int,
    gait_config: dict[str, Any],
    reward_overrides: dict[str, float] | None = None,
):
    def _thunk():
        env = Monitor(PupperEnv(
            max_steps=max_steps,
            reward_overrides=reward_overrides,
            gait_enabled=bool(gait_config["enabled"]),
            gait_types=tuple(gait_config["types"]),
            gait_contact_weight=float(gait_config["contact_reward_weight"]),
            gait_switch_steps=int(gait_config["switch_steps"]),
        ))
        env.reset(seed=seed + rank)
        return env

    return _thunk


def _make_model(
    vec_env: SubprocVecEnv,
    ppo_config: dict[str, Any],
    tensorboard_log: str | None,
    seed: int,
    device: str,
) -> PPO:
    return PPO(
        str(ppo_config["policy"]),
        vec_env,
        n_steps=int(ppo_config["n_steps"]),
        batch_size=int(ppo_config["batch_size"]),
        n_epochs=int(ppo_config["n_epochs"]),
        learning_rate=float(ppo_config["learning_rate"]),
        gamma=float(ppo_config["gamma"]),
        gae_lambda=float(ppo_config["gae_lambda"]),
        clip_range=float(ppo_config["clip_range"]),
        ent_coef=float(ppo_config["ent_coef"]),
        policy_kwargs={"net_arch": list(ppo_config["net_arch"])},
        tensorboard_log=tensorboard_log,
        verbose=int(ppo_config["verbose"]),
        seed=seed,
        device=device,
    )


def transfer_policy_weights(target: PPO, source_path: Path, device: str) -> None:
    """把窄观测策略迁移到宽观测策略，新增输入列权重初始化为零。"""
    source = PPO.load(source_path, device=device)
    source_state = source.policy.state_dict()
    target_state = target.policy.state_dict()
    migrated_inputs: list[str] = []
    incompatible: list[str] = []

    for name, target_tensor in target_state.items():
        source_tensor = source_state.get(name)
        if source_tensor is None:
            incompatible.append(f"{name}: source missing")
            continue
        if target_tensor.shape == source_tensor.shape:
            target_state[name] = source_tensor.clone()
            continue
        if (
            target_tensor.ndim == 2
            and source_tensor.ndim == 2
            and target_tensor.shape[0] == source_tensor.shape[0]
            and target_tensor.shape[1] > source_tensor.shape[1]
        ):
            expanded = target_tensor.clone()
            expanded.zero_()
            expanded[:, :source_tensor.shape[1]] = source_tensor
            target_state[name] = expanded
            migrated_inputs.append(name)
            continue
        incompatible.append(
            f"{name}: {tuple(source_tensor.shape)} -> {tuple(target_tensor.shape)}",
        )

    if incompatible:
        raise ValueError("checkpoint 结构不兼容：" + "; ".join(incompatible))
    if not migrated_inputs:
        raise ValueError("没有发现需要扩展的观测输入层")

    target.policy.load_state_dict(target_state, strict=True)
    print(
        "已迁移策略参数；新增观测输入权重置零："
        + ", ".join(migrated_inputs),
    )


def train(config: dict[str, Any]) -> Path:
    """按已经验证的配置训练并返回最终 checkpoint 路径。"""
    run_config = config["run"]
    env_config = config["environment"]
    ppo_config = config["ppo"]
    checkpoint_config = config["checkpoint"]
    gait_config = config["gait"]

    seed = int(run_config["seed"])
    timesteps = int(run_config["total_timesteps"])
    n_envs = int(run_config["n_envs"])
    device = str(run_config["device"])
    tensorboard = bool(run_config["tensorboard"])
    max_steps = int(env_config["max_steps"])
    out_dir = _lab_relative_path(run_config["output_dir"])
    resume_from = run_config.get("resume_from")
    checkpoint = _lab_relative_path(resume_from) if resume_from else None
    transfer_from = run_config.get("transfer_from")
    transfer_checkpoint = (
        _lab_relative_path(transfer_from) if transfer_from else None
    )

    if checkpoint and not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint 不存在：{checkpoint}")
    if transfer_checkpoint and not transfer_checkpoint.exists():
        raise FileNotFoundError(
            f"迁移 checkpoint 不存在：{transfer_checkpoint}",
        )
    if checkpoint and transfer_checkpoint:
        raise ValueError("resume_from 和 transfer_from 不能同时设置")

    # 可选的奖励权重覆盖，用于两阶段课程（bootstrap 轻惩罚，finetune 全额惩罚）。
    reward_overrides = env_config.get("reward_overrides") or None
    if reward_overrides is not None:
        reward_overrides = {
            str(key): float(value) for key, value in reward_overrides.items()
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    vec_env = SubprocVecEnv([
        make_env(seed, rank, max_steps, gait_config, reward_overrides)
        for rank in range(n_envs)
    ])
    try:
        tensorboard_log = str(out_dir / "tb") if tensorboard else None

        if checkpoint:
            print(f"从 checkpoint 继续训练：{checkpoint}")
            model = PPO.load(
                checkpoint,
                env=vec_env,
                tensorboard_log=tensorboard_log,
                device=device,
            )
            model.set_random_seed(seed)
        else:
            model = _make_model(
                vec_env, ppo_config, tensorboard_log, seed, device,
            )
            if transfer_checkpoint:
                print(f"从窄观测策略迁移参数：{transfer_checkpoint}")
                transfer_policy_weights(model, transfer_checkpoint, device)

        callbacks: list[BaseCallback] = [CheckpointCallback(
            save_freq=max(int(checkpoint_config["every_steps"]) // n_envs, 1),
            save_path=str(out_dir),
            name_prefix=str(checkpoint_config["name_prefix"]),
        )]
        ent_coef = float(ppo_config["ent_coef"])
        ent_coef_final = float(ppo_config["ent_coef_final"])
        if ent_coef != ent_coef_final:
            callbacks.append(
                EntropyCoefDecayCallback(ent_coef, ent_coef_final),
            )
        model.learn(total_timesteps=timesteps, callback=callbacks)
        final_name = str(checkpoint_config["final_name"])
        if not final_name.endswith(".zip"):
            final_name += ".zip"
        final_path = out_dir / final_name
        model.save(str(final_path))
    finally:
        vec_env.close()

    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 PPO 训练 Pupper")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    config = deepcopy(load_config(args.config))
    overrides = {
        "total_timesteps": args.timesteps,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "device": args.device,
        "tensorboard": args.tensorboard,
    }
    for key, value in overrides.items():
        if value is not None:
            config["run"][key] = value
    if args.out is not None:
        config["run"]["output_dir"] = str(Path(args.out).expanduser().resolve())
    if args.checkpoint is not None:
        config["run"]["resume_from"] = str(
            Path(args.checkpoint).expanduser().resolve(),
        )
        config["run"]["transfer_from"] = None

    print(f"加载配置：{Path(args.config).expanduser().resolve()}")
    print(
        f"训练参数：steps={config['run']['total_timesteps']}, "
        f"envs={config['run']['n_envs']}, device={config['run']['device']}, "
        f"gait={config['gait']['enabled']}",
    )
    path = train(config)
    print(f"模型已保存：{path}")


if __name__ == "__main__":
    main()
