import argparse
import random
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, capacity: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: float,
    ) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], device=device),
            torch.as_tensor(self.actions[idx], device=device),
            torch.as_tensor(self.rewards[idx], device=device),
            torch.as_tensor(self.next_obs[idx], device=device),
            torch.as_tensor(self.dones[idx], device=device),
        )

    def __len__(self) -> int:
        return self.size


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Actor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
    ) -> None:
        super().__init__()
        self.backbone = MLP(obs_dim, hidden_dim, action_dim)
        action_scale = (action_high - action_low) / 2.0
        action_bias = (action_high + action_low) / 2.0
        self.register_buffer("action_scale", torch.as_tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(action_bias, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raw_action = torch.tanh(self.backbone(obs))
        return raw_action * self.action_scale + self.action_bias


class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.q = MLP(obs_dim + action_dim, hidden_dim, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([obs, action], dim=-1))


@dataclass
class DDPGConfig:
    env_id: str
    total_steps: int
    start_steps: int
    batch_size: int
    buffer_size: int
    gamma: float
    tau: float
    actor_lr: float
    critic_lr: float
    hidden_dim: int
    exploration_noise: float
    seed: int
    eval_interval: int
    eval_episodes: int
    demo_episodes: int
    demo_render: bool
    record_path: Path | None
    record_episodes: int
    record_fps: int
    record_frame_skip: int
    save_path: Path
    device: str


class DDPGAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        config: DDPGConfig,
    ) -> None:
        if config.device == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_name = config.device
        self.device = torch.device(device_name)

        self.actor = Actor(obs_dim, action_dim, config.hidden_dim, action_low, action_high).to(self.device)
        self.actor_target = Actor(obs_dim, action_dim, config.hidden_dim, action_low, action_high).to(self.device)
        self.critic = Critic(obs_dim, action_dim, config.hidden_dim).to(self.device)
        self.critic_target = Critic(obs_dim, action_dim, config.hidden_dim).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        self.gamma = config.gamma
        self.tau = config.tau
        self.action_low = action_low
        self.action_high = action_high
        self.action_scale_np = (action_high - action_low) / 2.0

    @torch.no_grad()
    def act(self, obs: np.ndarray, noise_scale: float = 0.0) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor(obs_tensor).cpu().numpy()[0]
        if noise_scale > 0.0:
            noise = np.random.normal(0.0, self.action_scale_np * noise_scale, size=action.shape)
            action = action + noise
        return np.clip(action, self.action_low, self.action_high)

    def update(self, replay_buffer: ReplayBuffer, batch_size: int) -> tuple[float, float]:
        obs, actions, rewards, next_obs, dones = replay_buffer.sample(batch_size, self.device)

        with torch.no_grad():
            next_actions = self.actor_target(next_obs)
            target_q = self.critic_target(next_obs, next_actions)
            backup = rewards + self.gamma * (1.0 - dones) * target_q

        current_q = self.critic(obs, actions)
        critic_loss = F.mse_loss(current_q, backup)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss = -self.critic(obs, self.actor(obs)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)
        return float(actor_loss.item()), float(critic_loss.item())

    def soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(source_param.data, alpha=self.tau)

    def save_actor(self, save_path: Path) -> None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), save_path)


def evaluate_policy(
    env_id: str,
    agent: DDPGAgent,
    episodes: int,
    seed: int,
) -> float:
    eval_env = gym.make(env_id)
    returns: list[float] = []
    for episode_idx in range(episodes):
        obs, _ = eval_env.reset(seed=seed + 10_000 + episode_idx)
        done = False
        episode_return = 0.0
        while not done:
            action = agent.act(obs, noise_scale=0.0)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            episode_return += reward
        returns.append(episode_return)
    eval_env.close()
    return float(np.mean(returns))


def demo_policy(
    env_id: str,
    agent: DDPGAgent,
    episodes: int,
    seed: int,
    render: bool,
) -> None:
    render_mode = "human" if render else None
    demo_env = gym.make(env_id, render_mode=render_mode)
    for episode_idx in range(episodes):
        obs, _ = demo_env.reset(seed=seed + 20_000 + episode_idx)
        done = False
        episode_return = 0.0
        while not done:
            action = agent.act(obs, noise_scale=0.0)
            obs, reward, terminated, truncated, _ = demo_env.step(action)
            done = terminated or truncated
            episode_return += reward
        print(f"[demo] episode={episode_idx + 1} return={episode_return:.2f}")
    demo_env.close()


def record_policy(
    env_id: str,
    agent: DDPGAgent,
    episodes: int,
    seed: int,
    record_path: Path,
    fps: int,
    frame_skip: int,
) -> None:
    if record_path.suffix.lower() not in {".gif", ".mp4"}:
        raise ValueError("record_path must end with .gif or .mp4")

    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_env = gym.make(env_id, render_mode="rgb_array")
    frames: list[np.ndarray] = []

    for episode_idx in range(episodes):
        obs, _ = record_env.reset(seed=seed + 30_000 + episode_idx)
        done = False
        step_idx = 0
        initial_frame = record_env.render()
        if initial_frame is not None:
            frames.append(initial_frame)

        while not done:
            action = agent.act(obs, noise_scale=0.0)
            obs, _, terminated, truncated, _ = record_env.step(action)
            done = terminated or truncated
            step_idx += 1
            if step_idx % frame_skip == 0:
                frame = record_env.render()
                if frame is not None:
                    frames.append(frame)

        final_frame = record_env.render()
        if final_frame is not None:
            frames.append(final_frame)

    record_env.close()

    if record_path.suffix.lower() == ".gif":
        imageio.mimsave(record_path, frames, fps=fps, loop=0)
    else:
        with imageio.get_writer(
            record_path,
            fps=fps,
            codec="libx264",
            format="FFMPEG",
            macro_block_size=None,
        ) as writer:
            for frame in frames:
                writer.append_data(frame)

    print(f"saved_recording={record_path}")


def train(config: DDPGConfig) -> None:
    set_seed(config.seed)
    env = gym.make(config.env_id)
    env.action_space.seed(config.seed)

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    action_low = env.action_space.low.astype(np.float32)
    action_high = env.action_space.high.astype(np.float32)

    agent = DDPGAgent(obs_dim, action_dim, action_low, action_high, config)
    replay_buffer = ReplayBuffer(obs_dim, action_dim, config.buffer_size)

    obs, _ = env.reset(seed=config.seed)
    episode_return = 0.0
    episode_length = 0
    recent_returns: deque[float] = deque(maxlen=10)

    print(f"device={agent.device} env={config.env_id}", flush=True)
    use_tqdm = sys.stderr.isatty()
    progress = trange(1, config.total_steps + 1, desc="training", leave=True, disable=not use_tqdm)
    for step in progress:
        if step <= config.start_steps:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, noise_scale=config.exploration_noise)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Only true termination should stop TD bootstrapping.
        replay_done = float(terminated)
        replay_buffer.add(obs, action, reward, next_obs, replay_done)

        obs = next_obs
        episode_return += reward
        episode_length += 1

        actor_loss = None
        critic_loss = None
        if step > config.start_steps and len(replay_buffer) >= config.batch_size:
            actor_loss, critic_loss = agent.update(replay_buffer, config.batch_size)

        if done:
            recent_returns.append(episode_return)
            if use_tqdm:
                progress.set_postfix(
                    episode_return=f"{episode_return:.1f}",
                    episode_length=episode_length,
                    avg10=f"{np.mean(recent_returns):.1f}",
                    actor_loss=f"{actor_loss:.3f}" if actor_loss is not None else "na",
                    critic_loss=f"{critic_loss:.3f}" if critic_loss is not None else "na",
                )
            obs, _ = env.reset()
            episode_return = 0.0
            episode_length = 0

        if step % config.eval_interval == 0:
            avg_eval_return = evaluate_policy(config.env_id, agent, config.eval_episodes, config.seed)
            print(f"[eval] step={step} avg_return={avg_eval_return:.2f}")

    env.close()
    agent.save_actor(config.save_path)
    final_eval_return = evaluate_policy(config.env_id, agent, config.eval_episodes, config.seed)
    print(f"saved_actor={config.save_path}")
    print(f"final_eval_return={final_eval_return:.2f}")

    if config.demo_episodes > 0:
        demo_policy(
            config.env_id,
            agent,
            episodes=config.demo_episodes,
            seed=config.seed,
            render=config.demo_render,
        )

    if config.record_path is not None:
        record_policy(
            config.env_id,
            agent,
            episodes=config.record_episodes,
            seed=config.seed,
            record_path=config.record_path,
            fps=config.record_fps,
            frame_skip=config.record_frame_skip,
        )


def parse_args() -> DDPGConfig:
    parser = argparse.ArgumentParser(description="Train a pure PyTorch DDPG baseline on Pusher-v5.")
    parser.add_argument("--env-id", type=str, default="Pusher-v5")
    parser.add_argument("--total-steps", type=int, default=250_000)
    parser.add_argument("--start-steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--exploration-noise", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-interval", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--demo-episodes", type=int, default=0)
    parser.add_argument("--demo-render", action="store_true")
    parser.add_argument("--record-path", type=Path, default=None)
    parser.add_argument("--record-episodes", type=int, default=1)
    parser.add_argument("--record-fps", type=int, default=20)
    parser.add_argument("--record-frame-skip", type=int, default=3)
    parser.add_argument("--save-path", type=Path, default=Path("artifacts/ddpg_pusher_v5_actor.pt"))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    return DDPGConfig(
        env_id=args.env_id,
        total_steps=args.total_steps,
        start_steps=args.start_steps,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        hidden_dim=args.hidden_dim,
        exploration_noise=args.exploration_noise,
        seed=args.seed,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        demo_episodes=args.demo_episodes,
        demo_render=args.demo_render,
        record_path=args.record_path,
        record_episodes=args.record_episodes,
        record_fps=args.record_fps,
        record_frame_skip=args.record_frame_skip,
        save_path=args.save_path,
        device=args.device,
    )


if __name__ == "__main__":
    train(parse_args())
