"""Lab 6 PPO 自定义 callback。

只保留一个：`RewardComponentCallback`，把 env 的 18 项 reward 分量 + 跌倒率
按 rollout 平均写进 tensorboard，方便和聚合指标交叉验证。
"""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class RewardComponentCallback(BaseCallback):
    """读取 `info["r_*"]` 与 `dones` / `TimeLimit.truncated`，按 rollout dump 平均值。"""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._comp_sums: dict[str, float] = {}
        self._step_count: int = 0
        self._n_terminated: int = 0
        self._n_truncated: int = 0

    def _reset_buffers(self) -> None:
        self._comp_sums = {}
        self._step_count = 0
        self._n_terminated = 0
        self._n_truncated = 0

    def _on_rollout_start(self) -> None:
        self._reset_buffers()

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]
        for i, info in enumerate(infos):
            for k, v in info.items():
                if k.startswith("r_"):
                    self._comp_sums[k] = self._comp_sums.get(k, 0.0) + float(v)
            self._step_count += 1
            if dones[i]:
                if info.get("TimeLimit.truncated", False):
                    self._n_truncated += 1
                else:
                    self._n_terminated += 1
        return True

    def _on_rollout_end(self) -> None:
        n = max(self._step_count, 1)
        for k, total in self._comp_sums.items():
            self.logger.record(f"reward_components/{k}_mean", total / n)
        n_done = self._n_terminated + self._n_truncated
        if n_done > 0:
            self.logger.record("episodes/fall_rate", self._n_terminated / n_done)
            self.logger.record("episodes/n_finished", n_done)
