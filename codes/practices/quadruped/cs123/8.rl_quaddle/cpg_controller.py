"""Quaddle 查表 CPG 控制器（W2 交付物，W3 数据采集的专家执行层）。

设计对齐真机 OpenCat：离散技能档位（前进/后退/左转/右转/停），
每档是一组每腿 CPG 参数 (amp_deg, freq, phase0, skew)，相位钟全局连续，
切档时幅值 0.3s 斜坡防冲击。速度幅值取自实测标定（滚轮足端模型，
2026-11-08，见 scripts/wz_table_roller.json 与 README 版本记录）：

    forward    trot (0,π/2,0,π/2)  10.5° 2.5Hz skew−0.45  → vx ≈ +21 mm/s
    back       同上 skew+0.45                               → vx ≈ −46 mm/s
    turn_left  左腿 1.5Hz / 右腿 2.5Hz（差频）               → wz ≈ +0.33 rad/s
    turn_right 左右互换                                      → wz ≈ −0.33 rad/s
    stop       amp 0                                        → 静止站姿

接口：
    ctrl = CpgController(env)
    ctrl.set_skill("forward")
    vx, wz = ctrl.step()      # 每控制步调用一次（内部 mj_step 20 substeps）

命令语义是"档位"而非连续速度——与真机 kwkF/ktrL 一致；W3 数据采集时
脚本化专家按目标方位选择档位，VLA 学习"观测+指令 → 档位序列"。
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

CTRL_LIMIT = 2.44  # rad
AMP_RAMP_S = 0.3   # 切档幅值斜坡

# 每腿参数：(amp_deg, freq_hz, phase0, skew)，腿序 lf, rf, rb, lb
# 方向说明：滚轮足端下 skew 反号不再反向（滚动方向偏好压过波形偏斜），
# back 依赖 DIR 里的相位钟倒走（CPG 时间反演 = 轮式倒挡）。
TROT_PHASES = (0.0, math.pi / 2, 0.0, math.pi / 2)
SKILLS = {
    # 注意：仿真模型的滚轮是"单向棘轮"——正/倒相位、skew 反号全部产生 +x 位移
    # （2026-11-08 四配置对照实测），故无 back 档；后方目标用 180° 弧线转向替代。
    # 真机无此限制（OpenCat 固件有 kbwF 倒退步态表）。
    "forward": [(10.5, 2.5, p, -0.45) for p in TROT_PHASES],
    # turn 档位为差频区间端点：执行时按 turn_strength ∈ (0,1] 在
    # 弱转(1.0↔2.5Hz)与强转(1.0↔3.0Hz)间插值——连续转向强度
    "turn_left_weak": [
        (10.5, 1.5, 0.0, -0.45), (10.5, 2.5, math.pi / 2, -0.45),
        (10.5, 1.5, 0.0, -0.45), (10.5, 2.5, math.pi / 2, -0.45),
    ],
    "turn_left_strong": [
        (10.5, 1.0, 0.0, -0.45), (10.5, 3.0, math.pi / 2, -0.45),
        (10.5, 1.0, 0.0, -0.45), (10.5, 3.0, math.pi / 2, -0.45),
    ],
    "turn_right_weak": [
        (10.5, 2.5, 0.0, -0.45), (10.5, 1.5, math.pi / 2, -0.45),
        (10.5, 2.5, 0.0, -0.45), (10.5, 1.5, math.pi / 2, -0.45),
    ],
    "turn_right_strong": [
        (10.5, 3.0, 0.0, -0.45), (10.5, 1.0, math.pi / 2, -0.45),
        (10.5, 3.0, 0.0, -0.45), (10.5, 1.0, math.pi / 2, -0.45),
    ],
    "stop": [(0.0, 2.5, 0.0, 0.0)] * 4,
}
# 相位钟步进方向（+1 正走 / -1 倒走）
DIR = {"forward": 1.0, "turn_left": 1.0, "turn_right": 1.0, "stop": 1.0}


class CpgController:
    def __init__(self, env, settle_s: float = 1.2):
        self.env = env
        self.model, self.data = env.model, env.data
        self.dt = env.dt
        self.phase = 0.0
        self._phases = np.zeros(4)     # 每腿独立相位钟（差频转向的前提）
        self.skill = "stop"
        self._dir = 1.0
        self._amp_scale = 0.0          # 切档斜坡 0→1
        self._last_meas = np.zeros(2)
        # 沉降：与 env.reset 相同的稳定初始
        env.reset(seed=0)
        self.data.ctrl[:] = 0.0
        for _ in range(int(settle_s / env.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)

    def set_turn_skill(self, direction: str, strength: float) -> None:
        """连续转向：direction 'left'/'right'，strength ∈ (0,1] 映射差频强度。"""
        s = max(0.15, min(1.0, strength))
        tier = "weak" if s < 0.6 else "strong"
        self.set_skill(f"turn_{direction}_{tier}")

    def set_skill(self, name: str) -> None:
        if name not in SKILLS:
            raise ValueError(f"未知技能 {name}，可选 {sorted(SKILLS)}")
        if name != self.skill:
            self._amp_scale = 0.0      # 触发斜坡
        self._dir = DIR.get(name, 1.0)
        self.skill = name

    def reset_state(self) -> None:
        """episode 复位：相位钟清零、幅值斜坡复位（物理状态由外部 reset）。"""
        self._phases[:] = 0.0
        self._amp_scale = 0.0
        self._dir = 1.0
        self.skill = "stop"

    def step(self):
        table = SKILLS[self.skill]
        self._amp_scale = min(1.0, self._amp_scale + self.dt / AMP_RAMP_S)
        for k, (amp_deg, freq, ph0, skew) in enumerate(table):
            # 每腿独立相位钟：差频转向（左右腿不同频率）依赖此实现
            self._phases[k] += self._dir * 2.0 * math.pi * freq * self.dt
            s = self._phases[k] + ph0
            target = math.radians(amp_deg * self._amp_scale) * (
                math.sin(s) + skew * math.sin(2.0 * s)
            )
            self.data.ctrl[k] = float(np.clip(target, -CTRL_LIMIT, CTRL_LIMIT))
        for _ in range(self.env.n_substeps):
            mujoco.mj_step(self.model, self.data)
        r = self.data.body("body").xmat.reshape(3, 3)
        v_body = r.T @ self.data.qvel[:3]
        vx = float(-v_body[1])   # body -Y 是前进方向（URDF 系），世界 +x
        wz = float((r.T @ self.data.qvel[3:6])[2])
        self._last_meas = np.array([vx, wz])
        return vx, wz

    @property
    def fallen(self) -> bool:
        return bool(self.data.body("body").xpos[2] < 0.030)
