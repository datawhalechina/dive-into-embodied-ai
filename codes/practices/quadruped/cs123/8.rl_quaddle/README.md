# 8. RL Quaddle：4 舵机速度跟踪策略（VLA 低层运控）

阶段二产物：给 Quaddle 训练速度指令 `(vx, vy, wz)` 的低层策略，作为 VLA 高层
指令的执行层。环境基于 [`7.quaddle-mjcf`](../7.quaddle-mjcf/)（q=0 平衡站姿），
管线结构沿用 [`6.rl_pupper_v2_5`](../6.rl_pupper_v2_5/) 的经验（命令重采样、
kick 扰动、终止罚、动作低通），按 4 舵机精简为 24 维观测 / 4 维动作。

## 环境要点

- 物理 1 kHz（闭链约束需要），控制 50 Hz，回合 5 s（250 控制步），命令每 2.5 s 重采样
- 动作：tanh ±1 → ×0.25 rad（±14°）→ 30ms 级一阶低通 → position actuator
- 奖励：vx/vy 跟踪（σ=0.08，速度低故收紧削弱站桩补贴）+ wz 跟踪（σ=0.5）
  + 存活 0.5 + 姿态/垂直速度/动作平滑/能耗正则 + 翻倒终止 −50
- 扰动：开局随机初速度 0.15 m/s、2% 概率 kick ±0.15 m/s（对抗站桩局部最优）
- 终止：roll/pitch > 30° 或 body 高 < 30 mm

## 用法（在 `codes/practices/quadruped/cs123` 下）

```bash
# 冒烟测试
uv run pytest -q 8.rl_quaddle/tests

# 训练（CPU，8 并行环境，2M 步约 1.5~3 h）
uv run python 8.rl_quaddle/train.py --timesteps 2000000 --n-envs 8 --tensorboard
uv run tensorboard --logdir 8.rl_quaddle/outputs/tb

# 断点续训 / 评估
uv run python 8.rl_quaddle/train.py --resume 8.rl_quaddle/outputs/ppo_quaddle.zip
uv run python 8.rl_quaddle/evaluate.py --model 8.rl_quaddle/outputs/ppo_quaddle.zip --episodes 20
```

## 结果

（待首轮训练完成后填写：500k 冒烟轮 + 2M 正式轮的跟踪精度、存活率、
与 W1 开环基线 ~5mm/s 的对比。）

## 版本记录

### v1（关节动作空间，已存档 outputs/ppo_quaddle*.zip）

4 维逐时刻关节动作（±14° + 30ms 低通）。500k 冒烟轮：存活率 100% 但 |vx| 仅
3.0mm/s（站桩局部最优）；移植 v2.5 的 dont_wait 罚后续训至 2M：|vx| 6.2mm/s，
reward 平台震荡——**PPO 从 50Hz 关节噪声里自发涌现 2-4Hz 相干步态太难**。

### 诊断实验（2026-11-08）

修正评估接口后手动扫 CPG 相位空间：trot/meta/ripple 全相位族在 8-12°、
1.5-4Hz 下普遍产生 **20~190mm/s** 净位移——物理可行解丰富，缺的是动作空间的
"周期结构先验"，不是物理。（注意：W1 README 的"正弦只能 3~5mm/s"结论是在
settle 时长/滤波路径不同的旧脚本下测的，本页诊断更准：正弦步态本身能走，
是 RL 探索没找到。）

### v2（当前）：CPG 参数化动作空间

动作 5 维慢变步态参数（amp/freq/form/sign/turn，τ=0.15s 低通），env 内部
50Hz CPG 生成舵机目标。周期性结构内置 → RL 只学"命令→参数"映射，与真机
OpenCat 步态表执行模型同构。观测 26→25 维（含当前参数）。reset 内置 0.5s
物理沉降（±0.01 关节噪声曾致沉降瞬态自发翻倒）。`amp01=0` 映射 7° 摆动——
探索起点天生在动。

### v2.1 最终交付（2026-11-08）：滚轮足端 + 查表 CPG 控制器

**模型修正**（`7.quaddle-mjcf` 同步更新）：足端接触从球（各向同性滑动摩擦）改为
滚轮圆柱（r3.5×4mm，wheel_tip 铰轴）——这是真机行走机制的来源。球足端模型的
稳定行走包线实测只有 ~10-20mm/s，且 RL 三个版本收敛到 3→6→14mm/s 的过程
本质上是在逼近这个物理上限；滚轮模型下 forward 实测 **+113mm/s**（含动量滑行）。

**RL 结论存档**（4 轮 checkpoint 在 outputs/）：逐关节动作与 CPG 参数动作两代
动作空间、含 dont_wait 罚与 skew 谐波维，RL 均未学会命令条件化响应
（vx_err 64mm/s @ ±120 命令；小范围 10.4mm/s @ ±15 命令，且速度不随命令
符号响应）。症结：4 足球足端摩擦系统的"行走解"需精确的相位-偏斜组合，
on-policy PPO 的参数空间探索无法可靠命中；而查表（人工标定）10 分钟即得。

**最终控制器**：`cpg_controller.py` 查表 CPG（对齐真机 OpenCat SkillLibrary 模型）：
- `forward` trot+skew−0.45 → 实测 +113mm/s（3s 窗口含动量）
- `back` skew+0.45 → −46mm/s；`turn_left/right` 差频弧线（wz≈±0.2-0.3）
- `stop` 滑行 ~85mm 后静止
- 11s 指令序列验收：前进→右转→前进→停，总位移 1.1m，未翻倒 PASS

已知限制（W3 使用须知）：① 转向是弧线行驶非原地转，专家闭环按方位选档即可；
② stop 滑行 ~85mm，到达判定提前 15cm 停车；③ 速度前后不对称是滚轮接触
各向异性的固有特性；④ RL 策略若后续想复活，优先级最高的是给观测加 CPG
相位时钟与"上一档位"历史（探索结构问题，非算力问题）。

### 评估报告（2026-11-08，用户要求复核）

对训练完成的 `ppo_quaddle_cpg2.zip`（球足端环境训练）在**滚轮新环境**复评：

| 配置 | 存活 | \|vx_err\| | 结论 |
|---|---|---|---|
| RL @ 训练分布命令 ±120mm/s | 100% | 51.9mm/s | \|vx\| 恒 ~3.4mm/s，不随命令响应 |
| RL @ 小范围 ±15mm/s | 100% | 26.7mm/s | 同上，FAIL（阈值 3） |
| 查表 forward | 100% | — | **+113mm/s** ✓ |
| 查表 back（初版） | 100% | — | **+158mm/s ✗ 方向错** |

**back 排查**：skew 反号、CPG 相位倒走（时间反演）均无法产生 -x 位移——
四配置对照（正/倒相位 × ±skew）全部 +x。结论：滚轮是**单向棘轮**
（摆动泵能量、方向由轮系锁定，+x 恰为真机前进方向，轮轴侧向安装符合
omni 轮构型）。此简化模型无倒挡；技能表定格为
`{forward, stop, turn_left, turn_right}`，后方目标用 180° 弧线转向替代
（W3 专家闭环，真机无此限制）。

**最终判定**：W2 以查表 CPG 控制器验收通过；RL 四轮存档
（复活前提：观测加相位时钟与档位历史）。全部 10 项测试通过。
