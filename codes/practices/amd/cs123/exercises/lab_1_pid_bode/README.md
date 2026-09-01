# Lab 1：Pupper 腿踢脚——HFE 正弦追踪

教程 [§1.6.2](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/pid-control#162-单摆-pd) 让一根抽象单摆稳到一个固定角度，[§1.6.4](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/pid-control#164-实验扰动误差) 又加了一个常值外扰。两件事在脑子里都讲清楚了，但你手里还没"自己的狗"。

本 Lab 把单摆换成 **Pupper 真腿的 HFE 关节**——`HAA` 和 `KFE` 用 `<equality>` 锁在 0，整条 thigh + calf 以 HFE 为支点，像人抬腿一样前后摆。再把"稳到固定角度"换成"追一条 $q_d(t) = 0.3\sin(2\pi f t)$ 的正弦线"。扫七个频率，你会亲手画出这条 PD 回路的 Bode 图——不靠 `scipy.signal`，全靠仿真数据。

## 为什么是这件事

| 候选任务 | 教程已覆盖？ | 出片程度 | 留下的产物 |
|---|---|---|---|
| Bang-bang 对比 P | [§1.2](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/pid-control#12-控制系统) 讲了，没做实验 | 低 | 一段抖动视频 |
| 加采样延迟 | [§1.1.2](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/pid-control#112-执行器分类) 讲了 gap，没做实验 | 中 | 延迟 → 欠阻尼曲线 |
| **正弦追踪 + Bode 图** | 完全没做 | 高 | **2 段 GIF + 1 张图** |

正弦追踪赢在三件事：视觉最直观（红色目标 / 蓝色实际两条线同屏动）、图表最经典（学生第一次亲手画 Bode）、直接铺垫 [§3](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/inverse-kinematics)/[§5](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/gait-control)（画圆和踏步本质都是"跟踪一条时变轨迹"，正弦是最简形式）。

## 你交什么

`make_artifacts.py` 跑完后，`portfolio/` 里会有三件东西：

- `deliverable.png` = `bode_pupper_hfe.png`：7 个频点的幅值（dB）和相位（°），标出 −3 dB 带宽
- `deliverable.gif` = `hfe_sine_0p3hz.gif`：12 秒 / 640×400 / 15 fps，HFE 在 0.3 Hz 下几乎没可见滞后，动作更适合观察
- `notes.md`：50–100 字反思

## 起点

打开 `starter.py`：仿真主循环、MuJoCo 渲染、`imageio` 写 GIF、Bode 拟合（最小二乘 sin/cos）都已经写好。算法核心留了三处 `# TODO:`：

| TODO | 对应任务 | 干什么 |
|---|---|---|
| TODO 1 | 任务 B | 用 `mujoco.mj_fullM` 展开 `data.qM`，取 HFE 对角项 = 反射惯量 $I_\text{hfe}$ |
| TODO 2 | 任务 B | 套 $K_d = 2\zeta\sqrt{K_p\,I}$、$\zeta=0.7$ 自动算 $K_d$ |
| TODO 3 | 任务 A/C/D | 写 PD 控制律 $\tau = K_p(q_d - q) + K_d(\dot q_d - \dot q)$ |

填完这三处就能直接跑出 Bode 图和 GIF。本目录里不写 `solution.py`。

## 分步任务

1. **任务 A · 常值目标（20 min）**：先跑 `q_des = 0.3`，PD 把 HFE 稳到约 17°，确认控制器活着——对应教程 [§1.6.2](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/pid-control#162-单摆-pd) 的直觉。
2. **任务 B · 反射惯量 + 自动调 $K_d$（20 min）**：徒手 $mL^2/3$ 不准，混合连杆组合公式不干净；用 `mj_fullM` 读，再代二阶系统经验公式（[§1.5.4](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/pid-control#154-pd-调参)）。
3. **任务 C · 低频正弦 GIF（30 min）**：把目标换成 $q_d(t) = 0.3\sin(2\pi f t)$，用 0.3 Hz 跑 12 秒导出作品集 GIF；0.5 Hz 仍会在任务 D 的 Bode 扫频里测到。
4. **任务 D · 扫频 + Bode（40 min）**：固定 $K_p$，在 $f \in \{0.2, 0.5, 1, 2, 3, 5, 8\}$ Hz 各跑 5 秒，记录稳态幅值和相位差。横轴 $\log f$，左纵轴 $20\log_{10}(A_\text{out}/A_\text{in})$，右纵轴 $\Delta\phi$，标出 −3 dB 带宽。
5. **任务 E · 刚度对比（Stretch，20 min）**：$K_p \in \{K_p^*,\ 2K_p^*,\ 5K_p^*\}$ 三组叠到同一张图——刚度越高带宽越高，但更容易振荡。`starter.py` 里只留了 TODO 注释，不强制完成。

## MuJoCo 场景

`models/scene.xml` 通过 `<include>` 引用 `shared/models/leg.xml`（语义名 `HAA / HFE / KFE`），再用 `<equality>` 把 `HAA` 和 `KFE` 锁到 0：

```xml
<include file="../../shared/models/leg.xml"/>
<material name="lab1_grid" texture="lab1_grid" texrepeat="1 1"/>
<geom name="floor" type="plane" size="0.7 0.7 0.02" material="lab1_grid"/>
<equality>
  <joint joint1="HAA" polycoef="0 0 0 0 0"/>
  <joint joint1="KFE" polycoef="0 0 0 0 0"/>
</equality>
```

Lab 2 的升级路径很简单：**删掉这块 `<equality>`** 就解锁成 3 DoF 全腿——同一份 MJCF 一路走到 Lab 4。

## 评分点

- 任务 A 末段满足 $|q - 0.3| < 0.02$ rad（PD 把腿稳住了）
- 任务 D 在 0.2 Hz 增益 $|G| > 0.95$（低频几乎透传）
- 任务 D 在 8 Hz 增益 $|G| < 0.5$（高频明显衰减）
- Bode 图轴标签、频点和 −3 dB 带宽都标出来
- GIF 能一眼看出目标线和 HFE 实测线的差距

## 常见坑

1. $K_p$ 调到很大就发散——大概率不是 PD 错了，是 `timestep` 太粗。本 Lab 用 5 ms，安全档。
2. $K_d$ 不是 MJCF 里的 `damping`。前者是控制律里的阻尼项，后者是物理摩擦——两者不能互替。
3. 力矩饱和：本 Lab `ctrlrange=[-3, 3] N·m`，扫频到高频时会先撞限幅再撞带宽，画图前看一眼 `torque_log` 是否经常顶到 ±3。

## 与教程的衔接

- **复用**：[§1.6.2](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/pid-control#162-单摆-pd) 的脚本骨架、[§1.5.4](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/pid-control#154-pd-调参) 的 $\omega_n / \zeta$ 公式
- **扩展**：
  - [§1.1.3](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/pid-control#113-齿轮箱) 引入了"反射惯量"概念——本 Lab 第一次用 `mj_fullM` 把它读出来
  - 教程只做 step response（[§1.6.2](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/pid-control#162-单摆-pd) / [§1.6.3](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/pid-control#163-实验增益响应)）；本 Lab 第一次做 frequency response
  - 教程用抽象 `pendulum.xml`；本 Lab 第一次把 PD 接到 Pupper 真腿
- **不重复**：[§1.6.3](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/pid-control#163-实验增益响应) 的 $K_p / K_d$ 三组对照、[§1.6.4](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/pid-control#164-实验扰动误差) 的稳态误差实验，作为本 Lab 的前置练习，不再重做

## 运行

`exercises/` 是独立的 uv 项目，命令都从 `exercises/` 目录里跑：

```bash
uv run python lab_1_pid_bode/starter_todo.py    # 学生起点：会在 TODO 1 处报 NotImplementedError
uv run python lab_1_pid_bode/starter.py         # 参考答案：直接跑通，打印 I_hfe / Kp / Kd 与任务 A 误差
uv run python lab_1_pid_bode/make_artifacts.py  # 写出 portfolio/ 下三件交付物
uv run python lab_1_pid_bode/tests.py           # 三条数值 assert 全过
```
