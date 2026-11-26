# 在线仿真模型库

[English](README.en.md) · [Italiano](README.it.md)

入口为 `/simulation`，中、英、意界面共享同一模型目录。**自由操纵**把键盘和触屏方向按钮转换为运动指令；**策略展示**自动发送模型的 `demoCommand`。物理与策略均在浏览器 Worker 中运行，没有远程训练服务。

## 已接入

| 模型 | 控制方式 | 操纵范围 |
| --- | --- | --- |
| Unitree Go2 | 仓库中的 PPO v0 ONNX 权重，50 Hz 推理 | 前进、后退、转向、横移、外力推撞 |
| Unitree G1 (12-DoF) | `unitree_rl_gym` 下肢 LSTM ONNX 权重，50 Hz 推理 | 前进、后退、转向、横移、外力推撞 |
| Unitree H1 (10-DoF) | `unitree_rl_gym` 全尺寸人形 LSTM ONNX 权重，50 Hz 推理 | 前进、后退、转向、横移、外力推撞 |
| Unitree H1-2 (12-DoF) | `unitree_rl_gym` 全尺寸人形 LSTM ONNX 权重，50 Hz 推理 | 前进、后退、转向、横移、外力推撞 |
| Pupper v3 | 项目 800 万步 PPO checkpoint，50 Hz 推理 | 前进、后退、转向、横移、外力推撞 |
| MicroDuck 小黄鸭 | 14-DoF PPO 行走 ONNX + XL330 BAM M6 电机模型，50 Hz 推理 | 前进、后退、转向、横移、外力推撞 |
| Petoi Quaddle | 原有周期步态控制器，无训练权重 | 前进、行走转向、停止 |

Go2 按训练顺序 FL / FR / RL / RR 构造 47 维观测，包含机身角速度、投影重力、速度指令、步态相位、关节状态与上次动作。归一化在 ONNX 图中。策略动作经目标关节角和 PD 转为扭矩，使用名称映射到模型的执行器顺序。停止是零速度指令，暂停才会冻结物理。

Go2 模型来自原项目的 `scene_go2.xml`，保留惯性、关节、执行器和碰撞几何；构建时去掉需要另行下载的可视网格，以碰撞几何显示简化外观。模型来源、权重哈希和许可证随构建发布。该网页验证不代表真机或训练效果验证。

Unitree G1（12-DoF）、Unitree H1（10-DoF，1.8 m 全尺寸）与 Unitree H1-2（12-DoF，含双轴踝关节）均使用来自 `unitree_rl_gym`（BSD 3-Clause）的下肢循环神经网络（LSTM）策略，导出为无状态 ONNX 图（G1/H1-2 为 47 维观测、H1 为 41 维观测，隐状态 `h_in / c_in` 各 64 维，50 Hz 推理，2 ms 物理步长，与 TorchScript 误差 `< 1.5e-6`）。三款人形的视觉 STL 网格均做了轻量化减面（G1：`24.0 → 1.49 MB`；H1：`14.15 → 1.02 MB`；H1-2：`39.37 → 1.66 MB`），同时保留原惯性参数、关节限位与足底接触点。对于接入 RL 策略的模型，可使用工具栏**外力推撞**按钮（或画面获焦后按 `P`）施加瞬时水平速度冲击，测试闭环抗扰恢复。

## Pupper v3 与小黄鸭

Pupper v3 使用 `cs123/6.rl_pupper_v2_5` 的 800 万步 PPO checkpoint，保留原始网格。浏览器适配器按 FR / FL / RR / RL 构造 53 维观测，采用 4 ms 物理步长、20 ms 控制周期、0.5 动作缩放与 EMA 滤波、KP=5 / KD=0.25，复现训练环境的坐标变换与 trot 相位。零指令仍会有少量漂移。

导出的 ONNX、配置、原生对照数据与来源哈希位于实践项目的 `web/`。需要重新导出时，在包含 SB3 / PyTorch 的 cs123 环境安装 `onnx==1.23.2` 和 `onnxruntime==1.31.1`，从仓库根目录运行：

```bash
codes/practices/quadruped/cs123/.venv/bin/python scripts/export-pupper-web.py
```

脚本逐层导出确定性 actor，并比较 256 组输入的 SB3 与 ONNX 动作；构建直接使用保存的导出，无需 Python 或训练环境。

MicroDuck 使用 `scene_walk.xml` 原始网格与项目内训练并导出的 61 维观测 PPO 行走策略（`web/policy.onnx`，归一化已烘焙进计算图，与 PyTorch 误差 `< 1e-6`）。浏览器适配器复现训练时的 **XL330 BAM M6 直流电机电压/反电动势/电流限幅模型** 与 **4 步（20 ms）指令延迟**（5 ms 物理步长、50 Hz 策略控制），支持前进、后退、横移、原地转向、松键稳停与外力推撞恢复。原项目代码许可证和网格授权说明随资源保留。

## 添加训练好的模型

1. 把可分发的模型、资源、权重、推理配置及许可证放入对应实践项目。先导出浏览器可用的 ONNX；不能直接运行 `.pt`、`.pth` 或 SB3 `.zip`。
2. 在 `models.json` 注册唯一 ID、名称、`kind`、`adapter`、资源路径、跟踪机身、相机距离、展示指令与教程入口。同一 Go2 合约的其他权重可以复用适配器。
3. 对不同机器人或策略，在 `runtime/` 增加适配器，并在 `worker.mjs` 与 `scripts/simulation-assets.cjs` 中接入。必须明确观测顺序与坐标系、历史状态、归一化、动作缩放、关节顺序、控制周期、初始化和重置。ONNX 文件本身不足以确定这些信息。当前推理适配器支持 Go2 与 Pupper 的已知合约，并非通用权重上传器。
4. 同步中、英、意的模型说明。只改源文件，`static/simulation-assets/` 和 `static/quaddle/` 由构建生成；不在构建时下载未知版本模型或执行用户脚本。
5. 为新适配器增加真实物理与策略检查，覆盖运动、停止、重置及异常输出，再运行下列检查。

```bash
npm run check:simulation
npm run check:quaddle-web
npm run typecheck
npm run assets:webp:check
npm run build
npm run check:i18n
npm run check:page-markdown
```

浏览器验收应覆盖键盘与触屏、模式/模型切换、松键、窗口失焦、页面隐藏、暂停/重置、重复加载、加载失败重试、窄屏和深浅色。自由操纵仅在画面获得焦点后接收快捷键；离开时清除指令。低性能设备允许降低仿真速度，不追赶后台积欠时间。
