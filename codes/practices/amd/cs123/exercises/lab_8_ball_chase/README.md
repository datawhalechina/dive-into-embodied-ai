# Lab 8：追球 Demo（Ball-Chasing Demo）

到 [§8](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/perception) 为止，RL 的"腿"（Lab 5/6）、LLM 的"脑"（Lab 7）、相机的"眼"（[§8](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/perception) 教程）都做好了——但从来没有让它们在同一段视频里同时工作过。Lab 8 把视觉、跟踪、RL 步态拧到一起：Pupper 看到红球、走过去、在球前停下，20 秒追球视频做"毕业汇演"。

## 为什么选这个任务

| 候选 | 教程覆盖 | 门槛 | 作品价值 |
|---|---|---|---|
| 原地转头追颜色球 | [§8.6](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/perception#86-原地追球) 已覆盖 | 低 | 低（只转头） |
| YOLO 检测 + 追踪 | [§8.3](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/perception#83-目标检测) 方案 B | 中（GPU） | 中 |
| **HSV 检测 + RL 步态追球** | ❌ | 低（纯 CPU） | **高（毕业视频）** |
| VLM 视觉闭环 | [§8.8](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/perception#88-接入语言-agent) Stretch | 高（API） | 高 |

HSV + RL 追球赢在：它是整门课唯一一段"前面所有 Lab 的资产同时上场"的视频，不需要 GPU 也不需要 API key，纯 CPU 就能跑。

## 你交什么

- `portfolio/ball_chase.gif`：追球毕业视频（课程 cover image）
- `portfolio/tracking_error.png`：追踪误差曲线（e_yaw vs 时间 + FSM 状态色条）
- `portfolio/detection_vs_distance.png`：检测 vs 距离（bbox 高度 vs 球距离）
- `portfolio/notes.md`：50–100 字反思

## 起点与 TODO map

教师版 `starter.py` 已经写好 ChaseRobotState、GIF 录制管线、图表导出和球运动控制。学生补三处 TODO。

| TODO | 位置 | 要写什么 |
|---|---|---|
| TODO 1 | `perception/ball_detector.py` 的 `detect_red_ball()` | HSV 阈值检测：RGB → HSV → inRange（红色两段并集）→ findContours → 返回 bbox dict 或 None |
| TODO 2 | `perception/tracker_fsm.py` 的 `TrackerFSM.step()` + `visual_servo()` | P 控制器：`e_yaw = (cx - W/2) / (W/2)` → `wz = -KP_YAW * e_yaw`；FSM 三态切换 |
| TODO 3 | `starter.py` 的 `run_chase()` | 主循环组合：每步 camera → detect → fsm.step → (vx, wz) → robot._policy_tick → mj_step |

## 分步任务

1. **HSV 检测（30 min）**：补 `perception/ball_detector.py` 的 `detect_red_ball()`。（对应教程 [§8.3](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/perception#83-目标检测) 方案 A）
2. **P 控制器 + FSM（30 min）**：补 `perception/tracker_fsm.py` 的 `visual_servo()` 和 `TrackerFSM.step()`。（对应教程 [§8.4](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/perception#84-航向控制)–[§8.5](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/perception#85-跟踪状态机)）
3. **追球主循环（40 min）**：补 `starter.py` 的 `run_chase()`，理解 500 Hz physics / 50 Hz policy / 10 Hz vision 的三层频率。（对应教程 [§8.6](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/perception#86-原地追球)–[§8.7](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/perception#87-行走追球)）
4. **跑 demo（20 min）**：`uv run python lab_8_ball_chase/eval_chase.py`，观察追球行为。
5. **反思（10 min）**：填 `portfolio/notes.md`。

## MuJoCo scene

`models/pupper_chase.xml` = `pupper_v3_floating.xml` + 两个新增：

- `<camera name="head_cam">` 挂在 `base_link` 上（机器人头部第一视角）
- `<body name="ball" mocap="true">` 红色球体（Python 控制位置，每 3 秒随机移动）

物理系统严格复用 Lab 7 的 RobotState（test_policy.json + 50 Hz / 500 Hz）：

```python
from starter import ChaseRobotState
robot = ChaseRobotState(policy_name="test")
robot.reset()
```

## Rubric

- `detect_red_ball(红色块图像)` 返回 bbox dict，`detect_red_ball(蓝色图像)` 返回 None
- `visual_servo(中心 box)` → wz ≈ 0；`visual_servo(左侧 box)` → wz > 0
- FSM：检测到球 → TRACKING；丢失超时 → SEARCHING；长时间丢失 → STOPPED
- `ChaseRobotState` 加载 pupper_chase.xml，走 1 秒不摔
- `head_cam` 渲染出 (240, 320, 3) 图像
- 追球 GIF 中机器人能朝球方向移动

## 常见坑

- **HSV 红色两段**：红色在 HSV 横跨 0 度，需要 H∈[0,10] 和 H∈[170,180] 两段取并集，漏掉一段会丢一半红色。
- **P 控制器震荡**：KP_YAW 太大会左右摇摆，太小会追不上。默认 1.0 是个好起点。
- **视觉采样频率**：10 Hz 够了。每个物理步都跑视觉（500 Hz）会非常慢且没必要。
- **不要每次 walk 都 reset**：连续追球时保持物理状态，否则机器人会瞬移回原点。
- **is_fallen 阈值**：test policy 步态自然下沉到 z≈0.10，阈值设 0.05 不会误判。

## 教程衔接

- **复用**：教程 [§8.3](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/perception#83-目标检测) HSV 检测、[§8.4](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/perception#84-航向控制) P 控制器、[§8.5](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/perception#85-跟踪状态机) TrackerFSM。
- **扩展**：把"原地转头"扩展为"边走边追"，加 RL 步态闭环和 GIF 录制。
- **不重复**：教程 [§8.6](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/perception#86-原地追球) 原地追踪作为前置练习。

## 不做什么

- 不做 [§8.3](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/perception#83-目标检测) 方案 B（YOLOv8-nano，需要 GPU）
- 不做 [§8.8](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/perception#88-接入语言-agent) VLM/LLM 视觉接入（Stretch 留给学生）
- 不做真机 / 真相机
- 不开 mujoco viewer 录 GIF（统一 offscreen）
- 不训练新策略（直接用 Lab 5 的上游 RTNeural policy）

## Run

从 `exercises/` 运行。**首次运行前先拉取上游 RTNeural policy**（约 38 MB，已 `.gitignore`，仅需一次）：

```bash
bash shared/rl/fetch_policies.sh                    # 下载 test_policy.json 到 shared/rl/policies/
```

然后：

```bash
uv run python lab_8_ball_chase/tests.py             # 7 条断言（不需要 GPU / API key）
uv run python lab_8_ball_chase/starter_todo.py      # 学生起点：调用 detect_red_ball / TrackerFSM 时会触发 NotImplementedError
uv run python lab_8_ball_chase/starter.py           # 参考答案 / 快速环境检查
uv run python lab_8_ball_chase/eval_chase.py        # 跑追球 demo + 录 GIF
uv run python lab_8_ball_chase/make_artifacts.py    # 一键串
```
