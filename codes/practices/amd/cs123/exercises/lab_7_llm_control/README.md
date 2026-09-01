# Lab 7：Pupper 的自然语言简历 (Pupper's Natural-Language Resume)

教程 [§7.7.1](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/llm-control#771-文本输入版) 的 REPL 能跑两条命令，但作品集里需要更有说服力的展示——**5 张难度递增的任务卡 GIF + 一份消息轨迹 markdown**，从"停下"到"走到坐标 (2,1)"再到"以 3 m/s 飞奔"（LLM 自动降速重试），完整展示 agent 的理解、规划和容错能力。

Lab 5 的上游 RTNeural policy（test）在这里第一次被当作"工具"调用——LLM 说"往前走 1 米"，底层就是 50 Hz policy / 500 Hz physics 的同一套循环在跑。

## 为什么选这个任务

| 候选 | 教程覆盖 | 门槛 | 作品价值 |
|---|---|---|---|
| 文本 REPL 录屏 | [§7.7.1](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/llm-control#771-文本输入版) 已覆盖 | 低 | 低（截图） |
| 语音 Whisper 接入 | [§7.7.2](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/llm-control#772-语音版) | 中（音频依赖） | 中 |
| **5 级任务展柜 GIF** | ❌ | 低（API key） | **高（5 GIF + 轨迹）** |
| VLM 视觉闭环 | 留给 [§8](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/perception) | 高 | 高 |

5 级任务展柜赢在：它把 agent 的能力具象化为可视的 GIF，每一级考察不同的 LLM 能力（单步 / 参数转换 / 多步 / ReAct / 容错），比 REPL 截图有说服力得多。

## 你交什么

- `portfolio/agent_L1_stop.gif` ~ `agent_L5_retry.gif`：5 张任务卡 GIF，640×480，12 fps
- `portfolio/agent_traces.md`：消息轨迹表（自动生成）
- `portfolio/notes.md`：50–100 字反思

## 起点与 TODO map

教师版 `starter.py` 已经写好 GIF 渲染管线、任务定义和消息轨迹导出。学生补三处 TODO。

| TODO | task | what to write |
|---|---|---|
| TODO 1 | `dispatch()` | 实现 walk / stop / get_pose / wait 四个工具的 dispatch，含参数校验和软失败 |
| TODO 2 | `run_agent()` | 实现 tool_use 循环：发消息 → 检查 finish_reason → 执行 tool_calls → 组装 tool result → 喂回去 |
| TODO 3 | `render_task_gif()` | 实现单个任务的 GIF 录制：reset → 运行 agent（每步采帧）→ 加字幕 → 返回帧列表 |

## 分步任务

1. **工具 dispatch（30 min）**：读懂 `tools/robot_tools.py` 的 TOOLS schema，补 `dispatch()`。（对应教程 [§7.2](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/llm-control#72-工具-api) / [§7.5](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/llm-control#75-失败处理)）
2. **Agent 循环（30 min）**：读懂教程 [§7.3](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/llm-control#73-function-calling) 的 50 行 `run_agent`，补 `run_agent()`。（对应教程 [§7.3](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/llm-control#73-function-calling)）
3. **GIF 录制（30 min）**：补 `render_task_gif()`，理解 frame_callback 采帧机制。（对应教程 [§7.6](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/llm-control#76-接入策略) / [§7.7](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/llm-control#77-demo-录制)）
4. **跑 5 级任务（20 min）**：`uv run python lab_7_llm_control/eval_tasks.py`，观察 LLM 行为。
5. **反思（10 min）**：填 `portfolio/notes.md`。

## MuJoCo scene

复用 Lab 5 的上游 RTNeural policy（test），50 Hz policy / 500 Hz physics。不另起 MJCF。

```python
from tools.robot_tools import RobotState
robot = RobotState(policy_name="test")
robot.reset()
```

## Rubric

- `TOOLS` 列表有 4 个工具（walk / stop / get_pose / wait），每个有 type=function + function.name / description / parameters
- `dispatch("walk", {"vx": 2.0, ...})` 返回 `{"ok": False, "error": "vx out of range ..."}`，不 raise
- `dispatch("walk", {"vx": 0.3, "wz": 0.0, "duration": 1.0})` 返回 `{"ok": True, "final_pose": [...]}`
- `dispatch("get_pose", {})` 返回 `{"ok": True, "pose": [x, y, yaw]}`，三个值都 finite
- `dispatch("unknown_tool", {})` 返回 `{"ok": False, "error": "unknown tool: ..."}`
- 5 张 GIF 视觉判据：L1 机器人先走再停、L2 向前移动、L3 先前进再转弯、L4 多次调整、L5 先报错再降速

## 常见坑

- **API key 配置**：复制 `.env.example` 为 `.env`，填入 API key。默认用 DeepSeek（`OPENAI_BASE_URL=https://api.deepseek.com`），也支持 OpenAI / Ollama 等任何 OpenAI-compatible 接口。tests.py 不需要 API key，但 eval_tasks.py / make_artifacts.py 需要。
- **LLM 随机性**：同一条指令每次跑 LLM 的回复可能不同，GIF 内容会有差异，这是正常的。
- **dispatch 不要 raise**：这是整个 Lab 的核心设计——软失败让 LLM 能收到错误信息并自动重试。
- **不要每次 walk 都 reset**：连续多步控制时保持物理状态，否则机器人会瞬移回原点。
- **macOS 上 offscreen 渲染**：如果遇到 OpenGL 问题，确保 MuJoCo 版本 ≥ 3.1。

## 教程衔接

- **复用**：教程 [§7.2](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/llm-control#72-工具-api) 工具 schema、[§7.3](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/llm-control#73-function-calling) agent 循环、[§7.5](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/llm-control#75-失败处理) 软失败模式、[§7.6](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/llm-control#76-接入策略) 策略集成。
- **扩展**：把"跑两条命令"扩展为 5 级难度递增的任务展柜，加 GIF 录制和消息轨迹导出。
- **不重复**：教程 [§7.7.1](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/llm-control#771-文本输入版) REPL demo 作为前置练习。

## 不做什么

- 不做 [§7.7.2](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/llm-control#772-语音版) 语音 Whisper 接入（Stretch 留给学生自己加）
- 不做 VLM / VLA 视觉闭环（留给 [§8](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/perception) / Lab 8）
- 不做真机 / 真相机
- 不开 mujoco viewer 录 GIF（统一 offscreen）
- 不做 LangChain / LlamaIndex 框架集成（50 行原生循环就够）
- 不训练新策略（直接用 Lab 5 的上游 RTNeural policy）

## Run

从 `exercises/` 运行。**首次运行前先拉取上游 RTNeural policy**（约 38 MB，已 `.gitignore`，仅需一次）：

```bash
bash shared/rl/fetch_policies.sh                     # 下载 test_policy.json 到 shared/rl/policies/
```

然后：

```bash
uv run python lab_7_llm_control/tests.py             # 7 条断言（不需要 API key）
uv run python lab_7_llm_control/starter_todo.py      # 学生起点：调用 dispatch 时会触发 TODO 1 NotImplementedError
uv run python lab_7_llm_control/starter.py            # 参考答案 / 快速环境检查（不需要 API key）
uv run python lab_7_llm_control/eval_tasks.py         # 跑 L1–L5 + 录 GIF（需要 API key）
uv run python lab_7_llm_control/make_artifacts.py     # 一键串（需要 API key）
```
