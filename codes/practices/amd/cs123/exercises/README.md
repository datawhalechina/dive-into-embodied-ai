# CS123 Lab

这里集中放置各章配套 Lab。章节正文里的"演示"是作者带你跑代码；章末的
`Lab` 是你自己写、自己跑、最后放进作品集的任务。

## Lab 总览

| 章 | Lab | 作品集产物 |
|---|---|---|
| Ch1 PID | `lab_1_pid_bode` | `bode_pupper_hfe.png` + HFE 正弦踢腿 GIF |
| Ch2 FK | `lab_2_fk_teleop` | `leg_teleop.gif` + 单腿工作空间图 |
| Ch3 IK | `lab_3_stepping` | `leg_stepping.gif` |
| Ch4 URDF | `lab_4_urdf_surgery` | `pupper_zoo.gif` + PD 调参图 |
| Ch6 RL | `lab_6_rl_pupper` | `rl_pupper_commands.gif` + PPO checkpoint |
| Ch7 LLM | `lab_7_llm_control` | 5 张任务卡 GIF + 消息轨迹 |
| Ch8 视觉 | `lab_8_ball_chase` | `ball_chase.gif` |

## 命名约定

- **演示**: 章节正文里的完整代码和现象，作者带你看，照跑即可。
- **思考**: 不写代码的概念题，一段话回答。
- **Lab**: 章末延伸任务，你自己写、自己跑、有产出。
- **作品集**: 各 Lab 的成果合订本，课程结束后留档。

## 共享资产

`shared/` 是跨 Lab 资源树：

- `shared/models/leg.xml`: 语义关节名 `HAA/HFE/KFE` 的 Pupper 单腿
- `shared/models/meshes/`: Pupper V3 STL 的唯一副本
- `shared/controllers/`: 控制器 dataclass 和数值工具
- `shared/viz/`: GIF 与绘图工具

各 Lab 用相对路径 include/import，不要在自己的 `models/` 下复制 mesh。

## 预置策略（Lab 6/7/8 必做）

Lab 6/7/8 都要加载上游 CS123 的 RTNeural 预置策略 `test_policy.json`（约 38 MB）。
该文件不入库，首次跑这三个 Lab 之前在 `exercises/` 下执行一次：

```bash
bash shared/rl/fetch_policies.sh
```

脚本逻辑：优先用 `tmp/lab_source/lab_5_fall_2025/` 下的本地副本，否则从
`https://raw.githubusercontent.com/cs123-stanford/lab_5_fall_2025/main/test_policy.json`
下载；已存在则跳过（`FORCE=1` 强制重拉）。落盘路径
`shared/rl/policies/test_policy.json` 已被 `.gitignore` 忽略，不会进仓库。
