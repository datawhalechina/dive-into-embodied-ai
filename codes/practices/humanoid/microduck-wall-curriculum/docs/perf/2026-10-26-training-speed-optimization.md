# MicroDuck 训练提速优化报告（100x 目标调研与实测）

日期：2026-10-26 · 分支：`perf/100x`（基线封存于 `baseline-pre-100x` / tag `baseline-b24-pre100x`）
硬件：RTX 4070 Ti 12GB · 栈：mjlab 1.3.0 + mujoco-warp 3.8.1 + rsl_rl PPO · 配置：B24（5cm 墙全链）

## 一、基线画像（全部实测）

单轮训练：8000 iter × 1.32s ≈ 2.93h；吞吐 37k steps/s（2048 envs × 24 steps）。

| 环节 | 耗时 | 占比 | 性质 |
|---|---|---|---|
| rsl_rl collection（总） | 50.8 ms/step | 100% | |
| ├─ env.step | 33.4 ms/step | 66% | |
| │  ├─ 物理求解（4 substep，CUDA Graph 已启用） | 20.8 ms | 55-70% | **真计算**（接触丰富） |
| │  ├─ reward（41 项全量算） | ~6.7 ms | ~18% | 半 glue |
| │  └─ write/forward/obs/events | ~5 ms | ~15% | glue |
| └─ rsl_rl rollout glue（actor前向+buffer拷贝+TensorDict） | 17.4 ms/step | 34% | **纯 overhead** |
| PPO learning | 0.101 s/iter | 8% | 非瓶颈 |

## 二、关键实验与结论

| # | 实验 | 结果 | 结论 |
|---|---|---|---|
| 1 | envs 缩放 512/1024/2048 | 22.7/25.9/35.9 ms/step | 时间≈17ms 固定 + 9ms/1024envs，12GB 卡死在 2048 |
| 2 | mujoco-warp testspeed（裸 XML） | 378万~1137万 steps/s | ❌ **无效对照**：testspeed XML 无地板，零接触自由落体，solver 空转 |
| 3 | njmax 1500→128 | env.step 37.2→29.4ms（**-21%**） | ✅ 立即可用，零风险（实际约束数远低于 1500） |
| 4 | contact_sensor_maxmatch 64→16 | 无变化 | ❌ 非瓶颈 |
| 5 | CUDA Graph 检查 | mjlab 默认已开启（driver 560≥12.4） | 已是优化态，无免费收益 |
| 6 | sim.step 干净计时 | 5.2ms/substep | 接触丰富物理求解为真实计算 |

**核心结论：物理求解（20.8ms/step）是真计算而非 overhead。100x 单卡不可达。**

## 三、最终优化方案与预期

| 措施 | 倍数 | 状态 | 说明 |
|---|---|---|---|
| ① njmax 1500→128 | 1.27x | ✅ 已验证 | 改 `microduck_roulade_env_cfg.py` sim 配置 |
| ② rsl_rl rollout 瘦身 | ~1.3x | 待做 | obs 归一化/buffer 写入/每步 logging 优化，目标 17.4→<6ms |
| ③ reward 死项剪枝 | ~1.1x | 待做 | 41 项中 B1=0/B17=0 等版本块虽 mask 但照算，改为条件注册 |
| ④ 碰撞体精简（训练期） | 1.2-1.5x | 待验证 | robot_allcollisions → 精简碰撞集（需对比学习曲线） |
| ⑤ bf16 obs + envs 4096 | ~1.3x | 待验证 | 12GB 显存腾挪 |
| **复合（同卡）** | **2.2-2.8x** | | 3h → **65-80 min/轮** |

**升级路线（硬件叠加）**：A100/H100（warp 官方口径 5-10x，接触求解对大显存/算力敏感）× 上述本地优化 ≈ **15-25x**（3h → 7-12 min/轮）。GPUDrive 式 100 万 steps/s 属全定制管线，不适用于 MJWarp 通用栈。

## 四、调研参考

- Rudin et al. 2021《Learning to Walk in Minutes》：4096 envs 大规模并行不掉样本效率，ANYmal 20 分钟收敛
- MuJoCo-Warp 官方：比旧后端快 5-10x；testspeed 支持 graph；`opt.graph_conditional` 需较新版本
- 《Sim-to-Real Humanoid in 15 Minutes》(2025)：精简 reward + 大规模并行 + PPO 调参
- GPUDrive (2024)：100 万 steps/s（全定制、多智能体驾驶，不通用）

## 五、复现命令

```bash
# 基线计时（30 iter 短跑）
scripts/profile_b24_timing.sh
# env.step 三层拆解
scripts/bench_env_step.py
# manager 级 phase 插桩（含同步开销，仅看相对占比）
scripts/bench_step_phases.py
# 裸物理对照（注意：无地板，零接触，仅测 kernel 上限）
.venv/bin/python -m mujoco_warp.testspeed <xml> --nworld 2048 --nstep 500
```

## 六、遗留与下一步

1. 应用 njmax=128 → 30 iter 短跑验证学习曲线与基线一致（指标逐项对比）
2. rsl_rl rollout 瘦身 spike（不动算法，只动数据通路）
3. reward 条件注册改造（env_cfg 按 DIVEWALL_PHASE_* 开关注册，而非计算后置零）
4. 训练期精简碰撞集实验（保持 sim2real 部署用全碰撞模型）

## 七、njmax=128 实跑验证（2026-10-26 15:15）

30 iter scratch 短跑对比（同种子、同 B24 配置）：

| 指标 | 基线 njmax=1500 | njmax=128 |
|---|---|---|
| Iteration time | 1.30-1.32 s | **1.15-1.16 s** |
| Steps/s | 37,218 | **42,574** |
| Collection / Learning | 1.22 / 0.101 s | 1.05 / 0.101 s |
| Mean reward (iter 29) | -1.70 | -1.56 |
| wall_flight / heading_hold / air_time_mean | 0.0051 / 0.4411 / 0.5214 | 0.0050 / 0.4107 / 0.5085 |

**结论：端到端 1.14x（bench 预测 env.step 层 1.27x，learning 0.1s 固定开销摊薄后吻合），学习指标一致（差异在噪声级）。✅ 建议合入正式训练配置。**

## 八、落地记录（15:50）

1. **njmax=128 已设为 DiveWall 任务默认**（env_cfg 直改，无 flag 生效，43.3k steps/s 复测通过）
2. **check_for_nan=False 已设默认**——实测 5 iter 无可测差异（1.13-1.14s），但消除每步潜在同步点且环境侧有 nan_state termination 兜底，保留
3. rsl_rl glue 实测仍有 ~14ms/step（collection 1.05s − env.step 24×29ms ≈ 350ms），下一步用 torch profiler 对 alg.act / process_env_step / logger.process_env_step 逐项计时

## 九、logger D2H 延迟补丁（17:15，待复测）

- 定位：`rsl_rl/utils/logger.py::process_env_step` 每步 `nonzero() + .cpu()` = 强制 GPU 同步，是训练期 GPU util 75%（p50）气泡的来源之一
- 补丁：per-step 只做 GPU 侧 accumulate，D2H 拉取推迟到每 iter 一次的 `log()`；数值等价（同 mask 同值）
- 应用方式：直接改 .venv site-packages，diff 存 `patches/rsl_rl-logger-defer-d2h.patch` 可复现
- 初测 2.96s/iter 无效——GPU 被 mlebench GRPO 评估任务（65% util）抢占，测量污染；待 GPU 空闲后复测
- GPU util 实测（150 iter 无人抢占时）：p50=75%，p90=100%，均值 82% → 消气泡理论上限 1.25-1.33x

## 十、单 iteration torch profiler 分析（18:56，GPU 空闲时捕获）

捕获第 5 个 iteration（schedule: skip 3 / wait 1 / active 1），无 profiler 干扰时该 iter ≈ 1.14s。

**总量**：Self CUDA ≈ 695ms vs 墙钟 ~1.1s → **GPU 忙约 60-65%，空闲 35-40%**（dmon 的 75% 含编译期偏高）。

**CUDA 时间构成（自上而下）**：
| 类别 | 占 CUDA 时间 | 明细 |
|---|---|---|
| 物理求解（warp kernels） | ~45% | ccd_kernel_builder 151ms + narrowphase 36ms + site_local_to_global 62ms + cholesky_solve 42ms + body_jac 21ms + efc_contact_jac 9ms 等，120 次调用 = 24 步 × 4 substep + resets |
| 神经网络（actor/critic 前向+更新） | ~15% | aten::mm/addmm/sgemm ≈ 100ms |
| **glue 小 kernel（reward/obs 管理器）** | **~35%** | mul×14600 次 15ms、cat×4979 次 9ms、index×4780 12ms、index_put×4288 7ms、copy_×10353 10ms、sum×1400 7ms——**数万次微 kernel 派发** |

**结论**：
1. GPU 内部最大单项仍是物理（真计算），但其 45% 份额意味着非物理优化空间客观存在
2. reward/obs 管理器的 glue 在 **kernel 侧**也烧掉了 ~1/3 GPU 时间（此前只测到 CPU 侧 6.7ms）——死项剪枝价值上调
3. GPU 空闲 35-40% 气泡 = Python 循环 + 微 kernel 派发间隙，靠 CUDA Graph 化 reward/obs 计算可同时吃掉 kernel 数量和气泡

## 九、solver 调参与 unroll/envs 补充验证（2026-10-26 晚）

1. **solver 调参证伪**：testspeed 实测 scene.xml（真实接触）——CG 比 Newton 慢 35%（110k vs 168k @512）；iterations 50/30 vs 默认差异噪声级；ls_parallel 本版本不支持。保持 Newton 默认。
2. **unroll 24→48 证伪（性能侧）**：learning 时间随样本量线性增长（0.101→0.198s），steps/s 零提升（42.6k≈42.5k）。仅影响样本效率/更新频率权衡，属算法实验，不以性能理由合入。
3. **envs 4096 ✅**：30 iter scratch，collection 1.56s / learning 0.197s / iter ~1.76s → **55.8k steps/s（+31%）**，峰值显存 7.27/12.2 GB，曲线与基线一致。2x envs 仅多 48% collection 时间。
4. **同卡复合提速：njmax(1.14x) × envs4096(1.31x) ≈ 1.5x 端到端（3h → ~2h/轮）**。合入：跑批 ENVS 2048→4096，首轮盯前 100 iter reward 曲线。

## 十、rollout 瘦身 spike 与收官（2026-10-27 凌晨）

1. **rsl_rl glue 归因修正**：组件级实测 actor 0.17ms / critic 0.10ms / normalizer 0.12ms / log_prob 0.04ms，合计 <0.5ms/step。此前 profiler kernel 视图的"17.4ms rsl_rl glue"归因错误。
2. **reward 死项缩水**：41 项仅 3 项真死（roulade_sagittal / roulade_lateral_vel / roulade_flatness，~0.5ms）；arrival_damping / joint_torque_rate_l2 为课程延迟激活项（iter 2500/3500 起效），不能剪枝。
3. **真 glue 位于 mjlab manager 内部**（write/forward/obs/events ~5ms/step），fork 框架性价比不划算。
4. **最终结论：同卡 1.5x 收官**（njmax 1.14x × envs4096 1.31x；3h→~2h/轮）。原方案表 ②③④⑤ 作废或缩水：②≈1.0x、③≈1.02x、④不验证、⑤已实现其一半（envs4096）。剩余 manager glue 上限 ~1.15x，不做。硬件升级（A100 5-10x）是下一步唯一大头。
