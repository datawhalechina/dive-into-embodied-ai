# Pupper 结构变体

本目录提供第 4 章的 Pupper 形态与站姿探索代码。

代码从一份 `skeleton.xml` 派生三种模型：

- `original`：原始腿长和机身质量。
- `long-leg`：大腿和小腿长度放大 1.5 倍。
- `heavy`：机身质量放大 2 倍。

每种模型都会重新搜索站姿、扫描 PD 参数，并生成站姿与稳定性对比图。

## 文件

| 文件 | 作用 |
| --- | --- |
| `run_pupper_variants.py` | 生成模型、搜索站姿、扫描 PD 参数并生成结果 |
| `test_pupper_variants.py` | 检查模型缩放、质量、站姿与 PD 稳定性 |
| `utils.py` | 本实验使用的最小控制和绘图工具 |
| `skeleton.xml` | 三种变体共用的 12-DoF 浮动基座骨架 |

mesh 直接复用 `codes/practices/quadruped/cs123/assets/mjcfs/meshes/stl/`。生成文件写入 `4.quadruped-mjcf/outputs/pupper_variants/`，不会修改源模型。

## 运行

在 `codes/practices/quadruped/cs123` 目录执行：

```bash
uv run python 4.quadruped-mjcf/pupper_variants/run_pupper_variants.py
uv run python 4.quadruped-mjcf/pupper_variants/test_pupper_variants.py
```
