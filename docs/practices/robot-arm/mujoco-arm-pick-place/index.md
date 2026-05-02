---
sidebar_position: 1
title: MuJoCo 仿真入门
---

# MuJoCo 仿真入门

使用 MuJoCo 物理引擎搭建你的第一个机器人仿真环境。

## 项目简介

MuJoCo（Multi-Joint dynamics with Contact）是由 DeepMind 开源的高性能物理仿真引擎，广泛用于机器人学习和控制研究。

## 你将学到

- MuJoCo 环境安装与配置
- MJCF 模型文件编写
- 基础物理仿真与可视化
- 与 Python 的交互控制

## 快速开始

```bash
pip install mujoco
```

```python
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

mujoco.viewer.launch(model, data)
```
