
TODO：
1. 词云需要进一步优化一下（带上机器人和对应的一些技能点）
2. 标题的中英文展示
3. 项目的定位（xxx）
4. 项目受众
5. 内容大纲
    1. 理论知识
    2. 项目实践
    3. 求职面试
6. 组队学习 （存放历史的一些组队学习计划文档）
7. 贡献者团队
8. 关于xxx, LICENSE

理论知识
1. 


项目实践

求职面试


<div align="center">
    <img src="static/img/career.webp" width="100%" alt="Banner" />
</div>


<h1 align="center"> Dive into Embodied AI（⚠️ Alpha 内测版） </h1>

> [!CAUTION]
> ⚠️ Alpha 内测版本：仍在迁移和重构中，部分章节是占位页，欢迎提 Issue 反馈问题或建议。

面向求职与转行的具身智能开源教程，按「总览 + 基础篇 + 实践篇 + 求职篇」三篇制组织：先建立认知，再做项目，最后整理到简历和面试。

## 项目受众

- 想入门具身智能的应届生与在校生
- 从 ML / CV / NLP / 自动驾驶 / 传统机器人方向转入具身的工程师
- 想跳槽到更头部具身公司、需要补齐项目和面试准备的人

## 在线阅读

https://datawhalechina.github.io/dive-into-embodied-ai/

## 目录

| 章节 | 简介 | 状态 |
|------|------|------|
| [总览](docs/overview/intro.md) | 项目介绍、学习路径、公司图谱 | 🚧 部分施工中 |
| [基础篇](docs/foundations/intro.md) | 机器人学、ROS2、仿真、强化学习、VLM/VLA 等基础模块 | 🚧 部分施工中 |
| [实践篇](docs/practices/intro.md) | 机械臂、四足、人形、移动操作四类项目 | 🚧 部分施工中 |
| [求职篇](docs/career/intro.md) | 岗位拆解、面经、简历、公司技术栈、招聘信息、转岗路径、社区与内推 | 🚧 部分施工中 |

## 贡献者名单

| 姓名 | 职责 | 简介 |
| :--- | :--- | :--- |
| 罗如意 | 项目负责人 | 智能汽车竞赛国奖&多模态顶会Oral&FunRec开源项目负责人 |
| 江季  | 项目负责人 | xxx |
| 康博 | 核心贡献者 | xxx |

## 本地预览

仓库使用 Git LFS 存放视频和 GIF。clone 之后必须先装 `git-lfs` 再 `git lfs pull`，否则本地看到的图/视频是 pointer 文本，不是真内容。完整步骤见 [CONTRIBUTING.md](CONTRIBUTING.md#首次克隆必读)。

```bash
# 1. 装 git-lfs（每台机器只需一次）
# brew install git-lfs        # macOS
# sudo apt install git-lfs  # Ubuntu / Debian
# choco install git-lfs     # Windows

# 2. 初始化并拉取 LFS 文件
git lfs install
git lfs pull

# 3. 装依赖、起本地预览
npm install
npm run dev
```

## 参与贡献

- 贡献规则（LFS、图片格式、招聘数据、必跑命令）见 [CONTRIBUTING.md](CONTRIBUTING.md)。
- 发现问题请提 Issue，长期没人回复可联系 [Datawhale 保姆团队](https://github.com/datawhalechina/DOPMC/blob/main/OP.md)。
- 想参与贡献请提 Pull Request，长期没人回复同上。
- 想发起新的 Datawhale 项目，请参考 [Datawhale 开源项目指南](https://github.com/datawhalechina/DOPMC/blob/main/GUIDE.md)。

## 关注我们

<div align=center>
<p>扫描下方二维码关注公众号：Datawhale</p>
<img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width = "180" height = "180">
</div>

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。
