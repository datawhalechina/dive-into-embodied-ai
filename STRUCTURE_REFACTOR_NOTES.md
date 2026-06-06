# 教程结构重构交接说明

更新时间：2026-06-06（移除「求职面试」一级入口）

> **2026-06-06 更新**：用户要求整体删除「求职面试」一级入口。已删除 `docs/career/` 整个目录、`sidebars.ts` 全部 `career*` 侧边栏、mega menu 的 `career` 项、首页第三张卡片与页脚链接，并同步 `README.md`、`CLAUDE.md`、`scripts/check-doc-sidebars.cjs`。下文凡涉及「求职面试 / career」的历史描述均已作废，仅作变更留痕。

## 目标结构

当前教程为三个一级入口：

1. 零基础入门
2. 项目实战
3. 理论技能树

顶部导航当前应显示为：

```text
首页 / 零基础入门 / 项目实战 / 理论技能树
```

## 已提交 checkpoint

- `5c55056 💄 style: 调整首页导航位置和颜色`
- `e9ef0b2 ♻️ refactor: 重构教程四段入口骨架`

`e9ef0b2` 已提交第一批结构骨架，包括顶部菜单、首页入口卡片、页脚、入口概述页、技能树占位页等。

## 关键实现文件

- 顶部导航配置：`docusaurus.config.ts`
- 顶部 mega menu 数据：`src/components/NavbarMegaMenu/data.ts`
- 顶部 mega menu 样式：`src/components/NavbarMegaMenu/styles.css`
- 首页四个入口卡片：`src/components/HomepageFeatures/index.tsx`
- 文档侧边栏：`sidebars.ts`
- 零基础入门概述：`docs/overview/intro.md`
- 理论技能树概述：`docs/foundations/intro.md`
- 项目实战概述：`docs/practices/intro.md`
- 求职面试概述：`docs/career/intro.md`

## 当前设计

### 零基础入门

只保留两个入口：

- `学习路径`：`/docs/overview/learning-path`
- `从0到1搭建四足机器人`：`/docs/practices/quadruped/cs123/intro`

已从下拉、`overviewSidebar`、`docs/overview/intro.md` 中移除其他入口。旧内容文件暂时保留，避免历史链接断。

### 理论技能树

当前下拉拆成 4 列：

- `大脑：智能决策`
- `小脑：运动控制`
- `感知系统`
- `工程底座`

当前要求：导航顺序为零基础入门、项目实战、求职面试、理论技能树；仿真并入工程底座。

技能树概述页和侧边栏也已同步：

- 感知：`docs/foundations/vlm/intro`、`docs/foundations/perception/placeholder`
- 工程底座：仿真、ROS2、CAN/MCU、机械结构、模仿学习、LeRobot

### 项目实战

仍按项目方向组织：

- 课程入口：CS123 四足、LeRobot 中文课程
- 本体方向：机械臂、四足、人形、移动操作
- 部署与综合：SO-101 + LeRobot、移动操作

### 求职面试（已于 2026-06-06 整体移除）

该一级入口连同 `docs/career/` 目录、全部 `career*` 侧边栏、mega menu `career` 项、首页卡片与页脚链接已删除，不再展示。本节仅留作历史记录。

## 当前未提交变更

除 `.claude/worktrees/` 外，当前未提交结构变更主要是：

- `README.md` 已按当前「零基础入门 + 项目实战 + 求职面试 + 理论技能树」结构更新
- `docs/career/transition-paths/*` 三篇长文改为兼容入口页，指向 `docs/overview/*` 主版本
- 删除 `docs/career/transition-paths/figs/*` 下重复图片资源
- `sidebars.ts` 移除 `careerTransitionPathsSidebar`
- `docs/overview/intro.md` 收窄为零基础入门两个入口
- `src/components/NavbarMegaMenu/data.ts`：
  - 零基础入门只保留两个卡片
  - 理论技能树保留感知系统，并将仿真、ROS2、通信、硬件和数据合并为工程底座
  - 求职面试只保留三项核心准备
- `docs/foundations/intro.md` 同步技能树说明
- `src/components/HomepageFeatures/index.tsx` 同步求职面试卡片描述

## 验证状态

最近一次 `npm run build` 通过。

仍存在仓库既有警告：

- `docs/CLAUDE#anchor` broken anchor
- 之前有过 GIF / LFS 图片读取警告，通常与本地 LFS 文件状态有关

## 下一步建议

1. 提交当前这批未提交结构变更，但不要包含 `README.md`，除非用户明确要求。
2. 继续按新结构迁移内容：
   - 把 `docs/career/transition-paths` 完全退化为兼容入口，主内容只维护 `docs/overview`
   - 将简历、公司技术栈、社区与内推决定是否迁到求职面试外的补充资料区，或暂时不展示
   - 逐步补齐技能树中感知、CAN/MCU、机械结构等占位页
3. 每一小批改动后运行 `npm run build`。

## 注意事项

- `README.md` 已按用户要求更新；提交时可以纳入当前结构重构批次。
- `.claude/worktrees/` 是未跟踪目录，不要处理。
- 现阶段优先“先搭结构，再迁内容”，不要一次性大规模移动文件。
