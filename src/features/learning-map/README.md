# 学习地图模块

`src/pages/learning-map.tsx` 是 Docusaurus 路由入口，页面实现集中在本目录。

学习地图负责指路：概念部分只保留一句定义和具身导论的入口，后面是知识全景、入门路线和实践方向。具身智能的概念、技能与载体，以及足球、技能和载体的演示，都在 `docs/introduction/` 的具身导论中，修改时同步对应的英文译稿。

| 修改内容 | 对应文件 |
| --- | --- |
| 页面结构、概念入口、参考 | `index.tsx` |
| 目录名称、旧锚点的跳转、任务案例、学习路线与课程链接 | `content.ts` |
| 侧边目录与滚动定位 | `ChapterNavigation.tsx` |
| 知识全景的任务切换与展示 | `KnowledgeMap.tsx` |

页面与目录共享 `content.ts` 中的章节配置。

技能与载体章节已移到具身导论。已有外部链接使用的 `#robot-interaction`、`#ability-*` 和 `#embodied-platforms` 锚点，由 `content.ts` 中的 `movedSections` 跳转到导论对应的章节；导论中的标题锚点改名时，同步更新这张表。

## 导论中的演示

演示组件位于 `src/components/docs/introduction/`，由导论第 1、3、4 章引入：

| 演示 | 对应目录 |
| --- | --- |
| 足球闭环演示 | `what-is-embodied-ai/FootballDemo/` |
| 四类技能的演示 | `tasks-and-skills/SkillDemo/` |
| 载体演示的素材切换、播放与静态回退 | `embodiments/CarrierDemos/` |

动画目录中的 `index.tsx` 管理播放与控件，`scene.ts` 管理 3D 场景。技能动画的阶段文案在 `content.ts`，动作轨迹在 `motion.ts`。场景通过动态导入按需加载。技能内容通过显式的 `skill` 字段关联动画，避免从锚点字符串推断类型。

载体演示默认使用 `static/img/learning-map/carriers/official/` 中的官方片段 MP4 与 WebP 封面，可以切换到上一级目录中的 3D 示意 GIF 与 WebP 封面。两种演示按可见区域与动态效果偏好播放。官方素材来源、截取时间与校验值见 `scripts/carrier-gifs/official-sources.json`，两套素材的生成步骤见该目录的 `README.md`；网页无需为载体演示创建 WebGL 场景。
