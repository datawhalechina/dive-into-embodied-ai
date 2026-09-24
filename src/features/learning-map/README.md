# 学习地图模块

`src/pages/learning-map.tsx` 是 Docusaurus 路由入口，页面实现集中在本目录。

学习地图负责指路：概念部分只保留定义、演示和导论链接。技能的方法、代表工作与对比表，以及载体的研究问题，写在 `docs/introduction/` 的具身导论第 3、4 章，修改时同步对应的英文译稿。

| 修改内容 | 对应文件 |
| --- | --- |
| 页面结构、概念介绍、参考 | `index.tsx` |
| 目录名称、任务案例、学习路线与课程链接 | `content.ts` |
| 侧边目录与滚动定位 | `ChapterNavigation.tsx` |
| 知识全景的任务切换与展示 | `KnowledgeMap.tsx` |
| 四类技能的定义与导论链接 | `skills/content.tsx` |
| 技能章节的排版 | `skills/index.tsx` |
| 技能类型 | `skills/types.ts` |
| 具身智能载体的分类、平台与关联技能 | `carriers/content.ts` |
| 载体章节的排版与对照表 | `carriers/index.tsx` |
| 足球演示 | `demos/football/` |
| 四类技能的演示 | `demos/skills/` |
| 载体演示的素材切换、播放与静态回退 | `demos/carriers/` |

动画目录中的 `index.tsx` 管理播放与控件，`scene.ts` 管理 3D 场景。技能动画的阶段文案在 `content.ts`，动作轨迹在 `motion.ts`。场景仍通过动态导入按需加载。

载体演示默认使用 `static/img/learning-map/carriers/official/` 中的官方片段 MP4 与 WebP 封面，可以切换到上一级目录中的 3D 示意 GIF 与 WebP 封面。两种演示按可见区域与动态效果偏好播放。官方素材来源、截取时间与校验值见 `scripts/carrier-gifs/official-sources.json`，两套素材的生成步骤见该目录的 `README.md`；网页无需为载体演示创建 WebGL 场景。

页面与目录共享 `content.ts` 中的章节配置。技能内容通过显式的 `skill` 字段关联动画，避免从锚点字符串推断类型。

`#robot-interaction`、`#ability-*` 等锚点用于已有外部链接，内部文件和组件改名时保留这些地址。
