# 学习地图模块

`src/pages/learning-map.tsx` 是 Docusaurus 路由入口，页面实现集中在本目录。

| 修改内容 | 对应文件 |
| --- | --- |
| 页面结构、概念介绍、参考 | `index.tsx` |
| 目录名称、任务案例、学习路线与课程链接 | `content.ts` |
| 侧边目录与滚动定位 | `ChapterNavigation.tsx` |
| 知识全景的任务切换与展示 | `KnowledgeMap.tsx` |
| 四类技能的介绍与对比数据 | `skills/content.tsx` |
| 技能章节的排版与组合关系 | `skills/index.tsx` |
| 技能类型 | `skills/types.ts` |
| 足球演示 | `demos/football/` |
| 四类技能的演示 | `demos/skills/` |

动画目录中的 `index.tsx` 管理播放与控件，`scene.ts` 管理 3D 场景。技能动画的阶段文案在 `content.ts`，动作轨迹在 `motion.ts`。场景仍通过动态导入按需加载。

页面与目录共享 `content.ts` 中的章节配置。技能内容通过显式的 `skill` 字段关联动画，避免从锚点字符串推断类型。

`#robot-interaction`、`#ability-*` 等锚点用于已有外部链接，内部文件和组件改名时保留这些地址。
