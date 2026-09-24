import {translate} from '@docusaurus/Translate';
import React from 'react';
import Link from '@docusaurus/Link';
import useBrokenLinks from '@docusaurus/useBrokenLinks';
import Layout from '@theme/Layout';
import TranslationNotice from '@site/src/components/TranslationNotice';
import {ArrowDown, ArrowRight, ArrowUpRight, Check} from 'lucide-react';
import ChapterNavigation from './ChapterNavigation';
import KnowledgeMap from './KnowledgeMap';
import RobotSkills from './skills';
import EmbodiedCarriers from './carriers';
import FootballDemo from './demos/football';
import {chapters, directions, firstSteps} from './content';
import {robotSkillAnchors} from './skills/content';
import styles from './styles.module.css';

export default function LearningMap(): React.JSX.Element {
  const {collectAnchor} = useBrokenLinks();
  // Register section targets so links from MDX are checked during the build.
  chapters.forEach(({id}) => collectAnchor(id));
  robotSkillAnchors.forEach(collectAnchor);

  return (
    <Layout title={translate({message: "具身智能学习地图"})} description={translate({message: "面向新人的具身智能学习地图：理解感知、决策、控制、仿真、数据与硬件的关系，从第一个实验走向机器人项目。"})}>
      <main className={styles.page}>
        <div className={styles.container}>
          <aside className={styles.sidebar}><ChapterNavigation /></aside>
          <div className={styles.content}>
            <header className={styles.hero}>
              <p className={styles.kicker}>{translate({message: "新手指南 · 从这里开始"})}</p>
              <h1>{translate({message: "具身智能学习地图"})}</h1>
              <p className={styles.lead}>{translate({message: "先看全局，再找到你的第一步。"})}</p>
              <p className={styles.intro}>{translate({message: "这张地图以机器人为主线，帮你连接概念、方法与实践。"})}</p>
              <TranslationNotice translated />
              <nav className={styles.pageNav} aria-label={translate({message: "学习地图页内导航"})}>
                {chapters.map(({id, quickLabel}) => (
                  <a key={id} href={`#${id}`}>{quickLabel} <ArrowDown size={15} aria-hidden="true" /></a>
                ))}
              </nav>
            </header>
            <section id="what-is-embodied-ai" className={styles.section} aria-labelledby="definition-title">
              <div className={styles.sectionHeading}>
                <div><p className={styles.kicker}>{translate({message: "概念起点"})}</p><h2 id="definition-title">{translate({message: "什么是具身智能？"})}</h2></div>
              </div>
              <div className={styles.definition}>
                <p>{translate({message: "具身智能（Embodied AI）是指将人工智能技术集成到具有物理实体（如机器人、智能汽车等）的系统之中，让AI通过“身体”的"})}<strong>{translate({message: "感知、决策与行动闭环"})}</strong>{translate({message: "与真实物理世界进行动态交互。"})}</p>
                <p>{translate({message: "例如，让机器人踢足球，需要看清球和球门、判断防守位置、选择射门方向，再控制身体保持平衡并踢球。踢出后，它还要观察球的运动，判断结果并调整下一步。这就形成了“感知 → 决策 → 行动 → 反馈”的闭环。"})}</p>
              </div>
              <FootballDemo />
              <Link to="/docs/introduction/what-is-embodied-ai" className={`site-text-link ${styles.readMore}`}>{translate({message: "深入了解具身智能"})}<ArrowRight size={16} aria-hidden="true" /></Link>
            </section>
            <RobotSkills className={styles.section} />
            <EmbodiedCarriers className={styles.section} />
            <KnowledgeMap />
            <section id="first-steps" className={styles.section} aria-labelledby="steps-title">
              <div className={styles.sectionHeading}>
                <div><p className={styles.kicker}>{translate({message: "入门路线"})}</p><h2 id="steps-title">{translate({message: "第一次来，可以这样走。"})}</h2></div>
                <p>{translate({message: "先体验，再补基础，再完成一个项目。"})}<br />{translate({message: "每一步都有一个可以验证的小目标。"})}</p>
              </div>
              <ol className={styles.steps}>
                {firstSteps.map(({title, preparation, description, outcome, link}, index) => (
                  <li key={title}>
                    <span className={styles.stepNumber} aria-hidden="true">{String(index + 1).padStart(2, '0')}</span>
                    <div className={styles.stepBody}><p className={styles.preparation}>{preparation}</p><h3>{title}</h3><p>{description}</p><p className={styles.outcome}><Check size={16} aria-hidden="true" />{outcome}</p></div>
                    <Link to={link.to} className={styles.stepLink}>{link.label}<ArrowRight size={17} aria-hidden="true" /></Link>
                  </li>
                ))}
              </ol>
              <p className={styles.preparationNote}>{translate({message: "准备写代码时，按需补 Python、Linux / Git 与线性代数；进入模型训练后，再补 PyTorch、概率与深度学习。已有基础可以直接跳到对应阶段。"})}</p>
            </section>
            <section id="directions" className={styles.section} aria-labelledby="directions-title">
              <div className={styles.sectionHeading}>
                <div><p className={styles.kicker}>{translate({message: "按任务选择"})}</p><h2 id="directions-title">{translate({message: "你想让机器人做什么？"})}</h2></div>
                <p>{translate({message: "机器人形态、任务和算法是不同维度。"})}<br />{translate({message: "同一台机器人可以组合多种能力。"})}</p>
              </div>
              <div className={styles.directions}>
                {directions.map(({title, english, description, path, to, label, note}) => (
                  <article key={title}>
                    <p className={styles.directionEnglish}>{english}</p><h3>{title}</h3><p>{description}</p>
                    <p className={styles.directionPath}>{path}</p>
                    <Link to={to} className="site-text-link">{label}<ArrowUpRight size={16} aria-hidden="true" /></Link>
                    <p className={styles.directionNote}>{note}</p>
                  </article>
                ))}
              </div>
              <div className={styles.nextSteps}><div><strong>{translate({message: "想继续深入一个方向？"})}</strong><p>{translate({message: "把问题写清楚：机器人看到什么、输出什么动作、什么条件下算成功。再选择方法和对照实验。"})}</p></div><Link to="/docs/overview/learning-path" className="site-text-link">{translate({message: "查看进阶学习路径"})}<ArrowRight size={17} aria-hidden="true" /></Link></div>
            </section>
            <section id="references" className={styles.section} aria-labelledby="references-title">
              <div className={styles.sectionHeading}><h2 id="references-title">{translate({message: "参考"})}</h2></div>
              <ol className={styles.references}>
                <li><Link href="https://www.nvidia.cn/glossary/embodied-ai/">{translate({message: "NVIDIA：什么是具身智能？"})}<ArrowUpRight size={16} aria-hidden="true" /></Link></li>
                <li><Link href="https://modernrobotics.northwestern.edu/nu-gm-book-resource/grasping-and-manipulation/">{translate({message: "Modern Robotics：抓取与操作"})}<ArrowUpRight size={16} aria-hidden="true" /></Link></li>
                <li><Link href="https://modernrobotics.northwestern.edu/nu-gm-book-resource/13-1-wheeled-mobile-robots/">{translate({message: "Modern Robotics：轮式机器人与移动操作"})}<ArrowUpRight size={16} aria-hidden="true" /></Link></li>
              </ol>
            </section>
          </div>
        </div>
      </main>
    </Layout>
  );
}
