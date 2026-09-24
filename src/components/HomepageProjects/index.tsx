import {translate} from '@docusaurus/Translate';
import React from 'react';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import {ArrowRight, ArrowUpRight, Bot, Cog} from 'lucide-react';
import styles from './styles.module.css';

const projects = [
  {title: translate({message: "MicroDuck 双足机器人"}), category: translate({message: "双足行走 / 强化学习"}), description: translate({message: "从 PPO 训练到动作回放，探索行走、起身与轮式动作。"}), to: '/docs/practices/humanoid/microduck-rl', image: '/img/microduck-poster.webp', alt: translate({message: "MicroDuck 双足机器人在 MuJoCo 仿真中行走"}), tags: ['MuJoCo Warp', 'PPO']},
  {title: translate({message: "玩转 Pupper 四足机器人"}), category: translate({message: "AMD 实践 / 四足控制"}), description: translate({message: "在 AMD 平台上探索 Pupper 的强化学习运动策略与 VLA 实验。"}), to: '/docs/practices/amd/pupper-control/intro', image: '/img/pupper-poster.webp', alt: translate({message: "Pupper 四足机器人前进步态仿真"}), tags: ['AMD', translate({message: "运动控制"})]},
  {title: translate({message: "ACT 双臂操作训练"}), category: translate({message: "双臂协作 / 模仿学习"}), description: translate({message: "使用 ALOHA 仿真数据训练策略，在评测中验证双臂操作能力。"}), to: '/docs/practices/vla/act', image: '/img/act-poster.webp', alt: translate({message: "ALOHA 双臂机器人操作任务仿真"}), tags: ['ACT', 'ALOHA']},
];
const moreProjects = [
  {title: translate({message: "MuJoCo 机械臂与 DDPG"}), description: translate({message: "探索机械臂的连续控制任务"}), to: '/docs/practices/robot-arm/mujoco-arm-pick-place', icon: Bot, status: translate({message: "机械臂"})},
  {title: translate({message: "SO-101 + LeRobot 真机实践"}), description: translate({message: "硬件连通、基础测试与动作回放"}), to: '/docs/practices/robot-arm/data-collection/so101-lerobot-real', icon: Cog, status: translate({message: "真机"})},
];
function ProjectCard({project}: {project: typeof projects[number]}) {
  return (
    <Link className={styles.card} to={project.to}>
      <div className={styles.visual}><img src={useBaseUrl(project.image)} alt={project.alt} width="640" height="480" loading="lazy" /></div>
      <div className={styles.body}>
        <p className={styles.category}>{project.category}</p>
        <h3>{project.title}</h3><p className={styles.description}>{project.description}</p>
        <div className={styles.bottom}><span>{project.tags.join(' · ')}</span><ArrowUpRight size={20} aria-hidden="true" /></div>
      </div>
    </Link>
  );
}
export default function HomepageProjects(): React.JSX.Element {
  return (
    <section className={styles.projects} aria-labelledby="homepage-projects-title">
      <div className="home-container">
        <div className="home-section-heading">
          <div><p className="section-kicker">{translate({message: "精选实战"})}</p><h2 id="homepage-projects-title">{translate({message: "让知识，真的动起来。"})}</h2></div>
          <Link className="site-text-link" to="/docs/practices/intro">{translate({message: "查看全部项目"})}<ArrowRight size={17} aria-hidden="true" /></Link>
        </div>
        <div className={styles.grid}>{projects.map(project => <ProjectCard key={project.to} project={project} />)}</div>
        <div className={styles.moreProjects}>
          {moreProjects.map(({title, description, to, icon: Icon, status}) => (
            <Link className={styles.moreCard} key={to} to={to}>
              <span className={styles.moreIcon}><Icon size={24} strokeWidth={1.6} aria-hidden="true" /></span>
              <div><div className={styles.moreTitle}><h3>{title}</h3><span>{status}</span></div><p>{description}</p></div>
              <ArrowUpRight size={20} aria-hidden="true" />
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}
