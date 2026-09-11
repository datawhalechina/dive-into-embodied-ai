import React from 'react';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import {ArrowRight, ArrowUpRight, Bot, Cog} from 'lucide-react';
import styles from './styles.module.css';

const projects = [
  {title: 'MicroDuck 双足机器人', category: '双足行走 / 强化学习', description: '从 PPO 训练到动作回放，探索行走、起身与轮式动作。', to: '/docs/practices/humanoid/microduck-rl', image: '/img/microduck-poster.webp', alt: 'MicroDuck 双足机器人在 MuJoCo 仿真中行走', tags: ['MuJoCo Warp', 'PPO']},
  {title: '从零搭建四足机器人', category: '四足控制 / CS123', description: '从 PD 控制与运动学出发，一步步实现四足机器人的行走。', to: '/docs/practices/quadruped/cs123/intro', image: '/img/pupper-poster.webp', alt: 'Pupper 四足机器人前进步态仿真', tags: ['MuJoCo', '运动控制']},
  {title: 'ACT 双臂操作训练', category: '双臂协作 / 模仿学习', description: '使用 ALOHA 仿真数据训练策略，在评测中验证双臂操作能力。', to: '/docs/practices/vla/act', image: '/img/act-poster.webp', alt: 'ALOHA 双臂机器人操作任务仿真', tags: ['ACT', 'ALOHA']},
];
const moreProjects = [
  {title: 'MuJoCo 机械臂与 DDPG', description: '探索机械臂的连续控制任务', to: '/docs/practices/robot-arm/mujoco-arm-pick-place', icon: Bot, status: '机械臂'},
  {title: 'Flamingo 轮足机器人', description: 'Isaac Lab 训练与 Sim2Sim 迁移', to: '/docs/practices/wheel-legged/flamingo-isaaclab/preview', icon: Cog, status: '内容预告'},
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
          <div><p className="section-kicker">精选实战</p><h2 id="homepage-projects-title">让知识，真的动起来。</h2></div>
          <Link className="site-text-link" to="/docs/practices/intro">查看全部项目 <ArrowRight size={17} aria-hidden="true" /></Link>
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
