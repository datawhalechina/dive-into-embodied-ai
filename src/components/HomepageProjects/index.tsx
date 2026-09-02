import React from 'react';
import Link from '@docusaurus/Link';
import { ArrowRight, Bot, Boxes, BrainCircuit, Cog } from 'lucide-react';
import styles from './styles.module.css';

const microDuckDemo = require(
  '@site/docs/practices/humanoid/microduck-rl/figs/microduck-training-1500.gif',
).default as string;

type Project = {
  title: string;
  category: string;
  description: string;
  link: string;
  status: string;
  accent: 'cyan' | 'green' | 'amber' | 'blue';
  icon: string;
  image?: string;
  imageAlt?: string;
};

const projects: Project[] = [
  {
    title: 'MicroDuck RL 小黄鸭双足机器人',
    category: '双足 · GPU 仿真',
    description: 'mjlab + MuJoCo Warp 并行 PPO，展示从 MJCF 资产到稳定步态回放的完整链路。',
    link: '/docs/practices/humanoid/microduck-rl',
    status: '稳定步态 Demo',
    accent: 'cyan',
    icon: '🐥',
    image: microDuckDemo,
    imageAlt: 'MicroDuck 小黄鸭双足机器人稳定步态回放',
  },
  {
    title: '从零到一搭建四足机器人',
    category: '四足 · MuJoCo',
    description: '从 PD、运动学和 MJCF 开始，逐步走到 PPO 步态与 LLM 控制。',
    link: '/docs/practices/quadruped/cs123/intro',
    status: '可用',
    accent: 'green',
    icon: '🐕',
  },
  {
    title: 'MuJoCo 机械臂与 DDPG',
    category: '机械臂 · 连续控制',
    description: '搭建 MuJoCo 环境，比较 InvertedPendulum、Reacher 与 Pusher 任务。',
    link: '/docs/practices/robot-arm/mujoco-arm-pick-place',
    status: '可用',
    accent: 'blue',
    icon: '🦾',
  },
  {
    title: 'ACT 双臂操作训练',
    category: '双臂 · 模仿学习',
    description: '使用 ALOHA 仿真数据训练 ACT，并用多回合评测检查策略是否真正完成任务。',
    link: '/docs/practices/vla/act',
    status: '可用',
    accent: 'amber',
    icon: '🧠',
  },
  {
    title: '两轮足 Flamingo · Isaac Lab',
    category: '轮足 · Sim2Sim',
    description: '用 PPO / CaT 训练欠驱动轮足机器人，并验证跨仿真迁移的鲁棒性。',
    link: '/docs/practices/wheel-legged/flamingo-isaaclab/preview',
    status: '预告',
    accent: 'cyan',
    icon: '🛞',
  },
];

const projectIcons = [Bot, Boxes, Cog, BrainCircuit];

function ProjectVisual({ project, index }: { project: Project; index: number }) {
  const Icon = projectIcons[index % projectIcons.length];

  if (project.image) {
    return (
      <div className={styles.visual}>
        <img src={project.image} alt={project.imageAlt ?? project.title} loading="lazy" />
        <span className={styles.visualBadge}>{project.status}</span>
      </div>
    );
  }

  return (
    <div className={styles.visual} aria-hidden="true">
      <div className={styles.iconVisual}>
        <span className={styles.iconEmoji}>{project.icon}</span>
        <Icon size={22} strokeWidth={1.8} />
      </div>
      <span className={styles.visualBadge}>{project.status}</span>
    </div>
  );
}

export default function HomepageProjects(): React.JSX.Element {
  return (
    <section className={styles.projects} aria-labelledby="homepage-projects-title">
      <div className="container">
        <div className={styles.heading}>
          <p className="section-kicker">SIMULATION LABS</p>
          <h2 id="homepage-projects-title">仿真实战项目</h2>
          <p>从可视化 demo 开始，把机器人模型、控制器和强化学习训练串成一条可复现的工程链路。</p>
        </div>
        <div className={styles.grid}>
          {projects.map((project, index) => (
            <Link
              className={`${styles.card} ${styles[`card--${project.accent}`]}`}
              to={project.link}
              key={project.title}
            >
              <ProjectVisual project={project} index={index} />
              <div className={styles.body}>
                <p className={styles.category}>{project.category}</p>
                <h3>{project.title}</h3>
                <p className={styles.description}>{project.description}</p>
                <span className={styles.cardLink}>
                  查看项目 <ArrowRight size={16} aria-hidden="true" />
                </span>
              </div>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}
