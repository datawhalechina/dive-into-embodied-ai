import {translate} from '@docusaurus/Translate';
import React from 'react';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import {ArrowRight, ArrowUpRight, Bot, Cog, Cpu, Compass} from 'lucide-react';
import styles from './styles.module.css';

const projects = [
  {
    title: translate({id: "home.proj.g1Title", message: "Unitree G1 人形运控实战"}),
    badge: translate({id: "home.proj.g1Badge", message: "进阶 · 单卡 GPU / 网页可玩"}),
    description: translate({id: "home.proj.g1Short", message: "16 项奖励、万轮 PPO、Sim2Sim 与 50 Hz 浏览器 LSTM。"}),
    to: '/docs/practices/humanoid/unitree-g1',
    image: '/img/unitree-g1-poster.webp',
    alt: translate({id: "home.proj.g1Alt", message: "Unitree G1 双足人形机器人在 MuJoCo 仿真中行走"}),
    tags: ['12-DoF LSTM', 'Sim2Sim'],
  },
  {
    title: translate({message: "MicroDuck 双足机器人"}),
    badge: translate({id: "home.proj.duckBadge", message: "中级 · ROCm / CUDA"}),
    description: translate({id: "home.proj.duckShort", message: "MuJoCo Warp 并行 PPO，12 步课程学会跳墙与起身。"}),
    to: '/docs/practices/humanoid/microduck-rl',
    image: '/img/microduck-poster.webp',
    alt: translate({message: "MicroDuck 双足机器人在 MuJoCo 仿真中行走"}),
    tags: ['MuJoCo Warp', 'PPO'],
  },
  {
    title: translate({message: "ACT 双臂操作训练"}),
    badge: translate({id: "home.proj.actBadge", message: "中级 · 单卡 GPU"}),
    description: translate({id: "home.proj.actShort", message: "ALOHA 示教数据训练动作分块策略并闭环评测。"}),
    to: '/docs/practices/vla/act',
    image: '/img/act-poster.webp',
    alt: translate({message: "ALOHA 双臂机器人操作任务仿真"}),
    tags: ['ACT', 'ALOHA'],
  },
];

const moreProjects = [
  {
    title: translate({id: "home.more.go2Title", message: "Go2 四足运控 · MJX（ROCm / CUDA）"}),
    description: translate({id: "home.more.go2Short", message: "8GB 显存可训，R9700 4.5h 满训直通官方 Sim2Sim"}),
    to: '/docs/practices/amd/unitree-rl-mjx',
    icon: Cpu,
    status: translate({id: "home.more.go2Badge", message: "AMD / MJX"}),
  },
  {
    title: translate({id: "home.more.cs123Title", message: "从 0 到 1 搭建四足机器人（CS123）"}),
    description: translate({id: "home.more.cs123Short", message: "8 章完整闭环：PD、正逆运动学、步态到视觉追球"}),
    to: '/docs/practices/quadruped/cs123/intro',
    icon: Bot,
    status: translate({id: "home.more.cs123Badge", message: "入门 · CPU"}),
  },
  {
    title: translate({id: "home.more.mobileTitle", message: "移动操作与自主导航（SLAM / VLN）"}),
    description: translate({id: "home.more.mobileShort", message: "SLAM + Nav2、视觉语言导航与底盘-机械臂协同"}),
    to: '/docs/practices/mobile-manipulation/placeholder',
    icon: Compass,
    status: translate({id: "home.more.mobileBadge", message: "移动操作"}),
  },
  {
    title: translate({message: "SO-101 + LeRobot 真机实践"}),
    description: translate({message: "硬件连通、基础测试与动作回放"}),
    to: '/docs/practices/robot-arm/data-collection/so101-lerobot-real',
    icon: Cog,
    status: translate({message: "真机"}),
  },
];

function ProjectCard({project}: {project: typeof projects[number]}) {
  return (
    <Link className={styles.card} to={project.to}>
      <div className={styles.visual}>
        <img src={useBaseUrl(project.image)} alt={project.alt} width="640" height="480" loading="lazy" />
      </div>
      <div className={styles.body}>
        <span className={styles.metaPill}>{project.badge}</span>
        <h3>{project.title}</h3>
        <p className={styles.description}>{project.description}</p>
        <div className={styles.bottom}>
          <span>{project.tags.join(' · ')}</span>
          <ArrowUpRight size={19} aria-hidden="true" />
        </div>
      </div>
    </Link>
  );
}

export default function HomepageProjects(): React.JSX.Element {
  return (
    <section className={styles.projects} aria-labelledby="homepage-projects-title">
      <div className="home-container">
        <div className="home-section-heading">
          <div>
            <p className="section-kicker">{translate({message: "精选实战"})}</p>
            <h2 id="homepage-projects-title">{translate({id: "home.proj.heading", message: "直接看代表项目，挑一个动手跑。"})}</h2>
          </div>
          <Link className="site-text-link" to="/practices">
            {translate({message: "查看全部项目"})}
            <ArrowRight size={17} aria-hidden="true" />
          </Link>
        </div>
        <div className={styles.grid}>
          {projects.map(project => <ProjectCard key={project.to} project={project} />)}
        </div>
        <div className={styles.moreProjects}>
          {moreProjects.map(({title, description, to, icon: Icon, status}) => (
            <Link className={styles.moreCard} key={to} to={to}>
              <span className={styles.moreIcon}><Icon size={22} strokeWidth={1.6} aria-hidden="true" /></span>
              <div>
                <div className={styles.moreTitle}><h3>{title}</h3><span>{status}</span></div>
                <p>{description}</p>
              </div>
              <ArrowUpRight size={19} aria-hidden="true" />
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}
