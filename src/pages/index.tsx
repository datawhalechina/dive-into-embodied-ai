import React from 'react';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import {ArrowRight, ArrowUpRight, BookOpen, Code2, GitFork, SlidersHorizontal, Move3D, Crosshair} from 'lucide-react';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import HomepageProjects from '@site/src/components/HomepageProjects';
import styles from './index.module.css';

const microDuckVideo = require('@site/docs/practices/humanoid/microduck-rl/figs/microduck-velocity-flat.mp4').default as string;
const playgrounds = [
  {number: '01', title: '调好第一个控制器', subtitle: 'PD 控制', description: '调整增益，观察响应、超调与稳定性的变化。', to: '/cs123/pd-playground', icon: SlidersHorizontal},
  {number: '02', title: '看懂关节如何运动', subtitle: '正运动学', description: '转动关节，观察足端位置与坐标系的关系。', to: '/cs123/fk-playground', icon: Move3D},
  {number: '03', title: '让足端走到目标点', subtitle: '逆运动学', description: '移动目标点，探索机器人腿的可达空间。', to: '/cs123/ik-playground', icon: Crosshair},
];

function HomepageHeader() {
  const poster = useBaseUrl('/img/microduck-poster.webp');
  return (
    <header className={styles.hero}>
      <div className="home-container">
        <div className={styles.heroGrid}>
          <div className={styles.heroCopy}>
            <p className={styles.eyebrow}>DATAWHALE <span aria-hidden="true">/</span> 开源具身智能教程</p>
            <h1>动手学<br /><span>具身智能。</span></h1>
            <p className={styles.intro}>从理解一个算法，到迈出机器人的第一步。<br className={styles.desktopBreak} />在理论、仿真与真机之间，找到你的学习路线。</p>
            <div className={styles.actions}>
              <Link className="site-button site-button--primary" to="/docs/foundations/intro">开始学习 <ArrowRight size={18} aria-hidden="true" /></Link>
              <Link className="site-text-link" to="/docs/practices/intro">探索实战项目 <ArrowRight size={17} aria-hidden="true" /></Link>
            </div>
            <div className={styles.heroNotes} aria-label="课程特色">
              <span><BookOpen size={15} aria-hidden="true" />中文教程</span>
              <span><Code2 size={15} aria-hidden="true" />可运行代码</span>
              <span><GitFork size={15} aria-hidden="true" />社区共建</span>
            </div>
          </div>
          <figure className={styles.robotCard}>
            <div className={styles.robotHeading}><span>从仿真开始，迈出第一步</span><span className={styles.robotTag}>MuJoCo · PPO</span></div>
            <video className={styles.robotVideo} controls playsInline loop muted preload="none" poster={poster} aria-label="MicroDuck 双足机器人步态回放，无音频">
              <source src={microDuckVideo} type="video/mp4" />
              你的浏览器不支持视频播放，可在下方项目教程中查看实验结果。
            </video>
            <figcaption className={styles.robotCaption}>
              <div><strong>MicroDuck</strong><span>用强化学习，让小黄鸭学会行走。</span></div>
              <Link to="/docs/practices/humanoid/microduck-rl" aria-label="阅读 MicroDuck 项目教程"><ArrowUpRight size={22} aria-hidden="true" /></Link>
            </figcaption>
          </figure>
        </div>
      </div>
    </header>
  );
}

function Playgrounds() {
  return (
    <section className={styles.playgrounds} aria-labelledby="playgrounds-title">
      <div className="home-container">
        <div className="home-section-heading">
          <div><p className="section-kicker">交互实验</p><h2 id="playgrounds-title">动一下，原理就清楚了。</h2></div>
          <p>无需安装环境，在浏览器里试试控制与运动学。</p>
        </div>
        <div className={styles.playgroundGrid}>
          {playgrounds.map(({number, title, subtitle, description, to, icon: Icon}) => (
            <Link to={to} className={styles.playgroundCard} key={to}>
              <div className={styles.playgroundTop}><Icon size={26} strokeWidth={1.6} aria-hidden="true" /><span>CS123 / {number}</span></div>
              <p className={styles.playgroundSubtitle}>{subtitle}</p>
              <h3>{title}</h3><p>{description}</p>
              <span className="site-text-link">打开实验 <ArrowUpRight size={17} aria-hidden="true" /></span>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}

function Community() {
  return (
    <section className={styles.community} aria-labelledby="community-title">
      <div className="home-container">
        <div className={styles.communityInner}>
          <div><p className="section-kicker">与 DATAWHALE 一起学习</p><h2 id="community-title">每一次复现，都让开源更进一步。</h2><p>分享实验结果，修正一处笔误，或提出一个好问题。<br />你的学习经验，也可以成为下一个人的起点。</p></div>
          <div className={styles.communityLinks}>
            <a className="site-button site-button--primary" href="https://github.com/datawhalechina/dive-into-embodied-ai">在 GitHub 参与共建 <ArrowUpRight size={18} aria-hidden="true" /></a>
            <div><a href="https://github.com/datawhalechina/dive-into-embodied-ai/issues">反馈问题</a><a href="https://github.com/datawhalechina/dive-into-embodied-ai/discussions">参与讨论</a></div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home(): React.JSX.Element {
  return (
    <Layout title="动手学具身智能" description="Datawhale 开源具身智能教程。从机器人学、强化学习与 VLA 理论，到 MuJoCo 仿真、交互实验和真机项目。">
      <main className={styles.home}>
        <HomepageHeader />
        <HomepageFeatures />
        <HomepageProjects />
        <Playgrounds />
        <Community />
      </main>
    </Layout>
  );
}
