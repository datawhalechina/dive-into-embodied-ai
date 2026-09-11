import React, {useEffect, useRef, useState} from 'react';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Layout from '@theme/Layout';
import {ArrowRight, ArrowUpRight, BookOpen, Code2, GitFork, SlidersHorizontal, Move3D, Crosshair, Pause, Play, ArrowDown} from 'lucide-react';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import HomepageProjects from '@site/src/components/HomepageProjects';
import styles from './index.module.css';

const heroVideo = 'https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260508_215831_c6a8989c-d716-4d8d-8745-e972a2eec711.mp4';
const playgrounds = [
  {number: '01', title: '调好第一个控制器', subtitle: 'PD 控制', description: '调整增益，观察响应、超调与稳定性的变化。', to: '/cs123/pd-playground', icon: SlidersHorizontal},
  {number: '02', title: '看懂关节如何运动', subtitle: '正运动学', description: '转动关节，观察足端位置与坐标系的关系。', to: '/cs123/fk-playground', icon: Move3D},
  {number: '03', title: '让足端走到目标点', subtitle: '逆运动学', description: '移动目标点，探索机器人腿的可达空间。', to: '/cs123/ik-playground', icon: Crosshair},
];

function HomepageHeader() {
  const robotPoster = useBaseUrl('/img/microduck-poster.webp');
  const stageRef = useRef<HTMLElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [motionAllowed, setMotionAllowed] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [ready, setReady] = useState(false);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
    const update = () => {
      setMotionAllowed(!preference.matches);
      if (preference.matches) videoRef.current?.pause();
    };
    update();
    preference.addEventListener('change', update);
    return () => preference.removeEventListener('change', update);
  }, []);

  const resetPerspective = () => {
    stageRef.current?.style.setProperty('--scene-x', '0px');
    stageRef.current?.style.setProperty('--scene-y', '0px');
  };
  const movePerspective = (event: React.PointerEvent<HTMLElement>) => {
    if (!motionAllowed || event.pointerType !== 'mouse') return;
    const rect = event.currentTarget.getBoundingClientRect();
    event.currentTarget.style.setProperty('--scene-x', `${(event.clientX - rect.left - rect.width / 2) * .012}px`);
    event.currentTarget.style.setProperty('--scene-y', `${(event.clientY - rect.top - rect.height / 2) * .012}px`);
  };
  const toggleVideo = async () => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      try { await video.play(); } catch { setPlaying(false); }
    } else video.pause();
  };

  return (
    <header ref={stageRef} className={styles.hero} onPointerMove={movePerspective} onPointerLeave={resetPerspective}>
      <div className={styles.sceneFallback} style={{visibility: ready && !failed && motionAllowed ? 'hidden' : 'visible'}} aria-hidden="true">
        <div className={styles.orbit} /><div className={styles.orbitInner} />
        <div className={styles.core}><div /><div /><div /><div /><div /><div /></div>
        <div className={styles.groundShadow} />
      </div>
      <video
        ref={videoRef}
        className={`${styles.heroVideo} ${ready && !failed ? styles.videoReady : ''}`}
        src={motionAllowed ? heroVideo : undefined}
        autoPlay={motionAllowed} muted loop playsInline preload="none" tabIndex={-1} aria-hidden="true"
        onLoadedData={() => setReady(true)} onError={() => setFailed(true)}
        onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)}
      />
      <div className={styles.heroVeil} aria-hidden="true" />
      <div className={styles.sceneLabel} aria-hidden="true"><span>ROBOT LEARNING / 具身智能</span><span>感知 → 决策 → 控制</span></div>
      <div className={styles.heroContent}>
        <div className={styles.heroCopy}>
          <Link className={styles.eyebrow} to="/docs/overview/intro"><span className={styles.statusDot} /> DATAWHALE · 开源具身智能教程 <ArrowUpRight size={14} aria-hidden="true" /></Link>
          <h1>让机器人，<br /><span>感知并行动。</span></h1>
          <p className={styles.intro}>从理解一个算法，到迈出机器人的第一步。<br />动手连接感知、学习与控制，在仿真中走向真实。</p>
          <div className={styles.actions}>
            <Link className={styles.heroCta} to="/docs/foundations/intro">开始学习 <ArrowRight size={17} aria-hidden="true" /></Link>
            <Link className={styles.heroSecondary} to="/docs/practices/intro">探索实战 <ArrowUpRight size={16} aria-hidden="true" /></Link>
          </div>
          <div className={styles.heroNotes} aria-label="课程特色">
            <span><BookOpen size={14} aria-hidden="true" />中文教程</span>
            <span><Code2 size={14} aria-hidden="true" />可运行代码</span>
            <span><GitFork size={14} aria-hidden="true" />社区共建</span>
          </div>
        </div>
      </div>
      <Link className={styles.robotSpotlight} to="/docs/practices/humanoid/microduck-rl">
        <img src={robotPoster} alt="MicroDuck 双足机器人仿真" width="88" height="72" />
        <div><span>从这里开始一次机器人实验</span><strong>让 MicroDuck 学会行走</strong><small>MuJoCo 仿真 · 强化学习</small></div>
        <ArrowUpRight size={19} aria-hidden="true" />
      </Link>
      <div className={styles.heroBottom}>
        <a href="#learning-paths" className={styles.scrollHint}>向下探索学习路线 <ArrowDown size={14} aria-hidden="true" /></a>
        <div className={styles.sceneControls}>
          <span>从数字仿真，到物理世界</span>
          {motionAllowed && !failed && <button type="button" onClick={toggleVideo} aria-label={playing ? '暂停背景动画' : '播放背景动画'} title={playing ? '暂停背景动画' : '播放背景动画'}>{playing ? <Pause size={15} aria-hidden="true" /> : <Play size={15} aria-hidden="true" />}</button>}
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
    <Layout wrapperClassName={styles.homeLayout} title="动手学具身智能" description="Datawhale 开源具身智能教程。从机器人学、强化学习与 VLA 理论，到 MuJoCo 仿真、交互实验和真机项目。">
      <main className={styles.home}>
        <HomepageHeader />
        <div id="learning-paths" className={styles.learningPaths}><HomepageFeatures /></div>
        <HomepageProjects />
        <Playgrounds />
        <Community />
      </main>
    </Layout>
  );
}
