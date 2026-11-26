import {translate} from '@docusaurus/Translate';
import React, {useEffect, useState} from 'react';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Layout from '@theme/Layout';
import {ArrowRight, ArrowUpRight, Play, SlidersHorizontal, Move3D, Crosshair, Pause} from 'lucide-react';
import g1PushGif from '@site/docs/practices/humanoid/unitree-g1/figs/unitree-g1-push.gif';
import microduckWalkGif from '@site/docs/practices/amd/microduck-rl/figs/microduck-amd-r9700-model750.gif';
import actSuccessGif from '@site/docs/practices/vla/act/figs/act_50k_success.gif';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import HomepageProjects from '@site/src/components/HomepageProjects';
import TranslationNotice from '@site/src/components/TranslationNotice';
import styles from './index.module.css';

const playgrounds = [
  {number: '01', title: translate({message: "调好第一个控制器"}), subtitle: translate({message: "PD 控制"}), description: translate({id: "home.pg.pdDesc", message: "调增益，实时看响应与超调。"}), to: '/cs123/pd-playground', icon: SlidersHorizontal},
  {number: '02', title: translate({message: "看懂关节如何运动"}), subtitle: translate({message: "正运动学"}), description: translate({id: "home.pg.fkDesc", message: "转关节，直观看足端坐标变化。"}), to: '/cs123/fk-playground', icon: Move3D},
  {number: '03', title: translate({message: "让足端走到目标点"}), subtitle: translate({message: "逆运动学"}), description: translate({id: "home.pg.ikDesc", message: "拖目标点，探索机械腿可达空间。"}), to: '/cs123/ik-playground', icon: Crosshair},
];

function HomepageHeader() {
  const g1Poster = useBaseUrl('/img/unitree-g1-poster.webp');
  const microduckPoster = useBaseUrl('/img/microduck-poster.webp');
  const actPoster = useBaseUrl('/img/act-poster.webp');
  const [active, setActive] = useState(0);
  const [paused, setPaused] = useState(false);

  useEffect(() => {
    const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
    const sync = () => { if (preference.matches) setPaused(true); };
    sync();
    preference.addEventListener('change', sync);
    return () => preference.removeEventListener('change', sync);
  }, []);

  const showcase = [
    {
      id: 'g1',
      tab: translate({id: "home.stage.g1Tab", message: "Unitree G1 人形"}),
      badge: translate({id: "home.stage.g1Badge", message: "50 Hz LSTM · 浏览器可玩"}),
      caption: translate({id: "home.stage.g1Caption", message: "12-DoF 双足行走与 0.55 m/s 侧向推撞恢复"}),
      action: translate({id: "home.stage.g1Action", message: "去在线仿真操控"}),
      to: '/simulation',
      gif: g1PushGif,
      poster: g1Poster,
      alt: translate({id: "home.stage.g1Alt", message: "Unitree G1 人形机器人在 MuJoCo 仿真中抵抗侧向推撞"}),
    },
    {
      id: 'microduck',
      tab: translate({id: "home.stage.duckTab", message: "MicroDuck 双足"}),
      badge: translate({id: "home.stage.duckBadge", message: "PPO · ROCm / CUDA"}),
      caption: translate({id: "home.stage.duckCaption", message: "MuJoCo Warp 并行强化学习与 12 步跳墙课程"}),
      action: translate({id: "home.stage.duckAction", message: "进入实战教程"}),
      to: '/docs/practices/humanoid/microduck-rl',
      gif: microduckWalkGif,
      poster: microduckPoster,
      alt: translate({id: "home.stage.duckAlt", message: "MicroDuck 双足机器人在仿真中稳定行走"}),
    },
    {
      id: 'act',
      tab: translate({id: "home.stage.actTab", message: "ACT 双臂操作"}),
      badge: translate({id: "home.stage.actBadge", message: "ALOHA · 模仿学习"}),
      caption: translate({id: "home.stage.actCaption", message: "动作分块策略复现与 MuJoCo 闭环评测"}),
      action: translate({id: "home.stage.actAction", message: "查看双臂教程"}),
      to: '/docs/practices/vla/act',
      gif: actSuccessGif,
      poster: actPoster,
      alt: translate({id: "home.stage.actAlt", message: "ACT 双臂机器人在 ALOHA 仿真中完成方块交接"}),
    },
  ];
  const current = showcase[active];

  return (
    <header className={styles.hero}>
      <div className={styles.heroGrid}>
        <div className={styles.heroCopy}>
          <Link className={styles.eyebrow} to="/docs/introduction/intro">
            <span className={styles.statusDot} />
            {translate({message: "DATAWHALE · 开源具身智能教程"})}
            <ArrowUpRight size={14} aria-hidden="true" />
          </Link>
          <h1>
            {translate({message: "让机器人，"})}
            <br />
            <span>{translate({message: "感知并行动。"})}</span>
          </h1>
          <p className={styles.intro}>
            {translate({id: "home.heroLead", message: "浏览器零配置玩仿真，从零到一跑通人形、四足与机械臂。"})}
          </p>
          <div className={styles.actions}>
            <Link className="site-button site-button--primary" to="/simulation">
              <Play size={16} aria-hidden="true" />
              {translate({id: "home.ctaSim", message: "免费玩在线仿真"})}
            </Link>
            <Link className={styles.heroSecondaryBtn} to="/docs/introduction/intro">
              {translate({message: "新手入门"})}
            </Link>
            <a className={styles.heroAnchor} href="#learning-paths">
              {translate({id: "home.ctaPath", message: "30 秒选路线"})}
              <ArrowRight size={15} aria-hidden="true" />
            </a>
          </div>
          <ul className={styles.heroStats} aria-label={translate({message: "课程特色"})}>
            <li><strong>7</strong><span>{translate({id: "home.statRobots", message: "款浏览器机器人"})}</span></li>
            <li><strong>29</strong><span>{translate({id: "home.statProjects", message: "篇完整实战"})}</span></li>
            <li><strong>0</strong><span>{translate({id: "home.statSetup", message: "配置上手体验"})}</span></li>
          </ul>
        </div>

        <div className={styles.stageCard}>
          <div className={styles.stageViewport}>
            <img
              key={current.id}
              src={paused ? current.poster : current.gif}
              alt={current.alt}
              width="640"
              height="480"
            />
            <span className={styles.stageBadge}>{current.badge}</span>
            <button
              type="button"
              className={styles.stageMotionToggle}
              onClick={() => setPaused(v => !v)}
              aria-label={paused ? translate({message: "播放背景动画"}) : translate({message: "暂停背景动画"})}
              title={paused ? translate({message: "播放背景动画"}) : translate({message: "暂停背景动画"})}
            >
              {paused ? <Play size={14} aria-hidden="true" /> : <Pause size={14} aria-hidden="true" />}
            </button>
          </div>
          <div className={styles.stageFooter}>
            <div className={styles.stageTabs} role="tablist" aria-label={translate({id: "home.stageTabsLabel", message: "切换机器人演示"})}>
              {showcase.map((item, idx) => (
                <button
                  key={item.id}
                  type="button"
                  role="tab"
                  aria-selected={active === idx}
                  className={active === idx ? styles.stageTabActive : styles.stageTab}
                  onClick={() => setActive(idx)}
                >
                  {item.tab}
                </button>
              ))}
            </div>
            <div className={styles.stageMeta}>
              <p>{current.caption}</p>
              <Link to={current.to}>
                {current.action}
                <ArrowUpRight size={15} aria-hidden="true" />
              </Link>
            </div>
          </div>
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
          <div>
            <p className="section-kicker">{translate({message: "交互实验"})}</p>
            <h2 id="playgrounds-title">{translate({message: "动一下，原理就清楚了。"})}</h2>
          </div>
          <Link className="site-text-link" to="/simulation">
            {translate({id: "home.pg.trySim", message: "进入 7 款机器人在线仿真"})}
            <ArrowRight size={17} aria-hidden="true" />
          </Link>
        </div>
        <div className={styles.playgroundGrid}>
          {playgrounds.map(({number, title, subtitle, description, to, icon: Icon}) => (
            <Link to={to} className={styles.playgroundCard} key={to}>
              <div className={styles.playgroundTop}>
                <Icon size={24} strokeWidth={1.6} aria-hidden="true" />
                <span>CS123 / {number}</span>
              </div>
              <p className={styles.playgroundSubtitle}>{subtitle}</p>
              <h3>{title}</h3>
              <p>{description}</p>
              <span className="site-text-link">{translate({message: "打开实验"})}<ArrowUpRight size={16} aria-hidden="true" /></span>
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
          <div>
            <p className="section-kicker">{translate({message: "与 DATAWHALE 一起学习"})}</p>
            <h2 id="community-title">{translate({message: "每一次复现，都让开源更进一步。"})}</h2>
            <p>{translate({id: "home.communityShort", message: "完全免费开源（CC BY-NC-SA 4.0），欢迎提 Issue、分享复现结果或认领专题共建。"})}</p>
          </div>
          <div className={styles.communityLinks}>
            <a className="site-button site-button--primary" href="https://github.com/datawhalechina/dive-into-embodied-ai">
              {translate({message: "在 GitHub 参与共建"})}
              <ArrowUpRight size={18} aria-hidden="true" />
            </a>
            <div>
              <a href="https://github.com/datawhalechina/dive-into-embodied-ai/issues">{translate({message: "反馈问题"})}</a>
              <a href="https://github.com/datawhalechina/dive-into-embodied-ai/discussions">{translate({message: "参与讨论"})}</a>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home(): React.JSX.Element {
  return (
    <Layout wrapperClassName={styles.homeLayout} title={translate({message: "动手学具身智能"})} description={translate({message: "Datawhale 开源具身智能学习站：具身导论、理论基础与项目实战，连接机器人学习、论文、数据集和开源工具。"})}>
      <main className={styles.home}>
        <HomepageHeader />
        <div id="learning-paths" className={styles.learningPaths}>
          <div className="home-container"><TranslationNotice translated /></div>
          <HomepageFeatures />
        </div>
        <HomepageProjects />
        <Playgrounds />
        <Community />
      </main>
    </Layout>
  );
}
