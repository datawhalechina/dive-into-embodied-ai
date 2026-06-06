import React, { useEffect, useState } from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import {
  ArrowRight,
  BookOpen,
  Code2,
  CodeXml,
  Cpu,
  ExternalLink,
  GitFork,
  PlayCircle,
  Terminal,
  Users,
  Wrench,
} from 'lucide-react';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

const TYPEWRITER_PHRASES = [
  'open curriculum for embodied AI',
  'robotics · simulation · control · VLA',
  'from readable notes to runnable labs',
  'built in the open with Datawhale',
];

const PROJECT_SIGNALS = [
  { label: 'License', value: 'CC BY-NC-SA 4.0' },
  { label: 'Status', value: 'Alpha' },
  { label: 'Tracks', value: '4 learning routes' },
  { label: 'Labs', value: 'CS123 playgrounds' },
];

const LAB_PIPELINE = [
  { icon: BookOpen, title: '读懂主线', text: '从路线图、机器人学、仿真与 VLA 建立最小知识闭环。' },
  { icon: Terminal, title: '跑通实验', text: '用可交互 playground 和 MuJoCo / ROS2 项目把公式落到代码。' },
  { icon: GitFork, title: '参与共建', text: '围绕 Issue、章节修订和实验复现沉淀可复用的开源材料。' },
];

const CONSOLE_LINES = [
  '$ git clone datawhalechina/dive-into-embodied-ai',
  '$ npm run dev',
  '$ open /cs123/pd-playground',
];

const REPO_METRICS = [
  { label: 'foundation', value: 'ROS2 · RL · VLA' },
  { label: 'practice', value: 'robot arm · quadruped' },
  { label: 'onboarding', value: 'learning path · quadruped lab' },
];

function Typewriter() {
  const [text, setText] = useState('');
  useEffect(() => {
    let phraseIndex = 0;
    let charIndex = 0;
    let deleting = false;
    let timer: ReturnType<typeof setTimeout>;

    const tick = () => {
      const word = TYPEWRITER_PHRASES[phraseIndex];
      charIndex += deleting ? -1 : 1;
      setText(word.substring(0, charIndex));

      let delay = deleting ? 45 : 90;
      if (!deleting && charIndex === word.length) {
        delay = 2200;
        deleting = true;
      } else if (deleting && charIndex === 0) {
        deleting = false;
        phraseIndex = (phraseIndex + 1) % TYPEWRITER_PHRASES.length;
        delay = 400;
      }
      timer = setTimeout(tick, delay);
    };

    timer = setTimeout(tick, 800);
    return () => clearTimeout(timer);
  }, []);

  return <span className="typewriter">{text}</span>;
}

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
  const heroImage = useBaseUrl('/img/career.webp');
  return (
    <header className="site-hero" style={{ backgroundImage: `url(${heroImage})` }}>
      <div className="container site-hero__inner">
        <div className="site-hero__content">
          <div className="site-hero__eyebrow">
            <CodeXml size={16} aria-hidden="true" />
            <span>Datawhale open-source robotics curriculum</span>
          </div>
          <h1 className="site-hero__title" aria-label={siteConfig.title}>
            <span className="site-hero__titleLead">Dive into</span>
            <span className="site-hero__titleMain">Embodied AI</span>
          </h1>
          <p className="site-hero__subtitle">{siteConfig.tagline}</p>
          <p className="hero__typewriter">
            <Typewriter />
          </p>
          <p className="site-hero__copy">
            给想入门、转行或求职具身智能的同学准备。用开源教程、可运行实验和机器人项目，
            把「认知 → 项目 → 面试」连成一条能持续迭代的学习路线。
          </p>
          <div className="site-hero__actions">
            <Link className="site-button site-button--primary" to="/docs/overview/intro">
              <PlayCircle size={18} aria-hidden="true" />
              开始学习
            </Link>
            <a
              className="site-button site-button--ghost"
              href="https://github.com/datawhalechina/dive-into-embodied-ai"
            >
              <GitFork size={18} aria-hidden="true" />
              GitHub
              <ExternalLink size={15} aria-hidden="true" />
            </a>
          </div>
          <div className="site-hero__signals" aria-label="项目状态">
            {PROJECT_SIGNALS.map((item) => (
              <div className="site-hero__signal" key={item.label}>
                <span>{item.label}</span>
                <strong>{item.value}</strong>
              </div>
            ))}
          </div>
        </div>
      </div>
    </header>
  );
}

function OpenSourceBand() {
  return (
    <section className="oss-band" aria-label="开源项目协作入口">
      <div className="container oss-band__inner">
        <div>
          <p className="section-kicker">OPEN SOURCE FIRST</p>
          <h2>像维护工程仓库一样维护学习路线</h2>
          <p>
            章节、代码、实验和求职材料都围绕可复现、可讨论、可贡献来组织。读者不是只消费内容，
            也可以通过 Issue、PR 和组队学习把自己的复现经验回流到项目里。
          </p>
        </div>
        <div className="oss-band__links">
          <a href="https://github.com/datawhalechina/dive-into-embodied-ai/issues">
            <Code2 size={18} aria-hidden="true" />
            提 Issue
          </a>
          <a href="https://github.com/datawhalechina/dive-into-embodied-ai/discussions">
            <Users size={18} aria-hidden="true" />
            参与讨论
          </a>
          <Link to="/docs/practices/quadruped/cs123/intro">
            <Wrench size={18} aria-hidden="true" />
            进入实验
          </Link>
        </div>
      </div>
    </section>
  );
}

function RepoConsole() {
  return (
    <section className="repo-console" aria-label="开源仓库与机器人实验台">
      <div className="container repo-console__inner">
        <div className="repo-console__copy">
          <p className="section-kicker">REPO TO ROBOT</p>
          <h2>把教程做成可运行、可复现、可贡献的实验台</h2>
          <p>
            首页给路线，文档给原理，Playground 给可拖动的反馈。它应该看起来像一个正在生长的开源仓库，
            也像一套机器人实验记录，而不是静态宣传页。
          </p>
          <Link to="/docs/practices/intro" className="repo-console__link">
            查看实践项目 <ArrowRight size={16} aria-hidden="true" />
          </Link>
        </div>
        <div className="repo-console__terminal" aria-label="本地运行命令示例">
          <div className="repo-console__terminalTop">
            <span />
            <span />
            <span />
            <strong>embodied-ai.local</strong>
          </div>
          <div className="repo-console__terminalBody">
            {CONSOLE_LINES.map((line) => (
              <code key={line}>{line}</code>
            ))}
          </div>
          <div className="repo-console__metrics">
            {REPO_METRICS.map((item) => (
              <div key={item.label}>
                <span>{item.label}</span>
                <strong>{item.value}</strong>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

function LabPipeline() {
  return (
    <section className="lab-pipeline" aria-label="学习到实验的链路">
      <div className="container">
        <div className="section-heading">
          <p className="section-kicker">ROBOTICS WORKFLOW</p>
          <h2>从开源路线到机器人实验闭环</h2>
        </div>
        <div className="lab-pipeline__grid">
          {LAB_PIPELINE.map((item, index) => {
            const Icon = item.icon;
            return (
              <article className="lab-pipeline__item" key={item.title}>
                <div className="lab-pipeline__step">0{index + 1}</div>
                <Icon size={22} aria-hidden="true" />
                <h3>{item.title}</h3>
                <p>{item.text}</p>
              </article>
            );
          })}
        </div>
        <div className="lab-pipeline__cta">
          <Cpu size={18} aria-hidden="true" />
          <span>优先推荐从 CS123 Playground 开始，把控制、运动学和步态规划先拖动起来。</span>
          <Link to="/cs123/pd-playground">
            打开 Playground <ArrowRight size={16} aria-hidden="true" />
          </Link>
        </div>
      </div>
    </section>
  );
}

export default function Home(): React.JSX.Element {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout title={siteConfig.title} description={siteConfig.tagline}>
      <HomepageHeader />
      <main className="homepage-main">
        <OpenSourceBand />
        <RepoConsole />
        <HomepageFeatures />
        <LabPipeline />
      </main>
    </Layout>
  );
}
