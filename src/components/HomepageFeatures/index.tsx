import {translate} from '@docusaurus/Translate';
import React, {useState} from 'react';
import Link from '@docusaurus/Link';
import {ArrowUpRight, Compass, Brain, Wrench, Cpu} from 'lucide-react';
import styles from './styles.module.css';

type Step = {
  step: string;
  title: string;
  outcome: string;
  meta: string;
  tone: 'free' | 'gpu' | 'real';
  to: string;
};

type PathItem = {
  id: string;
  label: string;
  hint: string;
  icon: typeof Compass;
  steps: Step[];
};

export default function HomepageFeatures(): React.JSX.Element {
  const [selected, setSelected] = useState('beginner');

  const paths: PathItem[] = [
    {
      id: 'beginner',
      label: translate({id: "path.beginner.label", message: "零基础入门"}),
      hint: translate({id: "path.beginner.hint", message: "无需显卡 · 先玩再学"}),
      icon: Compass,
      steps: [
        {
          step: '01',
          title: translate({id: "path.beginner.s1Title", message: "3 分钟玩在线仿真"}),
          outcome: translate({id: "path.beginner.s1Desc", message: "操控 G1 / Go2 / 小黄鸭并体验外力推撞"}),
          meta: translate({id: "path.meta.browser", message: "零配置 · 浏览器"}),
          tone: 'free',
          to: '/simulation',
        },
        {
          step: '02',
          title: translate({id: "path.beginner.s2Title", message: "速览具身导论"}),
          outcome: translate({id: "path.beginner.s2Desc", message: "用 20 分钟看清任务、载体与技术栈"}),
          meta: translate({id: "path.meta.read", message: "入门 · 图文导读"}),
          tone: 'free',
          to: '/docs/introduction/intro',
        },
        {
          step: '03',
          title: translate({id: "path.beginner.s3Title", message: "从 0 搭第一台四足"}),
          outcome: translate({id: "path.beginner.s3Desc", message: "配交互式 PD/FK/IK 跑通完整控制闭环"}),
          meta: translate({id: "path.meta.cpu", message: "入门 · 仅 CPU 可跑"}),
          tone: 'free',
          to: '/docs/practices/quadruped/cs123/intro',
        },
      ],
    },
    {
      id: 'algo',
      label: translate({id: "path.algo.label", message: "算法 / AI 转具身"}),
      hint: translate({id: "path.algo.hint", message: "有 PyTorch 基础"}),
      icon: Brain,
      steps: [
        {
          step: '01',
          title: translate({id: "path.algo.s1Title", message: "按需补齐控制与坐标系"}),
          outcome: translate({id: "path.algo.s1Desc", message: "快速打通 PID→MPC、标定与 ROS 2"}),
          meta: translate({id: "path.meta.theory", message: "理论地图 · 速查"}),
          tone: 'free',
          to: '/foundations',
        },
        {
          step: '02',
          title: translate({id: "path.algo.s2Title", message: "复现 Unitree G1 人形运控"}),
          outcome: translate({id: "path.algo.s2Desc", message: "拆解 16 项奖励、万轮 PPO 与 Sim2Sim"}),
          meta: translate({id: "path.meta.gpu", message: "进阶 · 单卡 GPU"}),
          tone: 'gpu',
          to: '/docs/practices/humanoid/unitree-g1',
        },
        {
          step: '03',
          title: translate({id: "path.algo.s3Title", message: "实战 VLA（ACT / π₀.₅）"}),
          outcome: translate({id: "path.algo.s3Desc", message: "复现 ALOHA 双臂与 LIBERO 后训练"}),
          meta: translate({id: "path.meta.gpu10", message: "进阶 · 10GB+ 显存"}),
          tone: 'gpu',
          to: '/docs/practices/vla/act',
        },
      ],
    },
    {
      id: 'hardware',
      label: translate({id: "path.hw.label", message: "硬件 / 控制转具身"}),
      hint: translate({id: "path.hw.hint", message: "懂机械或嵌入式"}),
      icon: Wrench,
      steps: [
        {
          step: '01',
          title: translate({id: "path.hw.s1Title", message: "把控制直觉接到 RL"}),
          outcome: translate({id: "path.hw.s1Desc", message: "看懂奖励、策略网络与并行仿真训练"}),
          meta: translate({id: "path.meta.rl", message: "理论 · RL 入门"}),
          tone: 'free',
          to: '/docs/foundations/rl-for-robotics/intro',
        },
        {
          step: '02',
          title: translate({id: "path.hw.s2Title", message: "CAN / MCU 与真机标定"}),
          outcome: translate({id: "path.hw.s2Desc", message: "对齐通信带宽、时间戳与安全门控"}),
          meta: translate({id: "path.meta.eng", message: "工程 · 底层底座"}),
          tone: 'real',
          to: '/docs/foundations/communication/can-mcu',
        },
        {
          step: '03',
          title: translate({id: "path.hw.s3Title", message: "SO-101 + LeRobot 真机"}),
          outcome: translate({id: "path.hw.s3Desc", message: "完成机械臂连通、遥操作数采与回放"}),
          meta: translate({id: "path.meta.real", message: "真机 · 低成本套件"}),
          tone: 'real',
          to: '/docs/practices/robot-arm/data-collection/so101-lerobot-real',
        },
      ],
    },
    {
      id: 'amd',
      label: translate({id: "path.amd.label", message: "AMD / 云算力"}),
      hint: translate({id: "path.amd.hint", message: "ROCm 或免费云端"}),
      icon: Cpu,
      steps: [
        {
          step: '01',
          title: translate({id: "path.amd.s1Title", message: "打开 AUP 云算力环境"}),
          outcome: translate({id: "path.amd.s1Desc", message: "浏览器直连 JupyterHub 与 ROCm"}),
          meta: translate({id: "path.meta.cloud", message: "云端 · 免装驱动"}),
          tone: 'free',
          to: '/docs/practices/amd/aup-learning-cloud',
        },
        {
          step: '02',
          title: translate({id: "path.amd.s2Title", message: "Go2 四足 MJX（ROCm 10）"}),
          outcome: translate({id: "path.amd.s2Desc", message: "8GB 显存可训，R9700 4.5h 满训收敛"}),
          meta: translate({id: "path.meta.rocm", message: "进阶 · ROCm / CUDA"}),
          tone: 'gpu',
          to: '/docs/practices/amd/unitree-rl-mjx',
        },
        {
          step: '03',
          title: translate({id: "path.amd.s3Title", message: "MicroDuck / ACT ROCm"}),
          outcome: translate({id: "path.amd.s3Desc", message: "原生编译 MuJoCo Warp 与 BF16 训练"}),
          meta: translate({id: "path.meta.rocm2", message: "实战 · Radeon GPU"}),
          tone: 'gpu',
          to: '/docs/practices/amd/microduck-rl',
        },
      ],
    },
  ];

  const current = paths.find(item => item.id === selected) ?? paths[0];

  return (
    <section className={styles.features} aria-labelledby="learning-title">
      <div className="home-container">
        <div className={styles.header}>
          <div>
            <p className="section-kicker">{translate({id: "path.kicker", message: "30 秒选起点"})}</p>
            <h2 id="learning-title">{translate({id: "path.title", message: "按你的背景，直接开始第一步。"})}</h2>
          </div>
          <div className={styles.tabs} role="tablist" aria-label={translate({id: "path.tabsAria", message: "选择学习背景"})}>
            {paths.map(item => {
              const Icon = item.icon;
              const isActive = item.id === current.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  role="tab"
                  aria-selected={isActive}
                  className={isActive ? styles.tabActive : styles.tab}
                  onClick={() => setSelected(item.id)}
                >
                  <Icon size={17} aria-hidden="true" />
                  <span>
                    <strong>{item.label}</strong>
                    <small>{item.hint}</small>
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        <div className={styles.stepGrid} role="tabpanel" aria-label={current.label}>
          {current.steps.map(step => (
            <Link key={step.to} to={step.to} className={styles.stepCard}>
              <div className={styles.stepTop}>
                <span className={styles.stepNum}>{step.step}</span>
                <span className={`${styles.badge} ${styles[`badge_${step.tone}`]}`}>{step.meta}</span>
              </div>
              <h3>{step.title}</h3>
              <p>{step.outcome}</p>
              <span className={styles.stepLink}>
                {translate({id: "path.go", message: "立即开始"})}
                <ArrowUpRight size={16} aria-hidden="true" />
              </span>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}
