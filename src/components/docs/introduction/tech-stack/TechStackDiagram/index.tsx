import React, {useId, useState} from 'react';
import {translate} from '@docusaurus/Translate';
import {
  ArrowRight,
  ArrowUp,
  Box,
  BrainCircuit,
  Cog,
  Cpu,
  Database,
  Eye,
  HelpCircle,
  Layers,
  type LucideIcon,
  Repeat2,
  Rocket,
  Wrench,
} from 'lucide-react';
import styles from './styles.module.css';

interface StackModule {
  id: string;
  code: string;
  anchor: string;
  band: 'runtime' | 'support';
  name: string;
  pillar: string;
  rate: string;
  keywords: string;
  problem: string;
  tools: string;
  example: string;
  icon: LucideIcon;
  tone: 'blue' | 'purple' | 'amber' | 'green' | 'teal' | 'indigo' | 'slate';
}

const modules: StackModule[] = [
  {
    id: 'perception',
    code: '6.1',
    anchor: '#perception',
    band: 'runtime',
    name: translate({id: 'techStack.m1.name', message: '感知与定位'}),
    pillar: translate({id: 'techStack.m1.pillar', message: '运行闭环 · 感官'}),
    rate: translate({id: 'techStack.m1.rate', message: '数十 Hz'}),
    keywords: translate({id: 'techStack.m1.keywords', message: '标定 · 3D 视觉 · SLAM'}),
    problem: translate({
      id: 'techStack.m1.problem',
      message: '周围有什么、杯子在三维空间的什么位置、机器人自己走到了哪、机身有没有偏斜？',
    }),
    tools: translate({
      id: 'techStack.m1.tools',
      message: 'RGB-D / 激光 / IMU / 触觉、手眼标定、6D 位姿、状态估计、tf2、SLAM',
    }),
    example: translate({
      id: 'techStack.m1.example',
      message: '定位红杯子与盘子的三维坐标，持续跟踪机器人在厨房里的位姿。',
    }),
    icon: Eye,
    tone: 'blue',
  },
  {
    id: 'brain',
    code: '6.2',
    anchor: '#foundation-models',
    band: 'runtime',
    name: translate({id: 'techStack.m2.name', message: '大模型与策略'}),
    pillar: translate({id: 'techStack.m2.pillar', message: '运行闭环 · 大脑'}),
    rate: translate({id: 'techStack.m2.rate', message: '数～数十 Hz'}),
    keywords: translate({id: 'techStack.m2.keywords', message: 'VLM · IL / VLA · 世界模型'}),
    problem: translate({
      id: 'techStack.m2.problem',
      message: '怎样听懂自然语言指令、把语义与画面对齐，并从示范或交互中学会生成动作？',
    }),
    tools: translate({
      id: 'techStack.m2.tools',
      message: 'VLM、模仿学习（ACT / Diffusion Policy）、VLA（OpenVLA / π0）、世界模型',
    }),
    example: translate({
      id: 'techStack.m2.example',
      message: '将“红杯子放到盘子里”与画面中的物体对应起来，输出抓取与放置动作序列。',
    }),
    icon: BrainCircuit,
    tone: 'purple',
  },
  {
    id: 'control',
    code: '6.3',
    anchor: '#motion-control',
    band: 'runtime',
    name: translate({id: 'techStack.m3.name', message: '规划与运控'}),
    pillar: translate({id: 'techStack.m3.pillar', message: '运行闭环 · 小脑'}),
    rate: translate({id: 'techStack.m3.rate', message: '数十～上千 Hz'}),
    keywords: translate({id: 'techStack.m3.keywords', message: 'MoveIt 2 · MPC / 力控 · RL'}),
    problem: translate({
      id: 'techStack.m3.problem',
      message: '手臂怎么绕开障碍平滑伸过去？碰到杯子怎么控力不捏碎？双腿或底盘怎么稳稳走？',
    }),
    tools: translate({
      id: 'techStack.m3.tools',
      message: 'FK/IK、MoveIt 2 / Nav2、PID / LQR / MPC / WBC、阻抗力控、Pinocchio、PPO 运控',
    }),
    example: translate({
      id: 'techStack.m3.example',
      message: '生成无碰撞轨迹，抓取杯子时柔顺控力，并在移动过程中保持身体稳定。',
    }),
    icon: Cog,
    tone: 'amber',
  },
  {
    id: 'simulation',
    code: '6.4',
    anchor: '#simulation',
    band: 'support',
    name: translate({id: 'techStack.m4.name', message: '仿真工程'}),
    pillar: translate({id: 'techStack.m4.pillar', message: '支撑 · 虚拟试错'}),
    rate: translate({id: 'techStack.m4.rate', message: '大规模并行'}),
    keywords: translate({id: 'techStack.m4.keywords', message: 'MuJoCo · Isaac Lab'}),
    problem: translate({
      id: 'techStack.m4.problem',
      message: '真机昂贵、训练慢、摔坏难修，怎样先在虚拟世界里低成本并行试错与重复评测？',
    }),
    tools: translate({
      id: 'techStack.m4.tools',
      message: 'URDF / MJCF、MuJoCo、Isaac Sim / Isaac Lab、Gymnasium、域随机化',
    }),
    example: translate({
      id: 'techStack.m4.example',
      message: '先在虚拟厨房中并行跑上千次抓取和行走试验，验证稳定后再迁移真机。',
    }),
    icon: Box,
    tone: 'green',
  },
  {
    id: 'data',
    code: '6.5',
    anchor: '#data-engineering',
    band: 'support',
    name: translate({id: 'techStack.m5.name', message: '数据工程'}),
    pillar: translate({id: 'techStack.m5.pillar', message: '支撑 · 经验沉淀'}),
    rate: translate({id: 'techStack.m5.rate', message: '轨迹流水线'}),
    keywords: translate({id: 'techStack.m5.keywords', message: '遥操作 · LeRobot'}),
    problem: translate({
      id: 'techStack.m5.problem',
      message: '学习型策略高度依赖示教轨迹，怎样采集、对齐、清洗并持续管理高质量训练数据？',
    }),
    tools: translate({
      id: 'techStack.m5.tools',
      message: '主从遥操作、多路时间戳同步、数据质检与归一化、LeRobot、OXE / DROID',
    }),
    example: translate({
      id: 'techStack.m5.example',
      message: '录下人类示范抓杯子的多视角图像与关节轨迹，剔除掉帧和异常片段。',
    }),
    icon: Database,
    tone: 'teal',
  },
  {
    id: 'deployment',
    code: '6.6',
    anchor: '#deployment',
    band: 'support',
    name: translate({id: 'techStack.m6.name', message: '部署工程'}),
    pillar: translate({id: 'techStack.m6.pillar', message: '支撑 · 真机运行'}),
    rate: translate({id: 'techStack.m6.rate', message: '低延迟实时'}),
    keywords: translate({id: 'techStack.m6.keywords', message: 'ROS 2 · TensorRT · RTC'}),
    problem: translate({
      id: 'techStack.m6.problem',
      message: '显卡上跑通的模型怎样低延迟、不掉线、不撞机地接入真实传感器与执行器？',
    }),
    tools: translate({
      id: 'techStack.m6.tools',
      message: 'ROS 2 / DDS、ONNX Runtime / TensorRT、RTC 异步执行、Jetson 端侧、限幅急停',
    }),
    example: translate({
      id: 'techStack.m6.example',
      message: '把策略推理与高频控制器解耦部署到机载平台，并加上速度与力矩保护。',
    }),
    icon: Rocket,
    tone: 'indigo',
  },
  {
    id: 'hardware',
    code: '6.7',
    anchor: '#hardware',
    band: 'support',
    name: translate({id: 'techStack.m7.name', message: '硬件基础'}),
    pillar: translate({id: 'techStack.m7.pillar', message: '支撑 · 物理边界'}),
    rate: translate({id: 'techStack.m7.rate', message: '物理上限'}),
    keywords: translate({id: 'techStack.m7.keywords', message: '机构 · 驱动 · CAN'}),
    problem: translate({
      id: 'techStack.m7.problem',
      message: '算法算出的速度和力矩，骨架、电机、供电与总线扛不扛得住？出错时是算法还是硬件上限？',
    }),
    tools: translate({
      id: 'techStack.m7.tools',
      message: '连杆与减速器、电机与三环驱动、编码器、供电安全、MCU 与 CAN 总线',
    }),
    example: translate({
      id: 'techStack.m7.example',
      message: '保证机械臂负载、夹爪刚度、关节响应速度和总线带宽满足拿杯任务需求。',
    }),
    icon: Cpu,
    tone: 'slate',
  },
];

const runtimeModules = modules.filter(m => m.band === 'runtime');
const supportModules = modules.filter(m => m.band === 'support');

export default function TechStackDiagram(): React.JSX.Element {
  const [selectedId, setSelectedId] = useState<string>('perception');
  const id = useId().replace(/:/g, '');
  const current = modules.find(m => m.id === selectedId) ?? modules[0];

  return (
    <figure className={styles.figure} aria-labelledby={`${id}-title`}>
      <figcaption className={styles.header}>
        <div>
          <p className={styles.eyebrow}>
            {translate({id: 'techStack.eyebrow', message: '具身智能 · 技术栈全景'})}
          </p>
          <h3 className={styles.title} id={`${id}-title`}>
            {translate({id: 'techStack.title', message: '上层运行闭环，下层工程与物理支撑'})}
          </h3>
          <p className={styles.instruction}>
            {translate({
              id: 'techStack.instruction',
              message: '点选任意模块，查看它解决的问题、代表工具，以及在“拿红杯子”任务中的角色。',
            })}
          </p>
        </div>
        <span className={styles.badge}>
          <Layers size={15} aria-hidden="true" />
          {translate({id: 'techStack.badge', message: '7 个模块 · 两层协作'})}
        </span>
      </figcaption>

      <div className={styles.canvas}>
        <section
          className={styles.runtimeBand}
          aria-label={translate({id: 'techStack.runtimeAria', message: '运行时闭环模块'})}>
          <div className={styles.bandHead}>
            <span className={styles.bandTitle}>
              {translate({id: 'techStack.runtimeTitle', message: '运行时闭环'})}
            </span>
            <span className={styles.bandMeta}>
              {translate({id: 'techStack.runtimeMeta', message: '感官 → 大脑 → 小脑 · 慢决策与快执行协同'})}
            </span>
          </div>

          <div className={styles.runtimeGrid} role="group" aria-label={translate({id: 'techStack.selectRuntime', message: '选择运行闭环模块'})}>
            {runtimeModules.map((mod, index) => {
              const Icon = mod.icon;
              return (
                <div className={`${styles.cardWrap} ${styles[mod.tone]}`} key={mod.id}>
                  <button
                    type="button"
                    className={styles.card}
                    aria-pressed={selectedId === mod.id}
                    aria-controls={`${id}-detail`}
                    onClick={() => setSelectedId(mod.id)}>
                    <span className={styles.cardTop}>
                      <span className={styles.iconPill} aria-hidden="true">
                        <Icon size={18} strokeWidth={1.8} aria-hidden="true" />
                      </span>
                      <span className={styles.code}>{mod.code}</span>
                    </span>
                    <strong className={styles.cardName}>{mod.name}</strong>
                    <span className={styles.keywords}>{mod.keywords}</span>
                    <span className={styles.rate}>{mod.rate}</span>
                  </button>
                  {index < runtimeModules.length - 1 && (
                    <ArrowRight className={styles.arrow} size={18} aria-hidden="true" />
                  )}
                </div>
              );
            })}
          </div>

          <div className={styles.returnLoop} aria-hidden="true">
            <svg className={styles.returnSvg} viewBox="0 0 900 38" preserveAspectRatio="none">
              <path d="M 750 0 V 14 Q 750 26 736 26 H 164 Q 150 26 150 14 V 2" />
            </svg>
            <ArrowUp className={styles.returnArrow} size={14} aria-hidden="true" />
            <span className={styles.returnLabel}>
              <Repeat2 size={14} aria-hidden="true" />
              {translate({id: 'techStack.feedback', message: '物理交互产生新状态，回传感知进入下一拍'})}
            </span>
          </div>
        </section>

        <div className={styles.bridge} aria-hidden="true">
          <span className={styles.bridgeLine} />
          <span className={styles.bridgePill}>
            <ArrowUp size={13} aria-hidden="true" />
            {translate({id: 'techStack.bridge', message: '支撑训练、数据迭代与真机稳定运行'})}
            <ArrowUp size={13} aria-hidden="true" />
          </span>
          <span className={styles.bridgeLine} />
        </div>

        <section
          className={styles.supportBand}
          aria-label={translate({id: 'techStack.supportAria', message: '工程与物理支撑模块'})}>
          <div className={styles.bandHead}>
            <span className={styles.bandTitle}>
              {translate({id: 'techStack.supportTitle', message: '工程底座与物理边界'})}
            </span>
            <span className={styles.bandMeta}>
              {translate({id: 'techStack.supportMeta', message: '练得起 · 学得好 · 跑得稳 · 扛得住'})}
            </span>
          </div>

          <div className={styles.supportGrid} role="group" aria-label={translate({id: 'techStack.selectSupport', message: '选择支撑模块'})}>
            {supportModules.map(mod => {
              const Icon = mod.icon;
              return (
                <div className={`${styles.cardWrap} ${styles[mod.tone]}`} key={mod.id}>
                  <button
                    type="button"
                    className={`${styles.card} ${styles.supportCard}`}
                    aria-pressed={selectedId === mod.id}
                    aria-controls={`${id}-detail`}
                    onClick={() => setSelectedId(mod.id)}>
                    <span className={styles.cardTop}>
                      <span className={styles.iconPill} aria-hidden="true">
                        <Icon size={17} strokeWidth={1.8} aria-hidden="true" />
                      </span>
                      <span className={styles.code}>{mod.code}</span>
                    </span>
                    <strong className={styles.cardName}>{mod.name}</strong>
                    <span className={styles.keywords}>{mod.keywords}</span>
                    <span className={styles.rate}>{mod.rate}</span>
                  </button>
                </div>
              );
            })}
          </div>
        </section>
      </div>

      <div className={`${styles.detail} ${styles[current.tone]}`} id={`${id}-detail`}>
        <div className={styles.detailHead}>
          <div role="status" aria-live="polite" aria-atomic="true">
            <p className={styles.detailKicker}>
              <span className={styles.detailCode}>{current.code}</span>
              {current.pillar}
            </p>
            <h4 className={styles.detailTitle}>{current.name}</h4>
          </div>
          <a className={styles.jumpLink} href={current.anchor}>
            {translate({id: 'techStack.jump', message: '查看本节详解'})}
            <ArrowRight size={15} aria-hidden="true" />
          </a>
        </div>

        <dl className={styles.detailGrid}>
          <div>
            <dt>
              <HelpCircle size={15} aria-hidden="true" />
              {translate({id: 'techStack.labelProblem', message: '解决什么问题'})}
            </dt>
            <dd>{current.problem}</dd>
          </div>
          <div>
            <dt>
              <Wrench size={15} aria-hidden="true" />
              {translate({id: 'techStack.labelTools', message: '代表技术与工具'})}
            </dt>
            <dd>{current.tools}</dd>
          </div>
        </dl>

        <p className={styles.example}>
          <span>{translate({id: 'techStack.exampleLabel', message: '在“拿红杯子”任务中'})}</span>
          {current.example}
        </p>
      </div>
    </figure>
  );
}
