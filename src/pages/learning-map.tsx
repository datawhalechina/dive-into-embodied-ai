import React, {useState} from 'react';
import Link from '@docusaurus/Link';
import useBrokenLinks from '@docusaurus/useBrokenLinks';
import Layout from '@theme/Layout';
import {
  ArrowDown, ArrowRight, ArrowUpRight, BrainCircuit, Check,
  Database, Eye, FlaskConical, Move3D, RotateCcw, Settings2,
} from 'lucide-react';
import styles from './learning-map.module.css';

const scenarios = [
  {
    name: '机械臂拿杯子',
    goal: '把桌上的杯子拿起，放进指定区域。',
    success: '杯子放对位置，保持直立，手臂不碰撞。',
    stages: ['从相机图像中定位杯子，估计它与机械臂的相对位置。', '根据目标选择抓取位置，安排靠近、夹取和放置动作。', '把动作转成关节指令，让手臂移动、夹爪闭合。'],
    feedback: '杯子移动了吗？抓稳了吗？用新的观测修正下一步动作。',
  },
  {
    name: '四足向前走',
    goal: '让四足机器人按给定速度向前行走。',
    success: '跟上目标速度，保持平衡，并减少滑倒。',
    stages: ['读取关节角度、机身姿态与接触信息，估计运动状态。', '根据速度目标和当前状态，生成步态或选择运动策略。', '跟踪关节目标，驱动电机完成抬腿、落脚和支撑。'],
    feedback: '速度偏了多少？身体是否倾斜？根据反馈调整下一步。',
  },
  {
    name: '导航到门口',
    goal: '从房间里出发，绕过障碍物到达门口。',
    success: '到达目标区域，途中不碰撞，并记录耗时。',
    stages: ['用相机或激光雷达观察障碍物，估计自身位置。', '确定门口的位置，规划路线，并随环境变化重新规划。', '把路径转成底盘的速度与转向指令，跟踪局部轨迹。'],
    feedback: '前方出现障碍了吗？位置有偏差吗？持续更新路径与动作。',
  },
];

const capabilities = [
  {
    title: '感知与理解', question: '世界和自己，现在是什么状态？', icon: Eye,
    concepts: '视觉与三维几何 · 状态估计 · 视觉语言模型',
    links: [
      {label: '传感器、坐标系与标定', to: '/docs/foundations/perception/sensor-calibration-sim2real'},
      {label: '多模态模型基础（VLM）', to: '/docs/foundations/vlm/intro'},
    ],
  },
  {
    title: '学习与决策', question: '为了完成任务，下一步做什么？', icon: BrainCircuit,
    concepts: '模仿学习 · 强化学习 · VLA · 世界模型',
    links: [
      {label: '模仿学习与强化学习', to: '/docs/foundations/rl-for-robotics/intro'},
      {label: '视觉、语言到动作（VLA）', to: '/docs/foundations/vla/vla-intro'},
      {label: '世界模型：预测动作后果', to: '/docs/foundations/world-model/intro'},
    ],
  },
  {
    title: '运动与控制', question: '怎样把动作准确、稳定地做出来？', icon: Move3D,
    concepts: '运动学与动力学 · 运动规划 · 反馈控制',
    links: [
      {label: '坐标变换与正逆运动学', to: '/docs/foundations/robotics-and-ros2/kinematics_transform'},
      {label: '从 PID 到模型预测控制', to: '/docs/foundations/controllers/intro'},
      {label: 'MoveIt 2 运动规划', to: '/docs/foundations/robotics-and-ros2/moveit2_basics'},
    ],
  },
];

const foundations = [
  {
    title: '仿真与评测', icon: FlaskConical,
    description: '在可重复的环境里验证方法，用成功率、误差和失败案例判断效果。',
    concepts: '物理仿真 · 评测基准 · 仿真到真实（Sim2Real）',
    links: [
      {label: '认识仿真工具', to: '/docs/foundations/simulation/intro'},
      {label: '数据集与评测基准', to: '/docs/information/datasets'},
    ],
  },
  {
    title: '数据与训练', icon: Database,
    description: '把示教或交互记录变成训练数据，检查观测、动作与时间是否对齐。',
    concepts: '遥操作 · 数据质量 · 策略训练与验证',
    links: [
      {label: 'LeRobot 数据与工具链', to: '/docs/practices/robot-arm/data-collection/lerobot-course'},
      {label: '从示教数据学习动作', to: '/docs/foundations/rl-for-robotics/imitation-learning'},
    ],
  },
  {
    title: '本体与系统', icon: Settings2,
    description: '让传感器、计算机、通信与执行器协同工作，把算法接到机器人身体上。',
    concepts: '机械结构 · 传感器与电机 · ROS2 · 嵌入式',
    links: [
      {label: 'ROS2 系统导览', to: '/docs/foundations/robotics-and-ros2/course_introduction'},
      {label: 'SO-101 真机连接与调试', to: '/docs/practices/robot-arm/data-collection/so101-lerobot-real'},
    ],
  },
];

const firstSteps = [
  {
    title: '建立系统概念', preparation: '从这里开始 · 无需设备',
    description: '用一个机器人任务，分清观测、动作、目标和反馈。',
    outcome: '能说明一个动作背后有哪些模块在协作。',
    link: {label: '阅读机器人系统全景', to: '/docs/foundations/robotics-and-ros2/course_introduction'},
  },
  {
    title: '亲手体验一个原理', preparation: '浏览器即可',
    description: '调整 PD 控制器的参数，观察响应、超调与稳定性。',
    outcome: '能解释参数变化为什么会让关节抖动或变慢。',
    link: {label: '打开 PD 控制实验', to: '/cs123/pd-playground'},
  },
  {
    title: '跑通最小仿真', preparation: '准备 Python 基础',
    description: '安装 MuJoCo，加载模型，用代码推进仿真并施加控制。',
    outcome: '保存一份自己能运行、能修改的仿真实验。',
    link: {label: '开始 MuJoCo 教程', to: '/docs/foundations/simulation/mujoco'},
  },
  {
    title: '沿一条主线完成项目', preparation: '按章节补齐数学与算法',
    description: '从单关节控制走到四足建模、步态与策略训练。',
    outcome: '记录实验条件、结果和失败原因，形成完整复现。',
    link: {label: '从零到一搭建四足机器人', to: '/docs/practices/quadruped/cs123/intro'},
  },
];

const directions = [
  {
    title: '抓取与操作', english: 'Manipulation',
    description: '让机器人抓取、推拉、装配或协调双臂，改变物体的状态。',
    path: '运动学 → 示教数据 → 模仿学习 → ACT → VLA',
    to: '/docs/practices/vla/act', label: '做一次 ACT 双臂实验',
    note: '建议先有 Python / PyTorch 基础，再按教程准备训练环境。',
  },
  {
    title: '足式运动', english: 'Locomotion',
    description: '让四足、双足机器人保持平衡，学会行走、转向和起身。',
    path: '反馈控制 → 运动学 → 仿真 → 强化学习 → 迁移验证',
    to: '/docs/practices/quadruped/cs123/intro', label: '进入四足机器人课程',
    note: '先用仿真建立直觉，再继续 MicroDuck 等独立项目。',
  },
  {
    title: '导航与移动操作', english: 'Navigation',
    description: '找到目标、绕开障碍，再把移动与抓取连接成一个任务。',
    path: '定位与建图 → 路径规划 → 导航 → 移动与操作协同',
    to: '/docs/overview/robotics-and-ros2-roadmap', label: '先补机器人学与 ROS2',
    note: '本站已有基础路线；完整导航与移动操作项目仍待补充。',
  },
];

function TopicLinks({links}: {links: {label: string; to: string}[]}) {
  return (
    <ul className={styles.topicLinks}>
      {links.map(({label, to}) => (
        <li key={to}><Link to={to}>{label}<ArrowUpRight size={15} aria-hidden="true" /></Link></li>
      ))}
    </ul>
  );
}

function KnowledgeMap() {
  const [scenarioIndex, setScenarioIndex] = useState(0);
  const scenario = scenarios[scenarioIndex];

  return (
    <section id="knowledge-map" className={styles.section} aria-labelledby="map-title">
      <div className={styles.sectionHeading}>
        <div><p className={styles.kicker}>知识全景</p><h2 id="map-title">从一个任务，看懂整个系统。</h2></div>
        <p>上层串起行动闭环，下层支撑开发与验证。<br />选择知识点，就能进入对应教程。</p>
      </div>
      <div className={styles.map}>
        <div className={styles.scenarioBar}>
          <span id="scenario-label">换个任务理解</span>
          <div className={styles.scenarioOptions} role="group" aria-labelledby="scenario-label">
            {scenarios.map(({name}, index) => (
              <button key={name} type="button" aria-pressed={scenarioIndex === index}
                aria-controls="task-example" onClick={() => setScenarioIndex(index)}>
                {scenarioIndex === index && <Check size={14} aria-hidden="true" />}{name}
              </button>
            ))}
          </div>
        </div>
        <div id="task-example" className={styles.taskDefinition}>
          <p><span>任务目标</span>{scenario.goal}</p>
          <p><span>怎样算完成</span>{scenario.success}</p>
        </div>
        <p className={styles.srOnly} role="status">
          {scenario.name}。{scenario.goal}{scenario.stages.join('')}{scenario.feedback}完成标准：{scenario.success}
        </p>
        <div className={styles.loop}>
          {capabilities.map(({title, question, icon: Icon, concepts, links}, index) => (
            <article className={styles.capability} key={title}>
              <div className={styles.capabilityHeading}><Icon size={23} strokeWidth={1.7} aria-hidden="true" /><h3>{title}</h3></div>
              <p className={styles.question}>{question}</p>
              <p className={styles.example}>{scenario.stages[index]}</p>
              <p className={styles.concepts}>{concepts}</p>
              <TopicLinks links={links} />
              {index < capabilities.length - 1 && <ArrowRight className={styles.flowArrow} size={21} aria-hidden="true" />}
            </article>
          ))}
        </div>
        <div className={styles.feedback}><RotateCcw size={18} aria-hidden="true" /><p><strong>行动改变环境，新的观测再次进入闭环。</strong><span>{scenario.feedback}</span></p></div>
        <p className={styles.modelNote}>这是按功能理解系统的方式；实际系统可以用多个模块协作，也可以用一个模型承担多种功能。</p>
        <div className={styles.foundationLabel}><span>支撑整个闭环</span></div>
        <div className={styles.foundations}>
          {foundations.map(({title, icon: Icon, description, concepts, links}) => (
            <article key={title}>
              <div className={styles.capabilityHeading}><Icon size={21} strokeWidth={1.7} aria-hidden="true" /><h3>{title}</h3></div>
              <p>{description}</p><p className={styles.concepts}>{concepts}</p>
              <TopicLinks links={links} />
            </article>
          ))}
        </div>
      </div>
      <div className={styles.methodNote}>
        <strong>算法在地图里的位置</strong>
        <p>模仿学习（IL）从示范中学动作，强化学习（RL）用奖励改进策略；VLA 把视觉与语言连接到动作，世界模型预测动作的后果。控制和规划同样可以独立解决任务，也可以与学习方法组合。</p>
      </div>
    </section>
  );
}

export default function LearningMap(): React.JSX.Element {
  const {collectAnchor} = useBrokenLinks();
  // Register section targets so links from MDX are checked during the build.
  ['knowledge-map', 'first-steps', 'directions'].forEach(collectAnchor);

  return (
    <Layout title="具身智能学习地图" description="面向新人的具身智能学习地图：理解感知、决策、控制、仿真、数据与硬件的关系，从第一个实验走向机器人项目。">
      <main className={styles.page}>
        <div className={styles.container}>
          <header className={styles.hero}>
            <p className={styles.kicker}>新手指南 · 从这里开始</p>
            <h1>具身智能学习地图</h1>
            <p className={styles.lead}>先看全局，再找到你的第一步。</p>
            <p className={styles.intro}>具身智能，让智能体通过身体感知环境、采取行动，并在交互中完成任务。<br className={styles.desktopBreak} />这张地图以机器人为主线，帮你连接概念、方法与实践。</p>
            <nav className={styles.pageNav} aria-label="学习地图页内导航">
              <a href="#knowledge-map">看知识全景 <ArrowDown size={15} aria-hidden="true" /></a>
              <a href="#first-steps">从零开始学 <ArrowDown size={15} aria-hidden="true" /></a>
              <a href="#directions">选择实践方向 <ArrowDown size={15} aria-hidden="true" /></a>
            </nav>
          </header>
          <KnowledgeMap />
          <section id="first-steps" className={styles.section} aria-labelledby="steps-title">
            <div className={styles.sectionHeading}>
              <div><p className={styles.kicker}>入门路线</p><h2 id="steps-title">第一次来，可以这样走。</h2></div>
              <p>先体验，再补基础，再完成一个项目。<br />每一步都有一个可以验证的小目标。</p>
            </div>
            <ol className={styles.steps}>
              {firstSteps.map(({title, preparation, description, outcome, link}, index) => (
                <li key={title}>
                  <span className={styles.stepNumber} aria-hidden="true">{String(index + 1).padStart(2, '0')}</span>
                  <div className={styles.stepBody}><p className={styles.preparation}>{preparation}</p><h3>{title}</h3><p>{description}</p><p className={styles.outcome}><Check size={16} aria-hidden="true" />{outcome}</p></div>
                  <Link to={link.to} className={styles.stepLink}>{link.label}<ArrowRight size={17} aria-hidden="true" /></Link>
                </li>
              ))}
            </ol>
            <p className={styles.preparationNote}>准备写代码时，按需补 Python、Linux / Git 与线性代数；进入模型训练后，再补 PyTorch、概率与深度学习。已有基础可以直接跳到对应阶段。</p>
          </section>
          <section id="directions" className={styles.section} aria-labelledby="directions-title">
            <div className={styles.sectionHeading}>
              <div><p className={styles.kicker}>按任务选择</p><h2 id="directions-title">你想让机器人做什么？</h2></div>
              <p>机器人形态、任务和算法是不同维度。<br />同一台机器人可以组合多种能力。</p>
            </div>
            <div className={styles.directions}>
              {directions.map(({title, english, description, path, to, label, note}) => (
                <article key={title}>
                  <p className={styles.directionEnglish}>{english}</p><h3>{title}</h3><p>{description}</p>
                  <p className={styles.directionPath}>{path}</p>
                  <Link to={to} className="site-text-link">{label}<ArrowUpRight size={16} aria-hidden="true" /></Link>
                  <p className={styles.directionNote}>{note}</p>
                </article>
              ))}
            </div>
            <div className={styles.nextSteps}><div><strong>想继续深入一个方向？</strong><p>把问题写清楚：机器人看到什么、输出什么动作、什么条件下算成功。再选择方法和对照实验。</p></div><Link to="/docs/overview/learning-path" className="site-text-link">查看进阶学习路径 <ArrowRight size={17} aria-hidden="true" /></Link></div>
          </section>
          <aside className={styles.references} aria-labelledby="references-title">
            <h2 id="references-title">参考与延伸</h2>
            <p>本地图结合以下两份指南的总纲，按本站课程重新组织。感谢开源作者与社区的整理。</p>
            <ul>
              <li><a href="https://github.com/TianxingChen/Embodied-AI-Guide/blob/main/README.md">Embodied-AI-Guide <ArrowUpRight size={14} aria-hidden="true" /></a><span>算法、基础设施、控制与硬件的广度。</span></li>
              <li><a href="https://github.com/jiangranlv/embodied-ai-start">PKU EPIC Lab · embodied-ai-start <ArrowUpRight size={14} aria-hidden="true" /></a><span>从任务定义出发，理解机器人技能与研究方法。</span></li>
            </ul>
            <p className={styles.coverageNote}>继续探索：灵巧手、触觉、三维视觉和移动操作也是重要方向；本站的专题覆盖会逐步补齐，可先通过上述指南了解全貌。</p>
          </aside>
        </div>
      </main>
    </Layout>
  );
}
