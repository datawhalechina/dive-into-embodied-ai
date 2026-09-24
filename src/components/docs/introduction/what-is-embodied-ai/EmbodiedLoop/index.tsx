import React, {useId, useState} from 'react';
import {translate} from '@docusaurus/Translate';
import {ArrowDown, ArrowRight, BrainCircuit, Camera, Cog, MoveUpRight, Repeat2, ScanLine, Target} from 'lucide-react';
import styles from './styles.module.css';

const stages = [
  {
    name: translate({message: '感知与理解'}),
    parts: translate({message: '相机 · 本体传感器'}),
    signal: translate({message: '观测 → 状态'}),
    question: translate({message: '世界和自己，现在是什么状态？'}),
    input: translate({message: '相机图像、关节角度与身体姿态。'}),
    process: translate({message: '识别足球、球门和守门员，估计它们相对机器人的位置。'}),
    output: translate({message: '场景与身体的状态估计。'}),
    example: translate({message: '看见足球在前方，守门员挡住了球门右侧。'}),
    icon: Camera,
    style: 'perception',
  },
  {
    name: translate({message: '学习与决策'}),
    parts: translate({message: '策略 · 规划'}),
    signal: translate({message: '状态 → 目标'}),
    question: translate({message: '为了完成任务，下一步做什么？'}),
    input: translate({message: '场景状态、身体状态与进球目标。'}),
    process: translate({message: '比较可用的射门空当，选择靠近路线与射门方向。'}),
    output: translate({message: '目标站位、射门方向与期望运动。'}),
    example: translate({message: '选择球门左侧空当，规划站位与踢球动作。'}),
    icon: BrainCircuit,
    style: 'decision',
  },
  {
    name: translate({message: '运动与控制'}),
    parts: translate({message: '控制器 · 执行器'}),
    signal: translate({message: '目标 → 动作'}),
    question: translate({message: '怎样把动作准确、稳定地做出来？'}),
    input: translate({message: '期望运动与实时关节反馈。'}),
    process: translate({message: '调整站位和重心，协调关节，跟踪目标动作。'}),
    output: translate({message: '关节力矩与身体运动。'}),
    example: translate({message: '支撑腿保持平衡，摆动腿触球，把球踢向左侧。'}),
    icon: Cog,
    style: 'action',
  },
  {
    name: translate({message: '环境与结果'}),
    parts: translate({message: '足球 · 球门 · 守门员'}),
    signal: translate({message: '动作 → 新观测'}),
    question: translate({message: '行动之后发生了什么？'}),
    input: translate({message: '机器人与足球、地面的物理交互。'}),
    process: translate({message: '足球运动，守门员与环境继续变化，传感器获得新的观测。'}),
    output: translate({message: '更新后的场景观测与身体反馈。'}),
    example: translate({message: '观察球是否进门；如果踢偏，就根据球的新位置重新调整。'}),
    icon: Target,
    style: 'environment',
  },
];

export default function EmbodiedLoop(): React.JSX.Element {
  const [selected, setSelected] = useState(0);
  const id = useId().replace(/:/g, '');
  const stage = stages[selected];

  return (
    <figure className={styles.figure} aria-labelledby={`${id}-title`}>
      <figcaption className={styles.header}>
        <div>
          <p className={styles.eyebrow}>{translate({message: '具身智能 · 信息与动作的闭环'})}</p>
          <h3 className={styles.title} id={`${id}-title`}>{translate({message: '从观察世界，到改变世界'})}</h3>
          <p className={styles.instruction}>{translate({message: '点选一个环节，查看它的输入、处理与输出。'})}</p>
        </div>
        <span className={styles.loopBadge}><Repeat2 size={16} aria-hidden="true" />{translate({message: '持续循环'})}</span>
      </figcaption>

      <div className={styles.diagram}>
        <div className={styles.pipeline} role="group" aria-label={translate({message: '选择闭环环节'})}>
          <div className={styles.robotBoundary} aria-hidden="true" />
          <div className={styles.worldBoundary} aria-hidden="true" />
          <p className={styles.robotLabel}>{translate({message: '机器人内部'})}</p>
          <p className={styles.worldLabel}>{translate({message: '外部环境'})}</p>
          {stages.map(({name, parts, signal, icon: Icon, style}, index) => (
            <div className={`${styles.stage} ${styles[style]}`} key={style}>
              <button
                type="button"
                className={styles.node}
                aria-pressed={selected === index}
                aria-controls={`${id}-detail`}
                onClick={() => setSelected(index)}>
                <span className={styles.nodeTop}><Icon size={25} strokeWidth={1.7} aria-hidden="true" /><span className={styles.number} aria-hidden="true">0{index + 1}</span></span>
                <strong>{name}</strong>
                <span className={styles.parts}>{parts}</span>
                <span className={styles.signal}>{signal}</span>
              </button>
              {index < stages.length - 1 && <ArrowRight className={styles.connector} size={20} aria-hidden="true" />}
            </div>
          ))}
        </div>

        <div className={styles.feedback}>
          <svg className={styles.feedbackPath} viewBox="0 0 1000 64" preserveAspectRatio="none" aria-hidden="true">
            <path d="M 875 0 V 20 Q 875 40 855 40 H 145 Q 125 40 125 20 V 0" />
          </svg>
          <ArrowDown className={styles.feedbackArrow} size={18} aria-hidden="true" />
          <p><Repeat2 size={17} aria-hidden="true" /><span>{translate({message: '新的观测与反馈'})}<small>{translate({message: '再次感知，修正下一次行动'})}</small></span></p>
        </div>
      </div>

      <div className={`${styles.detail} ${styles[stage.style]}`} id={`${id}-detail`}>
        <div className={styles.detailHeader}>
          <div role="status" aria-live="polite" aria-atomic="true">
            <p className={styles.currentStage}><span aria-hidden="true">0{selected + 1}</span>{stage.name}</p>
            <h4 className={styles.question}>{stage.question}</h4>
          </div>
          <button className={styles.next} type="button" onClick={() => setSelected((selected + 1) % stages.length)}>
            {selected === stages.length - 1 ? translate({message: '回到感知'}) : translate({message: '下一环节'})}<ArrowRight size={17} aria-hidden="true" />
          </button>
        </div>
        <dl className={styles.dataFlow}>
          <div><dt><ScanLine size={16} aria-hidden="true" />{translate({message: '输入'})}</dt><dd>{stage.input}</dd></div>
          <div><dt><Cog size={16} aria-hidden="true" />{translate({message: '处理'})}</dt><dd>{stage.process}</dd></div>
          <div><dt><MoveUpRight size={16} aria-hidden="true" />{translate({message: '输出'})}</dt><dd>{stage.output}</dd></div>
        </dl>
        <p className={styles.example}><span>{translate({message: '在足球场上'})}</span>{stage.example}</p>
      </div>
      <p className={styles.caption}>{translate({message: '按信息流理解系统；真实机器人中的各环节以不同频率持续运行。'})}</p>
    </figure>
  );
}
