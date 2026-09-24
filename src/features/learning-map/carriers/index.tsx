import React from 'react';
import Link from '@docusaurus/Link';
import {translate} from '@docusaurus/Translate';
import {embodiedCarriers} from './content';
import CarrierDemos from '../demos/carriers';
import styles from './styles.module.css';

export default function EmbodiedCarriers({className}: {className?: string}) {
  return (
    <section id="embodied-platforms" className={className} aria-labelledby="carriers-title">
      <header className={styles.heading}>
        <p className={styles.kicker}>{translate({message: '从技能到身体'})}</p>
        <h2 id="carriers-title">{translate({message: '具身智能的载体'})}</h2>
      </header>
      <div className={styles.introduction}>
        <p>{translate({message: '具身智能的载体，是智能体用来感知环境、执行动作并与环境交互的“身体”。对物理机器人而言，这个身体包含机械结构、传感器和执行器。不同的身体，决定了机器人能看到什么、能做出什么动作，以及在哪些环境中工作。'})}</p>
        <p>{translate({message: '围绕前面的四类技能，可以先认识以下五类常见物理载体，再了解用于研究的仿真载体。'})}</p>
      </div>

      <CarrierDemos />

      <div className={styles.tableScroll} role="region" aria-labelledby="carriers-title" tabIndex={0}>
        <table className={styles.table}>
          <caption className={styles.srOnly}>{translate({message: '具身智能常见载体的代表平台、关联技能与研究问题'})}</caption>
          <thead>
            <tr>
              <th scope="col">{translate({message: '载体'})}</th>
              <th scope="col">{translate({message: '代表平台 / 工具'})}</th>
              <th scope="col">{translate({message: '关联技能'})}</th>
              <th scope="col">{translate({message: '常见研究问题'})}</th>
            </tr>
          </thead>
          <tbody>
            {embodiedCarriers.map(({name, body, platforms, skills, skillNote, research}) => (
              <tr key={name}>
                <th scope="row">{name}<span className={styles.note}>{body}</span></th>
                <td>
                  <ul className={styles.platforms}>
                    {platforms.map(platform => <li key={platform.href}><Link href={platform.href}>{platform.name}</Link></li>)}
                  </ul>
                </td>
                <td>{skills}{skillNote && <span className={styles.note}>{skillNote}</span>}</td>
                <td>{research}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className={styles.tableHint}>{translate({message: '窄屏可左右滚动查看完整表格。'})}</p>

      <div className={styles.context}>
        <p>{translate({message: '这些分类有交叉：人形机器人、带臂四足机器人也可以是移动操作平台。'})}<strong>{translate({message: '身体形态提供了实现技能的条件，实际能力仍需要在具体任务中验证。'})}</strong></p>
        <p>{translate({message: '仿真载体指仿真中的机器人或智能体；表中的 Habitat、Isaac Lab、MuJoCo 是构建或运行这些载体的平台与工具。虚拟身体同样需要明确传感器、动作空间及其与环境的交互方式。'})}</p>
      </div>
    </section>
  );
}
