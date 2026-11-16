import React from 'react';
import styles from './styles.module.css';

/** Lightweight animated illustrations for the eight learning categories. */
export default function CategoryMotion({kind}: {kind: string}) {
  return <svg className={styles.categoryMotion} viewBox="0 0 88 56" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    {kind === 'quadruped' && <>
      <path d="M22 18 H58 L66 12 L74 16 V25 H60 V27 H22 Z" />
      <g className={styles.stepFront}><path d="M55 27 L61 37 L56 48 M27 27 L33 37 L28 48" /></g>
      <g className={styles.stepBack}><path d="M49 27 L43 37 L48 48 M21 27 L15 37 L20 48" /></g>
      <path d="M22 18 L15 13 M11 50 H77" opacity=".22" /><circle cx="68" cy="18" r="1" fill="currentColor" />
    </>}
    {kind === 'humanoid' && <>
      <rect x="35" y="5" width="18" height="12" rx="4" /><path d="M39 18 V29 H49 V18 M35 20 L27 31 M53 20 L61 31" />
      <g className={styles.stepFront}><path d="M39 29 L32 38 L35 48 H27" /></g>
      <g className={styles.stepBack}><path d="M49 29 L56 38 L53 48 H61" /></g>
      <path d="M20 51 H68" opacity=".22" /><circle cx="41" cy="10" r=".8" fill="currentColor" /><circle cx="47" cy="10" r=".8" fill="currentColor" />
    </>}
    {(kind === 'robot-arm' || kind === 'control') && <>
      <path d="M22 49 H48 M27 49 V43 H43 V49" />
      <g className={styles.armSwing}>
        <path d="M35 43 L25 25 L48 11 L65 23 M62 20 L68 17 L74 22 M65 23 L64 30 L70 34" />
        <circle cx="35" cy="43" r="3" className={styles.motionNode} /><circle cx="25" cy="25" r="3" className={styles.motionNode} /><circle cx="48" cy="11" r="3" className={styles.motionNode} />
      </g>
      {kind === 'control' && <path className={styles.signalTrace} d="M7 44 V37 L12 28 L17 37 H23" opacity=".5" />}
    </>}
    {kind === 'amd' && <>
      <rect x="28" y="12" width="32" height="32" rx="5" /><rect x="35" y="19" width="18" height="18" rx="2" />
      {[0, 1, 2, 3].map(i => <path key={i} d={`M${34 + i * 7} 6 V12 M${34 + i * 7} 44 V50 M22 ${18 + i * 7} H28 M60 ${18 + i * 7} H66`} />)}
      <circle className={styles.chipPulse} cx="44" cy="28" r="4" fill="currentColor" stroke="none" />
    </>}
    {kind === 'decision' && <>
      <path d="M14 17 L43 9 L74 28 L43 47 L14 39 L43 28 Z M14 17 L43 28 L74 28 M14 39 L43 9 M43 47 L43 28" opacity=".35" />
      <path className={styles.signalTrace} d="M14 17 L43 28 L74 28" />
      {[[14, 17], [14, 39], [43, 9], [43, 28], [43, 47], [74, 28]].map(([x, y], i) => <circle key={i} cx={x} cy={y} r="3.5" className={styles.motionNode} />)}
      <circle className={styles.chipPulse} cx="43" cy="28" r="2" fill="currentColor" stroke="none" />
    </>}
    {kind === 'perception' && <>
      <circle cx="44" cy="28" r="21" opacity=".35" /><circle cx="44" cy="28" r="12" opacity=".25" />
      <path d="M19 28 H69 M44 3 V53" opacity=".18" />
      <g className={styles.radarSweep}><path d="M44 28 L44 7" /><path d="M44 28 L44 7 A21 21 0 0 1 61 16 Z" fill="currentColor" stroke="none" opacity=".12" /></g>
      <circle cx="55" cy="17" r="2.5" fill="currentColor" stroke="none" /><circle className={styles.chipPulse} cx="35" cy="36" r="2" fill="currentColor" stroke="none" />
    </>}
    {kind === 'engineering' && <>
      <path d="M16 36 L44 49 L72 36 M16 28 L44 41 L72 28" opacity=".45" />
      <g className={styles.layerFloat}><path d="M16 19 L44 6 L72 19 L44 32 Z" /><path d="M30 19 L44 13 L58 19 L44 25 Z" opacity=".4" /></g>
      <path className={styles.signalTrace} d="M44 32 V49" />
    </>}
  </svg>;
}
