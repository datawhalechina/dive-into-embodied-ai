import React from 'react';
import styles from './styles.module.css';

/** Small conceptual diagrams: network, feedback loop, sensor rays, system layers. */
export default function ModuleDiagram({kind}: {kind: string}) {
  return <svg className={styles.diagram} viewBox="0 0 420 170" fill="none" aria-hidden="true">
    {kind === 'decision' && <>
      {[0, 1, 2].flatMap(row => [0, 1, 2, 3].map(col => <path key={`${row}-${col}`} d={`M90 ${45 + row * 40} L210 ${25 + col * 40} L330 ${45 + row * 40}`} className={styles.diagramLine} />))}
      {[90, 210, 330].flatMap((x, col) => Array.from({length: col === 1 ? 4 : 3}, (_, row) => <circle key={`${x}-${row}`} cx={x} cy={(col === 1 ? 25 : 45) + row * 40} r={col === 1 ? 9 : 6} className={styles.diagramNode} />))}
    </>}
    {kind === 'control' && <>
      <path d="M44 65 H125 M185 65 H265 M325 65 H380 M350 65 V133 H76 V65" className={styles.diagramLine} />
      <rect x="125" y="37" width="60" height="56" rx="12" className={styles.diagramNode} />
      <rect x="265" y="37" width="60" height="56" rx="12" className={styles.diagramNode} />
      <circle cx="76" cy="65" r="10" className={styles.diagramNode} />
      <path d="M100 59 L110 65 L100 71 M239 59 L249 65 L239 71 M220 127 L210 133 L220 139" className={styles.diagramArrow} />
      <text x="155" y="71">PID</text><text x="295" y="71">θ</text>
    </>}
    {kind === 'perception' && <>
      {Array.from({length: 11}, (_, i) => <path key={i} d={`M95 85 L${270 + Math.cos(i * .7) * 25} ${15 + i * 14}`} className={styles.diagramLine} />)}
      {Array.from({length: 7}, (_, i) => <circle key={i} cx={275 + Math.cos(i) * 23} cy={26 + i * 20} r="4" className={styles.diagramNode} />)}
      <circle cx="95" cy="85" r="22" className={styles.diagramNode} />
      <path d="M325 130 H375 M325 130 V80 M325 130 L298 150" className={styles.diagramArrow} />
      <circle cx="95" cy="85" r="7" className={styles.diagramDot} />
    </>}
    {kind === 'engineering' && <>
      {[0, 1, 2].map(i => <g key={i} transform={`translate(0 ${i * 37})`}>
        <path d="M210 15 L325 43 L210 72 L95 43 Z" className={styles.diagramNode} />
        {[0, 1, 2, 3, 4].map(j => <circle key={j} cx={155 + j * 27} cy="43" r="2.5" className={styles.diagramDot} />)}
      </g>)}
    </>}
  </svg>;
}
