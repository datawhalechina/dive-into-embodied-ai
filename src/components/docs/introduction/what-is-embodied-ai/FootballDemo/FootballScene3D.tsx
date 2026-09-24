import {translate} from '@docusaurus/Translate';
import React, {useEffect, useRef, useState} from 'react';
import {RotateCcw} from 'lucide-react';
import type {FootballFrame, FootballScene} from './scene';
import styles from './styles.module.css';

export default function FootballScene3D(frame: FootballFrame): React.JSX.Element {
  const host = useRef<HTMLDivElement>(null);
  const scene = useRef<FootballScene | null>(null);
  const latest = useRef(frame);
  latest.current = frame;
  const [state, setState] = useState<'loading' | 'ready' | 'unavailable'>('loading');

  useEffect(() => {
    let cancelled = false;
    import('./scene').then(({createFootballScene}) => {
      if (cancelled || !host.current) return;
      try {
        scene.current = createFootballScene(host.current, () => setState('unavailable'));
        scene.current.update(latest.current);
        setState('ready');
      } catch {
        setState('unavailable');
      }
    }).catch(() => {if (!cancelled) setState('unavailable');});
    return () => {cancelled = true; scene.current?.dispose(); scene.current = null;};
  }, []);

  useEffect(() => {scene.current?.update(frame);}, [frame.elapsed, frame.keeperNear, frame.stage]);

  return (
    <div className={styles.world}>
      <div ref={host} className={styles.canvasHost} />
      <div className={styles.worldLabels}><span><i />{translate({message: "射门机器人"})}</span><span><i />{translate({message: "守门员"})}</span></div>
      {state !== 'ready' && <p className={styles.sceneMessage} role="status">{state === 'loading' ? translate({message: "正在准备 3D 球场…"}) : translate({message: "当前浏览器未能显示 3D 场景，仍可使用下方阶段说明。"})}</p>}
      {state === 'ready' && <div className={styles.viewControls}><span>{translate({message: "拖动旋转 · 方向键调整视角"})}</span><button type="button" onClick={() => scene.current?.resetView()}><RotateCcw size={15} aria-hidden="true" />{translate({message: "复位视角"})}</button></div>}
    </div>
  );
}
