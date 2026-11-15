import {useEffect, useRef, useState} from 'react';
import useBaseUrl from '@docusaurus/useBaseUrl';
import type {createScene, Snapshot} from './scene';

export type Mode = 'manual' | 'showcase';
export type Joint = {name: string; min: number; max: number};
export function useSimulation(id: string, distance: number, mode: Mode, dark: boolean, canvasLabel: string) {
  const assetBase = useBaseUrl('/simulation-assets/');
  const host = useRef<HTMLDivElement>(null);
  const surface = useRef<HTMLDivElement>(null);
  const worker = useRef<Worker | null>(null);
  const scene = useRef<ReturnType<typeof createScene> | null>(null);
  const latest = useRef({mode, dark}); latest.current = {mode, dark};
  const [attempt, setAttempt] = useState(0);
  const [status, setStatus] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle');
  const [error, setError] = useState('');
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [suspended, setSuspended] = useState(false);
  const [joints, setJoints] = useState<Joint[]>([]);
  const send = (message: object) => worker.current?.postMessage(message);

  useEffect(() => {
    if (!attempt) return;
    let disposed = false, failed = false, localWorker: Worker | null = null;
    let localScene: ReturnType<typeof createScene> | null = null;
    let inView = true, lastUi = 0;
    setStatus('loading'); setSnapshot(null); setError(''); setJoints([]);
    const fail = (detail: string) => {
      if (disposed || failed) return;
      failed = true;
      setError(detail); setStatus('error'); localWorker?.terminate();
      localScene?.setVisible(false);
    };
    const visibility = () => {
      const visible = inView && !document.hidden;
      setSuspended(!visible);
      localWorker?.postMessage({type: 'visibility', value: visible});
      localScene?.setVisible(visible);
    };
    const observer = new IntersectionObserver(([entry]) => { inView = entry.isIntersecting; visibility(); });
    if (surface.current) observer.observe(surface.current);
    document.addEventListener('visibilitychange', visibility);
    const timeout = setTimeout(() => fail('Loading timed out. Check the connection and retry.'), 90000);
    import('./scene').then(({createScene}) => {
      if (disposed || failed || !host.current) return;
      localScene = createScene(host.current, distance, canvasLabel, () => fail('WebGL context lost.'));
      localScene.setDark(latest.current.dark); scene.current = localScene;
      localWorker = new Worker(`${assetBase}worker.mjs`, {type: 'module'});
      worker.current = localWorker;
      localWorker.onerror = event => fail(event.message);
      localWorker.onmessage = ({data}) => {
        if (disposed || failed) return;
        try {
          if (data.type === 'error') { clearTimeout(timeout); fail(data.detail); }
          if (data.type === 'ready') {
            clearTimeout(timeout); localScene!.load(data.geometry); setJoints(data.joints ?? []); setStatus('ready');
            localWorker!.postMessage({type: 'mode', value: latest.current.mode}); visibility();
          }
          if (data.type === 'state') {
            localScene!.update(data);
            if (performance.now() - lastUi > 100 || data.time === 0) { setSnapshot(data); lastUi = performance.now(); }
          }
        } catch (reason) { fail(String(reason)); }
      };
      localWorker.postMessage({type: 'init', id, mode: latest.current.mode}); visibility();
    }).catch(reason => fail(String(reason)));
    return () => {
      disposed = true; clearTimeout(timeout); observer.disconnect();
      document.removeEventListener('visibilitychange', visibility);
      localWorker?.terminate(); localScene?.dispose(); worker.current = null; scene.current = null;
    };
  }, [attempt, id, distance, assetBase, canvasLabel]);
  useEffect(() => { send({type: 'mode', value: mode}); }, [mode]);
  useEffect(() => { scene.current?.setDark(dark); }, [dark]);
  return {host, surface, scene, status, error, snapshot, suspended, joints, send, start: () => setAttempt(value => value + 1)};
}
