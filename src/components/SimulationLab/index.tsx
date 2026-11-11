import React, {useCallback, useEffect, useRef, useState} from 'react';
import Link from '@docusaurus/Link';
import {useColorMode} from '@docusaurus/theme-common';
import {ArrowUp, ArrowDown, ArrowLeft, ArrowRight, Play, Pause, RotateCcw, Crosshair, Gamepad2, Clapperboard, Box, Maximize2, Square} from 'lucide-react';
import catalog from '@site/simulation/models.json';
import {useSimulation, type Mode, type Joint} from './useSimulation';
import {messages} from './messages';
import styles from './styles.module.css';

const keyMap: Record<string, string> = {KeyW: 'forward', ArrowUp: 'forward', KeyS: 'back', ArrowDown: 'back', KeyA: 'left', ArrowLeft: 'left', KeyD: 'right', ArrowRight: 'right', KeyQ: 'strafeLeft', KeyE: 'strafeRight'};

function JointControl({joint, label, value, disabled, onChange}: {joint: Joint; label: string; value: number; disabled: boolean; onChange: (value: number) => void}) {
  const editing = useRef(false);
  const [degrees, setDegrees] = useState(value * 180 / Math.PI);
  useEffect(() => { if (!editing.current) setDegrees(value * 180 / Math.PI); }, [value]);
  return <label className={styles.jointControl}><span>{label}</span><output>{degrees.toFixed(0)}°</output>
    <input type="range" min={joint.min * 180 / Math.PI} max={joint.max * 180 / Math.PI} step="1" value={degrees} disabled={disabled}
      onPointerDown={() => { editing.current = true; }} onPointerUp={() => { editing.current = false; }} onPointerCancel={() => { editing.current = false; }}
      onKeyDown={() => { editing.current = true; }} onKeyUp={() => { editing.current = false; }} onBlur={() => { editing.current = false; }}
      onChange={event => { const next = Number(event.target.value); setDegrees(next); onChange(next * Math.PI / 180); }} />
  </label>;
}

export default function SimulationLab() {
  const text = messages();
  const [modelId, setModelId] = useState(catalog[0].id);
  const model = catalog.find(item => item.id === modelId)!;
  const [mode, setMode] = useState<Mode>('manual');
  const [speed, setSpeed] = useState(.6);
  const [rate, setRate] = useState(1);
  const [follow, setFollow] = useState(true);
  const {colorMode} = useColorMode();
  const sim = useSimulation(model.id, model.cameraDistance, mode, colorMode === 'dark', text.canvas);
  const arena = useRef<HTMLDivElement>(null);
  const workspace = useRef<HTMLDivElement>(null);
  const keys = useRef(new Set<string>());
  const pointers = useRef(new Map<number, string>());
  const [pressed, setPressed] = useState<string[]>([]);
  const ready = sim.status === 'ready';
  const paused = sim.snapshot?.paused ?? false;
  const manual = mode === 'manual';
  const duck = model.adapter === 'microduck';
  const policy = model.kind === 'policy';
  const modelText: Record<string, {brief: string; about: string; demo: string; demoInfo: string}> = {
    go2: {brief: text.go2Brief, about: text.go2Info, demo: text.walkPolicy, demoInfo: text.policyInfo},
    pupper: {brief: text.pupperBrief, about: text.pupperInfo, demo: text.pupperPolicy, demoInfo: text.pupperPolicyInfo},
    microduck: {brief: text.duckBrief, about: text.duckInfo, demo: '', demoInfo: text.duckNoPolicy},
    quaddle: {brief: text.quaddleBrief, about: text.quaddleInfo, demo: text.gaitDemo, demoInfo: text.gaitInfo},
  };
  const description = manual ? (duck ? text.duckDescription : text.manualDescription) : text.showcaseDescription;
  const commandRef = useRef((active: string[]) => {});
  commandRef.current = active => {
    const has = (key: string) => active.includes(key) ? 1 : 0;
    const x = duck ? has('forward') - has('back') : policy ? (has('forward') - has('back')) * Math.min(speed, has('back') ? .5 : 1) : has('back') ? 0 : has('forward');
    const y = duck ? has('strafeLeft') - has('strafeRight') : policy ? (has('strafeLeft') - has('strafeRight')) * Math.min(speed, .5) : 0;
    const z = (has('left') - has('right')) * .7;
    sim.send({type: 'command', value: [x, y, policy || duck || !has('back') ? z : 0]});
  };
  const syncInput = useCallback(() => {
    const active = [...new Set([...keys.current].map(code => keyMap[code]).filter(Boolean).concat([...pointers.current.values()]))];
    setPressed(active); commandRef.current(active);
  }, []);
  const stopInput = useCallback(() => {
    keys.current.clear(); pointers.current.clear(); setPressed([]); commandRef.current([]);
  }, []);
  useEffect(() => {
    const hidden = () => { if (document.hidden) stopInput(); };
    window.addEventListener('blur', stopInput);
    document.addEventListener('visibilitychange', hidden);
    return () => { window.removeEventListener('blur', stopInput); document.removeEventListener('visibilitychange', hidden); };
  }, [stopInput]);
  useEffect(() => { stopInput(); }, [modelId, mode, paused, sim.suspended, ready, stopInput]);
  useEffect(() => { commandRef.current(pressed); }, [speed]);
  useEffect(() => { if (ready) { sim.send({type: 'rate', value: rate}); sim.scene.current?.setFollow(follow); } }, [ready, rate, follow]);

  const chooseMode = (value: Mode) => { if (duck && value === 'showcase') return; stopInput(); setMode(value); };
  const reset = () => { stopInput(); sim.send({type: 'reset'}); sim.scene.current?.resetView(); };
  const togglePause = () => { stopInput(); sim.send({type: 'pause', value: !paused}); };
  const pointerButton = (direction: string, label: string, icon: React.ReactNode, key: string) => (
    <button type="button" className={styles.direction} aria-label={label} title={label} aria-pressed={pressed.includes(direction)} disabled={!ready || paused || !manual}
      onPointerDown={event => {
        if (event.button !== 0) return;
        event.preventDefault(); event.currentTarget.setPointerCapture(event.pointerId);
        pointers.current.set(event.pointerId, direction); syncInput();
      }}
      onPointerUp={event => { pointers.current.delete(event.pointerId); syncInput(); }}
      onPointerCancel={event => { pointers.current.delete(event.pointerId); syncInput(); }}
      onLostPointerCapture={event => { pointers.current.delete(event.pointerId); syncInput(); }}
      onKeyDown={event => { if ([' ', 'Enter'].includes(event.key)) { event.preventDefault(); pointers.current.set(-1, direction); syncInput(); } }}
      onKeyUp={event => { if ([' ', 'Enter'].includes(event.key)) { event.preventDefault(); pointers.current.delete(-1); syncInput(); } }}
      onBlur={stopInput}>
      {icon}<kbd aria-hidden="true">{key}</kbd>
    </button>
  );
  return (
    <main className={styles.lab}>
      <header className={styles.heading}><div><p className={styles.eyebrow}>MUJOCO LAB</p><h1>{text.title}</h1><p>{text.subtitle}</p></div><span className={styles.localBadge}>{text.browser}</span></header>
      <div ref={workspace} className={styles.workspace}>
        <aside className={styles.library} aria-label={text.library}>
          <div className={styles.sectionTitle}><h2>{text.library}</h2><span>{catalog.length}</span></div>
          <div className={styles.modelList}>
            {catalog.map(item => <button key={item.id} type="button" className={styles.model} aria-pressed={modelId === item.id} onClick={() => { stopInput(); setModelId(item.id); setSpeed(item.demoCommand[0] || .4); if (item.kind === 'manual') setMode('manual'); }}>
              <span className={styles.modelGlyph}><Box size={25} aria-hidden="true" /></span>
              <strong>{item.name}</strong><span>{item.kind === 'policy' ? text.trained : item.kind === 'manual' ? text.manualOnly : text.controller}</span>
              <small>{modelText[item.adapter].brief}</small>
            </button>)}
          </div>
          <p className={styles.libraryNote}>{text.libraryNote}</p>
        </aside>
        <section className={styles.experiment} aria-label={model.name}>
          <div className={styles.topbar}>
            <div className={styles.modeSwitch} role="group" aria-label={text.mode}>
              <button type="button" aria-pressed={manual} onClick={() => chooseMode('manual')}><Gamepad2 size={17} aria-hidden="true" />{text.manual}</button>
              <button type="button" aria-pressed={!manual} disabled={duck} title={duck ? text.duckNoPolicy : undefined} onClick={() => chooseMode('showcase')}><Clapperboard size={17} aria-hidden="true" />{text.showcase}</button>
            </div>
            <span className={styles.modelName}>{model.name}</span>
          </div>
          <div ref={sim.surface} className={styles.stage}>
            <div ref={arena} className={styles.arena} tabIndex={ready && manual ? 0 : -1} role="group" aria-label={text.keyboardArea} aria-describedby="simulation-controls-help"
              onBlur={stopInput}
              onPointerDown={() => { if (ready && manual) arena.current?.focus({preventScroll: true}); }}
              onKeyDown={event => {
                if (event.altKey || event.ctrlKey || event.metaKey || event.target !== event.currentTarget) return;
                if (event.code === 'Space' && !event.repeat && ready) { event.preventDefault(); togglePause(); return; }
                if (event.code === 'KeyR' && !event.repeat && ready) { event.preventDefault(); reset(); return; }
                if (!manual || !ready || paused || !keyMap[event.code]) return;
                event.preventDefault(); if (!event.repeat) { keys.current.add(event.code); syncInput(); }
              }}
              onKeyUp={event => { if (keyMap[event.code]) { event.preventDefault(); keys.current.delete(event.code); syncInput(); } }}>
              <div ref={sim.host} className={styles.canvas} />
            </div>
            {ready && <>
              <div className={styles.stageStatus}><span className={styles.liveDot} data-paused={paused || sim.suspended} />{sim.suspended ? text.suspended : paused ? text.paused : manual ? text.manual : text.showcase}</div>
              <div className={styles.viewTools}>
                <button type="button" onClick={() => sim.scene.current?.resetView()} aria-label={text.resetCamera} title={text.resetCamera}><Crosshair size={18} /></button>
                <button type="button" onClick={() => {
                  if (document.fullscreenElement) document.exitFullscreen?.().catch(() => {});
                  else workspace.current?.requestFullscreen?.().then(() => arena.current?.focus({preventScroll: true})).catch(() => {});
                }} aria-label={text.fullscreen} title={text.fullscreen}><Maximize2 size={18} /></button>
              </div>
              <p className={styles.stageHint}>{manual ? text.focusHint : text.cameraHint}</p>
            </>}
            {sim.status !== 'ready' && <div className={styles.cover} role="status">
              <Box size={42} strokeWidth={1.3} aria-hidden="true" />
              <h2>{sim.status === 'loading' ? text.loading : sim.status === 'error' ? text.error : text.welcome}</h2>
              <p>{sim.status === 'error' ? text.errorHint : description}</p>
              {sim.status !== 'loading' && <button type="button" className={styles.primary} onClick={sim.start}><Play size={17} aria-hidden="true" />{sim.status === 'error' ? text.retry : text.load}</button>}
              <small>{text.download}</small>
              {sim.status === 'error' && <details><summary>{text.errorDetails}</summary><pre>{sim.error}</pre></details>}
            </div>}
          </div>
          <div className={styles.transport}>
            <div><button type="button" disabled={!ready} onClick={togglePause}>{paused ? <Play size={16} /> : <Pause size={16} />}{paused ? text.resume : text.pause}</button><button type="button" disabled={!ready} onClick={reset}><RotateCcw size={16} />{text.reset}</button></div>
            <label>{text.rate}<select disabled={!ready} value={rate} onChange={event => setRate(Number(event.target.value))}><option value=".25">0.25×</option><option value=".5">0.5×</option><option value="1">1×</option></select></label>
          </div>
          <dl className={styles.telemetry} aria-label={text.telemetry}>
            <div><dt>{text.time}</dt><dd>{(sim.snapshot?.time ?? 0).toFixed(1)} <small>s</small></dd></div>
            <div><dt>{text.velocity}</dt><dd>{(sim.snapshot?.speed ?? 0).toFixed(2)} <small>m/s</small></dd></div>
            <div><dt>{text.position}</dt><dd>{(sim.snapshot?.position[0] ?? 0).toFixed(2)}, {(sim.snapshot?.position[1] ?? 0).toFixed(2)} <small>m</small></dd></div>
          </dl>
        </section>
        <aside className={styles.panel} aria-label={text.controls}>
          <h2>{manual ? text.manual : text.showcase}</h2>
          <p id="simulation-controls-help">{description}</p>
          {manual ? <>
            <div className={styles.pad} role="group" aria-label={text.directionPad}>
              <span />{pointerButton('forward', duck ? text.headUp : text.forward, <ArrowUp size={20} />, 'W')}<span />
              {pointerButton('left', duck ? text.headLeft : text.left, <ArrowLeft size={20} />, 'A')}
              <button type="button" className={styles.stop} disabled={!ready || paused} onClick={stopInput} aria-label={text.stop} title={text.stop}><Square size={14} /></button>
              {pointerButton('right', duck ? text.headRight : text.right, <ArrowRight size={20} />, 'D')}
              <span />{pointerButton('back', duck ? text.headDown : policy ? text.back : text.stop, <ArrowDown size={20} />, 'S')}<span />
            </div>
            {(policy || duck) && <div className={styles.strafe}>
              {pointerButton('strafeLeft', duck ? text.rollLeft : text.strafeLeft, <ArrowLeft size={18} />, 'Q')}
              <span>{duck ? text.headRoll : text.strafe}</span>
              {pointerButton('strafeRight', duck ? text.rollRight : text.strafeRight, <ArrowRight size={18} />, 'E')}
            </div>}
            <p className={styles.small}>{duck ? text.duckKeys : policy ? text.keys : text.quaddleKeys}</p>
            {policy && <label className={styles.speedLabel}>{text.speed}<output>{speed.toFixed(1)} m/s</output><input type="range" min={model.speedRange?.[0] ?? .2} max={model.speedRange?.[1] ?? 1} step=".1" value={speed} onChange={event => setSpeed(Number(event.target.value))} /></label>}
          </> : <div className={styles.showcaseInfo}><Play size={22} aria-hidden="true" /><strong>{modelText[model.adapter].demo}</strong><p>{modelText[model.adapter].demoInfo}</p></div>}
          {duck && <>
            <p className={styles.manualNotice}>{text.duckNoPolicy}</p>
            <label className={styles.follow}><input type="checkbox" disabled={!ready} checked={sim.snapshot?.supported ?? true} onChange={event => { stopInput(); sim.send({type: 'support', value: event.target.checked}); }} />{text.support}</label>
            <details className={styles.jointPanel}><summary>{text.jointControls} · 14</summary>
              {sim.joints.map((joint, index) => <JointControl key={joint.name} joint={joint} label={text.jointNames[joint.name] ?? joint.name} value={sim.snapshot?.jointTargets?.[index] ?? 0} disabled={!ready || paused}
                onChange={value => sim.send({type: 'joint', index, value})} />)}
              {!ready && <p>{text.load}</p>}
            </details>
          </>}
          <label className={styles.follow}><input type="checkbox" checked={follow} onChange={event => setFollow(event.target.checked)} />{text.follow}</label>
          <div className={styles.modelInfo}><h3>{text.about}</h3><p>{modelText[model.adapter].about}</p><Link to={model.tutorial}>{text.tutorial} →</Link></div>
        </aside>
      </div>
      <p className={styles.footnote}>{text.physicsNote}</p>
    </main>
  );
}
