import React, { useEffect, useRef, useState } from 'react';
import { Play, RotateCcw, Dice5, ChevronDown, ChevronRight } from 'lucide-react';
import { CARTPOLE_DISPLAY_SCRIPT } from './cartpoleScript';
import { PyodideRunner } from './pyodideRunner';

const SPEED_OPTIONS = [
  { label: '0.5×', value: 0.5 },
  { label: '1×', value: 1 },
  { label: '2×', value: 2 },
];

const PHYSICS_STEPS_PER_SEC = 50; // CartPole-v1 内部 tau=0.02
const MAX_STEPS = 500;            // 镜像 CartPole-v1 TimeLimit

const INITIAL_OBS = [0, 0, 0, 0];

const stageLabel = {
  idle: '',
  'loading-pyodide': '加载 Python 运行时…(约 16MB,只需一次)',
  'loading-numpy': '安装 numpy…',
  'loading-micropip': '加载 micropip…',
  'installing-gymnasium': '安装 gymnasium…',
  'injecting-script': '就绪…',
  ready: '',
  error: '',
};

export default function CartPolePyodideClient() {
  const canvasRef = useRef(null);
  const runnerRef = useRef(null);
  const [seed, setSeed] = useState(42);
  const [speed, setSpeed] = useState(1);
  const [stepCount, setStepCount] = useState(0);
  const [totalReward, setTotalReward] = useState(0);
  const [terminated, setTerminated] = useState(false);
  const [truncated, setTruncated] = useState(false);
  const [obs, setObs] = useState(INITIAL_OBS);
  const [showCode, setShowCode] = useState(false);
  const [loadStage, setLoadStage] = useState('idle');
  const [errorMsg, setErrorMsg] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [episodeEnded, setEpisodeEnded] = useState(false);

  const mountedRef = useRef(true);
  const animationIdRef = useRef(0);
  const accumulatorRef = useRef(0);
  const lastTimestampRef = useRef(0);
  const speedRef = useRef(speed);
  const stepCountRef = useRef(0);
  const totalRewardRef = useRef(0);
  const isRunningRef = useRef(false);

  useEffect(() => {
    speedRef.current = speed;
  }, [speed]);

  const drawScene = (nextObs, isDone) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    // 背景
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, w, h);

    // 地面线 + 刻度
    const groundY = h - 40;
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, groundY);
    ctx.lineTo(w, groundY);
    ctx.stroke();
    for (let i = -2; i <= 2; i += 1) {
      const px = w / 2 + (i / 2.4) * (w / 2);
      ctx.beginPath();
      ctx.moveTo(px, groundY);
      ctx.lineTo(px, groundY + 6);
      ctx.stroke();
    }

    // 小车
    const [x, , theta] = nextObs;
    const cartPx = w / 2 + (x / 2.4) * (w / 2);
    const cartW = 60;
    const cartH = 24;
    ctx.fillStyle = isDone ? '#f87171' : '#22d3ee';
    ctx.fillRect(cartPx - cartW / 2, groundY - cartH, cartW, cartH);

    // 杆子
    const poleLenPx = h * 0.45;
    const pivotX = cartPx;
    const pivotY = groundY - cartH;
    const tipX = pivotX + Math.sin(theta) * poleLenPx;
    const tipY = pivotY - Math.cos(theta) * poleLenPx;
    ctx.strokeStyle = isDone ? '#f87171' : '#fde047';
    ctx.lineWidth = 6;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(pivotX, pivotY);
    ctx.lineTo(tipX, tipY);
    ctx.stroke();

    // 轴承小圆点
    ctx.fillStyle = '#e2e8f0';
    ctx.beginPath();
    ctx.arc(pivotX, pivotY, 3, 0, Math.PI * 2);
    ctx.fill();
  };

  const stopLoop = () => {
    isRunningRef.current = false;
    if (animationIdRef.current) {
      cancelAnimationFrame(animationIdRef.current);
      animationIdRef.current = 0;
    }
  };

  const tick = async (timestamp) => {
    if (!isRunningRef.current) return;
    if (!lastTimestampRef.current) lastTimestampRef.current = timestamp;
    const delta = (timestamp - lastTimestampRef.current) / 1000;
    lastTimestampRef.current = timestamp;
    accumulatorRef.current +=
      delta * PHYSICS_STEPS_PER_SEC * speedRef.current;

    let stepsThisFrame = 0;
    let lastResult = null;
    while (accumulatorRef.current >= 1 && stepsThisFrame < 4) {
      accumulatorRef.current -= 1;
      stepsThisFrame += 1;
      try {
        const r = await runnerRef.current.step();
        lastResult = r;
        stepCountRef.current += 1;
        totalRewardRef.current += r.reward;
        if (
          r.terminated ||
          r.truncated ||
          stepCountRef.current >= MAX_STEPS
        ) {
          break;
        }
      } catch (err) {
        stopLoop();
        setErrorMsg(err instanceof Error ? err.message : String(err));
        return;
      }
    }

    if (lastResult) {
      setObs(lastResult.obs);
      setStepCount(stepCountRef.current);
      setTotalReward(totalRewardRef.current);
      const hitCap = stepCountRef.current >= MAX_STEPS;
      const done = lastResult.terminated || lastResult.truncated || hitCap;
      drawScene(lastResult.obs, done);
      if (done) {
        setTerminated(lastResult.terminated);
        setTruncated(lastResult.truncated || hitCap);
        setEpisodeEnded(true);
        stopLoop();
        return;
      }
    }

    animationIdRef.current = requestAnimationFrame(tick);
  };

  const ensureRunnerLoaded = async () => {
    if (isReady && runnerRef.current) return runnerRef.current;
    setIsLoading(true);
    setErrorMsg('');
    const runner = new PyodideRunner();
    runnerRef.current = runner;
    try {
      await runner.load((stage) => {
        if (!mountedRef.current) return;
        setLoadStage(stage);
      });
      if (!mountedRef.current) throw new Error('unmounted');
      setIsReady(true);
      return runner;
    } catch (err) {
      runnerRef.current?.dispose();
      runnerRef.current = null;
      setLoadStage('error');
      setErrorMsg(err instanceof Error ? err.message : String(err));
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  const handleRun = async () => {
    setEpisodeEnded(false);
    try {
      const runner = await ensureRunnerLoaded();
      const initialObs = await runner.initEnv(seed);
      if (!mountedRef.current) return;
      setObs(initialObs);
      drawScene(initialObs, false);
      setStepCount(0);
      setTotalReward(0);
      setTerminated(false);
      setTruncated(false);
      stepCountRef.current = 0;
      totalRewardRef.current = 0;
      accumulatorRef.current = 0;
      lastTimestampRef.current = 0;
      isRunningRef.current = true;
      animationIdRef.current = requestAnimationFrame(tick);
    } catch {
      // ensureRunnerLoaded 已设置 errorMsg,这里吞掉
    }
  };

  const handleReset = async () => {
    stopLoop();
    setEpisodeEnded(false);
    setStepCount(0);
    setTotalReward(0);
    setTerminated(false);
    setTruncated(false);
    setObs(INITIAL_OBS);
    if (isReady && runnerRef.current) {
      try {
        const initialObs = await runnerRef.current.initEnv(seed);
        if (!mountedRef.current) return;
        setObs(initialObs);
        drawScene(initialObs, false);
      } catch (err) {
        setErrorMsg(err instanceof Error ? err.message : String(err));
      }
    }
  };

  const handleRandomSeed = () => {
    setSeed(Math.floor(Math.random() * 1_000_000));
  };

  useEffect(() => {
    drawScene(INITIAL_OBS, false);
    return () => {
      mountedRef.current = false;
      stopLoop();
      runnerRef.current?.dispose();
      runnerRef.current = null;
    };
  }, []);

  return (
    <div className="tw-my-6 tw-rounded-2xl tw-border tw-border-slate-700/60 tw-bg-slate-900 tw-p-4 tw-text-slate-200">
      <div className="tw-flex tw-flex-wrap tw-items-center tw-justify-between tw-gap-3 tw-mb-3">
        <div className="tw-flex tw-items-center tw-gap-2 tw-text-sm tw-font-medium">
          <span>🕹 在线试试 · CartPole-v1</span>
          <span className="tw-text-xs tw-text-slate-400">(浏览器里跑)</span>
        </div>
        <div className="tw-flex tw-flex-wrap tw-items-center tw-gap-2">
          <button
            type="button"
            onClick={handleRun}
            disabled={isLoading}
            className="tw-inline-flex tw-items-center tw-gap-1.5 tw-rounded-lg tw-bg-cyan-500/90 hover:tw-bg-cyan-500 disabled:tw-bg-slate-700 disabled:tw-cursor-not-allowed tw-px-3 tw-py-1.5 tw-text-sm tw-text-slate-900 tw-font-medium"
          >
            <Play size={14} />
            {isLoading
              ? '加载中…'
              : episodeEnded
                ? 'Run again'
                : 'Run'}
          </button>
          <button
            type="button"
            onClick={handleReset}
            className="tw-inline-flex tw-items-center tw-gap-1.5 tw-rounded-lg tw-bg-slate-700 hover:tw-bg-slate-600 tw-px-3 tw-py-1.5 tw-text-sm"
          >
            <RotateCcw size={14} />
            Reset
          </button>
          <label className="tw-inline-flex tw-items-center tw-gap-1 tw-text-xs tw-text-slate-400">
            seed
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value) || 0)}
              className="tw-w-20 tw-rounded tw-bg-slate-800 tw-px-2 tw-py-1 tw-text-slate-100"
            />
          </label>
          <button
            type="button"
            onClick={handleRandomSeed}
            title="随机 seed"
            className="tw-inline-flex tw-items-center tw-rounded tw-bg-slate-700 hover:tw-bg-slate-600 tw-p-1.5"
          >
            <Dice5 size={14} />
          </button>
          <label className="tw-inline-flex tw-items-center tw-gap-1 tw-text-xs tw-text-slate-400">
            speed
            <select
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
              className="tw-rounded tw-bg-slate-800 tw-px-2 tw-py-1 tw-text-slate-100"
            >
              {SPEED_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>

      {stageLabel[loadStage] && (
        <div className="tw-mb-3 tw-text-xs tw-text-slate-400">
          {stageLabel[loadStage]}
        </div>
      )}
      {errorMsg && (
        <div className="tw-mb-3 tw-rounded-lg tw-border tw-border-red-500/40 tw-bg-red-500/10 tw-px-3 tw-py-2 tw-text-xs tw-text-red-200">
          Python 运行时加载失败：{errorMsg}。可以直接用上面的代码在本地跑。
        </div>
      )}

      <div className="tw-grid tw-grid-cols-1 md:tw-grid-cols-[480px_1fr] tw-gap-4">
        <canvas
          ref={canvasRef}
          width={480}
          height={260}
          className="tw-w-full tw-rounded-lg tw-border tw-border-slate-700/60 tw-bg-slate-950"
        />
        <div className="tw-rounded-lg tw-border tw-border-slate-700/60 tw-bg-slate-950 tw-p-3 tw-text-xs tw-font-mono tw-text-slate-300 tw-space-y-1">
          <div>
            step       : <span className="tw-text-cyan-300">{stepCount}</span>
          </div>
          <div>
            reward     : <span className="tw-text-cyan-300">{totalReward.toFixed(1)}</span>
          </div>
          <div>
            terminated : <span className="tw-text-cyan-300">{String(terminated)}</span>
          </div>
          <div>
            truncated  : <span className="tw-text-cyan-300">{String(truncated)}</span>
          </div>
          <div className="tw-pt-2 tw-text-slate-400">obs:</div>
          <div> x  : <span className="tw-text-cyan-300">{obs[0].toFixed(3)}</span></div>
          <div> ẋ  : <span className="tw-text-cyan-300">{obs[1].toFixed(3)}</span></div>
          <div> θ  : <span className="tw-text-cyan-300">{obs[2].toFixed(3)}</span></div>
          <div> θ̇  : <span className="tw-text-cyan-300">{obs[3].toFixed(3)}</span></div>
        </div>
      </div>

      {episodeEnded && (
        <div className="tw-mt-3 tw-rounded-lg tw-border tw-border-amber-500/40 tw-bg-amber-500/10 tw-px-3 tw-py-2 tw-text-xs tw-text-amber-200">
          Episode over · total reward: {totalReward.toFixed(1)} · steps: {stepCount}
          {truncated && !terminated ? ' · truncated' : ''}
          {terminated ? ' · terminated' : ''}
        </div>
      )}

      <div className="tw-mt-3">
        <button
          type="button"
          onClick={() => setShowCode((v) => !v)}
          className="tw-inline-flex tw-items-center tw-gap-1 tw-text-xs tw-text-slate-400 hover:tw-text-slate-200"
        >
          {showCode ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          Web 版适配代码
        </button>
        {showCode && (
          <pre className="tw-mt-2 tw-rounded-lg tw-border tw-border-slate-700/60 tw-bg-slate-950 tw-p-3 tw-text-xs tw-overflow-x-auto">
            <code>{CARTPOLE_DISPLAY_SCRIPT}</code>
          </pre>
        )}
      </div>
    </div>
  );
}
