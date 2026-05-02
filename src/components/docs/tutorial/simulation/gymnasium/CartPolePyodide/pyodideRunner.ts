import { CARTPOLE_RUNTIME_SCRIPT } from './cartpoleScript';

// 实施时如有新版,更新此处常量并同步更新 PYODIDE_SCRIPT_URL
const PYODIDE_VERSION = 'v0.28.3';
const PYODIDE_CDN_BASE = `https://cdn.jsdelivr.net/pyodide/${PYODIDE_VERSION}/full`;
const PYODIDE_SCRIPT_URL = `${PYODIDE_CDN_BASE}/pyodide.js`;

export type LoadStage =
  | 'idle'
  | 'loading-pyodide'
  | 'loading-numpy'
  | 'loading-micropip'
  | 'installing-gymnasium'
  | 'injecting-script'
  | 'ready'
  | 'error';

export interface StepResult {
  obs: [number, number, number, number];
  reward: number;
  terminated: boolean;
  truncated: boolean;
  action: 0 | 1;
}

type PyodideInstance = {
  loadPackage: (names: string[]) => Promise<void>;
  runPythonAsync: (code: string) => Promise<unknown>;
  globals: { get: (name: string) => (...args: unknown[]) => unknown };
};

declare global {
  interface Window {
    loadPyodide?: (config: { indexURL: string }) => Promise<PyodideInstance>;
  }
}

export class PyodideRunner {
  private pyodide: PyodideInstance | null = null;
  private disposed = false;

  async load(onProgress: (stage: LoadStage) => void): Promise<void> {
    if (typeof WebAssembly === 'undefined') {
      throw new Error('当前浏览器不支持 WebAssembly');
    }

    onProgress('loading-pyodide');
    await this.injectPyodideScript();
    if (!window.loadPyodide) {
      throw new Error('Pyodide 脚本加载后 window.loadPyodide 仍不存在');
    }
    this.pyodide = await window.loadPyodide({ indexURL: `${PYODIDE_CDN_BASE}/` });
    if (this.disposed) return;

    onProgress('loading-numpy');
    await this.pyodide.loadPackage(['numpy']);
    if (this.disposed) return;

    onProgress('loading-micropip');
    await this.pyodide.loadPackage(['micropip']);
    if (this.disposed) return;

    onProgress('installing-gymnasium');
    await this.pyodide.runPythonAsync(
      `import micropip\nawait micropip.install("gymnasium")`
    );
    if (this.disposed) return;

    onProgress('injecting-script');
    await this.pyodide.runPythonAsync(CARTPOLE_RUNTIME_SCRIPT);
    if (this.disposed) return;

    onProgress('ready');
  }

  private injectPyodideScript(): Promise<void> {
    return new Promise((resolve, reject) => {
      const existing = document.querySelector<HTMLScriptElement>(
        `script[data-pyodide="${PYODIDE_VERSION}"]`
      );
      if (existing && window.loadPyodide) {
        resolve();
        return;
      }
      if (existing) {
        existing.addEventListener('load', () => resolve());
        existing.addEventListener('error', () =>
          reject(new Error('Pyodide CDN 脚本加载失败'))
        );
        return;
      }
      const script = document.createElement('script');
      script.src = PYODIDE_SCRIPT_URL;
      script.async = true;
      script.dataset.pyodide = PYODIDE_VERSION;
      script.addEventListener('load', () => resolve());
      script.addEventListener('error', () => {
        script.remove();
        reject(new Error('Pyodide CDN 脚本加载失败'));
      });
      document.head.appendChild(script);
    });
  }

  async initEnv(seed: number): Promise<[number, number, number, number]> {
    if (!this.pyodide) throw new Error('Pyodide 未就绪');
    const envInit = this.pyodide.globals.get('env_init') as (
      s: number
    ) => Promise<unknown> | unknown;
    const result = envInit(seed);
    const list = await Promise.resolve(result);
    const arr = this.toJS(list) as number[];
    return [arr[0], arr[1], arr[2], arr[3]];
  }

  async step(): Promise<StepResult> {
    if (!this.pyodide) throw new Error('Pyodide 未就绪');
    const envStep = this.pyodide.globals.get('env_step') as () =>
      | Promise<unknown>
      | unknown;
    const raw = await Promise.resolve(envStep());
    const obj = this.toJS(raw) as {
      obs: number[];
      reward: number;
      terminated: boolean;
      truncated: boolean;
      action: number;
    };
    return {
      obs: [obj.obs[0], obj.obs[1], obj.obs[2], obj.obs[3]],
      reward: obj.reward,
      terminated: obj.terminated,
      truncated: obj.truncated,
      action: obj.action === 0 ? 0 : 1,
    };
  }

  private toJS(value: unknown): unknown {
    if (
      value &&
      typeof value === 'object' &&
      'toJs' in (value as Record<string, unknown>)
    ) {
      const proxy = value as {
        toJs: (opts: { dict_converter: typeof Object.fromEntries }) => unknown;
        destroy?: () => void;
      };
      const plain = proxy.toJs({ dict_converter: Object.fromEntries });
      proxy.destroy?.();
      return plain;
    }
    return value;
  }

  dispose(): void {
    this.disposed = true;
    if (this.pyodide) {
      try {
        const envClose = this.pyodide.globals.get('env_close') as () => unknown;
        envClose();
      } catch {
        // 忽略清理期异常
      }
    }
    this.pyodide = null;
  }
}
