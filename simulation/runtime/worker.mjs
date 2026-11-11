import loadMujoco from '../quaddle/mujoco.js';
import {createSimulation} from '../quaddle/simulation.mjs';
import {createGo2} from './go2.mjs';
import {createPupper} from './pupper.mjs';
import {createMicroDuck} from './microduck.mjs';
import {extractGeometry} from './geometry.mjs';

let sim, model, data, mj, controller, session, body, entry;
let mode = 'showcase', paused = false, visible = true, rate = 1, previous = 0, debt = 0;
let command = [0, 0, 0], heading = 0, gait;
const pending = [];
let started = false;
const get = async (name, json = false) => {
  const response = await fetch(new URL(name, import.meta.url));
  if (!response.ok) throw new Error(`Asset ${name}: HTTP ${response.status}`);
  return json ? response.json() : new Uint8Array(await response.arrayBuffer());
};
function fail(error) {
  paused = true;
  postMessage({type: 'error', detail: error instanceof Error ? error.message : String(error)});
}
function sendState() {
  const position = data.xpos.slice(body * 3, body * 3 + 3);
  const positions = data.geom_xpos.slice(), rotations = data.geom_xmat.slice();
  postMessage({type: 'state', time: data.time, mode, paused, position, positions, rotations,
    speed: Math.hypot(data.qvel[0], data.qvel[1]), upright: data.xmat[body * 9 + 8], command,
    jointTargets: controller?.targets ? Array.from(controller.targets) : undefined, supported: controller?.supported,
  }, [position.buffer, positions.buffer, rotations.buffer]);
}
function applyCommand() {
  const desired = mode === 'showcase' ? entry.demoCommand : command;
  if (controller) controller.setCommand(desired);
  else sim.setMode(desired[0] > 0 || desired[2] !== 0 ? 'walk' : 'stand');
}
function reset() {
  if (controller) controller.reset(); else sim.reset();
  command = [0, 0, 0]; heading = 0;
  if (gait) gait.heading_target_rad = 0;
  debt = 0;
  applyCommand();
}
function processMessages() {
  for (const message of pending.splice(0)) {
    if (message.type === 'mode' && ['manual', 'showcase'].includes(message.value) && !(entry.kind === 'manual' && message.value === 'showcase')) {
      mode = message.value; command = [0, 0, 0]; applyCommand();
    }
    if (message.type === 'command' && mode === 'manual' && Array.isArray(message.value) && message.value.length === 3 && message.value.every(Number.isFinite)) {
      command = message.value; applyCommand();
    }
    if (message.type === 'pause') {
      paused = !!message.value; command = [0, 0, 0]; applyCommand();
    }
    if (message.type === 'visibility') {
      visible = !!message.value; command = [0, 0, 0]; applyCommand();
    }
    if (message.type === 'reset') reset();
    if (message.type === 'joint' && controller?.setJoint && mode === 'manual') controller.setJoint(message.index, message.value);
    if (message.type === 'support' && controller?.setSupported) controller.setSupported(!!message.value);
    if (message.type === 'rate' && [.25, .5, 1].includes(message.value)) rate = message.value;
    if (['pause', 'visibility', 'reset', 'rate'].includes(message.type)) debt = 0;
  }
}
async function tick() {
  try {
    processMessages();
    const now = performance.now();
    if (!paused && visible) {
      debt = Math.min(debt + (now - previous) / 1000 * rate, .04);
      const count = Math.floor(debt / model.opt.timestep);
      if (count) {
        if (controller) await controller.step(count);
        else {
          for (let i = 0; i < count; i++) {
            if (mode === 'manual') heading += command[2] * model.opt.timestep;
            gait.heading_target_rad = heading;
            sim.step();
          }
        }
        debt -= count * model.opt.timestep;
      }
    }
    previous = now;
    sendState();
    setTimeout(tick, 16);
  } catch (error) { fail(error); }
}
async function initialize(message) {
  const catalog = await get('models.json', true);
  entry = catalog.find(item => item.id === message.id);
  if (!entry) throw new Error('Unknown model');
  mode = message.mode === 'manual' || entry.kind === 'manual' ? 'manual' : 'showcase';
  mj = await loadMujoco();
  let geometry;
  if (entry.adapter === 'quaddle') {
    const [manifest, config] = await Promise.all([get('../quaddle/manifest.json', true), get('../quaddle/gait.json', true)]);
    const files = await Promise.all(manifest.files.map(async name => [name, await get(`../quaddle/${name}`)]));
    gait = {...config};
    sim = createSimulation(mj, files, gait);
    ({model, data, geometry} = sim);
    geometry = geometry.map(item => ({...item, type: 7}));
  } else if (['go2', 'pupper', 'microduck'].includes(entry.adapter)) {
    const manifest = entry.adapter === 'go2' ? {files: ['scene.xml']} : await get(`${entry.id}/manifest.json`, true);
    const files = await Promise.all(manifest.files.map(async name => [name, await get(`${entry.id}/${name}`)]));
    const vfs = new mj.MjVFS();
    try {
      for (const [name, bytes] of files) vfs.addBuffer(name, bytes);
      model = mj.MjModel.from_xml_path('scene.xml', vfs);
    } finally { vfs.delete(); }
    data = new mj.MjData(model);
    if (entry.adapter === 'microduck') controller = createMicroDuck(mj, model, data);
    else {
      const [ort, config, policy] = await Promise.all([
        import('./ort/ort.wasm.min.mjs'), get(`${entry.id}/config.json`, true), get(`${entry.id}/policy.onnx`),
      ]);
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy = false;
      ort.env.wasm.wasmPaths = new URL('./ort/', import.meta.url).href;
      session = await ort.InferenceSession.create(policy, {executionProviders: ['wasm']});
      const createController = entry.adapter === 'go2' ? createGo2 : createPupper;
      controller = createController(mj, model, data, config, async obs => {
        const tensor = new ort.Tensor('float32', obs, [1, entry.observationSize]);
        let output;
        try {
          output = await session.run({obs: tensor});
          return Float32Array.from(output.action.data);
        } finally {
          tensor.dispose();
          if (output) Object.values(output).forEach(t => t.dispose());
        }
      });
    }
    geometry = extractGeometry(model, entry.adapter !== 'go2');
  } else throw new Error('Unsupported model adapter');
  body = mj.mj_name2id(model, 1, entry.body);
  if (body < 0) throw new Error('Missing tracking body');
  reset();
  postMessage({type: 'ready', geometry, joints: controller?.joints});
  previous = performance.now();
  tick();
}
onmessage = ({data: message}) => {
  if (message.type === 'init' && !started) { started = true; initialize(message).catch(fail); }
  else pending.push(message);
};
