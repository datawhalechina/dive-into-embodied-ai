import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import loadMujoco from '@mujoco/mujoco';
import * as ort from 'onnxruntime-web/wasm';
import {generateSimulationAssets} from './simulation-assets.cjs';
import {createGo2} from '../simulation/runtime/go2.mjs';
import {extractGeometry} from '../simulation/runtime/geometry.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
generateSimulationAssets(root);
const source = path.join(root, 'static/simulation-assets/go2');
const mj = await loadMujoco();
const vfs = new mj.MjVFS();
let model;
try {
  vfs.addBuffer('scene.xml', new Uint8Array(fs.readFileSync(path.join(source, 'scene.xml'))));
  model = mj.MjModel.from_xml_path('scene.xml', vfs);
} finally { vfs.delete(); }
const data = new mj.MjData(model);
ort.env.wasm.numThreads = 1;
const session = await ort.InferenceSession.create(new Uint8Array(fs.readFileSync(path.join(source, 'policy.onnx'))), {executionProviders: ['wasm']});
let inferenceCount = 0;
const controller = createGo2(mj, model, data, JSON.parse(fs.readFileSync(path.join(source, 'config.json'))), async obs => {
  assert.equal(obs.length, 47);
  assert(obs.every(Number.isFinite));
  const input = new ort.Tensor('float32', obs, [1, 47]);
  const output = await session.run({obs: input});
  inferenceCount++;
  try { return Float32Array.from(output.action.data); }
  finally { input.dispose(); output.action.dispose(); }
});
const report = [];
try {
  assert.equal(model.nu, 12);
  assert(extractGeometry(model).length > 20);
  for (const command of [[.6, 0, 0], [-.3, 0, 0], [0, .3, 0], [0, 0, .6], [0, 0, 0]]) {
    controller.reset();
    assert.equal(data.time, 0);
    assert.deepEqual(Array.from(controller.observation().slice(-12)), Array(12).fill(0));
    controller.setCommand(command);
    for (let i = 0; i < 300; i++) {
      await controller.step(Math.round(.02 / model.opt.timestep));
      assert(data.qpos.every(Number.isFinite));
      assert(data.qpos[2] > .18, 'Go2 must stay above ground');
      assert(data.xfrc_applied.every(x => x === 0), 'Motion must come from joint torques');
      assert(data.xmat[model.body('base_link').id * 9 + 8] > .8, 'Go2 must stay upright');
    }
    const position = Array.from(data.qpos.slice(0, 3));
    report.push({command, position});
    if (command[0] > 0) assert(position[0] > 1, 'Forward command must move forward');
    if (command[0] < 0) assert(position[0] < -.4, 'Reverse command must move backward');
    if (command[1] > 0) assert(position[1] > .3, 'Lateral command must move sideways');
    if (command[2] > 0) assert(Math.abs(data.qpos[6]) > .3, 'Turn command must change heading');
    if (command.every(x => x === 0)) assert(Math.hypot(...position.slice(0, 2)) < .1, 'Zero command must remain near rest');
  }
  assert.equal(inferenceCount, 1500, 'Policy must run at exactly 50 Hz');
  assert.throws(() => controller.setCommand([NaN, 0, 0]));
  console.log(JSON.stringify({passed: true, inferenceCount, report}, null, 2));
} finally { data.delete(); model.delete(); await session.release(); }
