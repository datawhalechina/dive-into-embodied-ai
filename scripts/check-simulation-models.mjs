import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import loadMujoco from '@mujoco/mujoco';
import * as ort from 'onnxruntime-web/wasm';
import {generateSimulationAssets} from './simulation-assets.cjs';
import {createPupper} from '../simulation/runtime/pupper.mjs';
import {createMicroDuck} from '../simulation/runtime/microduck.mjs';
import {extractGeometry} from '../simulation/runtime/geometry.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
generateSimulationAssets(root);
const mj = await loadMujoco();
function load(id) {
  const dir = path.join(root, 'static/simulation-assets', id);
  const vfs = new mj.MjVFS();
  try {
    for (const name of JSON.parse(fs.readFileSync(path.join(dir, 'manifest.json'))).files) {
      vfs.addBuffer(name, new Uint8Array(fs.readFileSync(path.join(dir, name))));
    }
    const model = mj.MjModel.from_xml_path('scene.xml', vfs);
    assert(extractGeometry(model, true).some(geom => geom.type === 7), `${id}: visual meshes required`);
    return {model, data: new mj.MjData(model), dir};
  } finally { vfs.delete(); }
}
const pupper = load('pupper-v3');
ort.env.wasm.numThreads = 1;
const session = await ort.InferenceSession.create(new Uint8Array(fs.readFileSync(path.join(pupper.dir, 'policy.onnx'))), {executionProviders: ['wasm']});
const infer = async obs => {
  assert.equal(obs.length, 53); assert(obs.every(Number.isFinite));
  const input = new ort.Tensor('float32', obs, [1, 53]);
  const result = await session.run({obs: input});
  try { return Float32Array.from(result.action.data); }
  finally { input.dispose(); result.action.dispose(); }
};
const controller = createPupper(mj, pupper.model, pupper.data, JSON.parse(fs.readFileSync(path.join(pupper.dir, 'config.json'))), infer);
try {
  const reference = JSON.parse(fs.readFileSync(path.join(root, 'codes/practices/quadruped/cs123/6.rl_pupper_v2_5/web/reference.json')));
  for (const fixture of reference) {
    controller.reset(); pupper.data.qpos.set(fixture.qpos); pupper.data.qvel.set(fixture.qvel);
    controller.setCommand(fixture.command); mj.mj_forward(pupper.model, pupper.data);
    const obs = controller.observation();
    assert(Math.max(...obs.map((v, i) => Math.abs(v - fixture.observation[i]))) < 1e-5, 'Pupper observation must match Python at multiple headings');
    const action = await infer(obs);
    assert(Math.max(...action.map((v, i) => Math.abs(v - fixture.action[i]))) < 1e-5, 'Pupper actor must match SB3');
  }
  const report = [];
  for (const command of [[.4, 0, 0], [-.2, 0, 0], [0, .2, 0], [0, 0, .4], [0, 0, 0]]) {
    controller.reset(); controller.setCommand(command);
    let minZ = 1, minUp = 1;
    for (let i = 0; i < 500; i++) {
      await controller.step(5);
      minZ = Math.min(minZ, pupper.data.qpos[2]);
      minUp = Math.min(minUp, pupper.data.xmat[pupper.model.body('base_link').id * 9 + 8]);
      assert(pupper.data.qpos.every(Number.isFinite));
      assert(pupper.data.xfrc_applied.every(v => v === 0));
    }
    const position = Array.from(pupper.data.qpos.slice(0, 3));
    assert(minZ > .12 && minUp > .86, 'Pupper must remain upright and above the training fall threshold');
    if (command[0] > 0) assert(position[0] > 1, 'Forward command must advance');
    if (command[0] < 0) assert(position[0] < -.5, 'Reverse command must retreat');
    if (command[1] > 0) assert(position[1] > .5, 'Strafe command must move sideways');
    if (command[2] > 0) assert(Math.abs(pupper.data.qpos[6]) > .3, 'Turn command must change heading');
    if (command.every(v => v === 0)) assert(Math.hypot(...position.slice(0, 2)) < .25, 'Zero command must limit drift');
    report.push({command, position, minZ, minUp});
  }
  controller.reset();
  assert.equal(pupper.data.time, 0);
  assert.deepEqual(Array.from(controller.observation().slice(33, 45)), Array(12).fill(0));
  assert.throws(() => controller.setCommand([NaN, 0, 0]));
  console.log('Pupper:', JSON.stringify(report));
} finally { pupper.data.delete(); pupper.model.delete(); await session.release(); }

const duck = load('microduck');
try {
  const manual = createMicroDuck(mj, duck.model, duck.data);
  manual.reset();
  assert.equal(manual.joints.length, 14);
  assert(manual.supported);
  const start = duck.data.qpos.slice(0, 3);
  const yaw = manual.joints.findIndex(j => j.name === 'head_yaw');
  const joint = duck.model.actuator_trnid[yaw * 2];
  const qposId = duck.model.jnt_qposadr[joint];
  manual.setJoint(yaw, .3);
  manual.step(Math.round(2 / duck.model.opt.timestep));
  assert(Math.abs(duck.data.qpos[qposId] - .3) < .08, 'Head must physically follow the servo target');
  assert(Math.hypot(...start.map((v, i) => v - duck.data.qpos[i])) < .01, 'Support must hold the base');
  manual.setCommand([1, 1, 1]);
  manual.step(Math.round(3 / duck.model.opt.timestep));
  manual.setCommand([0, 0, 0]);
  for (let i = 0; i < manual.joints.length; i++) {
    assert.throws(() => manual.setJoint(i, Infinity));
    manual.setJoint(i, 100);
    assert.equal(manual.targets[i], manual.joints[i].max);
    manual.setJoint(i, -100);
    assert.equal(manual.targets[i], manual.joints[i].min);
  }
  assert.throws(() => manual.setJoint(-1, 0));
  assert.throws(() => manual.setCommand([NaN, 0, 0]));
  manual.reset(); manual.setSupported(false);
  assert(!manual.supported);
  manual.step(Math.round(2 / duck.model.opt.timestep));
  assert(duck.data.qpos.every(Number.isFinite));
  manual.setSupported(true);
  assert(manual.supported); assert.equal(duck.data.time, 0);
  assert.deepEqual(Array.from(duck.data.qpos.slice(0, 3)), Array.from(start));
  console.log('MicroDuck: 14 servos, target tracking, limits, keyboard head control, support release and reset passed.');
} finally { duck.data.delete(); duck.model.delete(); }
