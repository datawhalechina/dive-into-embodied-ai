import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import loadMujoco from '@mujoco/mujoco';
import * as ort from 'onnxruntime-web/wasm';
import {generateSimulationAssets} from './simulation-assets.cjs';
import {createG1} from '../simulation/runtime/g1.mjs';
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

async function verifyUnitreeHumanoid({id, label, refPath, obsDim, minZLimit, pushMinZ, fwdMin, standMaxDrift}) {
  const bot = load(id);
  const cfg = JSON.parse(fs.readFileSync(path.join(bot.dir, 'config.json')));
  const sess = await ort.InferenceSession.create(new Uint8Array(fs.readFileSync(path.join(bot.dir, 'policy.onnx'))), {executionProviders: ['wasm']});
  const infer = async (obs, hIn, cIn) => {
    assert.equal(obs.length, obsDim); assert(obs.every(Number.isFinite));
    const obsT = new ort.Tensor('float32', obs, [1, obsDim]);
    const hT = new ort.Tensor('float32', hIn, [1, 1, cfg.hiddenSize]);
    const cT = new ort.Tensor('float32', cIn, [1, 1, cfg.hiddenSize]);
    const out = await sess.run({obs: obsT, h_in: hT, c_in: cT});
    try {
      return {
        action: Float32Array.from(out.action.data),
        h: Float32Array.from(out.h_out.data),
        c: Float32Array.from(out.c_out.data),
      };
    } finally {
      obsT.dispose(); hT.dispose(); cT.dispose();
      Object.values(out).forEach(t => t.dispose());
    }
  };
  const ctrl = createG1(mj, bot.model, bot.data, cfg, infer);
  try {
    const reference = JSON.parse(fs.readFileSync(path.join(root, refPath)));
    for (const fixture of reference) {
      ctrl.reset();
      bot.data.time = fixture.time;
      bot.data.qpos.set(fixture.qpos);
      bot.data.qvel.set(fixture.qvel);
      ctrl.setCommand(fixture.cmd);
      ctrl.setHistory(fixture.lastAction, fixture.hIn, fixture.cIn);
      mj.mj_forward(bot.model, bot.data);
      const obs = ctrl.observation();
      assert(Math.max(...obs.map((v, i) => Math.abs(v - fixture.observation[i]))) < 1e-5, `${label} observation must match Python reference`);
      const out = await infer(obs, Float32Array.from(fixture.hIn), Float32Array.from(fixture.cIn));
      assert(Math.max(...out.action.map((v, i) => Math.abs(v - fixture.action[i]))) < 1e-5, `${label} LSTM ONNX must match TorchScript`);
    }
    const report = [];
    for (const command of [[.5, 0, 0], [-.2, 0, 0], [0, .2, 0], [0, 0, .5], [0, 0, 0]]) {
      ctrl.reset(); ctrl.setCommand(command);
      let minZ = 10, minUp = 10, maxAbsQz = 0;
      for (let i = 0; i < 300; i++) {
        await ctrl.step(cfg.decimation);
        minZ = Math.min(minZ, bot.data.qpos[2]);
        minUp = Math.min(minUp, bot.data.xmat[bot.model.body('pelvis').id * 9 + 8]);
        maxAbsQz = Math.max(maxAbsQz, Math.abs(bot.data.qpos[6]));
        assert(bot.data.qpos.every(Number.isFinite));
        assert(bot.data.xfrc_applied.every(v => v === 0));
      }
      const position = Array.from(bot.data.qpos.slice(0, 3));
      assert(minZ > minZLimit && minUp > .95, `${label} must stay upright for ${JSON.stringify(command)} (minZ=${minZ}, minUp=${minUp})`);
      if (command[0] > 0) assert(position[0] > fwdMin, `${label} forward command must advance`);
      if (command[0] < 0) assert(position[0] < -.5, `${label} reverse command must retreat`);
      if (command[1] > 0) assert(position[1] > .4, `${label} strafe command must move sideways`);
      if (command[2] > 0) assert(maxAbsQz > .4, `${label} turn command must change heading`);
      if (command.every(v => v === 0)) assert(Math.hypot(...position.slice(0, 2)) < standMaxDrift, `${label} zero command must limit drift`);
      report.push({command, position, minZ, minUp});
    }
    // Verify lateral push recovery during walking
    ctrl.reset(); ctrl.setCommand([.5, 0, 0]);
    for (let i = 0; i < 50; i++) await ctrl.step(cfg.decimation);
    bot.data.qvel[1] += .55; mj.mj_forward(bot.model, bot.data);
    for (let i = 0; i < 100; i++) {
      await ctrl.step(cfg.decimation);
      assert(bot.data.qpos[2] > pushMinZ && bot.data.xmat[bot.model.body('pelvis').id * 9 + 8] > .9, `${label} must recover from lateral push`);
    }
    ctrl.reset();
    assert.equal(bot.data.time, 0);
    assert(ctrl.hidden.h.every(v => v === 0) && ctrl.hidden.c.every(v => v === 0));
    assert.throws(() => ctrl.setCommand([NaN, 0, 0]));
    console.log(`${label}:`, JSON.stringify(report));
  } finally { bot.data.delete(); bot.model.delete(); await sess.release(); }
}

await verifyUnitreeHumanoid({id: 'g1', label: 'G1', refPath: 'codes/practices/humanoid/unitree-g1-web/reference.json', obsDim: 47, minZLimit: .65, pushMinZ: .65, fwdMin: 1.5, standMaxDrift: .3});
await verifyUnitreeHumanoid({id: 'h1', label: 'H1', refPath: 'codes/practices/humanoid/unitree-h1-web/reference.json', obsDim: 41, minZLimit: .90, pushMinZ: .90, fwdMin: 1.5, standMaxDrift: .65});
await verifyUnitreeHumanoid({id: 'h1-2', label: 'H1-2', refPath: 'codes/practices/humanoid/unitree-h1-2-web/reference.json', obsDim: 47, minZLimit: .90, pushMinZ: .85, fwdMin: 1.5, standMaxDrift: .35});

const duck = load('microduck');
const duckConfig = JSON.parse(fs.readFileSync(path.join(duck.dir, 'config.json')));
const duckSession = await ort.InferenceSession.create(new Uint8Array(fs.readFileSync(path.join(duck.dir, 'policy.onnx'))), {executionProviders: ['wasm']});
const duckInfer = async obs => {
  assert.equal(obs.length, 61); assert(obs.every(Number.isFinite));
  const input = new ort.Tensor('float32', obs, [1, 61]);
  const out = await duckSession.run({obs: input});
  try { return Float32Array.from(out.action.data); }
  finally { input.dispose(); out.action.dispose(); }
};
const duckController = createMicroDuck(mj, duck.model, duck.data, duckConfig, duckInfer);
try {
  const duckReference = JSON.parse(fs.readFileSync(path.join(root, 'codes/practices/humanoid/microduck-rl/web/reference.json')));
  for (const fixture of duckReference) {
    duckController.reset();
    duck.data.qpos.set(fixture.qpos);
    duck.data.qvel.set(fixture.qvel);
    duckController.setCommand(fixture.cmd);
    duckController.setHistory(fixture.lastAction);
    mj.mj_forward(duck.model, duck.data);
    const obs = duckController.observation();
    assert(Math.max(...obs.map((v, i) => Math.abs(v - fixture.observation[i]))) < 1e-5, 'MicroDuck observation must match Python reference');
    const act = await duckInfer(obs);
    assert(Math.max(...act.map((v, i) => Math.abs(v - fixture.action[i]))) < 1e-5, 'MicroDuck ONNX actor must match PyTorch');
  }
  const duckReport = [];
  for (const command of [[.3, 0, 0], [-.3, 0, 0], [0, .3, 0], [0, 0, .8], [0, 0, 0]]) {
    duckController.reset(); duckController.setCommand(command);
    let minZ = 10, minUp = 10, maxAbsQz = 0;
    for (let i = 0; i < 250; i++) {
      await duckController.step(duckConfig.decimation);
      minZ = Math.min(minZ, duck.data.qpos[2]);
      minUp = Math.min(minUp, duck.data.xmat[duck.model.body('trunk_base').id * 9 + 8]);
      maxAbsQz = Math.max(maxAbsQz, Math.abs(duck.data.qpos[6]));
      assert(duck.data.qpos.every(Number.isFinite));
      assert(duck.data.xfrc_applied.every(v => v === 0));
    }
    const position = Array.from(duck.data.qpos.slice(0, 3));
    assert(minZ > .10 && minUp > .95, `MicroDuck must remain upright for ${JSON.stringify(command)}`);
    if (command[0] > 0) assert(position[0] > .6, 'MicroDuck forward command must advance');
    if (command[0] < 0) assert(position[0] < -.5, 'MicroDuck reverse command must retreat');
    if (command[1] > 0) assert(position[1] > .15, 'MicroDuck strafe command must move sideways');
    if (command[2] > 0) assert(maxAbsQz > .8, 'MicroDuck turn command must change heading');
    if (command.every(v => v === 0)) assert(Math.hypot(...position.slice(0, 2)) < .08, 'MicroDuck zero command must stay still');
    duckReport.push({command, position, minZ, minUp, maxAbsQz});
  }
  // Verify lateral push recovery during walking
  duckController.reset(); duckController.setCommand([.25, 0, 0]);
  for (let i = 0; i < 60; i++) await duckController.step(duckConfig.decimation);
  duck.data.qvel[1] += .25; mj.mj_forward(duck.model, duck.data);
  for (let i = 0; i < 120; i++) {
    await duckController.step(duckConfig.decimation);
    assert(duck.data.qpos[2] > .10 && duck.data.xmat[duck.model.body('trunk_base').id * 9 + 8] > .9, 'MicroDuck must recover from lateral push');
  }
  duckController.reset();
  assert.equal(duck.data.time, 0);
  assert.throws(() => duckController.setCommand([NaN, 0, 0]));
  console.log('MicroDuck:', JSON.stringify(duckReport));
} finally { duck.data.delete(); duck.model.delete(); await duckSession.release(); }
