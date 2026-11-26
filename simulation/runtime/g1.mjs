const clip = (value, low, high) => Math.max(low, Math.min(high, value));

// Unitree humanoid (G1 12-DoF, H1 10-DoF, H1-2 12-DoF) recurrent (LSTM) velocity controller.
// Expects a stateless ONNX wrapper returning {action, h, c} for inputs (obs, h, c).
export function createG1(mj, model, data, config, infer) {
  model.opt.timestep = config.simDt;
  const numActions = config.joints.length;
  const pelvis = mj.mj_name2id(model, 1, config.body ?? 'pelvis');
  const joints = config.joints.map(name => mj.mj_name2id(model, 3, name));
  if (pelvis < 0 || model.nu !== numActions || joints.some(id => id < 0)) {
    throw new Error('Unitree humanoid joint contract does not match the model');
  }
  const qids = joints.map(id => model.jnt_qposadr[id]);
  const vids = joints.map(id => model.jnt_dofadr[id]);
  const actuators = joints.map(id => Array.from({length: model.nu}, (_, a) => a).find(a => model.actuator_trnid[2 * a] === id));
  if (actuators.some(id => id === undefined)) throw new Error('Unitree humanoid actuator contract does not match the model');

  const hiddenSize = config.hiddenSize;
  let action = new Float32Array(numActions);
  let h = new Float32Array(hiddenSize);
  let c = new Float32Array(hiddenSize);
  const target = new Float64Array(config.defaults);
  let ticks = 0;
  let command = [0, 0, 0];

  function observation() {
    const r = pelvis * 9, rotation = data.xmat;
    const phase = (data.time % config.period) / config.period;
    const sinPhase = Math.sin(2 * Math.PI * phase);
    const cosPhase = Math.cos(2 * Math.PI * phase);
    return Float32Array.from([
      data.qvel[3] * config.angVelScale,
      data.qvel[4] * config.angVelScale,
      data.qvel[5] * config.angVelScale,
      -rotation[r + 6],
      -rotation[r + 7],
      -rotation[r + 8],
      command[0] * config.cmdScale[0],
      command[1] * config.cmdScale[1],
      command[2] * config.cmdScale[2],
      ...qids.map((id, i) => (data.qpos[id] - config.defaults[i]) * config.dofPosScale),
      ...vids.map(id => data.qvel[id] * config.dofVelScale),
      ...action,
      sinPhase,
      cosPhase,
    ]);
  }

  return {
    observation,
    get hidden() { return {h: Float32Array.from(h), c: Float32Array.from(c)}; },
    setHistory(lastAction, hIn, cIn) {
      if (lastAction) action = Float32Array.from(lastAction);
      if (hIn) h = Float32Array.from(hIn);
      if (cIn) c = Float32Array.from(cIn);
    },
    reset() {
      mj.mj_resetData(model, data);
      data.qpos[2] = config.initHeight;
      qids.forEach((id, i) => { data.qpos[id] = config.defaults[i]; });
      action.fill(0);
      h.fill(0);
      c.fill(0);
      target.set(config.defaults);
      ticks = 0;
      command = [0, 0, 0];
      mj.mj_forward(model, data);
    },
    setCommand(value) {
      if (value.length !== 3 || !value.every(Number.isFinite)) throw new Error('Expected three finite commands');
      command = [clip(value[0], -.4, .8), clip(value[1], -.4, .4), clip(value[2], -.8, .8)];
    },
    async step(count) {
      for (let n = 0; n < count; n++) {
        if (ticks % config.decimation === 0) {
          mj.mj_forward(model, data);
          const result = await infer(observation(), h, c);
          if (!result || result.action?.length !== numActions || !result.action.every(Number.isFinite)) {
            throw new Error('Invalid Unitree humanoid policy action');
          }
          if (result.h?.length !== hiddenSize || result.c?.length !== hiddenSize || !result.h.every(Number.isFinite) || !result.c.every(Number.isFinite)) {
            throw new Error('Invalid Unitree humanoid recurrent state');
          }
          action = Float32Array.from(result.action);
          h = Float32Array.from(result.h);
          c = Float32Array.from(result.c);
          for (let i = 0; i < numActions; i++) target[i] = config.defaults[i] + action[i] * config.actionScale;
        }
        actuators.forEach((id, i) => {
          let torque = config.stiffness[i] * (target[i] - data.qpos[qids[i]]) - config.damping[i] * data.qvel[vids[i]];
          if (model.actuator_ctrllimited[id]) {
            torque = clip(torque, model.actuator_ctrlrange[2 * id], model.actuator_ctrlrange[2 * id + 1]);
          }
          data.ctrl[id] = torque;
        });
        const before = data.time;
        mj.mj_step(model, data);
        ticks++;
        if (data.time <= before || !data.qpos.every(Number.isFinite)) throw new Error('Nonfinite G1 simulation state');
      }
      mj.mj_forward(model, data);
    },
  };
}
