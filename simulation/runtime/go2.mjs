const clip = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

// Observation/action order is the training order (FL, FR, RL, RR), not the
// SDK actuator order (FR, FL, RR, RL). Resolve every mapping by joint ID.
export function createGo2(mj, model, data, config, infer) {
  const joints = config.joints.map(name => mj.mj_name2id(model, 3, name));
  if (joints.some(id => id < 0) || model.nu !== 12) throw new Error('Go2 joint contract does not match the model');
  const qids = joints.map(id => model.jnt_qposadr[id]);
  const vids = joints.map(id => model.jnt_dofadr[id]);
  const actuators = joints.map(id => Array.from({length: model.nu}, (_, i) => i).find(i => model.actuator_trnid[2 * i] === id));
  if (actuators.some(id => id === undefined)) throw new Error('Go2 actuator contract does not match the model');
  const base = mj.mj_name2id(model, 1, 'base_link');
  let action = new Float32Array(12), ticks = 0, command = [0, 0, 0];
  const target = new Float64Array(12);
  const decimation = Math.round(config.stepDt / model.opt.timestep);
  if (Math.abs(decimation * model.opt.timestep - config.stepDt) > 1e-8) throw new Error('Invalid control decimation');
  function observation() {
    const r = base * 9, rotation = data.xmat;
    const phase = 2 * Math.PI * (ticks * model.opt.timestep % config.period) / config.period;
    const standing = Math.hypot(...command) < .1;
    return Float32Array.from([
      ...data.qvel.slice(3, 6), // MuJoCo free-joint angular velocity is body-local.
      -rotation[r + 6], -rotation[r + 7], -rotation[r + 8],
      ...command, standing ? 0 : Math.sin(phase), standing ? 0 : Math.cos(phase),
      ...qids.map((id, i) => data.qpos[id] - config.defaults[i]),
      ...vids.map(id => data.qvel[id]), ...action,
    ]);
  }
  return {
    observation,
    reset() {
      mj.mj_resetData(model, data);
      data.qpos[2] = .32;
      qids.forEach((id, i) => { data.qpos[id] = config.defaults[i]; });
      action.fill(0); ticks = 0; target.set(config.defaults); command = [0, 0, 0];
      mj.mj_forward(model, data);
    },
    setCommand(value) {
      if (value.length !== 3 || !value.every(Number.isFinite)) throw new Error('Expected three finite commands');
      command = [clip(value[0], -.5, 1), clip(value[1], -.5, .5), clip(value[2], -1, 1)];
    },
    async step(count) {
      for (let n = 0; n < count; n++) {
        if (ticks % decimation === 0) {
          // Align observations with the current qpos, including after perturbations.
          mj.mj_forward(model, data);
          const result = await infer(observation());
          if (result.length !== 12 || !result.every(Number.isFinite)) throw new Error('Invalid policy action');
          action = Float32Array.from(result);
          target.set(action.map((a, i) => config.defaults[i] + config.actionScale * a));
        }
        actuators.forEach((id, i) => {
          const torque = config.stiffness[i] * (target[i] - data.qpos[qids[i]]) - config.damping[i] * data.qvel[vids[i]];
          data.ctrl[id] = clip(torque, model.actuator_ctrlrange[2 * id], model.actuator_ctrlrange[2 * id + 1]);
        });
        const before = data.time;
        mj.mj_step(model, data); ticks++;
        if (data.time <= before || !data.qpos.every(Number.isFinite)) throw new Error('Nonfinite simulation state');
      }
      mj.mj_forward(model, data);
    },
  };
}
