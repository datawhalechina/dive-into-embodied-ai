const clip = (value, low, high) => Math.max(low, Math.min(high, value));
export function createPupper(mj, model, data, config, infer) {
  model.opt.timestep = config.physicsDt;
  const body = mj.mj_name2id(model, 1, 'base_link');
  const joints = config.joints.map(name => mj.mj_name2id(model, 3, name));
  if (model.nu !== 12 || joints.some(id => id < 0)) throw new Error('Pupper joint contract mismatch');
  const qids = joints.map(id => model.jnt_qposadr[id]), vids = joints.map(id => model.jnt_dofadr[id]);
  const actuatorIds = joints.map(id => Array.from({length: model.nu}, (_, a) => a).find(a => model.actuator_trnid[2 * a] === id));
  if (actuatorIds.some(id => id === undefined)) throw new Error('Missing Pupper actuator');
  for (const id of actuatorIds) {
    model.actuator_gainprm[id * 10] = config.kp;
    model.actuator_biasprm[id * 10 + 1] = -config.kp;
    model.actuator_biasprm[id * 10 + 2] = -config.kd;
  }
  let ticks = 0, command = [0, 0, 0], action = new Float32Array(12);
  const filtered = new Float64Array(12);
  const decimation = Math.round(config.stepDt / config.physicsDt);
  function observation() {
    const r = body * 9, m = data.xmat;
    const local = v => [0, 1, 2].map(i => m[r + i] * v[0] + m[r + 3 + i] * v[1] + m[r + 6 + i] * v[2]);
    const gravity = local([0, 0, -1]), norm = Math.hypot(...gravity);
    const phase = 2 * Math.PI * (Math.floor(ticks / decimation) * config.stepDt % config.period) / config.period;
    // Preserve the training environment's angular-velocity transformation,
    // including its additional R^T rotation of free-joint angular qvel.
    return Float32Array.from([
      ...local(data.qvel.slice(3, 6)), ...gravity.map(v => v / norm), ...command,
      ...qids.map((id, i) => data.qpos[id] - config.defaults[i]), ...vids.map(id => data.qvel[id]),
      ...action, ...local(data.qvel.slice(0, 3)), 0, 1, 0, Math.sin(phase), Math.cos(phase),
    ].map(v => clip(v, -100, 100)));
  }
  return {
    observation,
    reset() {
      mj.mj_resetData(model, data); data.qpos[2] = config.height;
      qids.forEach((id, i) => {data.qpos[id] = config.defaults[i]; data.ctrl[actuatorIds[i]] = config.defaults[i];});
      ticks = 0; action.fill(0); filtered.fill(0); command = [0, 0, 0]; mj.mj_forward(model, data);
    },
    setCommand(value) {
      if (value.length !== 3 || !value.every(Number.isFinite)) throw new Error('Expected finite commands');
      command = [clip(value[0], -.75, .75), clip(value[1], -.5, .5), clip(value[2], -2, 2)];
    },
    async step(count) {
      for (let n = 0; n < count; n++) {
        if (ticks % decimation === 0) {
          const result = await infer(observation());
          if (result.length !== 12 || !result.every(Number.isFinite)) throw new Error('Invalid Pupper action');
          action = Float32Array.from(result, x => clip(x, -1, 1));
          for (let i = 0; i < 12; i++) {
            filtered[i] = config.alpha * action[i] + (1 - config.alpha) * filtered[i];
            data.ctrl[actuatorIds[i]] = clip(config.defaults[i] + config.actionScale * filtered[i], config.lowers[i], config.uppers[i]);
          }
        }
        const before = data.time; mj.mj_step(model, data); ticks++;
        if (data.time <= before || !data.qpos.every(Number.isFinite)) throw new Error('Invalid Pupper physics state');
      }
      mj.mj_forward(model, data);
    },
  };
}
