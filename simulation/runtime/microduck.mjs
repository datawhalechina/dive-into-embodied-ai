const clip = (value, low, high) => Math.max(low, Math.min(high, value));

// MicroDuck 14-DoF PPO walking controller with BAM M6 XL330 voltage/current-limited
// DC motor dynamics and command lag matching mjlab training physics.
export function createMicroDuck(mj, model, data, config, infer) {
  model.opt.timestep = config.simDt;
  const trunk = mj.mj_name2id(model, 1, 'trunk_base');
  const joints = config.joints.map(name => mj.mj_name2id(model, 3, name));
  if (trunk < 0 || model.nu !== 14 || joints.some(id => id < 0)) {
    throw new Error('MicroDuck joint contract does not match the model');
  }
  const qids = joints.map(id => model.jnt_qposadr[id]);
  const vids = joints.map(id => model.jnt_dofadr[id]);
  const actuators = joints.map(id => Array.from({length: model.nu}, (_, a) => a).find(a => model.actuator_trnid[2 * a] === id));
  if (actuators.some(id => id === undefined)) throw new Error('MicroDuck actuator contract does not match the model');

  // Switch XML position servos into raw torque actuators so we can apply the
  // trained XL330 BAM M6 DC-motor law (PWM duty, back-EMF, current limiter).
  const bm = config.bam;
  for (let i = 0; i < 14; i++) {
    const aid = actuators[i], vid = vids[i];
    model.actuator_gainprm[aid * 10] = 1.0;
    for (let k = 0; k < 10; k++) model.actuator_biasprm[aid * 10 + k] = 0.0;
    model.actuator_biastype[aid] = 0; // mjBIAS_NONE
    model.actuator_forcerange[2 * aid] = -bm.forceLimit;
    model.actuator_forcerange[2 * aid + 1] = bm.forceLimit;
    model.dof_damping[vid] = bm.frictionViscous;
    model.dof_frictionloss[vid] = bm.frictionBase;
    model.dof_armature[vid] = bm.armature;
  }

  let action = new Float32Array(14);
  const target = new Float64Array(config.defaults);
  let delayBuffer = [];
  let ticks = 0;
  let command = [0, 0, 0];

  function resetBuffer() {
    delayBuffer = Array.from({length: config.delaySteps + 1}, () => Float64Array.from(config.defaults));
  }

  function observation() {
    const r = trunk * 9, m = data.xmat;
    return Float32Array.from([
      data.qvel[3], data.qvel[4], data.qvel[5],
      -m[r + 6], -m[r + 7], -m[r + 8],
      ...qids.map((id, i) => data.qpos[id] - config.defaults[i]),
      ...vids.map(id => data.qvel[id]),
      ...action,
      ...command,
      0, 0, 0, 0,             // head_pose command (neutral)
      0, 0, 0, 0, 0, 0,       // body_pose command (neutral)
    ]);
  }

  resetBuffer();

  return {
    observation,
    setHistory(lastAction) {
      if (lastAction) action = Float32Array.from(lastAction);
    },
    reset() {
      mj.mj_resetData(model, data);
      data.qpos[2] = config.initHeight;
      qids.forEach((id, i) => {
        data.qpos[id] = config.defaults[i];
        data.ctrl[actuators[i]] = 0;
      });
      action.fill(0);
      target.set(config.defaults);
      resetBuffer();
      ticks = 0;
      command = [0, 0, 0];
      mj.mj_forward(model, data);
    },
    setCommand(value) {
      if (value.length !== 3 || !value.every(Number.isFinite)) throw new Error('Expected three finite commands');
      command = [clip(value[0], -.35, .35), clip(value[1], -.3, .3), clip(value[2], -1, 1)];
    },
    async step(count) {
      for (let n = 0; n < count; n++) {
        if (ticks % config.decimation === 0) {
          mj.mj_forward(model, data);
          const result = await infer(observation());
          if (result.length !== 14 || !result.every(Number.isFinite)) throw new Error('Invalid MicroDuck policy action');
          action = Float32Array.from(result);
          for (let i = 0; i < 14; i++) target[i] = config.defaults[i] + action[i] * config.actionScale;
        }
        delayBuffer.push(Float64Array.from(target));
        const delayed = delayBuffer.shift();
        for (let i = 0; i < 14; i++) {
          const q = data.qpos[qids[i]], dq = data.qvel[vids[i]];
          const dutyRaw = (delayed[i] - q) * bm.kpFw * bm.errorGain;
          const center = (bm.kt * dq) / bm.vin;
          const span = (bm.R * bm.maxCurrent) / bm.vin;
          const duty = clip(clip(dutyRaw, center - span, center + span), -bm.maxPwm, bm.maxPwm);
          data.ctrl[actuators[i]] = (bm.kt * bm.vin * duty - bm.kt * bm.kt * dq) / bm.R;
        }
        const before = data.time;
        mj.mj_step(model, data);
        ticks++;
        if (data.time <= before || !data.qpos.every(Number.isFinite)) throw new Error('Nonfinite MicroDuck simulation state');
      }
      mj.mj_forward(model, data);
    },
  };
}
