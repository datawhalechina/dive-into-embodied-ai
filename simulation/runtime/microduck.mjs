export function createMicroDuck(mj, model, data) {
  const clip = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const stand = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY.value, 'STAND');
  const support = mj.mj_name2id(model, mj.mjtObj.mjOBJ_EQUALITY.value, 'manual_support');
  if (stand < 0 || support < 0 || model.nu !== 14) throw new Error('MicroDuck pose contract mismatch');
  let command = [0, 0, 0];
  const targets = new Float64Array(model.nu);
  const joints = Array.from({length: model.nu}, (_, i) => {
    const joint = model.actuator_trnid[i * 2];
    return {name: mj.mj_id2name(model, 3, joint), min: model.jnt_range[joint * 2], max: model.jnt_range[joint * 2 + 1]};
  });
  const headYaw = joints.findIndex(j => j.name === 'head_yaw');
  const headPitch = joints.findIndex(j => j.name === 'head_pitch');
  const headRoll = joints.findIndex(j => j.name === 'head_roll');
  function setJoint(index, value) {
    if (!Number.isInteger(index) || index < 0 || index >= joints.length || !Number.isFinite(value)) throw new Error('Invalid joint target');
    targets[index] = clip(value, joints[index].min, joints[index].max);
  }
  return {
    joints, setJoint,
    get targets() { return targets; },
    get supported() { return !!data.eq_active[support]; },
    setSupported(value) { if (value) this.reset(); else data.eq_active[support] = 0; },
    setCommand(value) {
      if (value.length !== 3 || !value.every(Number.isFinite)) throw new Error('Expected finite commands');
      command = value.map(v => clip(v, -1, 1));
    },
    reset() {
      mj.mj_resetDataKeyframe(model, data, stand); targets.set(data.ctrl);
      data.eq_active[support] = 1; command = [0, 0, 0]; mj.mj_forward(model, data);
    },
    step(count) {
      for (let i = 0; i < count; i++) {
        const dt = model.opt.timestep;
        setJoint(headYaw, targets[headYaw] + command[2] * dt);
        setJoint(headPitch, targets[headPitch] - command[0] * dt);
        setJoint(headRoll, targets[headRoll] + command[1] * dt);
        // Native position servos, with bounded target slew. No learned gait.
        for (let a = 0; a < model.nu; a++) data.ctrl[a] += clip(targets[a] - data.ctrl[a], -1.5 * dt, 1.5 * dt);
        const before = data.time; mj.mj_step(model, data);
        if (data.time <= before || !data.qpos.every(Number.isFinite)) throw new Error('Invalid MicroDuck physics state');
      }
      mj.mj_forward(model, data);
    },
  };
}
