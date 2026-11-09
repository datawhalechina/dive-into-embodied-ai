"""Measure physical progress, foot liftoff, support, limits and numeric stability."""
import argparse
import json
import math
import time
import hashlib

import mujoco
import numpy as np

from gait_controller import JointController, CONFIG
from run_sim import ROOT, SIDES, load, closure_error
from terrain import configure_ground


def validate(seconds=60, friction=None, stop=False, slope_deg=0.):
    if not math.isfinite(seconds) or seconds <= 4:
        raise ValueError("seconds must be finite and greater than 4 for footfall statistics")
    if friction is not None and (not math.isfinite(friction) or friction <= 0):
        raise ValueError("friction must be finite and positive")
    m, d = load()
    ground_rotation = configure_ground(m, d, slope_deg)
    forward_axis = ground_rotation @ np.array([0., -1., 0.])
    ground_normal = ground_rotation[:, 2]
    if friction is not None:
        m.geom_friction[:, 0] = friction
    controller = JointController("walk")
    duration = CONFIG["settle_seconds"] + seconds + (3 if stop else 0)
    dt = m.opt.timestep
    start = None
    start_stop = None
    clock = time.perf_counter()
    records = []
    previous_contact = np.zeros(4, dtype=bool)
    stable_contact = np.zeros(4, dtype=bool)
    stable_counts = np.zeros(4, dtype=int)
    liftoffs = np.zeros(4, dtype=int)
    contact_samples = np.zeros(4, dtype=int)
    max_tip_height = np.zeros(4)
    max_yaw, max_side, max_closure, max_force, max_speed = 0., 0., 0., 0., 0.
    min_up, min_z, max_z = 1., np.inf, -np.inf
    body_ground_steps, leg_ground_steps, external_force_steps = 0, 0, 0
    samples = 0
    finite, monotonic = True, True
    last_time = -1.
    body_id = m.geom("body_contact").id
    foot_geoms = [set(i for i in range(m.ngeom) if (m.geom(i).name or '').startswith(s + '_wheel') and '_contact_' in (m.geom(i).name or '')) for s in SIDES]
    shin_geoms = {m.geom(s + '_leg_contact').id for s in SIDES}
    driven_addresses = [m.jnt_dofadr[m.joint(s+'_thigh').id] for s in SIDES]
    for step in range(round(duration / dt)):
        if stop and d.time >= CONFIG["settle_seconds"] + seconds and start_stop is None:
            controller.mode = "stand"
            start_stop = d.qpos[:3].copy()
        controller(d)
        mujoco.mj_step(m, d)
        finite &= bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
        monotonic &= bool(d.time > last_time)
        last_time = d.time
        external_force_steps += int(np.any(d.xfrc_applied) or np.any(d.qfrc_applied))
        max_force = max(max_force, float(np.max(np.abs(d.actuator_force))))
        max_speed = max(max_speed, float(np.max(np.abs(d.qvel[driven_addresses]))))
        if step % 5:
            continue
        mujoco.mj_forward(m, d)
        max_closure = max(max_closure, closure_error(d))
        if d.time < CONFIG["settle_seconds"]:
            continue
        if start is None:
            start = d.qpos[:3].copy()
        active = np.zeros(4, dtype=bool)
        ground_ids = {c.geom2 if c.geom1 == m.geom('ground').id else c.geom1 for c in d.contact if m.geom('ground').id in (c.geom1, c.geom2)}
        body_ground_steps += int(body_id in ground_ids)
        leg_ground_steps += int(bool(shin_geoms & ground_ids))
        for j, ids in enumerate(foot_geoms):
            active[j] = bool(ids & ground_ids)
            max_tip_height[j] = max(max_tip_height[j], float(d.body(SIDES[j]+'_wheel_tip').xpos @ ground_normal))
        if d.time >= 4 and start_stop is None:
            # Require 20 ms in a new contact state so solver chatter is not
            # counted as repeated steps within one footfall.
            stable_counts = np.where(active == previous_contact, stable_counts + 1, 1)
            changed = (stable_counts >= 4) & (stable_contact != active)
            liftoffs += changed & stable_contact & ~active
            stable_contact[changed] = active[changed]
            contact_samples += active
            samples += 1
        previous_contact = active
        rot = ground_rotation.T @ d.body('body').xmat.reshape(3, 3)
        max_yaw = max(max_yaw, abs(math.atan2(rot[1,0], rot[0,0])))
        min_up = min(min_up, rot[2,2])
        height = float(d.qpos[:3] @ ground_normal)
        min_z, max_z = min(min_z, height), max(max_z, height)
        max_side = max(max_side, abs(d.qpos[0] - start[0]))
        if step % 1000 == 0:
            records.append(dict(t_s=round(d.time, 3), forward_m=float((d.qpos[:3] - start) @ forward_axis), lateral_m=float(d.qpos[0]-start[0]), height_m=height))
    end = start_stop if start_stop is not None else d.qpos[:3]
    forward = float((end - start) @ forward_axis)
    step_progress = np.diff([r['forward_m'] for r in records if 4 <= r['t_s'] < CONFIG['settle_seconds']+seconds])
    checks = dict(finite=bool(finite), monotonic_time=bool(monotonic),
                  no_warnings=bool(d.warning.number.sum() == 0), no_external_forces=external_force_steps == 0,
                  no_body_contact=body_ground_steps == 0, upright=bool(min_up > .8),
                  body_clearance=bool(min_z > .025), forward_speed=forward / seconds > .03,
                  lateral_drift=bool(max_side < max(.1, forward * .1)), heading=math.degrees(max_yaw) < 25,
                  all_feet_lift=bool(np.all(liftoffs >= 5)),
                  foot_support=bool(np.all(contact_samples / samples > .05) and np.all(contact_samples / samples < .95)),
                  loop_closure=max_closure < .002, torque_limit=max_force <= .245001,
                  joint_speed=max_speed <= 10.,
                  continuous_progress=bool(len(step_progress)>0 and step_progress.min() > .01))
    final_speed = float(np.linalg.norm(d.qvel[:3]))
    if stop:
        checks["stop_speed"] = final_speed < .01
    return dict(passed=all(checks.values()), checks=checks, simulation_seconds=float(d.time), commanded_walk_seconds=seconds,
                forward_distance_m=forward, average_forward_speed_m_s=forward/seconds,
                max_lateral_drift_m=float(max_side), max_heading_deviation_deg=math.degrees(max_yaw),
                min_upright_cosine=float(min_up), root_height_range_m=[float(min_z), float(max_z)],
                body_ground_contact_samples=body_ground_steps, shin_ground_contact_samples=leg_ground_steps,
                foot_liftoffs=dict(zip(SIDES,map(int,liftoffs))), foot_contact_fraction=dict(zip(SIDES,(contact_samples/samples).tolist())),
                max_foot_tip_height_m=dict(zip(SIDES,max_tip_height.tolist())), max_closure_error_m=max_closure,
                max_actuator_torque_Nm=max_force, max_driven_joint_speed_rad_s=max_speed,
                finite=finite, monotonic_time=monotonic, warning_count=int(d.warning.number.sum()),
                external_force_samples=external_force_steps, friction_override=friction,
                stop_test=stop, final_linear_speed_m_s=final_speed,
                model_sha256=hashlib.sha256((ROOT/'models/scene.xml').read_bytes()).hexdigest(),
                gait_sha256=hashlib.sha256((ROOT/'gait.json').read_bytes()).hexdigest(),
                controller_sha256=hashlib.sha256((ROOT/'gait_controller.py').read_bytes()).hexdigest(),
                wall_seconds=time.perf_counter()-clock, trajectory=records,
                slope_deg=slope_deg, measurement_frame="ground",
                ground_normal_world=ground_normal.tolist(),
                controller_feedback="simulated world pose and angular velocity; not a hardware estimator")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seconds',type=float,default=60)
    parser.add_argument('--sensitivity',action='store_true')
    parser.add_argument('--slope-deg',type=float,default=0.,help='Positive: uphill along initial forward (-Y)')
    args=parser.parse_args()
    (ROOT/'outputs').mkdir(exist_ok=True)
    result={'nominal':validate(args.seconds,slope_deg=args.slope_deg), 'stop':validate(8,stop=True,slope_deg=args.slope_deg)}
    if args.sensitivity:
        result['friction_0.6']=validate(12,friction=.6,slope_deg=args.slope_deg)
        result['friction_1.0']=validate(12,friction=1.,slope_deg=args.slope_deg)
    (ROOT/'outputs'/'walk_validation.json').write_text(json.dumps({k:result[k] for k in ('nominal','stop')},indent=2)+'\n')
    if args.sensitivity:
        (ROOT/'outputs'/'walk_sensitivity.json').write_text(json.dumps({k:v for k,v in result.items() if k.startswith('friction_')},indent=2)+'\n')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='trajectory'} for k,v in result.items()},indent=2))
    raise SystemExit(0 if all(v['passed'] for v in result.values()) else 1)


if __name__=='__main__':
    main()
