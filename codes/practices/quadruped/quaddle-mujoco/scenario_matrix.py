"""Run the same controller in explicit small terrain tasks; save failures too."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
import hashlib
from html import escape
import json
import math
from pathlib import Path
import platform
import time

import mujoco
import numpy as np

from gait_controller import CONFIG, JointController
from run_sim import ROOT, SIDES, camera_and_options, closure_error
from scenarios import SCENES, ShallowWater, load_scene, make_scene, plan_path


def profile(mode):
    config = copy.deepcopy(CONFIG)
    if mode == 'run_attempt':
        config = json.loads((ROOT/'reference/run-candidate.json').read_text())
    elif mode != 'walk':
        raise ValueError(f'Unknown profile: {mode}')
    return config


class Recorder:
    def __init__(self, model, data, scene, mode, output, route):
        import imageio.v2 as imageio
        from PIL import ImageFont
        self.renderer = mujoco.Renderer(model, height=720, width=960)
        self.writer = imageio.get_writer(output/'video.mp4', fps=25, codec='libx264', quality=7,
            macro_block_size=16, ffmpeg_params=['-threads', '1'])
        self.cam, self.opt = camera_and_options(data)
        self.opt.geomgroup[4] = 1  # Physical obstacles, hidden by MuJoCo's default.
        self.opt.geomgroup[5] = 1  # Non-colliding water surface.
        self.cam.distance = .46
        self.controller_label = "Periodic controller"
        self.scene, self.mode, self.output, self.route = scene, mode, output, route
        font = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
        self.font = ImageFont.truetype(str(font), 20) if font.exists() else ImageFont.load_default(size=20)
        self.small = ImageFont.truetype(str(font), 15) if font.exists() else ImageFont.load_default(size=15)

    def frame(self, data, rotation, forward, phase, diagnostics=None):
        from PIL import Image, ImageDraw
        self.cam.lookat[:] = data.body('body').xpos
        self.renderer.update_scene(data, camera=self.cam, scene_option=self.opt)
        image = Image.fromarray(self.renderer.render())
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 960, 82), fill=(14, 21, 33))
        draw.text((20, 12), f'QUADDLE / {self.scene.label}', font=self.font, fill='white')
        draw.text((20, 45), f'{self.mode} | {self.controller_label} | Uncalibrated simulation', font=self.small, fill=(180, 200, 220))
        if diagnostics:
            speed_peak,yaw_peak=diagnostics
            colour=(255,160,125) if speed_peak>10. or yaw_peak>=25. else (160,220,180)
            draw.text((20,64), f'Joint peak {speed_peak:.2f}/10 rad/s | Yaw peak {yaw_peak:.1f}/25 deg',font=self.small,fill=colour)
        rot = rotation.T @ data.body('body').xmat.reshape(3, 3)
        pitch = math.degrees(math.atan2(-rot[2, 1], np.linalg.norm(rot[:2, 1])))
        draw.rectangle((0, 660, 960, 720), fill=(14, 21, 33))
        draw.text((20, 669), f't={data.time:.2f}s | Progress={forward:.2f}m | Pitch/reference={pitch:+.1f}deg | {phase}', font=self.small, fill=(235, 195, 125))
        note = 'RUN CANDIDATE: speed alone does not establish running' if self.mode=='run_attempt' else 'Full startup retained | Green line = task goal'
        if self.scene.water_depth_m:
            note = 'WATER APPROXIMATION: buoyancy + drag on assumed spheres; no CFD or waterproofing'
        draw.text((20, 694), note, font=self.small, fill=(180, 200, 220))
        # A map inset exposes actual obstacle locations, terrain extent and progress.
        draw.rectangle((715, 90, 950, 330), fill=(20, 30, 42))
        def point(x, y):
            return (832+x*320, 316+y*150)
        for box in self.scene.boxes:
            colour = (185, 75, 50) if box['obstacle'] else (85, 100, 115)
            s0, s1 = max(-.05, box['s0']), min(1.25, box['s1'])
            if s1>s0:
                draw.rectangle((*point(box['xmin'], -s1), *point(box['xmax'], -s0)), fill=colour)
        if self.scene.water_depth_m:
            draw.rectangle((*point(-.22, -1.05), *point(.22, -.15)), fill=(25, 85, 140))
        if self.route:
            draw.line([point(x, y) for x, y in self.route], fill=(250, 205, 80), width=2)
        draw.line([point(-.27, -self.scene.goal_m), point(.27, -self.scene.goal_m)], fill=(60, 210, 130), width=3)
        x, y = point(data.qpos[0], data.qpos[1])
        x, y = float(np.clip(x, 720, 945)), float(np.clip(y, 123, 325))
        draw.ellipse((x-4, y-4, x+4, y+4), fill='white')
        draw.text((724, 97), 'Map / simulation pose', font=self.small, fill='white')
        self.writer.append_data(np.asarray(image))
        if data.time < .05:
            image.save(self.output/'preview.webp')

    def close(self):
        self.writer.close()
        self.renderer.close()


def run_case(name, mode, seconds=20., output=None, video=False, policy_path=None, config_override=None, initial_x_offset=0., scene_definition=None, initial_yaw_deg=0., friction_scale=1.):
    if not math.isfinite(seconds) or seconds < 6:
        raise ValueError('seconds must be finite and at least 6')
    scene = scene_definition if scene_definition is not None else make_scene(name)
    model, data, rotation = load_scene(scene)
    if not math.isfinite(initial_x_offset) or abs(initial_x_offset) > .02:
        raise ValueError('initial_x_offset must be within +/-2 cm')
    if not math.isfinite(initial_yaw_deg) or abs(initial_yaw_deg)>10.:
        raise ValueError('initial yaw must be within +/-10 degrees')
    if not math.isfinite(friction_scale) or not .5<=friction_scale<=1.5:
        raise ValueError('friction scale must be within [.5,1.5]')
    if friction_scale != 1.:
        model.geom_friction[:,0] *= friction_scale
    if initial_x_offset:
        data.qpos[0] += initial_x_offset
    if initial_yaw_deg:
        angle=math.radians(initial_yaw_deg)/2
        quat=np.empty(4)
        mujoco.mju_mulQuat(quat,np.array([math.cos(angle),0.,0.,math.sin(angle)]),data.qpos[3:7].copy())
        data.qpos[3:7]=quat
    if initial_x_offset or initial_yaw_deg or friction_scale != 1.:
        mujoco.mj_forward(model, data)
    config = profile(mode)
    if config_override:
        config.update(config_override)
    route = plan_path(scene) if name == 'avoid' else []
    waypoint = 1
    if route:
        config.update(heading_target_rad=0., lateral_kp=0., heading_kp=.8,
                      steering_limit=.45, max_target_speed_rad_s=5.)
    controller = JointController('walk', config=config)
    if policy_path:
        from terrain_rl import PolicyController
        policy_meta = json.loads((Path(policy_path)/'metadata.json').read_text())
        config.update(policy_meta.get('controller_config', {}))
        if config_override:
            config.update(config_override)
        if policy_meta.get('environment_version') == 2:
            from curriculum_rl import CurriculumPolicyController as PolicyController
        controller = PolicyController(model, scene, rotation, config, policy_path)
    fluid = ShallowWater(model, scene) if scene.water_depth_m else None
    output = Path(output or ROOT/'outputs/scenario_matrix')/mode/name
    output.mkdir(parents=True, exist_ok=True)
    recorder = Recorder(model, data, scene, mode, output, route) if video else None
    if recorder and policy_path:
        recorder.controller_label = 'Residual PPO'
    support_geoms = {i for i in range(model.ngeom) if model.geom_bodyid[i]==0 and model.geom_contype[i]!=0}
    obstacle_geoms = {i for i in support_geoms if model.geom_group[i]==4}
    body_geom = model.geom('body_contact').id
    foot_geoms = [{i for i in range(model.ngeom) if (model.geom(i).name or '').startswith(s+'_wheel') and '_contact_' in (model.geom(i).name or '')} for s in SIDES]
    driven = [model.jnt_dofadr[model.joint(s+'_thigh').id] for s in SIDES]
    axis = rotation@np.array([0., -1., 0.])
    initial = data.qpos[:3].copy(); start = None
    maximum_pitch = 0.; min_up = 1.; min_clearance = float('inf')
    max_yaw = 0.; max_torque = 0.; max_joint_speed = 0.; max_closure = 0.
    body_contacts = obstacle_contacts = external_steps = water_body_samples = 0
    foot_counts = np.zeros(4, dtype=int); contact_samples = 0
    flight_duration = 0.; flight_durations = []; longest_flight = 0.
    finite = True; records = []; complete = True
    first_failure = {}
    clock = time.perf_counter()
    try:
        for step in range(round((config['settle_seconds']+seconds)/model.opt.timestep)):
            if route and step % 50 == 0:
                position = data.qpos[:2]
                while waypoint < len(route)-1 and np.linalg.norm(position-np.array(route[waypoint])) < .07:
                    waypoint += 1
                delta = np.array(route[waypoint])-position
                config['heading_target_rad'] = math.atan2(delta[0], -delta[1])
            controller(data)
            if fluid:
                fluid.apply(data)
                external_steps += int(np.any(data.xfrc_applied))
            elif np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
                external_steps += 1
            mujoco.mj_step(model, data)
            finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
            if not finite:
                complete = False
                break
            max_torque = max(max_torque, float(np.max(np.abs(data.actuator_force))))
            max_joint_speed = max(max_joint_speed, float(np.max(np.abs(data.qvel[driven]))))
            if max_joint_speed > 10. and 'joint_speed' not in first_failure:
                first_failure['joint_speed'] = dict(time_s=float(data.time), peak_rad_s=max_joint_speed, position_m=data.qpos[:3].tolist())
            contacts = [(c.geom1, c.geom2) for c in data.contact if (c.geom1 in support_geoms) != (c.geom2 in support_geoms)]
            # Count flight only after settling/ramp, and include shin/body contacts.
            if data.time >= config['settle_seconds']+config['ramp_seconds']+.5:
                if not contacts:
                    flight_duration += model.opt.timestep
                elif flight_duration:
                    if flight_duration >= .01-1e-9:
                        flight_durations.append(flight_duration)
                    longest_flight = max(longest_flight, flight_duration); flight_duration = 0.
            if step % 5 == 0:
                mujoco.mj_forward(model, data)
                max_closure = max(max_closure, closure_error(data))
                rot = rotation.T @ data.body('body').xmat.reshape(3, 3)
                pitch = math.degrees(math.atan2(-rot[2, 1], np.linalg.norm(rot[:2, 1])))
                min_up = min(min_up, float(rot[2, 2]))
                body_contacts += int(any(body_geom in pair for pair in contacts))
                obstacle_contacts += int(any(set(pair)&obstacle_geoms for pair in contacts))
                for issue, bad in [('upright', min_up <= .8), ('body_contact', body_contacts > 0)]:
                    if bad and issue not in first_failure:
                        first_failure[issue] = dict(time_s=float(data.time), position_m=data.qpos[:3].tolist())
                if fluid and -.22 <= data.geom_xpos[body_geom, 0] <= .22 and .15 <= -data.geom_xpos[body_geom, 1] <= 1.05:
                    body_rotation = data.geom_xmat[body_geom].reshape(3, 3)
                    bottom = data.geom_xpos[body_geom, 2]-np.abs(body_rotation[2])@model.geom_size[body_geom]
                    water_body_samples += int(bottom < scene.water_depth_m)
                if data.time < config['settle_seconds']+config['ramp_seconds']:
                    maximum_pitch = max(maximum_pitch, abs(pitch))
                if data.time >= config['settle_seconds']:
                    if start is None:
                        start = data.qpos[:3].copy()
                    min_clearance = min(min_clearance, float(data.qpos[2]-scene.surface_height(data.qpos[0], data.qpos[1])))
                    max_yaw = max(max_yaw, abs(math.degrees(math.atan2(rot[1, 0], rot[0, 0]))))
                    for issue, bad in [('heading', max_yaw >= 25.)]:
                        if bad and issue not in first_failure:
                            first_failure[issue] = dict(time_s=float(data.time), position_m=data.qpos[:3].tolist())
                    contact_samples += 1
                    foot_counts += [any(set(pair)&ids for pair in contacts) for ids in foot_geoms]
            forward = float((data.qpos[:3]-(start if start is not None else initial))@axis)
            if step % 100 == 0:
                records.append(dict(t_s=float(data.time), x_m=float(data.qpos[0]), y_m=float(data.qpos[1]),
                    z_m=float(data.qpos[2]), forward_m=forward, pitch_reference_deg=pitch,
                    terrain_contact_count=len(contacts), heading_target_rad=config.get('heading_target_rad', 0.)))
            if recorder and step % 40 == 0:
                # Share the existing 5 ms forward pass; extra solver calls for
                # video can change warm-start state and alter the experiment.
                phase = 'support' if data.time<2 else ('transition' if data.time<3.5 else 'locomotion')
                recorder.frame(data, rotation, forward, phase, (max_joint_speed,max_yaw))
    finally:
        if recorder:
            recorder.close()
    # An unfinished aerial interval at the end is not a completed flight.
    longest_flight = max(longest_flight, flight_duration)
    distance = float((data.qpos[:3]-(start if start is not None else initial))@axis)
    goal_distance = float(np.linalg.norm(data.qpos[:2]-np.array([0., -scene.goal_m])))
    speed = distance/seconds
    goal_reached = goal_distance < .1 if route else distance >= scene.goal_m
    checks = dict(complete=complete, finite=finite, no_warnings=bool(data.warning.number.sum()==0),
        upright=min_up>.8, body_clearance=min_clearance>.025, no_body_contact=body_contacts==0,
        no_obstacle_contact=obstacle_contacts==0, startup_pitch=maximum_pitch<15.,
        loop_closure=max_closure<.002, torque_limit=max_torque<=.245001, joint_speed=max_joint_speed<=10.,
        forward_speed=speed>.03, task_goal=bool(goal_reached),
        heading=True if route else max_yaw<25., force_accounting=external_steps==0)
    if fluid:
        checks['entered_water'] = fluid.wet_steps > 0
        checks['drag_dissipative'] = fluid.max_drag_power <= 1e-12
        checks['body_above_water'] = water_body_samples == 0
    if mode=='run_attempt':
        # This project uses a conservative aerial-running criterion, not a universal gait definition.
        checks['running_speed'] = speed >= .1
        checks['repeated_flight'] = len(flight_durations) >= 3
    result = dict(scene=scene.metadata(), mode=mode, policy_path=str(policy_path) if policy_path else None,
        initial_x_offset_m=initial_x_offset, initial_yaw_deg=initial_yaw_deg, friction_scale=friction_scale, first_failure=first_failure,
        policy_sha256={name:hashlib.sha256((Path(policy_path)/name).read_bytes()).hexdigest() for name in
                       ['policy.zip', 'obs_stats.npz', 'metadata.json']} if policy_path else None, passed=all(checks.values()), checks=checks,
        simulation_seconds=float(data.time), commanded_seconds=seconds, forward_m=distance,
        average_forward_speed_m_s=speed, final_position_m=data.qpos[:3].tolist(), goal_distance_xy_m=goal_distance,
        startup_max_abs_pitch_deg=maximum_pitch, min_upright_cosine=min_up, min_root_clearance_m=min_clearance,
        max_heading_deviation_deg=max_yaw, body_contact_samples=body_contacts, obstacle_contact_samples=obstacle_contacts,
        max_torque_Nm=max_torque, max_driven_joint_speed_rad_s=max_joint_speed, max_closure_error_m=max_closure,
        foot_contact_fraction=dict(zip(SIDES, (foot_counts/max(1, contact_samples)).tolist())),
        completed_flight_intervals_s=flight_durations, longest_flight_s=longest_flight,
        controller_config=config, route_xy_m=route, localization='simulator ground truth',
        navigation='known-map inflated-obstacle A* and heading feedback' if route else None,
        wall_seconds=time.perf_counter()-clock, video='video.mp4' if video else None,
        fluid=dict(wet_steps=fluid.wet_steps, max_sample_force_N=fluid.max_force, max_drag_power_W=fluid.max_drag_power,
                   drag_energy_dissipated_J=fluid.drag_energy, body_water_samples=water_body_samples) if fluid else None)
    (output/'metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    with (output/'trajectory.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys()); writer.writeheader(); writer.writerows(records)
    return result


def write_gallery(output, report):
    labels = dict(flat='平地', uphill3='3°上坡', uphill5='5°上坡', downhill3='3°下坡',
        stairs_up='上台阶：4毫米×3级', stairs_down='下台阶：4毫米×3级',
        rough='坑洼起伏：±3毫米', avoid='已知地图绕障', water='8毫米浅水近似')
    cards = []
    for result in report['results']:
        name, mode = result['scene']['name'], result['mode']
        path = f'{mode}/{name}'
        failed = ', '.join(key for key, value in result['checks'].items() if not value) or '全部通过'
        status = '通过' if result['passed'] else '未通过'
        kind = '行走' if mode=='walk' else '跑步候选'
        media = f'<video controls playsinline preload="none" poster="{path}/preview.webp"><source src="{path}/video.mp4" type="video/mp4"></video>' if result['video'] else '<p>本组未录制视频。</p>'
        cards.append(f'''<article data-mode="{mode}"><h2>{labels[name]} · {kind}</h2>
<p class="{'pass' if result['passed'] else 'fail'}">{status} · 前进{result['forward_m']:.2f}米 · 均速{result['average_forward_speed_m_s']:.3f}米/秒</p>
{media}<details><summary>检查与数据</summary><p>{escape(failed)}</p>
<p>完整腾空段数：{len(result['completed_flight_intervals_s'])}；最大驱动关节速度：{result['max_driven_joint_speed_rad_s']:.2f} rad/s。</p>
<a href="{path}/metrics.json">指标JSON</a> · <a href="{path}/trajectory.csv">轨迹CSV</a></details></article>''')
    passed = sum(result['passed'] for result in report['results'])
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Quaddle 多场景仿真实验</title><style>
body{margin:auto;max-width:1400px;padding:24px;background:#101923;color:#e4ecf5;font:16px/1.65 system-ui,sans-serif}h1{font-size:30px}h2{font-size:20px;margin:0}a{color:#8cceff}.note{max-width:1000px;color:#c1cfdd}nav{display:flex;gap:12px;margin:24px 0}button{padding:10px 18px;border:1px solid #607080;border-radius:7px;background:#23374a;color:white;cursor:pointer}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:24px}article{background:#1b2a3a;padding:18px;border-radius:12px;min-width:0}video{width:100%;border-radius:6px}.pass{color:#8ce3ae}.fail{color:#ffd092}details{overflow-wrap:anywhere}article[hidden]{display:none}
</style><h1>Quaddle：同一模型，换个环境会怎样？</h1>
<p class="note">真实MuJoCo回放，完整保留启动；控制方式见视频标签和指标JSON。上下台阶均为4毫米高、12厘米深，共3级；坑洼起伏为±3毫米。绕障使用已知地图A*规划和仿真真值定位。浅水使用未标定的等效球浮力、阻力与湿地摩擦近似，无流体求解器。</p>
<p class="note">跑步候选额外要求：至少3个持续10毫秒的完整腾空段、均速≥0.1米/秒，并通过其他任务检查。提高频率本身不计为会跑。每条件只有一次确定性运行，尚无真机验证。</p>'''
    page += f'<p>{len(report["results"])}组任务中，{passed}组通过当前检查。失败条件与完整数据均保留。</p>'
    page += '<nav><button data-filter="all">全部</button><button data-filter="walk">行走</button><button data-filter="run_attempt">跑步候选</button></nav><main class="grid">'
    page += '\n'.join(cards) + '''</main><script>
document.querySelectorAll('button[data-filter]').forEach(button=>button.onclick=()=>{
document.querySelectorAll('article').forEach(card=>{card.hidden=button.dataset.filter!=='all'&&card.dataset.mode!==button.dataset.filter;if(card.hidden)card.querySelector('video')?.pause();});});
</script></html>'''
    (output/'index.html').write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=['all', *SCENES], default='all')
    parser.add_argument('--mode', choices=['both', 'walk', 'run_attempt'], default='both')
    parser.add_argument('--seconds', type=float, default=20.)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/scenario_matrix')
    parser.add_argument('--policy', type=Path, help='Terrain PPO directory with policy and matching normalization')
    parser.add_argument('--config', type=Path, help='Explicit gait JSON; keeps task thresholds unchanged')
    parser.add_argument('--video', action='store_true', help='Linux headless: set MUJOCO_GL=egl before launch')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 4:
        parser.error('--workers must be between 1 and 4')
    names = SCENES if args.scene=='all' else [args.scene]
    modes = ['walk', 'run_attempt'] if args.mode=='both' else [args.mode]
    results = []
    config_override = json.loads(args.config.read_text()) if args.config else None
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run_case, name, mode, args.seconds, args.output, args.video, args.policy, config_override) for mode in modes for name in names]
        for future in as_completed(futures):
            result = future.result(); results.append(result)
            print(result['mode'], result['scene']['name'], 'PASS' if result['passed'] else 'FAIL',
                f"{result['forward_m']:.3f}m", [key for key, value in result['checks'].items() if not value], flush=True)
    results.sort(key=lambda result:(result['mode'], result['scene']['name']))
    hashes = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
              ['models/scene.xml', 'gait.json', 'reference/run-candidate.json', 'gait_controller.py', 'scenarios.py', 'scenario_matrix.py']}
    report = dict(python=platform.python_version(), mujoco=mujoco.__version__, hardware_tested=False,
        learned_policy=bool(args.policy), trials_per_condition=1, running_definition='>=3 completed no-terrain-contact intervals of >=10ms, speed>=0.1m/s, plus all task checks',
        source_sha256=hashes, results=results)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    write_gallery(args.output, report)
    print('Saved', args.output/'summary.json', flush=True)
    # Failed physical tasks are results, not runner crashes; record them explicitly.


if __name__ == '__main__':
    main()
