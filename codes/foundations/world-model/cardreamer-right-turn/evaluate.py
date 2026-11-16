"""Bounded public-checkpoint evaluation. HD camera is display-only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import random
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--town', default='Town03')
p.add_argument('--port', type=int, default=22000)
p.add_argument('--seed', type=int, default=2024)
p.add_argument('--episodes', type=int, default=5)
p.add_argument('--max-steps', type=int, default=500)
a = p.parse_args()
sys.path[:0] = [str(a.source), str(a.source / 'dreamerv3')]
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
import numpy as np
import carla
import cv2
import imageio.v2 as imageio
import ruamel.yaml
import embodied
import car_dreamer
import dreamerv3
from dreamerv3.eval import wrap_env
from embodied.envs.from_gym import FromGym

# The upstream monitor starts a non-daemon HTTP server even with display disabled.
# Evaluation does not need it; avoid opening a network port or hanging on errors.
class NoMonitor:
    def __init__(self, config): pass
    def render(self, obs, info): pass
    def stop(self): pass
import car_dreamer.carla_base_env as base_module
base_module.EnvMonitorOpenCV = NoMonitor

random.seed(a.seed)
np.random.seed(a.seed)
a.output.mkdir(parents=True, exist_ok=True)
def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    return repr(value)

configs = ruamel.yaml.YAML(typ='safe').load((a.source/'dreamerv3/dreamerv3.yaml').read_text())
cfg = embodied.Config({'dreamerv3': configs['defaults']}).update({'dreamerv3': configs['small']})
raw, taskcfg = car_dreamer.create_task('carla_right_turn_simple', [
    '--env.world.carla_port', str(a.port), '--env.world.town', a.town, '--env.world.traffic.tm_seed', str(a.seed),
    '--env.display.enable', 'False'])
cfg = cfg.update(taskcfg).dreamerv3.update({'seed': a.seed, 'logdir': str(a.output)})
env = wrap_env(FromGym(raw), cfg)
step = embodied.Counter()
agent = dreamerv3.Agent(env.obs_space, env.act_space, step, cfg)
ckpt = embodied.Checkpoint()
ckpt.agent = agent
ckpt.load(str(a.checkpoint), keys=['agent'])
base = raw.unwrapped
world = base._world._world
write_json(a.output/'metadata.json', {
    'task': 'carla_right_turn_simple', 'town_override': a.town, 'checkpoint_sha256': hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
    'seed': a.seed, 'episodes': a.episodes, 'max_steps': a.max_steps,
    'model_source': 'https://huggingface.co/ucd-dare/CarDreamer',
    'policy_input': 'birdeye_wpt: simulator ground-truth BEV with route and background waypoints',
    'own_training': False, 'task_config': plain(dict(taskcfg)), 'model_config': plain(dict(cfg)),
})
results = []
for episode in range(a.episodes):
    sensor = None
    writer = None
    episode_dir = a.output/f'episode-{episode:02d}'
    episode_dir.mkdir(exist_ok=True)
    try:
        obs, info = env.step({**{k: np.zeros(v.shape, v.dtype) for k,v in env.act_space.items()}, 'reset': True})
        ego = base.get_ego_vehicle()
        frames = queue.Queue()
        bp = world.get_blueprint_library().find('sensor.camera.rgb')
        for k,v in {'image_size_x':'960', 'image_size_y':'540', 'fov':'90', 'sensor_tick':'0.1'}.items():
            bp.set_attribute(k,v)
        sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=-6,z=3), carla.Rotation(pitch=-15)), attach_to=ego)
        sensor.listen(frames.put)
        writer = imageio.get_writer(str(episode_dir/'replay.mp4'), fps=10, codec='libx264', quality=8, macro_block_size=2)
        state = None
        score = 0.0
        observed_collision = False
        rows = []
        bev_frames = {}
        started = time.time()
        for t in range(a.max_steps):
            batched = {k: np.asarray(v)[None] for k,v in obs.items()}
            actions, state = agent.policy(batched, state, mode='eval')
            actions = {k: np.asarray(v)[0] for k,v in actions.items()}
            actions['reset'] = False
            obs, info = env.step(actions)
            step.increment()
            observed_collision |= bool(np.asarray(obs.get('collision', False)).any())
            score += float(obs['reward'])
            target_frame = world.get_snapshot().frame
            speed_vector = ego.get_velocity()
            speed = float((speed_vector.x**2+speed_vector.y**2+speed_vector.z**2)**.5)
            control = ego.get_control()
            bev_frames[target_frame] = obs['birdeye_wpt'].copy()
            row = {'step':t+1,'frame':target_frame,'speed_mps':speed,'steer':control.steer,'throttle':control.throttle,'brake':control.brake,'reward':float(obs['reward']),'info':plain(info)}
            rows.append(row)
            if bool(obs['is_last']):
                break
        # GPU cameras can arrive several ticks late. Match by simulator frame ID
        # after the episode instead of blocking the synchronous world.
        time.sleep(2)
        recorded = {}
        while not frames.empty():
            frame = frames.get_nowait()
            if frame.frame in bev_frames:
                recorded[frame.frame] = np.frombuffer(frame.raw_data, np.uint8).reshape(frame.height,frame.width,4)[:,:,:3][:,:,::-1].copy()
        for row in rows:
            frame_id = row['frame']
            if frame_id not in recorded:
                continue
            canvas = np.zeros((600,1344,3), np.uint8)
            canvas[:540,:960] = recorded[frame_id]
            canvas[50:434,960:] = cv2.resize(bev_frames[frame_id], (384,384), interpolation=cv2.INTER_NEAREST)
            cv2.putText(canvas, 'Display camera | CARLA physics rollout', (12,568), cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(canvas, 'Policy input: truth BEV', (965,30), cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(canvas, f"Episode {episode+1} | {row['step']*.1:.1f}s | {row['speed_mps']:.2f}m/s | steer {row['steer']:+.2f}", (12,592), cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
            writer.append_data(canvas)
        flags = {k: bool(np.asarray(info.get(k,False)).any()) for k in ['destination_reached','is_collision','out_of_lane','time_exceeded']}
        result = {'episode':episode, 'steps':len(rows), 'return':score, 'wall_seconds':time.time()-started, **flags,
                  'recorded_frames':len(recorded), 'missing_camera_frames':len(rows)-len(recorded), 'observed_collision':observed_collision, 'runner_truncated':not bool(obs['is_last']), 'video':f'episode-{episode:02d}/replay.mp4'}
        result['success'] = flags['destination_reached'] and not any(flags[k] for k in ['is_collision','out_of_lane','time_exceeded']) and not result['runner_truncated'] and not observed_collision
        write_json(episode_dir/'steps.json', rows)
        write_json(episode_dir/'result.json', result)
        results.append(result)
        write_json(a.output/'results.json', results)
        print('EPISODE_RESULT', json.dumps(result), flush=True)
    finally:
        if writer is not None: writer.close()
        if sensor is not None:
            sensor.stop()
            sensor.destroy()
print('EVALUATION_COMPLETE', flush=True)
