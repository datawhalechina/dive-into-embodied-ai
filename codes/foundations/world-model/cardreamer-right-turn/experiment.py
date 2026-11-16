"""Finite CarDreamer evaluation with scene audits and action-conditioned forecasts."""
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
p.add_argument('--task', choices=['carla_right_turn_simple','carla_right_turn_medium','carla_right_turn_hard'], default='carla_right_turn_simple')
p.add_argument('--port', type=int, default=22100)
p.add_argument('--episodes', type=int, default=3)
p.add_argument('--seed', type=int, default=2024)
p.add_argument('--max-steps', type=int, default=350)
a = p.parse_args()
sys.path[:0] = [str(a.source),str(a.source/'dreamerv3')]
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import numpy as np
import carla
import cv2
import imageio.v2 as imageio
import ruamel.yaml
import jax
import jax.numpy as jnp
import embodied
import dreamerv3
import car_dreamer
from dreamerv3.eval import wrap_env
from embodied.envs.from_gym import FromGym
from dreamerv3 import ninjax as nj

class NoMonitor:
    def __init__(self, config): pass
    def render(self,*args): pass
    def stop(self): pass
import car_dreamer.carla_base_env as base_module
base_module.EnvMonitorOpenCV=NoMonitor
random.seed(a.seed)
np.random.seed(a.seed)
a.output.mkdir(parents=True,exist_ok=True)
def dump(path,data):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,default=lambda x:x.tolist() if isinstance(x,(np.ndarray,np.generic)) else str(x)))
    temp.replace(path)

client=carla.Client('127.0.0.1',a.port);client.set_timeout(60)
world=client.get_world()
assert world.get_map().name.endswith('/Town03_Opt')
prior=[{'id':x.id,'type':x.type_id} for x in world.get_actors() if x.type_id.startswith(('vehicle.','sensor.'))]
dump(a.output/'scene-before.json',prior)
if prior:
    raise RuntimeError('Dedicated scene must contain no vehicles/sensors before evaluation')
modelcfg=ruamel.yaml.YAML(typ='safe').load((a.source/'dreamerv3/dreamerv3.yaml').read_text())
cfg=embodied.Config({'dreamerv3':modelcfg['defaults']}).update({'dreamerv3':modelcfg['small']})
raw,taskcfg=car_dreamer.create_task(a.task,[
    '--env.world.carla_port',str(a.port),'--env.world.town','Town03_Opt',
    '--env.world.traffic.tm_seed',str(a.seed),'--env.display.enable','False'])
cfg=cfg.update(taskcfg).dreamerv3.update({'seed':a.seed,'logdir':str(a.output)})
env=wrap_env(FromGym(raw),cfg)
base=raw.unwrapped
world=base._world._world
base._world._client.set_timeout(60)
agent=dreamerv3.Agent(env.obs_space,env.act_space,embodied.Counter(),cfg)
ckpt=embodied.Checkpoint();ckpt.agent=agent;ckpt.load(str(a.checkpoint),keys=['agent'])

# Only a filtered latent state and recorded actions enter this function.
# It does not encode any future images. action[t] maps latent[t] to latent[t+1].
def forecast(latent,actions):
    imagined=agent.agent.wm.rssm.imagine(actions,latent)
    return agent.agent.wm.heads['decoder'](imagined)['birdeye_wpt'].mode()
forecast_fn=nj.jit(nj.pure(forecast),device=agent.policy_devices[0])

meta={'task':a.task,'town':'Town03_Opt','seed':a.seed,'planned_episodes':a.episodes,
      'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
      'training':False,'frame_seconds':.1,'task_configuration':dict(taskcfg),
      'prediction':'20-step open-loop latent rollout conditioned on recorded actions; no future image input'}
dump(a.output/'metadata.json',meta)
results=[]
try:
    for episode in range(a.episodes):
        folder=a.output/f'episode-{episode:02d}';folder.mkdir(exist_ok=True)
        sensor=None;collision_sensor=None;writer=None
        rows=[];bevs=[];actions_record=[];latents=[];observations=[];collisions=[]
        frames=queue.Queue()
        started=time.monotonic()
        try:
            obs,_=env.step({**{k:np.zeros(v.shape,v.dtype) for k,v in env.act_space.items()},'reset':True})
            ego=base.get_ego_vehicle()
            bp=world.get_blueprint_library().find('sensor.camera.rgb')
            for key,value in {'image_size_x':'960','image_size_y':'540','fov':'90','sensor_tick':'0.1'}.items():bp.set_attribute(key,value)
            sensor=world.spawn_actor(bp,carla.Transform(carla.Location(x=-6,z=3),carla.Rotation(pitch=-15)),attach_to=ego)
            sensor.listen(frames.put)
            collision_bp=world.get_blueprint_library().find('sensor.other.collision')
            collision_sensor=world.spawn_actor(collision_bp,carla.Transform(),attach_to=ego)
            def collision_callback(event):
                impulse=event.normal_impulse
                collisions.append({'frame':event.frame,'other_id':event.other_actor.id,'other_type':event.other_actor.type_id,
                                   'impulse':float((impulse.x**2+impulse.y**2+impulse.z**2)**.5)})
            collision_sensor.listen(collision_callback)
            dump(folder/'scene-start.json',[{'id':x.id,'type':x.type_id} for x in world.get_actors() if x.type_id.startswith(('vehicle.','sensor.'))])
            state=None;total=0.;observed_collision=False
            for step in range(a.max_steps):
                bevs.append(obs['birdeye_wpt'].copy())
                observations.append({k:np.asarray(v).copy() for k,v in obs.items()})
                policy,state=agent.policy({k:np.asarray(v)[None] for k,v in obs.items()},state,mode='eval')
                action={k:np.asarray(v)[0] for k,v in policy.items()};action['reset']=False
                actions_record.append(action['action'].copy())
                latents.append(jax.tree_util.tree_map(lambda x:np.asarray(x).copy(),state[0][0]))
                obs,info=env.step(action)
                frame=world.get_snapshot().frame
                v=ego.get_velocity();speed=float((v.x*v.x+v.y*v.y+v.z*v.z)**.5)
                loc=ego.get_location();other_dist=[]
                for other in world.get_actors().filter('vehicle.*'):
                    if other.id!=ego.id:
                        other_dist.append({'id':other.id,'type':other.type_id,'center_distance_m':float(loc.distance(other.get_location()))})
                other_dist.sort(key=lambda x:x['center_distance_m'])
                control=ego.get_control();total+=float(obs['reward'])
                observed_collision |= bool(np.asarray(obs['collision']).any())
                rows.append({'step':step+1,'frame':frame,'speed_mps':speed,'steer':control.steer,
                    'throttle':control.throttle,'brake':control.brake,'reward':float(obs['reward']),
                    'ego_x':loc.x,'ego_y':loc.y,'other_vehicles':other_dist,'info':info,
                    'observed_bev_index':step+1})
                if step%20==0:dump(folder/'steps-partial.json',rows)
                if bool(obs['is_last']):break
            bevs.append(obs['birdeye_wpt'].copy())
            np.savez_compressed(folder/'trajectory.npz',bev=np.stack(bevs),actions=np.stack(actions_record),
                                frames=np.array([x['frame'] for x in rows],np.int64))
            dump(folder/'steps.json',rows)
            time.sleep(1)
            dump(folder/'collisions.json',collisions)
            recorded={}
            while not frames.empty():
                im=frames.get_nowait()
                recorded[im.frame]=np.frombuffer(im.raw_data,np.uint8).reshape(im.height,im.width,4)[:,:,:3][:,:,::-1].copy()
            writer=imageio.get_writer(str(folder/'replay.mp4'),fps=10,codec='libx264',quality=6,macro_block_size=2)
            matched=0
            for index,row in enumerate(rows):
                if row['frame'] not in recorded:continue
                matched+=1
                canvas=np.zeros((600,1344,3),np.uint8);canvas[:540,:960]=recorded[row['frame']]
                canvas[50:434,960:]=cv2.resize(bevs[index+1],(384,384),interpolation=cv2.INTER_NEAREST)
                title=f"{a.task.replace('carla_right_turn_','')} | episode {episode+1} | {row['step']*.1:.1f}s | {row['speed_mps']:.2f}m/s"
                cv2.putText(canvas,title,(12,568),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
                cv2.putText(canvas,'Real simulator BEV; not a model prediction',(12,592),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
                writer.append_data(canvas)
            writer.close();writer=None
            flags={k:bool(np.asarray(info.get(k,False)).any()) for k in ['destination_reached','is_collision','out_of_lane','time_exceeded']}
            waiting=sum(x['speed_mps']<.2 for x in rows)*.1
            distances=[v['center_distance_m'] for x in rows for v in x['other_vehicles']]
            result={'episode':episode,'steps':len(rows),'return':total,**flags,'sensor_collisions':len(collisions),
                'observed_collision':observed_collision,'runner_truncated':not bool(obs['is_last']),
                'waiting_seconds_below_0.2mps':waiting,'minimum_vehicle_center_distance_m':min(distances) if distances else None,
                'recorded_frames':matched,'missing_camera_frames':len(rows)-matched,'wall_seconds':time.monotonic()-started}
            result['success']=flags['destination_reached'] and not any(flags[k] for k in ['is_collision','out_of_lane','time_exceeded']) and not observed_collision and not collisions and not result['runner_truncated']
            dump(folder/'result.json',result);results.append(result);dump(a.output/'results.json',results)
            print('EPISODE_RESULT',json.dumps(result),flush=True)
            # Keep forecast windows fixed in advance; skip windows beyond termination.
            prediction_stats=[]
            for anchor in [20,60,100]:
                horizon=20
                if anchor+horizon>len(actions_record):continue
                latent=jax.device_put(latents[anchor],agent.policy_devices[0])
                seq=jax.device_put(np.stack(actions_record[anchor:anchor+horizon])[None],agent.policy_devices[0])
                rng=jax.device_put(np.array([0,42+anchor],np.uint32),agent.policy_devices[0])
                pred,_=forecast_fn(agent.varibs,rng,latent,seq)
                prediction=np.asarray(jax.device_get(pred))[0].astype(np.float32)
                truth=np.stack(bevs[anchor+1:anchor+horizon+1]).astype(np.float32)/255
                hold=np.repeat(bevs[anchor][None],horizon,axis=0).astype(np.float32)/255
                assert prediction.shape==truth.shape and np.isfinite(prediction).all()
                np.savez_compressed(folder/f'forecast-{anchor:03d}.npz',prediction=prediction,truth=truth,
                    context=bevs[anchor],actions=np.stack(actions_record[anchor:anchor+horizon]))
                stats={'anchor_observation':anchor,'context_seconds':anchor*.1,'horizon_steps':horizon,
                    'normalized_mae':[float(np.abs(prediction[t]-truth[t]).mean()) for t in range(horizon)],
                    'hold_frame_mae':[float(np.abs(hold[t]-truth[t]).mean()) for t in range(horizon)]}
                prediction_stats.append(stats)
            dump(folder/'forecasts.json',prediction_stats)
            print('FORECAST_COMPLETE',episode,flush=True)
        except Exception as error:
            dump(folder/'error.json',{'type':type(error).__name__,'message':str(error),'completed_steps':len(rows)})
            raise
        finally:
            if writer is not None:writer.close()
            for item in [sensor,collision_sensor]:
                if item is not None:
                    try:item.stop();item.destroy()
                    except RuntimeError:pass
            # Destroy only actors owned by this task, then audit for leaked vehicles.
            base._observer.destroy()
            for handler in base._observer._data_handlers.values():
                if hasattr(handler,'_sensor'):handler._sensor=None
            owned=list(base._world.actor_dict)
            base._world._client.apply_batch_sync([carla.command.DestroyActor(i) for i in owned])
            base._world.actor_dict={}
            world.tick()  # Apply queued destruction before auditing the snapshot.
            left=[{'id':x.id,'type':x.type_id} for x in world.get_actors() if x.type_id.startswith(('vehicle.','sensor.'))]
            dump(folder/'scene-after-cleanup.json',left)
            if left:raise RuntimeError('Actor cleanup audit failed; stop to avoid scene contamination')
finally:
    try:
        base._world._vehicle_manager.set_synchronous_mode(False)
        settings=world.get_settings();settings.synchronous_mode=False;world.apply_settings(settings)
    except RuntimeError:pass
print('EXPERIMENT_COMPLETE',flush=True)
