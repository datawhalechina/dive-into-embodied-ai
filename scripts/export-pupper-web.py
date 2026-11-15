"""Export the archived Pupper v3 PPO actor without requiring GPU training.

Run with codes/practices/quadruped/cs123/.venv/bin/python after installing
onnx and onnxruntime in that environment. Website builds use the saved export.
"""
from pathlib import Path
import hashlib
import importlib.util
import json

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort
import torch
from stable_baselines3 import PPO
import mujoco

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / 'codes/practices/quadruped/cs123/6.rl_pupper_v2_5'
CHECKPOINT = LAB / 'reports/raw/checkpoints/pupper_ppo_8000000_steps.zip'
OUT = LAB / 'web'
OUT.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location('pupper_env', LAB / 'pupper_env.py')
env_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(env_module)
policy = PPO.load(CHECKPOINT, device='cpu')
assert policy.observation_space.shape == (53,)
assert not policy.policy.squash_output

nodes, tensors = [], []
x = 'obs'
layers = list(policy.policy.mlp_extractor.policy_net) + [policy.policy.action_net]
for i, layer in enumerate(layers):
    y = f'layer_{i}'
    if isinstance(layer, torch.nn.Linear):
        tensors += [numpy_helper.from_array(layer.weight.detach().numpy(), f'w{i}'), numpy_helper.from_array(layer.bias.detach().numpy(), f'b{i}')]
        nodes.append(helper.make_node('Gemm', [x, f'w{i}', f'b{i}'], [y], transB=1))
    elif isinstance(layer, torch.nn.Tanh):
        nodes.append(helper.make_node('Tanh', [x], [y]))
    else:
        raise ValueError(f'Unsupported actor layer: {layer}')
    x = y
tensors += [numpy_helper.from_array(np.array(-1, np.float32), 'low'), numpy_helper.from_array(np.array(1, np.float32), 'high')]
nodes.append(helper.make_node('Clip', [x, 'low', 'high'], ['action']))
graph = helper.make_graph(nodes, 'pupper_v3_ppo_v2_5', [helper.make_tensor_value_info('obs', TensorProto.FLOAT, [1, 53])], [helper.make_tensor_value_info('action', TensorProto.FLOAT, [1, 12])], tensors)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=10)
onnx.checker.check_model(model)
onnx.save(model, OUT / 'policy.onnx')
session = ort.InferenceSession(str(OUT / 'policy.onnx'), providers=['CPUExecutionProvider'])
rng = np.random.default_rng(42)
samples = rng.normal(0, 1, (256, 53)).astype(np.float32)
expected = policy.predict(samples, deterministic=True)[0]
actual = np.concatenate([session.run(None, {'obs': row[None]})[0] for row in samples])
error = float(np.max(np.abs(expected - actual)))
assert error < 1e-5, error

config = dict(joints=env_module.JOINT_NAMES, defaults=env_module.DEFAULT_POSE.tolist(), lowers=env_module.JOINT_LOWERS.tolist(), uppers=env_module.JOINT_UPPERS.tolist(), kp=env_module.KP, kd=env_module.KD, physicsDt=env_module.DT_PHYSICS, stepDt=env_module.DT_CONTROL, actionScale=env_module.ACTION_SCALE, alpha=env_module.ACTION_FILTER_ALPHA, height=env_module.INIT_HEIGHT, period=env_module.GAIT_SPECS['trot']['cycle_time'])
(OUT / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
env = env_module.PupperEnv(perturb_enabled=False, gait_enabled=True, gait_types=('trot',), gait_switch_steps=0, cmd_resample_steps=0)
fixtures = []
for yaw in [0., .7, -.4]:
    env.reset(seed=42)
    env.data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    env.data.qvel[:] = rng.normal(0, .1, env.model.nv)
    env.gait_phase = 0.
    env.cmd = np.array([.4, .1, .2], np.float32)
    mujoco.mj_forward(env.model, env.data)
    obs = env._get_obs()
    fixtures.append(dict(qpos=env.data.qpos.tolist(), qvel=env.data.qvel.tolist(), command=env.cmd.tolist(), observation=obs.tolist(), action=policy.predict(obs, deterministic=True)[0].tolist()))
(OUT / 'reference.json').write_text(json.dumps(fixtures, indent=2) + '\n')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
(OUT / 'provenance.json').write_text(json.dumps(dict(checkpoint=str(CHECKPOINT.relative_to(ROOT)), checkpointSha256=sha(CHECKPOINT), environmentSha256=sha(LAB / 'pupper_env.py'), policySha256=sha(OUT / 'policy.onnx'), maxActionError=error, verificationSamples=256, onnx=onnx.__version__, onnxruntime=ort.__version__, torch=torch.__version__, observationSize=53, gait='trot', note='Deterministic clipped SB3 actor; no observation normalization is used by this training pipeline.'), indent=2) + '\n')
print(json.dumps({'export': str(OUT), 'maxActionError': error, 'samples': 256}))
