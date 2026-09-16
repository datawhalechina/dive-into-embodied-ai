"""Opt-in kick recovery rewards. Keep the actor's observation layout unchanged."""
import torch
from mjlab_microduck.tasks import mdp


def reset_ball(env, env_ids, **kwargs):
    mdp.reset_ball_in_front_of_foot(env, env_ids, **kwargs)
    if not hasattr(env, '_completion_ball_start'):
        env._completion_ball_start = torch.zeros(env.num_envs, 2, device=env.device)
    ball = env.scene[kwargs.get('asset_name', 'ball')]
    # Read qpos, since root_link_pos_w is stale inside the reset callback.
    root = env.sim.data.qpos[env_ids][:, ball.indexing.free_joint_q_adr]
    env._completion_ball_start[env_ids] = root[:, :2]


def kick_window_velocity(env, asset_name='ball', max_speed=1.0):
    t = env.episode_length_buf * env.step_dt
    # Smoothly end the rolling reward after the initial strike window.
    window = ((1.5 - t) / 0.5).clamp(0.0, 1.0)
    return mdp.ball_forward_velocity(env, asset_name, max_speed) * window


def kick_recovery(env):
    robot = env.scene['robot']
    q = robot.data.root_link_quat_w
    upright = (1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)).clamp(0, 1)
    height = mdp.height_target_gaussian(env, target_height=0.115, std=0.02)
    pose = mdp.pose_target_match(env, std=0.35)
    # A product requires pose AND height AND uprightness, unlike a sum that
    # permits one good term to pay for a collapsed leg posture.
    stable = torch.exp(-robot.data.root_link_ang_vel_w.square().sum(-1) / 2)
    displacement = ((env.scene['ball'].data.root_link_pos_w[:, :2] -
                     env._completion_ball_start) * mdp._ball_kick_dir(env)).sum(-1)
    progress = (displacement / 0.20).clamp(0, 1)
    phase = ((env.episode_length_buf * env.step_dt - 0.8) / 0.7).clamp(0, 1)
    return torch.nan_to_num(phase * height * upright * pose * stable * (0.25 + 0.75 * progress))


def apply(cfg):
    from mjlab.managers import RewardTermCfg
    cfg.events['reset_ball'].func = reset_ball
    cfg.rewards['ball_forward_velocity'].func = kick_window_velocity
    cfg.rewards['ball_forward_velocity'].weight = 8.0
    cfg.rewards['kick_recovery'] = RewardTermCfg(func=kick_recovery, weight=8.0)


def mouth_height_error(env, asset_cfg, command_name='twist', descent_end=0.25,
                       hold_end=0.35, rise_end=0.60, target_height=0.015):
    asset = env.scene[asset_cfg.name]
    z = asset.data.site_pos_w[:, asset_cfg.site_ids[0], 2] - env.scene.terrain.env_origins[:, 2]
    gate = mdp.phase_pose_blend(mdp._gp_phase(env, command_name), descent_end, hold_end, rise_end)
    return -gate * torch.nan_to_num((z-target_height).abs()/0.20, nan=2.0).clamp(max=2.0)


def apply_ground(cfg):
    from mjlab.managers import RewardTermCfg
    p=dict(cfg.rewards['mouth_ground_proximity'].params)
    p.pop('std', None); p['target_height']=0.015
    # Keep the completion signal subordinate to the original stable reaching
    # objective.  The previous 5.0/4.0/1.0 combination made the policy dive
    # for the mouth target and lose the support polygon before returning.
    cfg.rewards['mouth_height_error']=RewardTermCfg(func=mouth_height_error,weight=1.5,params=p)
    cfg.rewards['mouth_ground_proximity'].weight=3.0
    cfg.rewards['mouth_ground_proximity'].params['target_height']=0.015
    cfg.rewards['neck_action_rate_l2'].weight=-0.15
    cfg.rewards['action_rate_l2'].weight=-0.3
    for stage in cfg.curriculum['action_rate_weight'].params['weight_stages']:
        stage['weight']=max(stage['weight'],-0.3)
