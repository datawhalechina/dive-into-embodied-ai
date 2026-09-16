"""MicroDuck laugh gesture task.

The current robot model has a visual jaw but no actuated jaw joint.  This task
therefore teaches a deployable laugh gesture with the existing 14 servos:
small body bounces, a raised head, and a gentle head sway, followed by a clean
return to the standing pose.  It is phase-conditioned so the gesture has a
repeatable beginning and end instead of learning to hold a funny pose forever.
"""

from copy import deepcopy

from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_ground_pick_env_cfg import (
    make_microduck_ground_pick_env_cfg,
    MicroduckGroundPickRlCfg,
)


LAUGH_PERIOD = 3.2

# Explicit HOME/source pose.  Keeping the source pose in the task makes the
# target independent of a future XML default-pose change.
HOME_POSE = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": -0.0873,
    "left_hip_pitch": -0.4579,
    "left_knee": -0.0049,
    "left_ankle": 0.4530,
    "neck_pitch": 0.3491,
    "head_pitch": 0.3491,
    "head_yaw": 0.0,
    "head_roll": 0.0,
    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.0873,
    "right_hip_pitch": 0.4579,
    "right_knee": 0.0049,
    "right_ankle": -0.4530,
}

# A visible two-bob trajectory around HOME.  The knees and ankles provide the
# body bounce; the neck/head joints make the gesture readable in a short
# tutorial video. Values are amplitudes in radians.
LAUGH_AMPLITUDE = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": 0.0,
    "left_hip_pitch": 0.04,
    "left_knee": 0.28,
    "left_ankle": -0.10,
    "neck_pitch": 0.20,
    "head_pitch": 0.40,
    "head_yaw": 0.32,
    "head_roll": -0.18,
    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.0,
    "right_hip_pitch": -0.04,
    "right_knee": -0.28,
    "right_ankle": 0.10,
}


def make_microduck_laugh_env_cfg(play: bool = False, rough: bool = False):
    """Create the phase-conditioned laugh gesture environment."""

    # GroundPick already contains the tested 61D observations, BAM actuator
    # setup, full-collision robot, foot sensors, and sim2real randomization.
    # Replace only the task objective and phase timing.
    cfg = make_microduck_ground_pick_env_cfg(play=play, rough=rough)

    keep_rewards = {
        "upright",
        "body_ang_vel",
        "angular_momentum",
        "soft_landing",
        "feet_grounded",
        "feet_flat",
        "action_rate_l2",
        "neck_action_rate_l2",
        "joint_torques_l2",
        "self_collisions",
    }
    for name in list(cfg.rewards):
        if name not in keep_rewards:
            del cfg.rewards[name]

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 1.5
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["soft_landing"].weight = -1e-5
    cfg.rewards["feet_grounded"].weight = 3.0
    cfg.rewards["feet_flat"].weight = -2.0
    cfg.rewards["action_rate_l2"].weight = -0.8 if not play else -1.0
    cfg.rewards["neck_action_rate_l2"].weight = -0.4
    cfg.rewards["joint_torques_l2"].weight = -5e-3
    cfg.rewards["self_collisions"].weight = -1.0

    pose_params = {
        "command_name": "twist",
        "source_pose": HOME_POSE,
        "amplitude_pose": LAUGH_AMPLITUDE,
        "gesture_start": 0.06,
        "gesture_end": 0.94,
        "asset_cfg": SceneEntityCfg("robot"),
    }
    cfg.rewards["laugh_pose"] = RewardTermCfg(
        func=microduck_mdp.laugh_pose_track,
        weight=12.0,
        params={**pose_params, "std": 0.18},
    )
    cfg.rewards["laugh_pose_l1"] = RewardTermCfg(
        func=microduck_mdp.laugh_pose_track_l1,
        weight=3.0,
        params=pose_params,
    )

    # The source task's phase object is the already tested cyclic command.  A
    # shorter period makes one complete laugh visible in a 3–4 second clip.
    command = cfg.commands["twist"]
    cfg.commands["twist"] = microduck_mdp.GroundPickPhaseCommandCfg(
        **{
            **vars(command),
            "class_type": microduck_mdp.GroundPickPhaseCommand,
            "period": LAUGH_PERIOD,
            # Each episode begins at the standing pose.  Random phase starts
            # are useful for ground-pick's multi-env decorrelation, but they
            # pair a standing reset with a mid-gesture target and make this
            # new gesture learn falls before it learns the motion.
            "randomize_phase": False,
        }
    )

    # A laugh is a self-contained gesture; there is no payload in the mouth
    # and no reason to apply ground-pick's payload event.
    cfg.events.pop("sample_mouth_payload", None)
    cfg.viewer.body_name = "trunk_base"
    return cfg


# Keep the proven PPO architecture and hyperparameters, but give this task its
# own default experiment/run names for checkpoint discovery.
MicroduckLaughRlCfg: RslRlOnPolicyRunnerCfg = deepcopy(MicroduckGroundPickRlCfg)
MicroduckLaughRlCfg.experiment_name = "laugh"
MicroduckLaughRlCfg.run_name = "laugh"
