"""Full laugh choreography for the arm-equipped MicroDuck model.

The legacy laugh task stays on the original 14-servo robot.  This task uses a
separate 18-servo XML with procedural arms, allowing the policy to learn the
visible sequence requested for the tutorial: belly-hug and forward laugh,
throw the body back, then alternate left/right palm and foot taps.
"""

from copy import deepcopy

from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_LAUGH_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_laugh_env_cfg import (
    make_microduck_laugh_env_cfg,
    MicroduckLaughRlCfg,
)


LAUGH_CHOREO_PERIOD = 5.0


def _pose(**values: float) -> dict[str, float]:
    """Keep every choreography frame in one explicit, stable joint order."""

    return values


# Source pose: hands are already folded toward the belly, so the two palms do
# not touch the floor at reset.  A tap frame sets the corresponding arm to
# shoulder=elbow=0, placing its palm on the plane.
HOME_POSE = _pose(
    left_hip_yaw=0.0,
    left_hip_roll=-0.0873,
    left_hip_pitch=-0.4579,
    left_knee=-0.0049,
    left_ankle=0.4530,
    left_shoulder_pitch=0.82,
    left_elbow_pitch=-1.62,
    right_shoulder_pitch=0.82,
    right_elbow_pitch=-1.62,
    neck_pitch=0.3491,
    head_pitch=0.3491,
    head_yaw=0.0,
    head_roll=0.0,
    right_hip_yaw=0.0,
    right_hip_roll=0.0873,
    right_hip_pitch=0.4579,
    right_knee=0.0049,
    right_ankle=-0.4530,
)

# The first half reads as “捧腹前仰后倒”; the second half synchronizes the
# corresponding hand and foot in two distinct tap windows.
HUG_POSE = {**HOME_POSE, "left_shoulder_pitch": 1.05, "left_elbow_pitch": -1.85,
            "right_shoulder_pitch": 1.05, "right_elbow_pitch": -1.85}
FORWARD_POSE = {
    **HUG_POSE,
    "left_hip_pitch": -0.54,
    "left_knee": -0.13,
    "left_ankle": 0.52,
    "right_hip_pitch": 0.54,
    "right_knee": 0.13,
    "right_ankle": -0.52,
    "neck_pitch": 0.24,
    "head_pitch": 0.52,
    "head_yaw": 0.12,
    "head_roll": -0.06,
}
BACK_POSE = {
    **HUG_POSE,
    "left_hip_pitch": -0.38,
    "left_knee": 0.12,
    "left_ankle": 0.37,
    "right_hip_pitch": 0.38,
    "right_knee": -0.12,
    "right_ankle": -0.37,
    "neck_pitch": 0.43,
    "head_pitch": 0.19,
    "head_yaw": -0.12,
    "head_roll": 0.08,
}
LEFT_TAP_POSE = {
    **BACK_POSE,
    "left_shoulder_pitch": 0.0,
    "left_elbow_pitch": 0.0,
    "right_shoulder_pitch": 1.05,
    "right_elbow_pitch": -1.85,
    "left_knee": -0.15,
    "left_ankle": 0.58,
    "right_knee": 0.08,
    "right_ankle": -0.50,
    "head_yaw": 0.14,
}
RIGHT_TAP_POSE = {
    **BACK_POSE,
    "left_shoulder_pitch": 1.05,
    "left_elbow_pitch": -1.85,
    "right_shoulder_pitch": 0.0,
    "right_elbow_pitch": 0.0,
    "left_knee": -0.08,
    "left_ankle": 0.50,
    "right_knee": 0.15,
    "right_ankle": -0.58,
    "head_yaw": -0.14,
}

LAUGH_KEYFRAMES = (
    (0.00, HOME_POSE),
    (0.12, HUG_POSE),
    (0.29, FORWARD_POSE),
    (0.45, BACK_POSE),
    (0.58, BACK_POSE),
    (0.68, BACK_POSE),
    (0.72, LEFT_TAP_POSE),
    (0.80, LEFT_TAP_POSE),
    (0.84, RIGHT_TAP_POSE),
    (0.92, RIGHT_TAP_POSE),
    (0.96, HUG_POSE),
    (1.00, HOME_POSE),
)

ARM_JOINTS = (
    "left_shoulder_pitch",
    "left_elbow_pitch",
    "right_shoulder_pitch",
    "right_elbow_pitch",
)
ARM_KEYFRAMES = tuple(
    (phase, {name: pose[name] for name in ARM_JOINTS})
    for phase, pose in LAUGH_KEYFRAMES
)
BODY_JOINTS = tuple(name for name in HOME_POSE if name not in ARM_JOINTS)
BODY_HOME_POSE = {name: HOME_POSE[name] for name in BODY_JOINTS}
BODY_LOCK_KEYFRAMES = ((0.0, BODY_HOME_POSE), (1.0, BODY_HOME_POSE))

# Projected-gravity x targets: upright → forward laugh → gentle backward throw
# → upright before the alternating taps.  Values are dimensionless sin(pitch)
# proxies, roughly 6–9 degrees at the extrema.
TRUNK_LEAN_KEYFRAMES = (
    (0.00, 0.00),
    (0.12, 0.00),
    (0.29, 0.12),
    (0.45, -0.10),
    (0.58, -0.10),
    (0.68, -0.10),
    (0.72, 0.00),
    (0.80, 0.00),
    (0.84, 0.00),
    (0.92, 0.00),
    (0.96, 0.00),
    (1.00, 0.00),
)


def make_microduck_laugh_choreo_env_cfg(play: bool = False, rough: bool = False):
    """Create the arm-enabled, phase-conditioned laugh choreography."""

    cfg = make_microduck_laugh_env_cfg(play=play, rough=rough)
    cfg.scene.entities["robot"] = MICRODUCK_LAUGH_ROBOT_CFG

    hands_ground_cfg = ContactSensorCfg(
        name="laugh_hand_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_hand_collision|right_hand_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (hands_ground_cfg,)

    cfg.rewards.pop("laugh_pose", None)
    cfg.rewards.pop("laugh_pose_l1", None)
    # The legacy helper indexes the original 14-joint neck slice.  This model
    # inserts four arm joints, so the all-joint action-rate term below is the
    # correct regularizer for this task.
    cfg.rewards.pop("neck_action_rate_l2", None)
    cfg.rewards["laugh_choreography"] = RewardTermCfg(
        func=microduck_mdp.laugh_choreography_track,
        weight=14.0,
        params={
            "command_name": "twist",
            "keyframes": LAUGH_KEYFRAMES,
            "std": 0.22,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["laugh_choreography_l1"] = RewardTermCfg(
        func=microduck_mdp.laugh_choreography_track_l1,
        weight=3.0,
        params={
            "command_name": "twist",
            "keyframes": LAUGH_KEYFRAMES,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["laugh_arm_choreography"] = RewardTermCfg(
        func=microduck_mdp.laugh_choreography_track,
        weight=60.0,
        params={
            "command_name": "twist",
            "keyframes": ARM_KEYFRAMES,
            "std": 0.18,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["laugh_arm_choreography_l1"] = RewardTermCfg(
        func=microduck_mdp.laugh_choreography_track_l1,
        weight=20.0,
        params={
            "command_name": "twist",
            "keyframes": ARM_KEYFRAMES,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["laugh_hand_taps"] = RewardTermCfg(
        func=microduck_mdp.laugh_alternating_hand_contact_reward,
        weight=20.0,
        params={
            "sensor_name": hands_ground_cfg.name,
            "command_name": "twist",
            "left_start": 0.72,
            "left_end": 0.82,
            "right_start": 0.84,
            "right_end": 0.94,
        },
    )
    cfg.rewards["laugh_hand_height"] = RewardTermCfg(
        func=microduck_mdp.laugh_hand_height_track,
        weight=30.0,
        params={
            "command_name": "twist",
            "left_start": 0.72,
            "left_end": 0.82,
            "right_start": 0.84,
            "right_end": 0.94,
            "target_height": 0.011,
            "std": 0.025,
            "asset_cfg": SceneEntityCfg(
                "robot", site_names=("left_hand", "right_hand")
            ),
        },
    )
    cfg.rewards["laugh_trunk_lean"] = RewardTermCfg(
        func=microduck_mdp.laugh_trunk_lean_track,
        weight=2.5,
        params={
            "command_name": "twist",
            "keyframes": TRUNK_LEAN_KEYFRAMES,
            "std": 0.12,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # During the tap section one foot/hand pair is intentionally moving.  Keep
    # support and flatness as soft stabilizers rather than forcing both feet to
    # remain planted throughout the whole gesture.
    cfg.rewards["feet_grounded"].weight = 0.8
    cfg.rewards["feet_flat"].weight = -0.8
    cfg.rewards["action_rate_l2"].weight = -0.55 if not play else -0.75
    cfg.rewards["joint_torques_l2"].weight = -3e-3
    cfg.rewards["self_collisions"].weight = -1.5
    cfg.rewards["upright"].weight = 1.0

    command = cfg.commands["twist"]
    cfg.commands["twist"] = microduck_mdp.GroundPickPhaseCommandCfg(
        **{
            **vars(command),
            "class_type": microduck_mdp.GroundPickPhaseCommand,
            "period": LAUGH_CHOREO_PERIOD,
            "randomize_phase": False,
        }
    )
    cfg.viewer.body_name = "trunk_base"
    return cfg


MicroduckLaughChoreoRlCfg: RslRlOnPolicyRunnerCfg = deepcopy(MicroduckLaughRlCfg)
MicroduckLaughChoreoRlCfg.experiment_name = "laugh_choreo"
MicroduckLaughChoreoRlCfg.run_name = "laugh_choreo"


def make_microduck_laugh_arm_stage_env_cfg(play: bool = False, rough: bool = False):
    """Bootstrap arm contacts while the lower body tracks a fixed home pose."""

    cfg = make_microduck_laugh_choreo_env_cfg(play=play, rough=rough)
    cfg.rewards.pop("laugh_choreography", None)
    cfg.rewards.pop("laugh_choreography_l1", None)
    cfg.rewards.pop("laugh_trunk_lean", None)
    cfg.rewards["laugh_body_lock"] = RewardTermCfg(
        func=microduck_mdp.laugh_choreography_track,
        weight=35.0,
        params={
            "command_name": "twist",
            "keyframes": BODY_LOCK_KEYFRAMES,
            "std": 0.20,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["laugh_arm_choreography"].weight = 80.0
    cfg.rewards["laugh_arm_choreography_l1"].weight = 20.0
    cfg.rewards["laugh_hand_taps"].weight = 40.0
    cfg.rewards["laugh_hand_height"].weight = 50.0
    cfg.rewards["upright"].weight = 3.0
    cfg.rewards["feet_grounded"].weight = 2.0
    cfg.rewards["feet_flat"].weight = -1.0
    cfg.rewards["action_rate_l2"].weight = -0.25 if not play else -0.35
    return cfg


MicroduckLaughArmStageRlCfg: RslRlOnPolicyRunnerCfg = deepcopy(
    MicroduckLaughChoreoRlCfg
)
MicroduckLaughArmStageRlCfg.experiment_name = "laugh_arm_stage"
MicroduckLaughArmStageRlCfg.run_name = "laugh_arm_stage"
