"""Microduck forward-roll (roulade) task — attempt 3, run 6.

Episodic policy: robot starts standing, rolls forward over the flat top of
its head, and lands back on its feet. Triggered at deployment like sit/standup
(policy switch = roll starts immediately; no phase clock, no reference motion).

RUN-6 KIP-UP REWORK (run-5 eval: 100% roll+rise at every spawn class, but the
rise was a slow crunch — measured rise vz ~0.35 m/s, latency OK — and the
deterministic play policy had a frozen-fold attractor (play rollouts park in
the tucked pose for whole episodes) plus unbounded action-norm drift at the
standing hold, a deployment risk):
  • 腾空 flight credit: NEW roulade_airborne pays per airborne step (feet off
    the ground) in the post-roll window below STAND_Z — the kip-up "pop" now
    earns; hopping once upright does not (above max_height the instant the
    feet leave). The roll itself stays support-gated (run-1/2 lesson).
  • rise_velocity gate extended 260°→360° (was 180–260°, which missed the
    actual rise — run-5 episode sum ≈0.01, a dead term).
  • stand_tax ramp: the post-roll height L1 now grows 0→1 over 1.5 s after
    the frontier crosses 260° (_roulade_open_step in mdp.py) — parking in
    the fold becomes progressively expensive; late midroll spawns start the
    clock at birth.
  • landing composite pose now covers the neck joints too (hold-gradient on
    the head against the action drift).

Design (see the roulade section of mdp.py for the full history):
  • ONE dense progress signal — paid increments of the max-so-far cumulative
    forward rotation (potential-based: full roll pays 2π worth total, camping
    anywhere pays zero per step).
  • Landing rewards gated on ROLL COMPLETION (rotation frontier ≥ ~260°), not
    on a clock — "do nothing" earns nothing, the standing spawn cannot farm
    them, and no upright/height pressure ever opposes the flip.
  • Reverse curriculum via mid-roll spawns (the trick that fixed face-up
    recovery in standup): a slice of episodes starts 50°–185° into the roll,
    tucked, with forward angular momentum, accumulator pre-set to the spawn
    angle. The second half of a roulade IS the face-up recovery problem, which
    we know is learnable.
  • Élan hook for later: reset_roulade_state.forward_vel_range gives standing
    spawns an initial forward base velocity — set ROULADE_FORWARD_VEL_RANGE
    to e.g. (0.0, 0.3) to train rolls out of a walk. (0, 0) = standstill-only.

DR / obs / regularisers mirror the standup env (velocity sim2real parity),
with the motion-blockers (body_ang_vel, |a_z|, arrival damping) kept near zero
during discovery and introduced late by curriculum — the roll IS a large
angular-velocity, large-impact event; taxing attempts prevents discovery
(proven twice on standup).
"""

import math
import os as _os
from copy import deepcopy

# Symmetry — the roll is sagittal / left-right symmetric; the mirror loss
# directly fights the sideways-collapse failure seen in run 2. Enabled after
# migrating symmetry.py to the 61-dim layout (2026-08-13, includes the
# "policy" → "actor" output-key fix; roulade is the first env to use it).
ENABLE_SYMMETRY = True

# ── Domain randomisation (matched to standup/velocity for sim2real parity) ───
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_KP_RANDOMIZATION              = False  # match velocity (OFF)
ENABLE_KD_RANDOMIZATION              = False  # match velocity (OFF)
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = False  # a push mid-roll is incoherent
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True

# ── Ranges (matched to the standup env) ───────────────────────────────────────
COM_RANDOMIZATION_RANGE             = 0.003   # ramped to 0.015 via curriculum
HEAD_COM_RANDOMIZATION_RANGE        = 0.003   # ramped to 0.01 via curriculum
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
KP_RANDOMIZATION_RANGE              = (0.85, 1.15)  # unused (kp DR off)
KD_RANDOMIZATION_RANGE              = (0.9, 1.1)    # unused (kd DR off)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# Episode: a CONTROLLED roll takes ~2 s + rise ~1.5 s + settle. Run-3: 4 → 5 s
# (4 s left no room for the rise after a paced roll).
EPISODE_LENGTH_S = 5.0

# Empirically-measured standing trunk height (standup lesson: don't guess).
STAND_Z = 0.115

# ── Run-6 tunable reward constants (algo-pi-agent sets these per round) ──────
# The roulade-kipup tuning scenario overwrites these module attrs from each
# round's config.json BEFORE the env factory runs — the factory must read the
# CONSTANTS (not literals) so per-round weights actually take effect.
# AIRBORNE_WEIGHT: now quality-weighted (vz × height per airborne step, max
# ~0.4×0.115≈0.046/step vs 1.0/step for the old boolean), so the weight is
# raised ~4× to keep the 打挺 incentive at the same relative strength.
AIRBORNE_WEIGHT       = 1.0    # quality-weighted airborne credit (vz × height)
RISE_VELOCITY_WEIGHT  = 0.75   # upward-CoM-velocity reward in the late-roll window
STAND_TAX_WEIGHT      = 5.0    # post-roll height shortfall (self-negating, ramped)
STAND_TAX_RAMP_STEPS  = 75.0   # steps for the stand-tax ramp 0→1 (~1.5 s)

# ── Élan (run-up) hook ────────────────────────────────────────────────────────
# (0, 0) = roll from a standstill (run 1). Widen to e.g. (0.0, 0.3) to train
# rolls entered with forward momentum — standing spawns then get a random
# initial forward base velocity, approximating a hand-off from the walking
# policy without simulating the walk itself.
ROULADE_FORWARD_VEL_RANGE = (0.0, 0.0)

# ── Mid-roll spawn (reverse curriculum) ───────────────────────────────────────
# 90° = balanced on the head, 180° = on the back, 270° = supine, ~340° = seated
# leaning back, >260° opens the landing gate. Run-3 change: MAX widened
# 185° → 340° — run-2 wandb showed the second half of the roll (supine →
# seated → rise) was never spawned and never learned; spawns past ~300° open
# the landing gate at birth, giving dense on-policy data on the crouch→stand
# last mile (the velstand run-5 crouch-basin lesson).
MIDROLL_PITCH_MIN   = math.radians(50.0)
MIDROLL_PITCH_MAX   = math.radians(340.0)
MIDROLL_OMEGA_RANGE = (0.0, 3.0)   # rad/s forward momentum at spawn
# Tuck anchor: legs folded (crouch-anchor values from the velstand crouch
# reset) + CHIN TUCK (run-5: neck_pitch −1 / head_pitch +1 puts the flat head
# top squarely on the floor — measured axis_z −0.99 vs +0.6 for the passive
# face-plant; the head-top latch requires this, so mid-roll spawns must
# demonstrate the tucked configuration). Servo-index keyed; mid-roll spawns
# lerp HOME→tuck by a per-env factor.
TUCK_OVERRIDES = {
    2:  -1.15,  # left  hip_pitch
    3:   1.25,  # left  knee
    4:   1.05,  # left  ankle
    5:  -1.0,   # neck_pitch  (chin tuck)
    6:   1.0,   # head_pitch  (chin tuck)
    11:  1.15,  # right hip_pitch
    12: -1.25,  # right knee
    13: -1.05,  # right ankle
}

# Rotation thresholds (rad) for the state-based gates.
LANDING_GATE_LO = math.radians(260.0)
LANDING_GATE_HI = math.radians(330.0)
RISE_GATE_LO    = math.radians(180.0)
RISE_GATE_HI    = math.radians(260.0)

_LEG_JOINTS  = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_NECK_JOINTS = [5, 6, 7, 8]

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_roulade_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Microduck forward-roll environment configuration."""

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    # Head-ground contact — the roll's pivot signal. jaw_soft is the body that
    # carries the head collision geoms (top_head_shell = the flat top, jaw,
    # bottom_head_shell) in robot_allcollisions.xml. NAME IS LOAD-BEARING:
    # _update_roulade_accum reads it for the over-the-head latch.
    head_ground_cfg = ContactSensorCfg(
        name="head_ground_contact",
        primary=ContactMatch(mode="body", pattern="jaw_soft", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    # Whole-robot ground contact — the SUPPORT GATE (run-2 fix): the rotation
    # accumulator only integrates while some robot geom touches the terrain,
    # so ballistic flips ("breakdance") earn no progress and never complete.
    # NAME IS LOAD-BEARING: _update_roulade_accum reads it.
    robot_ground_cfg = ContactSensorCfg(
        name="robot_ground_contact",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    # Perf 2026-10-26: 实测约束数远低于基类的 1500; njmax=128 → env.step -21%,
    # 30 iter 短跑指标一致 (见 docs/perf/2026-10-26-training-speed-optimization.md).
    # DIVEWALL_NJMAX_NONE=1 (2026-10-29 replica3): 复刻原版 B18 的 None→
    # mjwarp 启发式 njmax — 原版成功时用的是这个裁剪物理, 不是 128 也不是 1500.
    if not _os.environ.get("DIVEWALL_NJMAX_NONE"):
        cfg.sim.njmax = 128

    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors  = (feet_ground_cfg, self_collision_cfg, head_ground_cfg, robot_ground_cfg)
    cfg.viewer.body_name = "trunk_base"

    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Actions ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Rewards: drop walking-specific terms ──────────────────────────────────
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── Rewards: roulade task set ─────────────────────────────────────────────
    # Progress increments — the one dense task signal during the roll. During
    # a 1.5 s roll it averages ~0.7/step; total payout per full roll from a
    # standing spawn ≈ weight × (episode steps it took) × mean ≈ weight × 50.
    cfg.rewards["roulade_progress"] = RewardTermCfg(
        func=microduck_mdp.roulade_progress,
        weight=8.0,
        # max_paid_rate: run-4 raised 3 → 5 rad/s. Measured physics (run-3
        # checkpoint eval): the over-the-top transit runs at 3.5–5.5 rad/s —
        # this robot is 10 cm tall, its natural tumble timescale is fast, and
        # the 3 rad/s cap was forfeiting most of the physically-necessary
        # rotation. Style pressure lives in |a_z| / action_rate / the support
        # gate, not in fighting gravity's clock.
        params={"target_angle": 2 * math.pi, "max_paid_rate": 5.0},
    )

    # Whip-speed tax — run-4 threshold 4 → 7 rad/s (above the measured p90
    # transit speed of ~5.5): taxes genuine whips, not the natural tumble.
    cfg.rewards["roulade_overspeed"] = RewardTermCfg(
        func=microduck_mdp.roulade_overspeed_penalty,
        weight=-0.1,
        params={"omega_max": 7.0},
    )

    # Head-as-pivot shaping: contact × mid-roll window × forward-rate factor
    # (the rate factor kills the "rest face-down with head on floor" farm).
    cfg.rewards["roulade_head_pivot"] = RewardTermCfg(
        func=microduck_mdp.roulade_head_pivot,
        weight=0.5,
        params={
            "sensor_name": head_ground_cfg.name,
            "angle_lo": math.radians(30.0),
            "angle_hi": math.radians(240.0),
            "rate_norm": 2.0,
        },
    )

    # Completion-gated standing annuity — the dominant attractor. Broad stds
    # (standup composite lesson: partial landing must score visibly, ~0.2+).
    # Run-6: pose coverage now includes the neck joints — run-5 deterministic
    # rollouts showed the policy mean drifting unboundedly at the standing
    # hold (action norm ‖a‖→12 while the pose looked fine), with no pose
    # gradient on the head to pin it back.
    cfg.rewards["roulade_landing_composite"] = RewardTermCfg(
        func=microduck_mdp.roulade_landing_composite,
        weight=4.0,
        params={
            "target_height":    STAND_Z,
            "height_std":       0.04,
            "upright_std":      0.40,
            "pose_std":         0.40,
            "joint_indices":    _LEG_JOINTS + _NECK_JOINTS,
            "gate_lo":          LANDING_GATE_LO,
            "gate_hi":          LANDING_GATE_HI,
            "target_overrides": None,
        },
    )

    # Completion-gated bootstrap layers (gradient far from the goal, where the
    # composite product is ≈0): linear upright + broad height Gaussian.
    cfg.rewards["roulade_upright_after_roll"] = RewardTermCfg(
        func=microduck_mdp.roulade_upright_after_roll,
        weight=1.5,
        params={"gate_lo": LANDING_GATE_LO, "gate_hi": LANDING_GATE_HI},
    )
    cfg.rewards["roulade_height_after_roll"] = RewardTermCfg(
        func=microduck_mdp.roulade_height_after_roll,
        weight=1.0,
        params={
            "target_height": STAND_Z,
            "std":           0.04,
            "gate_lo":       LANDING_GATE_LO,
            "gate_hi":       LANDING_GATE_HI,
        },
    )

    # Sharp landing layer (run-4): tight-std upright × height product on top
    # of the broad composite. Run-3 eval showed EVERY completed episode
    # parking at the same z≈0.105 / 27°-lean pose — the broad stds score ~0.5
    # there, no gradient to finish. Sharp layer: ~0.1 at the basin, ~1.0
    # upright — 10× differential across the last mile.
    cfg.rewards["roulade_landing_sharp"] = RewardTermCfg(
        func=microduck_mdp.roulade_landing_sharp,
        weight=2.0,
        params={
            "target_height": STAND_Z,
            "height_std":    0.015,
            "upright_std":   0.3,
            "gate_lo":       LANDING_GATE_LO,
            "gate_hi":       LANDING_GATE_HI,
        },
    )

    # Completion-gated stand tax (run-3, THE standup lesson): once the
    # rotation is done, every step spent below STAND_Z costs — "crumple in a
    # heap after the roll" flips from free to net-negative, the same fix that
    # broke standup's static-sit basin (its height L1 at ÷4-scaled weight
    # 7.5). Gate closed during the roll, so the roll itself is never taxed;
    # mid/late-roll spawns are born with it active, which is the point.
    # Run-6 ramp: the tax grows 0→1 over 1.5 s after the frontier crosses the
    # gate — parking in the fold becomes progressively expensive while a
    # prompt rise barely feels it.
    cfg.rewards["roulade_stand_tax"] = RewardTermCfg(
        func=microduck_mdp.roulade_stand_tax,
        weight=STAND_TAX_WEIGHT,
        params={
            "target_height": STAND_Z,
            "gate_lo":       LANDING_GATE_LO,
            "gate_hi":       LANDING_GATE_HI,
            "ramp_steps":    STAND_TAX_RAMP_STEPS,
        },
    )

    # Exit-rise bootstrap: upward CoM velocity, gated to open from the late
    # roll (supine → up is the face-up-recovery problem; end-state rewards
    # have zero gradient at zero motion there — standup lesson #2).
    # Run-6: the gate now stays open through the whole rise (was 180–260°,
    # which missed the actual 260°+ rise — observed episode sum was ~0.01,
    # i.e. dead term). The z < max_height cap still prevents hop-farming.
    cfg.rewards["roulade_rise_velocity"] = RewardTermCfg(
        func=microduck_mdp.roulade_rise_velocity,
        weight=RISE_VELOCITY_WEIGHT,
        params={
            "max_height": STAND_Z + 0.01,
            "gate_lo":    RISE_GATE_LO,
            "gate_hi":    math.radians(360.0),
        },
    )

    # 腾空 flight-phase credit (run-6, the kip-up requirement): per airborne
    # step after the roll (gate ≥ 260°, head-latched) while below STAND_Z.
    # Pays the dynamic "pop" launch out of the fold; standing hops gain
    # nothing (they're above max_height the moment the feet leave), and the
    # roll itself stays support-gated (run-1 ballistic-whip lesson).
    cfg.rewards["roulade_airborne"] = RewardTermCfg(
        func=microduck_mdp.roulade_airborne,
        weight=AIRBORNE_WEIGHT,
        params={
            "max_height":  STAND_Z,
            "gate_lo":     LANDING_GATE_LO,
            "gate_hi":     math.radians(360.0),
            "sensor_name": "feet_ground_contact",
        },
    )

    # Pre-roll prep shaping (Phase-B14): fills the 20°–150° reward vacuum
    # between the belly-flop and the rise gate — fold the head down, kick
    # the supported pitch. Default OFF (weight 0); enabled by PHASE_B14.
    cfg.rewards["roulade_preroll_kick"] = RewardTermCfg(
        func=microduck_mdp.roulade_preroll_kick,
        weight=0.0,
        params={
            "gate_lo": math.radians(20.0),
            "gate_hi": math.radians(150.0),
            "wall_name": "wall",
        },
    )

    # Straightness — run-5: the run-4 policy rolled over the SHOULDER (lower
    # energy path than straight over the head — it avoids the fully-inverted
    # configuration, same cheat human beginners default to). The structural
    # fix is the flatness gate on the accumulator + the head-top latch (side
    # rolls no longer count as rotation at all); these penalties provide the
    # dense per-step gradient back toward the plane, weights raised 5× from
    # the run-2 values that were noise against progress@8.
    cfg.rewards["roulade_sagittal"] = RewardTermCfg(
        func=microduck_mdp.roulade_sagittal_penalty,
        weight=-0.1,
    )
    cfg.rewards["roulade_lateral_vel"] = RewardTermCfg(
        func=microduck_mdp.roulade_lateral_velocity_penalty,
        weight=-0.5,
    )
    cfg.rewards["roulade_flatness"] = RewardTermCfg(
        func=microduck_mdp.roulade_flatness_penalty,
        weight=-0.5,
    )

    # ── Sim2real regularisers ─────────────────────────────────────────────────
    # Motion-blockers stay near zero during discovery (the roll IS a large
    # angular-velocity + impact event); the settle/polish pressure comes from
    # the LATE-introduced gated terms below (arrival_damping, |a_z|, torque
    # rate) — the standup timing lesson.
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=microduck_mdp.action_rate_l2_bounded, weight=-0.01, params={"sq_cap": 25.0})
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2, weight=0.0
    )

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.002   # must stay ≈0: the roll is ω
    cfg.rewards["angular_momentum"].weight = -0.001
    cfg.rewards.pop("soft_landing", None)

    # Arrival damper — trunk ω_xy² gated on standing height AND low tilt, so
    # the roll itself is never taxed; introduced at 0 and ramped by curriculum.
    cfg.rewards["arrival_damping"] = RewardTermCfg(
        func=microduck_mdp.body_ang_vel_at_height,
        weight=0.0,
        params={
            "height_low":    0.09,
            "height_high":   0.11,
            "tilt_full_deg": 20.0,
            "tilt_zero_deg": 45.0,
            "asset_cfg":     SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # |a_z| impact shaping — active from step 0 (run-2 change: run 1
    # discovered a violent solution under zero impact cost and locked it in;
    # discovery is easy in this env, so shaping the style from the start is
    # the priority). Curriculum ramps it further.
    # NOTE: trunk_vertical_accel_penalty is SELF-NEGATING (returns -|a_z|) →
    # POSITIVE weight (penalty sign convention; a negative weight here would
    # reward violence — caught in the run-2 smoke test, sum was positive).
    cfg.rewards["gentle_landing"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.002,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # Self-collision — LIGHT: a tucked roll needs body-on-body contact
    # (knees against trunk); standup's -1.0 would fight the tuck.
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.1,
        params={"sensor_name": self_collision_cfg.name},
    )

    # Always-on upright would oppose the flip (the old attempt's core failure);
    # landing uprightness is handled by the completion-gated terms above.
    if "upright" in cfg.rewards:
        del cfg.rewards["upright"]

    # ── Observations (identical layout to walking / standup policies) ─────────
    del cfg.observations["actor"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )

    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    cfg.observations["actor"].terms["base_ang_vel"].noise    = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise       = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise       = Unoise(n_min=-0.25, n_max=0.25)

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # Command obs slots: zero padding for BOTH head (4) and body (6) — the head
    # is part of the task (it's the pivot), so no head_pose command here, but
    # the 61D obs layout parity with velocity/standup is kept so the runtime
    # stack works unchanged (send zeros).
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # ── Command: tiny noise around zero (kept for obs-shape parity) ──────────
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs  = 0.0
    command.heading_command   = False
    command.ranges.heading    = None
    command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)
    command.debug_vis = False
    command.ranges.lin_vel_x = (-0.01, 0.01)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-0.05, 0.05)
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))

    # ── Terminations ──────────────────────────────────────────────────────────
    # Falling over is the task — keep only the NaN guard + timeout.
    if "fell_over" in cfg.terminations:
        del cfg.terminations["fell_over"]
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )

    # ── Events ────────────────────────────────────────────────────────────────
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)

    # Standing start + mid-roll reverse-curriculum spawns; also resets the
    # rotation accumulator (must run after reset_robot_joints — dict insertion
    # order — since mid-roll tuck lerps FROM the HOME pose it wrote).
    cfg.events["set_roulade_state"] = EventTermCfg(
        func=microduck_mdp.reset_roulade_state,
        mode="reset",
        params={
            "standing_prob":      0.5,
            "midroll_prob":       0.5,
            "standing_z_min":     0.11,
            "standing_z_max":     0.12,
            "standing_tilt_max":  math.radians(5.0),
            "forward_vel_range":  ROULADE_FORWARD_VEL_RANGE,
            "midroll_pitch_min":  MIDROLL_PITCH_MIN,
            "midroll_pitch_max":  MIDROLL_PITCH_MAX,
            "midroll_z_min":      0.05,
            "midroll_z_max":      0.10,
            "midroll_omega_range": MIDROLL_OMEGA_RANGE,
            "tuck_overrides":     TUCK_OVERRIDES,
            "tuck_factor_range":  (0.3, 1.0),
            "joint_noise_std":    0.08,
        },
    )

    if "push_robot" in cfg.events:
        del cfg.events["push_robot"]

    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
        kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)
        cfg.events["randomize_motor_gains"] = EventTermCfg(
            func=microduck_mdp.randomize_delayed_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "kp_range": kp_range,
                "kd_range": kd_range,
            },
        )

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )

    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    # ── Terrain ───────────────────────────────────────────────────────────────
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # ── Curriculum ────────────────────────────────────────────────────────────
    if "terrain_levels" in cfg.curriculum:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # Reverse-curriculum mix: heavy mid-roll early (the completion sub-task is
    # learnable from day 0 — it overlaps face-up recovery), shift toward
    # standing starts as the full roll gets discovered. Mid-roll never goes to
    # zero: it keeps the second half practiced and is realistic DR anyway.
    # Run-3: stages pushed 1500/3000 → 3000/6000 — run 2 shifted away from
    # mid-roll BEFORE standing-spawn rolls were mastered (progress episode-sum
    # was ~20% of a full roll at iter 1876; curriculum-pacing failure, same
    # family as the 2026-07-28 standup regression).
    cfg.curriculum["roulade_spawn_mix"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "set_roulade_state",
            "param_stages": [
                {"step": 0,          "params": {"standing_prob": 0.50, "midroll_prob": 0.50}},
                {"step": 3000 * 24,  "params": {"standing_prob": 0.65, "midroll_prob": 0.35}},
                {"step": 6000 * 24,  "params": {"standing_prob": 0.80, "midroll_prob": 0.20}},
            ],
        },
    )

    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                    {"step": 1500 * 24, "range": 0.015},
                ],
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                ],
            },
        )

    # action_rate ramp — run-4: ceiling softened -0.6 → -0.4 and the -0.4
    # stage pushed 2000 → 3000. Run-3's landing metrics peaked at ~iter 2700
    # then declined, tracking the -0.4/-0.6 stages — the tightening was
    # squeezing the rise. (Run-2 note still holds: -0.1 minimum from step 0,
    # run 1 bred violence under near-zero smoothing.)
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "action_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": -0.1},
                {"step": 1500 * 24,  "weight": -0.2},
                {"step": 3000 * 24,  "weight": -0.4},
            ],
        },
    )

    # Smoothness polish — introduced only after the roll skill exists (standup
    # timing lesson: any attempt-tax active during discovery prevents the
    # maneuver from being found at all; fix is timing, not magnitude).
    cfg.curriculum["arrival_damping_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "arrival_damping",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 2500 * 24,  "weight": -0.025},
                {"step": 3500 * 24,  "weight": -0.05},
            ],
        },
    )
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 2500 * 24,  "weight": -5e-4},
                {"step": 3500 * 24,  "weight": -1e-3},
            ],
        },
    )
    cfg.curriculum["gentle_landing_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            # POSITIVE weights: the func is self-negating (returns -|a_z|).
            "reward_name":   "gentle_landing",
            "weight_stages": [
                {"step": 0,          "weight": 0.002},
                {"step": 2500 * 24,  "weight": 0.005},
            ],
        },
    )

    return cfg


# ── Dive-roll variant (前扑滚翻) ──────────────────────────────────────────────
# The physics probe (scripts/jump_probe.py, 2026-09-17) closed the true front
# flip: scripted max-effort squat→extend jumps never produced a clean hop (apex
# stayed BELOW standing height — position-controlled XL330s can't pop 800 g
# ballistically). So the dive-roll is built from FORWARD MOMENTUM, not jump
# height: the élan spawn hook gives standing spawns 0.1–0.3 m/s of forward
# speed, and a dedicated launch reward pays the brief airborne dive phase
# (feet off + forward vx + forward pitch rate, only before the roll frontier
# passes 90°). The roll itself stays support-gated (run-1 ballistic-whip
# lesson); landing/rise machinery is inherited unchanged.
# Run-2: élan raised from (0.1, 0.3) — a visible dive needs more incoming
# momentum (0.3 m/s ≈ 1.2 body-lengths/s barely reads as a lunge), and the
# reward vx gates are widened to match.
DIVE_FORWARD_VEL_RANGE = (0.5, 1.2) # élan: initial forward base velocity (m/s) — run-4: covers the 1.2 m/s trigger regime in-distribution
DIVE_LAUNCH_WEIGHT     = 6.0          # explicit flight credit (height gain + airborne steps <90°) — run-4
DIVE_APPROACH_WEIGHT   = 1.0          # dense shaping ramp toward the launch (run-2)


def make_microduck_roulade_dive_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Roulade with a dive entry: forward-momentum spawns + launch-phase reward."""
    cfg = make_microduck_roulade_env_cfg(play=play)

    # Élan: standing spawns start with forward speed (approximates a walk
    # hand-off without simulating the walk — see the base factory docstring).
    cfg.events["set_roulade_state"].params["forward_vel_range"] = DIVE_FORWARD_VEL_RANGE

    # Launch-phase credit (run-4): pay REAL flight inside the <90° window —
    # airborne steps pay a base component plus a bonus for base-z rising
    # ABOVE standing height (height_gain), vx gate widened to cover the
    # 1.2 m/s trigger regime. Replaces the run-3 capped-height product.
    cfg.rewards["roulade_dive_launch"] = RewardTermCfg(
        func=microduck_mdp.roulade_dive_launch_v2,
        weight=DIVE_LAUNCH_WEIGHT,
        params={
            "vx_lo":         0.4,
            "vx_hi":         1.25,
            "stand_z":       STAND_Z,
            "height_ref":    0.05,
            "frontier_max":  math.radians(90.0),
            "sensor_name":   "feet_ground_contact",
        },
    )

    # Run-2: dense approach shaping — run-1's pure product gate stayed at
    # ~0.0005 for 6000 iters (undiscoverable). This term pays the same three
    # factors continuously through the pre-roll window so the gradient exists
    # from step 0; the product term above remains the airborne bonus.
    cfg.rewards["roulade_dive_approach"] = RewardTermCfg(
        func=microduck_mdp.roulade_dive_approach,
        weight=DIVE_APPROACH_WEIGHT,
        params={
            "vx_lo":         0.3,
            "vx_hi":         0.75,
            "omega_ref":     3.0,
            "vz_ref":        0.3,
            "frontier_max":  math.radians(90.0),
        },
    )

    return cfg


# ── Dive-wall variant (障碍物强制前扑) ───────────────────────────────────────
# Run-4's dive reads as "momentum pushes it over", not an active leap: the
# élan is injected horizontal velocity and nothing forces an upward launch.
# The wall variant puts a heavy 6cm hurdle dead ahead of every spawn: the
# only way to a completed roll is to LEAP over it (collision penalty +
# cleared-plane bonus), so the launch becomes an active, duck-initiated
# vault. The wall is training scaffolding only — NOT in the obs, so the
# policy stays proprioceptive and the warm-start obs space is unchanged.
WALL_COLLISION_WEIGHT = -2.0   # run-3: FLAT any-contact penalty (vaulting priced in)
WALL_CLEARED_WEIGHT   = 0.2    # run-3: drip reduced 2.0 -> 0.2 (only pays "be past it")
WALL_FLIGHT_WEIGHT    = 15.0   # run-3: one-shot flight-over-wall jackpot
WALL_TAKEOFF_WEIGHT   = 6.0    # run-8: pitch-gated takeoff shaping (was 4.0 run-5..7)
WALL_TAKEOFF_PITCH    = (10.0, 50.0)  # stage-2d: was (30,70) — duck never pitched past 30° (push-fed), takeoff 0 for 16k iters
                                      # FREEZES at the wall (takeoff ate only 0.003 in 8k iters, gate
                                      # too strict to ever taste). Lower ramp so a partial forward-lean
                                      # takeoff starts paying.
WALL_PUSH_WEIGHT      = 6.0    # stage-2d: was 8.0 — GIF: duck farms grounded pitched kicking (扑倒蹬地), never airborne
                               # rising at the wall) is exactly the missing jump impulse; run-12
                               # duck stands planted at the wall — pay any vz>0 there
WALL_PUSH_PITCH       = (10.0, 30.0)  # run-14: run-13 lesson — vz-only push got farmed by tiny
                               # STATIONARY hops at the wall (no run-up, GIF: rolls from a
                               # standstill). Gate the push on a forward pitch ramp so "jump" and
                               # "dive" must co-occur to earn credit.
WALL_PUSH_VX_SCALE    = 1.0    # run-14: ×(0.3 + 0.7·clamp01(vx/vx_scale)) — stationary hop pays
                               # ≤30% of the term; a moving push-off pays in full
WALL_SHOVE_WEIGHT     = -3.0   # run-8b: per-step penalty while the wall is displaced/tilted
WALL_RUNUP_WEIGHT     = 4.0    # run-14: was 2.0 — run-13 runup stalled at 0.07 for 8k iters
                               # (user GIF: NO run-up at all, duck dives from a standstill);
                               # double the pull toward leaning forward speed in the runway
WALL_PREMATURE_ROLL_WEIGHT = -3.0  # run-14b: grounded pitch>45° more than 15cm before the
WALL_PREMATURE_DIVE_WEIGHT = -0.5  # stage-2c: -2.0 caused policy oscillation/collapse (10-05); soften to correct-not-crush
                               # wall costs — 在墙前远处翻滚甚至负向 (user 2026-10-04)
WALL_DISTANCE_RANGE   = (0.85, 1.20)  # run-10: spawn→wall distance — was 0.22-0.30 (2 body
                                      # lengths, no room for a visible approach); now 0.85-1.2 m
                                      # so the élan carries a real run-up before the dive


def make_microduck_roulade_divewall_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Dive entry forced by a hurdle the robot must vault to complete a roll."""
    cfg = make_microduck_roulade_dive_env_cfg(play=play)

    from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALL_CFG

    robot_cfg = cfg.scene.entities["robot"]
    cfg.scene.entities = {"robot": robot_cfg, "wall": MICRODUCK_WALL_CFG}

    # Robot-vs-wall contact sensor (force for the penalty).
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (
        ContactSensorCfg(
            name="robot_wall_contact",
            primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="wall", entity="wall"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
        ),
    )
    # Run-14 (user spec 2026-10-04): 墙前面翻滚不能给 reward —— 在墙附近前扑、
    # 墙后头落地翻滚. Gate every roll-PAYOFF term (progress, head pivot,
    # landing annuities) on the trunk already being PAST the wall plane.
    # Launch/dive terms (dive_launch, airborne, rise, push, takeoff, runup)
    # stay ungated — the pounce itself happens AT the wall by design.
    for _roll_term in (
        "roulade_progress",
        "roulade_head_pivot",
        "roulade_landing_composite",
        "roulade_upright_after_roll",
        "roulade_height_after_roll",
        "roulade_landing_sharp",
    ):
        if _roll_term in cfg.rewards:
            cfg.rewards[_roll_term].params["wall_name"] = "wall"
    # Extra contact headroom for robot-wall contacts.
    cfg.sim.nconmax = 60

    # All spawns standing (mid-roll spawns would roll INTO the wall for
    # free penalty — the kip-up skill is already baked in from run-4).
    cfg.events["set_roulade_state"].params["standing_prob"] = 1.0
    cfg.events["set_roulade_state"].params["midroll_prob"] = 0.0

    # Wall placement — AFTER set_roulade_state (dict insertion order).
    # Phase-B v2 (user 2026-10-16 10:46 GIF: 起点太远, 全速助跑撞墙挂住):
    # 出生距离可由脚本覆盖 (DIVEWALL_WALL_DIST="0.40,0.65") — 拉近后没跑
    # 满速就到墙, 撞墙动能减半, takeoff 分相对更划算。
    _wd = _os.environ.get("DIVEWALL_WALL_DIST", "")
    _wd_range = (tuple(float(x) for x in _wd.split(","))
                 if _wd else WALL_DISTANCE_RANGE)
    cfg.events["reset_wall"] = EventTermCfg(
        func=microduck_mdp.reset_wall_ahead,
        mode="reset",
        params={"distance_range": _wd_range,
                "wall_sink": float(_os.environ.get("DIVEWALL_WALL_SINK", "0") or 0)},
    )

    # Stage-4 v2 (user 2026-10-09): post-dive spawn — a fraction of envs
    # respawn PAST the wall in a mid-dive-fall state (pitch 120-200°, tucked,
    # residual forward ω). Reverse-curriculum seeding for the roll-completion
    # half of the full pipeline; must run AFTER reset_wall_ahead. Off (0)
    # unless DIVEWALL_POSTDIVE_PROB is set.
    _pd_prob = float(_os.environ.get("DIVEWALL_POSTDIVE_PROB", "0") or 0)
    if _pd_prob > 0.0:
        cfg.events["postdive_spawn"] = EventTermCfg(
            func=microduck_mdp.postdive_spawn,
            mode="reset",
            params={"prob": _pd_prob, "tuck_overrides": TUCK_OVERRIDES},
        )

    cfg.rewards["roulade_wall_collision"] = RewardTermCfg(
        func=microduck_mdp.wall_contact_penalty,
        weight=WALL_COLLISION_WEIGHT,
    )
    cfg.rewards["roulade_wall_cleared"] = RewardTermCfg(
        func=microduck_mdp.wall_cleared_bonus,
        weight=WALL_CLEARED_WEIGHT,
        params={"robot_fwd_offset": 0.05, "min_height": 0.12},
    )
    cfg.rewards["roulade_wall_flight"] = RewardTermCfg(
        func=microduck_mdp.wall_flight_clear_bonus,
        weight=WALL_FLIGHT_WEIGHT,
        params={"robot_fwd_offset": 0.05, "min_height": 0.12, "clearance": 0.02},
    )
    # Run-5: dense takeoff shaping in front of the wall (spread the sparse
    # flight jackpot into a continuous gradient: load -> push -> rise).
    cfg.rewards["roulade_wall_takeoff"] = RewardTermCfg(
        func=microduck_mdp.wall_takeoff_shaping,
        weight=WALL_TAKEOFF_WEIGHT,
        params={
            "zone_near": 0.02, "zone_far": 0.20,
            "vz_scale": 0.5, "vx_scale": 1.2,
            # Run-8 physics gate: wall top is only ~2 cm below standing COM
            # (0.146 m) — clearing it requires a dive pitch at takeoff, not
            # jump height. Pay the airborne rise ONLY when |pitch| has begun
            # the dive (ramp 50°→85°).
            "pitch_min_deg": WALL_TAKEOFF_PITCH[0],
            "pitch_full_deg": WALL_TAKEOFF_PITCH[1],
        },
    )
    # Run-6: grounded push-off credit (the leg extension itself — feet planted,
    # trunk rising — earns reward in the approach zone).
    cfg.rewards["roulade_wall_push"] = RewardTermCfg(
        func=microduck_mdp.wall_push_shaping,
        weight=WALL_PUSH_WEIGHT,
        params={
            "zone_near": 0.02, "zone_far": 0.25, "vz_scale": 0.3,
            "vx_scale": WALL_PUSH_VX_SCALE,
            "pitch_min_deg": WALL_PUSH_PITCH[0],
            "pitch_full_deg": WALL_PUSH_PITCH[1],
        },
    )
    # Run-8b: price the wall-shove exploit itself (probe 2026-10-02: the
    # policy bulldozed the 2 kg free-joint wall aside and "crossed" at
    # z≈0.05). Cleared/flight bonuses are also planted-gated (see mdp).
    cfg.rewards["roulade_wall_shove"] = RewardTermCfg(
        func=microduck_mdp.wall_shove_penalty,
        weight=WALL_SHOVE_WEIGHT,
    )
    # Run-14b (user spec 2026-10-04): rolling FAR before the wall is now
    # actively penalized (past-wall gating already zeroed its income).
    cfg.rewards["roulade_wall_premature_roll"] = RewardTermCfg(
        func=microduck_mdp.wall_premature_roll_penalty,
        weight=WALL_PREMATURE_ROLL_WEIGHT,
        params={"near_dist": 0.30, "pitch_thresh_deg": 30.0},
    )
    # Stage-2b (user 2026-10-05): 扑得太早 — penalize early pounces on the
    # runway (airborne-with-pitch or deep grounded lunge beyond 30cm).
    cfg.rewards["roulade_wall_premature_dive"] = RewardTermCfg(
        func=microduck_mdp.wall_premature_dive_penalty,
        weight=WALL_PREMATURE_DIVE_WEIGHT,
        params={"near_dist": 0.30, "pitch_hi_deg": 32.0, "pitch_air_deg": 15.0},
    )
    # Run-9/11: no walking AROUND the wall — lateral strays end the episode.
    # max_lat 0.7 sits INSIDE the 0.8m wall half-span (run-11: the 3m wall
    # spilled into neighbour env lanes on the shared plane and got bowled
    # across them — probe 2026-10-02 23:50; 0.7/0.8 is airtight + no spill).
    cfg.terminations["wall_lateral"] = TerminationTermCfg(
        func=microduck_mdp.wall_lateral_termination,
        params={"max_lat": 0.7},
    )
    # Run-10: leaning run-up credit in the extended runway (see mdp).
    # vx_lo 0.25 (not 0.4): the élan direction is not perfectly aligned with
    # the spawn heading, so the along-heading component is ~0.7x the injected
    # speed — a 0.4 gate pays ZERO at spawn (found 2026-10-02, runup stuck at
    # 0.0001 for 1000 iters).
    cfg.rewards["roulade_wall_runup"] = RewardTermCfg(
        func=microduck_mdp.wall_runup_shaping,
        weight=WALL_RUNUP_WEIGHT,
        params={"zone_near": 0.30, "vx_lo": 0.25, "vx_full": 1.2, "lean_full_deg": 30.0, "lean_max_deg": 35.0},
    )

    # ── Run-15 CURRICULUM (user 2026-10-04): reward-shaping scaffolding is
    # fighting itself — the duck still rolls BEFORE the wall. Split skill
    # acquisition into 3 stages via DIVEWALL_STAGE env var:
    #   stage 1  学跑步      — ONLY run-up credit (longer runway); every dive/
    #                         roll/launch/airborne term OFF, premature-roll
    #                         penalty ON, wall pushed back (far spawn).
    #   stage 2  带初速前扑  — + push/takeoff/flight/cleared (the pounce);
    #                         roll-payoff terms still OFF.
    #   stage 3  前扑接翻滚  — everything on (the run-14b full config).
    # Unset / 0 keeps the full config (backward compatible).
    # (_os imported at module top — do NOT re-import here: a local
    # `import os as _os` shadows the module-level one for the WHOLE
    # function scope and crashes earlier references with
    # "cannot access local variable '_os'").
    _stage = int(_os.environ.get("DIVEWALL_STAGE", "0") or 0)
    # CURRICULUM v3 (user 2026-10-07 编号定稿):
    #   Stage-1-run 纯跑步 | Stage-2-jump 跳远 | Stage-3-run&jump 前扑跳 |
    #   Stage-4-full 全流程 (unset/0 = full, backward compatible).
    # Legacy v2 ids alias: 25→1, 21→2 (3/4 already match v3 numbering).
    _stage = {25: 1, 21: 2}.get(_stage, _stage)
    if _stage in (1, 2, 3):
        # Curriculum v2 (user 2026-10-05 重新课程学习): school the JUMP in
        # isolation before re-introducing the wall. 18 runs of wall-side
        # shaping never bridged grounded-push → airborne-flight (mutual
        # exclusion).   stage 21 跳远学校 — wall parked far away, hop credit.
        #                      stage 22 前扑跳    — hop credit + pitch ramp.
        #   stage 25 纯跑步学校 (user 2026-10-06 19:52: GIF verdict "还是
        #   趴倒了…先学好第一步跑步"): reward 全部重置到跑步一件事——无 hop、
        #   无 flop 罚、无一切墙侧 shaping，只有 runup + 基础姿态正则。
        for _t in [
            "roulade_progress", "roulade_head_pivot",
            "roulade_landing_composite", "roulade_upright_after_roll",
            "roulade_height_after_roll", "roulade_landing_sharp",
            "roulade_stand_tax", "roulade_rise_velocity",
            "roulade_preroll_kick",
            "roulade_dive_launch", "roulade_dive_approach", "roulade_airborne",
            "roulade_wall_push", "roulade_wall_takeoff",
            "roulade_wall_flight", "roulade_wall_cleared",
            "roulade_wall_premature_dive", "roulade_wall_premature_roll",
            # v4 (user 2026-10-07 精简): wall parked 4.5m away in stages 1-3 —
            # wall-side penalties are dead weight, drop them.
            "roulade_wall_collision", "roulade_wall_shove",
            # v4: dead zero-weight terms.
            "body_ang_vel", "angular_momentum", "joint_torque_rate_l2",
            "arrival_damping", "gentle_landing",
            # v4: weight-schedule curriculum terms reference the popped
            # rewards above — drop them too or the manager raises.
            "roulade_overspeed", "roulade_flatness", "roulade_sagittal",
        ]:
            cfg.rewards.pop(_t, None)
        for _c in ("arrival_damping_weight", "torque_rate_weight",
                   "gentle_landing_weight"):
            cfg.curriculum.pop(_c, None)
        # v6 audit (user 2026-10-07 21:55 趴地帮凶清扫):
        # ① roulade_spawn_mix — reset 时把 standing_prob 打回 0.5/0.5
        #   midroll：一半 episode 从趴地旋转出生，直接吃 prone 罚，跑步样本
        #   被砍半。跑/跳学校全部移除，保持 standing_prob=1.0。
        # ② com_range / head_com_range — COM 随机化扩到 ±1.5cm，是翻滚
        #   鲁棒性设计，对跑步学校是纯难度加码；固定在初始 ±3mm。
        # ③ action_rate_weight — 罚从 -0.1 涨到 -0.4，跑步的大幅周期动作
        #   会被后期重罚（奖励静止）；固定 -0.01 不变。
        for _c in ("roulade_spawn_mix", "com_range", "head_com_range",
                   "action_rate_weight"):
            cfg.curriculum.pop(_c, None)
        # ④ 跌倒 termination — v5 躺平后要躺满 5s 才 reset，浪费样本；
        #   z 塌到 0.07 以下立即截断（出生 z 0.11-0.12 不误伤）。
        cfg.terminations["prone_cut"] = TerminationTermCfg(
            func=microduck_mdp.prone_termination,
            params={"z_thresh": 0.07},
        )
        cfg.terminations.pop("wall_lateral", None)
        cfg.events["reset_wall"].params["distance_range"] = (4.5, 5.5)
        # v5 (user 2026-10-07 19:54): 关掉 élan 初速注入——runup 被"出生自带
        # 0.5-1.2 m/s 的惯性滑行"刷分（deterministic probe 实锤：初速耗尽后
        # vx→0.05，蹲伏趴地）。跑步学校从静止起步，每一分都要自己蹬出来。
        cfg.events["set_roulade_state"].params["forward_vel_range"] = (0.0, 0.0)
        # RUN SCHOOL v7 (user 2026-10-08): walk-recipe port. v4-v6's gated
        # runup shaping has ~zero gradient from a standing start (the vx_lo
        # gate pays 0 below it); the recipe that actually taught this robot
        # to WALK is dense gaussian command tracking + posture anchors +
        # air_time. v7c (user 2026-10-08 review): stages 2/3 INHERIT a
        # down-weighted copy of the recipe (wider vx band, softer anchors)
        # so the running skill survives — jump terms stack on top instead
        # of replacing it. Actor obs stays untouched (constant forward
        # band, zero-padded command slot is harmless).
        cfg.rewards.pop("roulade_wall_runup", None)
        if _stage == 1:
            _vx_band = (0.2, 0.6)
            _w_track, _w_pose, _w_up, _w_air = 2.0, 1.0, 2.0, 3.0
        elif _stage == 2:
            _vx_band = (0.3, 0.9)
            _w_track, _w_pose, _w_up, _w_air = 2.0, 0.5, 1.0, 1.5
        else:
            _vx_band = (0.4, 1.2)
            _w_track, _w_pose, _w_up, _w_air = 2.0, 0.5, 1.0, 1.5
        _cmd = cfg.commands["twist"]
        _cmd.ranges.lin_vel_x = _vx_band   # forward-only run band
        _cmd.ranges.lin_vel_y = (0.0, 0.0)
        _cmd.ranges.ang_vel_z = (0.0, 0.0)
        _cmd.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S)
        if True:
            cfg.rewards["track_linear_velocity"] = RewardTermCfg(
                func=mdp.track_linear_velocity,
                weight=_w_track,
                params={"command_name": "twist", "std": math.sqrt(0.1)},
            )
            # Leg joints to home, walking tolerance (same std set as the
            # velocity env; command is always "walking" band).
            cfg.rewards["pose"] = RewardTermCfg(
                func=mdp.variable_posture,
                weight=_w_pose,
                params={
                    "std_standing": {
                        r".*hip_yaw.*": 0.1, r".*hip_roll.*": 0.05,
                        r".*hip_pitch.*": 0.15, r".*knee.*": 0.15,
                        r".*ankle.*": 0.1,
                    },
                    "std_walking": {
                        r".*hip_yaw.*": 0.3, r".*hip_roll.*": 0.05,
                        r".*hip_pitch.*": 0.4, r".*knee.*": 0.4,
                        r".*ankle.*": 0.25,
                    },
                    "std_running": {
                        r".*hip_yaw.*": 0.3, r".*hip_roll.*": 0.05,
                        r".*hip_pitch.*": 0.4, r".*knee.*": 0.4,
                        r".*ankle.*": 0.25,
                    },
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        joint_names=(r"^(?!passive_|.*neck.*|.*head.*).*",),
                    ),
                    "command_name": "twist",
                    "walking_threshold": 0.01,
                    "running_threshold": 1.5,
                },
            )
            cfg.rewards["upright"] = RewardTermCfg(
                func=mdp.upright,
                weight=_w_up,
                params={
                    "std": math.sqrt(0.05),
                    "asset_cfg": SceneEntityCfg(
                        "robot", body_names=("trunk_base",)
                    ),
                },
            )
            cfg.rewards["air_time"] = RewardTermCfg(
                func=mdp.feet_air_time,
                weight=_w_air,
                params={
                    "sensor_name": "feet_ground_contact",
                    "threshold_min": 0.125,
                    "threshold_max": 0.300,
                },
            )
            # v6 floors stay: falling down is expensive in any orientation.
            cfg.rewards["prone_fall"] = RewardTermCfg(
                func=microduck_mdp.prone_fall_penalty,
                weight=-2.0,
                params={"pitch_thresh_deg": 40.0, "z_thresh": 0.09},
            )
        # v7c: hop/flop terms STACK on the inherited walk recipe (was an
        # else-branch that dropped every walking anchor — the v4-v6 trap).
        if _stage >= 2:
            cfg.rewards["hop_forward"] = RewardTermCfg(
                func=microduck_mdp.hop_forward_credit,
                weight=6.0,
                params=(
                    # v5 (user 2026-10-07): vx 门控减弱 0.25→0.15（起跳瞬间
                    # 冲速会瞬时掉，门太苛梯度断）；soft gate 下限 0，不罚。
                    {"vz_scale": 0.4, "vx_scale": 1.2, "vx_gate": 0.15}
                    if _stage == 2 else
                    {"vz_scale": 0.4, "vx_scale": 1.0,
                     "pitch_min_deg": 15.0, "pitch_full_deg": 60.0}
                ),
            )
        if _stage >= 2:
            # Flop guard (2026-10-06): belly-flopes paid hop credit via the
            # push-off vz — jumps must stay upright in jump school.
            # v5 (user 2026-10-07): 50→65° — Stage-3 前扑要 pitch 60° 拿满
            # 分，罚线压到 65° 只打真拍地，不跟下一级打架。
            cfg.rewards["airborne_flop"] = RewardTermCfg(
                func=microduck_mdp.airborne_flop_penalty,
                weight=-2.0,
                params={"pitch_thresh_deg": 65.0},
            )
            # v5 (user 2026-10-07): 趴地=跌倒同代价。Stage-3 阈值放宽到
            # 50°——前扑跳落地瞬间 pitch 可能短暂冲过 40°，留余量防误罚。
            # v7c: Stage-3 exempt_airborne=True — 前扑腾空段 pitch 冲 60°
            # 是拿满 hop 分的正确姿势，空中不罚；落地趴地照罚。
            cfg.rewards["prone_fall"] = RewardTermCfg(
                func=microduck_mdp.prone_fall_penalty,
                weight=-2.0,
                params={
                    "pitch_thresh_deg": 50.0 if _stage >= 3 else 40.0,
                    "exempt_airborne": _stage >= 3,
                },
            )
        if _stage >= 3:
            # v5 (user 2026-10-07) roll preview: Stage-4 之前先预习空中俯仰
            # 旋转（paid rate 封顶 3 rad/s——roulade run-1 的教训：不奖励暴甩）。
            cfg.rewards["hop_forward"].params.update(
                # v7e (2026-10-08 22:25): pitch 起步线 8°→0 — v7d 2400 轮
                # hop 恒 0.0000：8° 门把 Stage-2 的直立小跳判零分，而带
                # 前倾的腾空从未被采样，梯度断档（同 v6 vx_lo 0.25 教训）。
                # 0→60° 线性 ramp：小跳拿部分分，坡道直通前扑。
                {"pitch_min_deg": 0.0}
            )
            cfg.rewards["airborne_spin"] = RewardTermCfg(
                func=microduck_mdp.airborne_spin_credit,
                weight=2.0,
                # v7d: spin 只在前扑姿态下计分 —— pitch≥30° ramp 到 60° 拿满，
                # 且 vx soft gate 0.3 m/s。直立小跳快转不再刷分（v7c 教训：
                # spin 0.88 而 hop 0.0008 的捷径）。
                params={
                    "omega_scale": 3.0,
                    "pitch_min_deg": 30.0,
                    "vx_gate": 0.3,
                },
            )
        # Phase-A v2 (user 2026-10-14 09:38 拍板, Phase-A v1 GIF verdict:
        # 跑步偏离墙、没有跳): ① 跑比跳划算(track 分 1.16 vs hop 0.30) →
        # hop 6→15、track 2→1，vx 命令带宽压到 (0.25, 0.6)——跑快没奖励；
        # ② yaw 跟踪误差 1.96 → 叠 track_angular_velocity 锚定航向
        # (ang_vel_z 命令=0)；③ air_time 换高度门控版 air_time_height —
        # 腾空且躯干 z>0.13 才计分，蹲姿贴地扑腾不蹭分。
        # 经 DIVEWALL_PHASE_A2 环境变量启用，不影响 v1 复现。
        if _stage == 2 and _os.environ.get("DIVEWALL_PHASE_A5"):
            # v5 (user 2026-10-14 16:30 GIF verdict + 抽帧实锤: t=9s 帧
            # z=0.145/AIR/v=0.46 — 真跳已发生但被 v4 门槛(z 0.15/t 0.18)
            # 卡成零分, air_time_height 6000 轮恒 0。门槛放到能力边缘,
            # 梯度先通再收紧: z 0.14 / t_min 0.13。注入起跳状态(方案②)
            # 暂缓 — élan 教训(2026-10-13): spawn 白送状态会被躺刷。
            # 航向加码: heading_hold 4→6, std 10→8 (v4 GIF 仍不朝墙)。
            cfg.rewards["hop_forward"].weight = 20.0
            cfg.rewards["hop_forward"].params.update({"z_thresh": 0.14})
            cfg.rewards["track_linear_velocity"].weight = 1.0
            cfg.commands["twist"].ranges.lin_vel_x = (0.25, 0.6)
            cfg.rewards["track_angular_velocity"] = RewardTermCfg(
                func=mdp.track_angular_velocity,
                weight=1.0,
                params={"command_name": "twist", "std": math.sqrt(0.25)},
            )
            cfg.rewards["heading_hold"] = RewardTermCfg(
                func=microduck_mdp.heading_hold_reward,
                weight=6.0,
                params={"std_deg": 8.0},
            )
            cfg.rewards.pop("air_time", None)
            cfg.rewards["air_time_height"] = RewardTermCfg(
                func=microduck_mdp.air_time_height_credit,
                weight=6.0,
                params={"z_thresh": 0.14, "t_min": 0.13, "t_max": 0.4},
            )
            cfg.rewards["overspeed"] = RewardTermCfg(
                func=microduck_mdp.overspeed_penalty,
                weight=-1.5,
                params={"vx_limit": 0.75, "vx_span": 0.5},
            )
        elif _stage == 2 and _os.environ.get("DIVEWALL_PHASE_A4"):
            # v4 (user 2026-10-14 14:12 GIF verdict: 跑得非常快但不朝墙跑
            # 不跳): v3 复盘 — 跑步弹跳的腾空段在蹭 hop/air_time_height 分
            # (跑越快腾空段越多分越高) → ① air_time_height 重写为落地结算,
            # 腾空时长 <0.18s(跑步弹跳)清零 + 峰值高度门; ② hop_forward 加
            # 躯干高度门 z>0.15; ③ 超速罚 vx>0.75; ④ heading_hold 加码。
            cfg.rewards["hop_forward"].weight = 20.0
            cfg.rewards["hop_forward"].params.update({"z_thresh": 0.15})
            cfg.rewards["track_linear_velocity"].weight = 1.0
            cfg.commands["twist"].ranges.lin_vel_x = (0.25, 0.6)
            cfg.rewards["track_angular_velocity"] = RewardTermCfg(
                func=mdp.track_angular_velocity,
                weight=1.0,
                params={"command_name": "twist", "std": math.sqrt(0.25)},
            )
            cfg.rewards["heading_hold"] = RewardTermCfg(
                func=microduck_mdp.heading_hold_reward,
                weight=4.0,
                params={"std_deg": 10.0},
            )
            cfg.rewards.pop("air_time", None)
            cfg.rewards["air_time_height"] = RewardTermCfg(
                func=microduck_mdp.air_time_height_credit,
                weight=6.0,
                params={"z_thresh": 0.15, "t_min": 0.18, "t_max": 0.5},
            )
            cfg.rewards["overspeed"] = RewardTermCfg(
                func=microduck_mdp.overspeed_penalty,
                weight=-1.5,
                params={"vx_limit": 0.75, "vx_span": 0.5},
            )
        elif _stage == 2 and _os.environ.get("DIVEWALL_PHASE_A3"):
            # v3 (user 2026-10-14 12:10 GIF verdict: 没有跳、转头远离墙体):
            # v2 复盘 — ① track_angular_velocity 只罚角速度不罚航向角，
            # 鸭子慢慢转头零成本漂走；② hop 2.09 靠探索噪声，确定性均值
            # 动作仍是跑步步态。v3: 航向角锚定(heading_hold, 锚=出生朝墙
            # 朝向) + 跳跃分再加码把均值动作推向跳 + 熵升档重开探索。
            cfg.rewards["hop_forward"].weight = 20.0
            cfg.rewards["track_linear_velocity"].weight = 1.0
            cfg.commands["twist"].ranges.lin_vel_x = (0.25, 0.6)
            cfg.rewards["track_angular_velocity"] = RewardTermCfg(
                func=mdp.track_angular_velocity,
                weight=1.0,
                params={"command_name": "twist", "std": math.sqrt(0.25)},
            )
            cfg.rewards["heading_hold"] = RewardTermCfg(
                func=microduck_mdp.heading_hold_reward,
                weight=2.0,
                params={"std_deg": 15.0},
            )
            cfg.rewards.pop("air_time", None)
            cfg.rewards["air_time_height"] = RewardTermCfg(
                func=microduck_mdp.air_time_height_credit,
                weight=6.0,
                params={"z_thresh": 0.13},
            )
        elif _stage == 2 and _os.environ.get("DIVEWALL_PHASE_A2"):
            cfg.rewards["hop_forward"].weight = 15.0
            cfg.rewards["track_linear_velocity"].weight = 1.0
            cfg.commands["twist"].ranges.lin_vel_x = (0.25, 0.6)
            cfg.rewards["track_angular_velocity"] = RewardTermCfg(
                func=mdp.track_angular_velocity,
                weight=0.5,
                params={"command_name": "twist", "std": math.sqrt(0.25)},
            )
            cfg.rewards.pop("air_time", None)
            cfg.rewards["air_time_height"] = RewardTermCfg(
                func=microduck_mdp.air_time_height_credit,
                weight=3.0,
                params={"z_thresh": 0.13},
            )
    if _stage == 4:
        # Stage-4 v3 (user 2026-10-11 GIF verdict: 无助跑直接翻滚+倒地蹬脚):
        # ① v1/v2 直接切回 run-14b full 配置，v7 跑步配方 (track/pose/upright/
        #    air_time) 整个消失 → 跑步技能在 stage4 失去奖励支撑被遗忘。
        #    v3 继承降权副本 (同 stage2/3 的 stack-not-replace 写法)。
        # ② 墙距 0.85-1.2m 没有助跑空间 → 拉远到 2.5-3.5m。
        # ③ 墙高由 DIVEWALL_WALL_SINK 控制 (v3 训练脚本设 0.09 → 有效 3cm)。
        # ④ wall_push 加 vx 硬门 0.3 (堵"走到墙边扒墙"刷分)。
        # ⑤ premature_roll 改 vx 门控模式 (零速贴墙翻滚即罚)。
        _vx_band = (0.4, 1.2)
        _w_track, _w_pose, _w_up, _w_air = 1.0, 0.5, 0.5, 1.0
        _cmd = cfg.commands["twist"]
        _cmd.ranges.lin_vel_x = _vx_band
        _cmd.ranges.lin_vel_y = (0.0, 0.0)
        _cmd.ranges.ang_vel_z = (0.0, 0.0)
        _cmd.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S)
        cfg.rewards["track_linear_velocity"] = RewardTermCfg(
            func=mdp.track_linear_velocity,
            weight=_w_track,
            params={"command_name": "twist", "std": math.sqrt(0.1)},
        )
        cfg.rewards["pose"] = RewardTermCfg(
            func=mdp.variable_posture,
            weight=_w_pose,
            params={
                "std_standing": {
                    r".*hip_yaw.*": 0.1, r".*hip_roll.*": 0.05,
                    r".*hip_pitch.*": 0.15, r".*knee.*": 0.15,
                    r".*ankle.*": 0.1,
                },
                "std_walking": {
                    r".*hip_yaw.*": 0.3, r".*hip_roll.*": 0.05,
                    r".*hip_pitch.*": 0.4, r".*knee.*": 0.4,
                    r".*ankle.*": 0.25,
                },
                "std_running": {
                    r".*hip_yaw.*": 0.3, r".*hip_roll.*": 0.05,
                    r".*hip_pitch.*": 0.4, r".*knee.*": 0.4,
                    r".*ankle.*": 0.25,
                },
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=(r"^(?!passive_|.*neck.*|.*head.*).*",),
                ),
                "command_name": "twist",
                "walking_threshold": 0.01,
                "running_threshold": 1.5,
            },
        )
        cfg.rewards["upright"] = RewardTermCfg(
            func=mdp.upright,
            weight=_w_up,
            params={
                "std": math.sqrt(0.05),
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            },
        )
        cfg.rewards["air_time"] = RewardTermCfg(
            func=mdp.feet_air_time,
            weight=_w_air,
            params={
                "sensor_name": "feet_ground_contact",
                "threshold_min": 0.125,
                "threshold_max": 0.300,
            },
        )
        # ② 墙距拉远：留 2.5-3.5m 助跑段。
        cfg.events["reset_wall"].params["distance_range"] = (2.5, 3.5)
        # ④ wall_push vx 硬门：不带 ≥0.3 m/s 接近速度，扒墙零分。
        cfg.rewards["roulade_wall_push"].params.update({"vx_gate": 0.3})
        # ⑤ premature_roll vx 门控模式：贴墙零速翻滚照罚 (不再受 15cm 豁免)。
        # v3.1 (user 2026-10-11 中期复盘): 直立跑不该吃 roll 专用罚分 —
        #    roulade_sagittal 加直立豁免 (|pitch|<20° 清零)：步态 ω_x/ω_z 摆动
        #    不再每步被税 (当时最大负项 -0.045/ep, track 卡 0.02)；
        #    premature_dive air 线 15°→25°：跑步腾空段轻微前倾免费。
        cfg.rewards["roulade_sagittal"].params.update({"upright_exempt_deg": 20.0})
        cfg.rewards["roulade_wall_premature_dive"].params.update({"pitch_air_deg": 25.0})
        # v3.2 (user 2026-10-11 12:50 定位): spawn_mix 是 10-07 审计过的"趴地
        # 帮凶"，stages 1-3 已清但 stage-4 漏了 —— full 配置开局 50% midroll
        # 出生 + postdive 0.25，导致 warm 模型的跑步技能 (track 毕业值 0.85)
        # 在 v3/v3.1 从第 1 个 iter 就只有 0.007：一半回合从出生在翻滚。
        # 移除 spawn_mix，站立出生；postdive 反向 spawn 是唯一的非站立来源。
        cfg.curriculum.pop("roulade_spawn_mix", None)
        cfg.events["set_roulade_state"].params["standing_prob"] = 1.0
        # v3.3: wall_lateral 只守过墙瞬间 (|dist|<1m)，跑道段不再处决跑偏学员
        # (probe: 86% 回合 1 秒内死于此项，track 攒不起来 → 跑步学不回来)。
        # v3.5 (user 2026-10-12 GIF: 绕墙): 窗口 1→1.8m —— 绕路在投入成本前
        # 就死。v3.4 末期的策略已能保持直线 (runup 0.4+)，2s 窗口内偏出 0.7m
        # 是真绕行不是学步漂移。
        cfg.terminations["wall_lateral"].params.update({"near_dist": 1.8})
        # v3.4 (user 2026-10-11 GIF 验收: 跑步不错，但没有前扑/翻滚):
        # stage-4 full 配置缺 stage-2/3 教会前扑的两个功臣，按 v7e 毕业参数
        # 叠回来 —— hop_forward(vz×vx×pitch 0→60° ramp) + airborne_spin(前扑
        # 姿态下的空中旋转)。跑步配方已在，跳跃/翻滚学校叠上去。
        cfg.rewards["hop_forward"] = RewardTermCfg(
            func=microduck_mdp.hop_forward_credit,
            weight=6.0,
            params={"vz_scale": 0.4, "vx_scale": 1.0,
                    "pitch_min_deg": 0.0, "pitch_full_deg": 60.0,
                    "vx_gate": 0.15},
        )
        cfg.rewards["airborne_spin"] = RewardTermCfg(
            func=microduck_mdp.airborne_spin_credit,
            weight=2.0,
            params={"omega_scale": 3.0, "pitch_min_deg": 30.0, "vx_gate": 0.3},
        )
        # v3.5 (user 2026-10-11 21:00 GIF: 跑过去撞墙): takeoff 项 pitch 门
        # (10,50) + 20cm 窗口对跑步的鸭子是零梯度 — 跑步腾空段 pitch<10°，
        # 永远尝不到分，从"跑"到"墙前起跳"没有桥。坡道 0→40° + 窗口 35cm
        # (1m/s 跑速下 0.35s)，跑步的轻微前倾就开始计分。
        cfg.rewards["roulade_wall_takeoff"].params.update(
            {"pitch_min_deg": 0.0, "pitch_full_deg": 40.0, "zone_far": 0.35}
        )
        cfg.rewards["roulade_wall_premature_roll"].params.update({"vx_gate": 0.2})
        # Phase-B v1 (user 2026-10-14 22:41 verdict: 6 版跳跃学校均值动作
        # 仍是跑步 — 放弃无墙苦练, 真墙当老师: 有效墙高 0.5cm(脚本
        # WALL_SINK 0.115), 现有小跳(vz~0.44)足以越过, 用过墙大分逼出跳。
        # 航向锚定从 Phase-A 移植(reset_wall_ahead 存 _spawn_yaw 无条件);
        # takeoff pitch 门 20→10 起(v3.7 的 20° 对小跳零梯度)。
        if _os.environ.get("DIVEWALL_PHASE_B1"):
            cfg.rewards["heading_hold"] = RewardTermCfg(
                func=microduck_mdp.heading_hold_reward,
                weight=4.0,
                params={"std_deg": 10.0},
            )
            cfg.rewards["roulade_wall_takeoff"].params.update(
                {"pitch_min_deg": 10.0, "pitch_full_deg": 60.0}
            )
        # (Phase-B v3 blocks MOVED to end-of-function — the v3.7/v3.8 blocks
        #  below previously overwrote takeoff/runup/track and spawn distance,
        #  so B2/B3 tweaks silently never took effect. bug found 2026-10-16.)
        # v3.8 (user 2026-10-12 15:52 GIF: 见到墙原路返回): 墙前变成"只有罚没
        # 有赏"的死区 — takeoff 20° 门吃不到、flight 概率低、runup 接近分一次
        # 性吃完就没了 → 留在墙边只剩挨罚，逃跑成理性选择。减半恐惧 + 加倍
        # 前进拉力: collision -2→-1、shove -3→-1.5、runup 4→6、track 1→1.5。
        cfg.rewards["roulade_wall_collision"].weight = -1.0
        cfg.rewards["roulade_wall_shove"].weight = -1.5
        cfg.rewards["roulade_wall_runup"].weight = 6.0
        cfg.rewards["track_linear_velocity"].weight = 1.5
        # v3.7 (user 2026-10-12 13:25 GIF: 跑得快但没跳起来前扑): takeoff 物理
        # 上是 PITCH 问题 — 墙顶仅低于站立 COM ~2cm，直立跳永远过不去，躯干
        # 起跳时须已前倾 ≥60° (wall_takeoff_shaping docstring)。v3.5/3.6 坡道
        # 0→40° 让直立小跳也能吃分 → 没有动力真前扑。收紧为 20→60°：<20° 的
        # 直立腾空零分，40°+ 才接近全额，把梯度集中到"扑出去"。
        cfg.rewards["roulade_wall_takeoff"].params.update(
            {"pitch_min_deg": 20.0, "pitch_full_deg": 60.0}
        )
        # ── Phase-B v3 FINAL overrides (must be LAST — see note above) ──
        # user 2026-10-16: 起点太远(被 2.5-3.5m 覆盖)/跑步速度不够/没有前扑。
        # 近墙 0.40-0.65m + takeoff ramp 0→50° + runup 8 + track 2。
        if _os.environ.get("DIVEWALL_PHASE_B3"):
            _wd3 = _os.environ.get("DIVEWALL_WALL_DIST", "")
            cfg.events["reset_wall"].params["distance_range"] = (
                tuple(float(x) for x in _wd3.split(",")) if _wd3 else (0.40, 0.65))
            cfg.rewards["heading_hold"] = RewardTermCfg(
                func=microduck_mdp.heading_hold_reward,
                weight=4.0,
                params={"std_deg": 10.0},
            )
            cfg.rewards["roulade_wall_takeoff"].params.update(
                {"pitch_min_deg": 0.0, "pitch_full_deg": 50.0, "zone_far": 0.40}
            )
            cfg.rewards["roulade_wall_runup"].weight = 8.0
            cfg.rewards["track_linear_velocity"].weight = 2.0
        # ── Phase-B v7 (2026-10-17): belly-crawl exploit fix ──
        # B6 probe: 6/8 ducks crossed the 2cm wall PRONE (z≈0.07, zero
        # airborne steps) — crawl over, stand up past the wall, farm the
        # cleared drip. Gate the drip on an in-flight crossing (flight latch)
        # so only a real jump pays. Wall raised to real 5cm (sink 0.07 via
        # script) to physically block crawling.
        if _os.environ.get("DIVEWALL_PHASE_B7"):
            cfg.rewards["roulade_wall_cleared"].params["air_cross_gate"] = True
        # ── Phase-B v8 (2026-10-17, user: "依旧没有前扑") ──
        # B7: hops at z~0.13 in front of the wall, 0/8 crossed, no pounce.
        # ① flight 大奖降到够得着: wall-top-relative (5cm 墙 → z≥0.09 腾空)
        # ② takeoff 加 vx 硬门 0.8 + pitch 门 20→60°: 原地跳零分, 必须
        #    带速度前扑起跳才计分。
        if _os.environ.get("DIVEWALL_PHASE_B8"):
            cfg.rewards["roulade_wall_cleared"].params["air_cross_gate"] = True
            cfg.rewards["roulade_wall_flight"].params.update(
                {"min_height": 0.09, "clearance": 0.0})
            cfg.rewards["roulade_wall_takeoff"].params.update(
                {"vx_gate": 0.8, "pitch_min_deg": 20.0, "pitch_full_deg": 60.0,
                 # user 15:10 GIF verdict: "提前跳 → 落在墙上 → 挂墙蹬腿"。
                 # B3 的 zone_far 0.40 付早跳分, 飞 ~30cm 正好砸墙顶。
                 # 收窄到墙前 2-15cm: 只有贴墙晚跳计分。
                 "zone_near": 0.02, "zone_far": 0.15})
        # ── Phase-B v9 (2026-10-17 18:26, user: "头挂墙蹬腿, 墙再降低") ──
        # 同 B8 四重门, 墙降回 2cm (sink 0.10, 真实几何): 腾空门槛不变,
        # 先在低墙把"腾空过墙"打成稳定行为, 再逐级升墙。
        if _os.environ.get("DIVEWALL_PHASE_B9"):
            cfg.rewards["roulade_wall_cleared"].params["air_cross_gate"] = True
            cfg.rewards["roulade_wall_flight"].params.update(
                {"min_height": 0.08, "clearance": 0.0})
            cfg.rewards["roulade_wall_takeoff"].params.update(
                {"vx_gate": 0.6, "pitch_min_deg": 10.0, "pitch_full_deg": 45.0,
                 "zone_near": 0.02, "zone_far": 0.15})
        # ── Phase-B v10 (2026-10-17 22:31, user: "前扑前没有跳, 前扑后未跟进
        # 翻滚") ── B9 已 4/8 过墙但动作链断在两头。B10 = B9 + ① takeoff
        # 加 vz 硬门 0.25 (必须先真跳再前扑, 贴地平扑零分) ② 过墙后翻滚链
        # 加权 (progress 8→12 / landing 4→8 / upright 1.5→4), 逼出"扑过去
        # 接滚再站起"的完整 follow-through。
        if _os.environ.get("DIVEWALL_PHASE_B10"):
            cfg.rewards["roulade_wall_cleared"].params["air_cross_gate"] = True
            cfg.rewards["roulade_wall_flight"].params.update(
                {"min_height": 0.08, "clearance": 0.0})
            cfg.rewards["roulade_wall_takeoff"].params.update(
                {"vx_gate": 0.6, "vz_gate": 0.25,
                 "pitch_min_deg": 10.0, "pitch_full_deg": 45.0,
                 "zone_near": 0.02, "zone_far": 0.15})
            if "roulade_progress" in cfg.rewards:
                cfg.rewards["roulade_progress"].weight = 12.0
            if "roulade_landing_composite" in cfg.rewards:
                cfg.rewards["roulade_landing_composite"].weight = 8.0
            if "roulade_upright_after_roll" in cfg.rewards:
                cfg.rewards["roulade_upright_after_roll"].weight = 4.0
        # ── Phase-B v11 (2026-10-18 11:35, user: "还缺接翻滚起立过去") ──
        # B10b probe 8/8 干净过墙 (flight 0.0596) 但落地后全部躺平: landing/
        # upright/height/sharp 四层全 0。根因分析: ① 落地门 260-330° 在
        # 滚过 330° 后关闭, progress 0.92 (~331°) 恰好错过整个 annuity 窗
        # ② 各层权重仍低, 梯度被 push/flight 占据。B11 = B10 同配置 +
        # ① 落地门 hi 330→360 (滚过头也持续付站起分) ② 站起栈全面加权
        # (composite 8→12 / upright 4→8 / height 1→4 / sharp 2→4)。
        if _os.environ.get("DIVEWALL_PHASE_B11"):
            cfg.rewards["roulade_wall_cleared"].params["air_cross_gate"] = True
            cfg.rewards["roulade_wall_flight"].params.update(
                {"min_height": 0.08, "clearance": 0.0})
            cfg.rewards["roulade_wall_takeoff"].params.update(
                {"vx_gate": 0.6, "vz_gate": 0.25,
                 "pitch_min_deg": 10.0, "pitch_full_deg": 45.0,
                 "zone_near": 0.02, "zone_far": 0.15})
            _gate_hi_360 = math.radians(360.0)
            if "roulade_progress" in cfg.rewards:
                cfg.rewards["roulade_progress"].weight = 12.0
            if "roulade_landing_composite" in cfg.rewards:
                cfg.rewards["roulade_landing_composite"].weight = 12.0
                cfg.rewards["roulade_landing_composite"].params["gate_hi"] = _gate_hi_360
            if "roulade_upright_after_roll" in cfg.rewards:
                cfg.rewards["roulade_upright_after_roll"].weight = 8.0
                cfg.rewards["roulade_upright_after_roll"].params["gate_hi"] = _gate_hi_360
            if "roulade_height_after_roll" in cfg.rewards:
                cfg.rewards["roulade_height_after_roll"].weight = 4.0
                cfg.rewards["roulade_height_after_roll"].params["gate_hi"] = _gate_hi_360
            if "roulade_landing_sharp" in cfg.rewards:
                cfg.rewards["roulade_landing_sharp"].weight = 4.0
                cfg.rewards["roulade_landing_sharp"].params["gate_hi"] = _gate_hi_360
            if "roulade_stand_tax" in cfg.rewards:
                cfg.rewards["roulade_stand_tax"].params["gate_hi"] = _gate_hi_360
        # ── Phase-B v12 (2026-10-18 14:05, diag: accum frontier 卡在 ~173°) ──
        # 诊断 probe (probe_diag_roulade.py): head_latch=1, frontier 0.96π≈173°,
        # 落地后翻到仰面就停 — 后半圈 (supine→起身) 从未发生。rise_velocity
        # gate_lo 180° / airborne gate_lo 260° 全部够不着 → 起身链死项。
        # B12 = B11 + ① rise gate_lo 180→150°, weight 0.75→3 (仰面就有起身
        # 梯度) ② airborne (打挺) gate_lo 260→180°, weight 1→3。
        if _os.environ.get("DIVEWALL_PHASE_B12"):
            cfg.rewards["roulade_wall_cleared"].params["air_cross_gate"] = True
            cfg.rewards["roulade_wall_flight"].params.update(
                {"min_height": 0.08, "clearance": 0.0})
            cfg.rewards["roulade_wall_takeoff"].params.update(
                {"vx_gate": 0.6, "vz_gate": 0.25,
                 "pitch_min_deg": 10.0, "pitch_full_deg": 45.0,
                 "zone_near": 0.02, "zone_far": 0.15})
            _gate_hi_360 = math.radians(360.0)
            if "roulade_progress" in cfg.rewards:
                cfg.rewards["roulade_progress"].weight = 12.0
            if "roulade_landing_composite" in cfg.rewards:
                cfg.rewards["roulade_landing_composite"].weight = 12.0
                cfg.rewards["roulade_landing_composite"].params["gate_hi"] = _gate_hi_360
            if "roulade_upright_after_roll" in cfg.rewards:
                cfg.rewards["roulade_upright_after_roll"].weight = 8.0
                cfg.rewards["roulade_upright_after_roll"].params["gate_hi"] = _gate_hi_360
            if "roulade_height_after_roll" in cfg.rewards:
                cfg.rewards["roulade_height_after_roll"].weight = 4.0
                cfg.rewards["roulade_height_after_roll"].params["gate_hi"] = _gate_hi_360
            if "roulade_landing_sharp" in cfg.rewards:
                cfg.rewards["roulade_landing_sharp"].weight = 4.0
                cfg.rewards["roulade_landing_sharp"].params["gate_hi"] = _gate_hi_360
            if "roulade_stand_tax" in cfg.rewards:
                cfg.rewards["roulade_stand_tax"].params["gate_hi"] = _gate_hi_360
            if "roulade_rise_velocity" in cfg.rewards:
                cfg.rewards["roulade_rise_velocity"].weight = 3.0
                cfg.rewards["roulade_rise_velocity"].params["gate_lo"] = math.radians(150.0)
            if "roulade_airborne" in cfg.rewards:
                cfg.rewards["roulade_airborne"].weight = 3.0
                cfg.rewards["roulade_airborne"].params["gate_lo"] = math.radians(180.0)
        # ── Phase-B v13 (2026-10-18 22:30, B12b 加练平台) ──
        # B12b 12000 iter 加练后 rise 分回落 (0.0093→0.0058), 站立率停在
        # ~2/8 —— 起身梯度有但不解渴。上第二杠杆: stand_tax (躺=负分)
        # 门 260→180° 提前生效 + 权重 5→8: 过墙落地翻到仰面后, 每步
        # 躺着都在流血, 起身从"可选"变"必选"。
        if _os.environ.get("DIVEWALL_PHASE_B13"):
            cfg.rewards["roulade_wall_cleared"].params["air_cross_gate"] = True
            cfg.rewards["roulade_wall_flight"].params.update(
                {"min_height": 0.08, "clearance": 0.0})
            cfg.rewards["roulade_wall_takeoff"].params.update(
                {"vx_gate": 0.6, "vz_gate": 0.25,
                 "pitch_min_deg": 10.0, "pitch_full_deg": 45.0,
                 "zone_near": 0.02, "zone_far": 0.15})
            _gate_hi_360 = math.radians(360.0)
            if "roulade_progress" in cfg.rewards:
                cfg.rewards["roulade_progress"].weight = 12.0
            if "roulade_landing_composite" in cfg.rewards:
                cfg.rewards["roulade_landing_composite"].weight = 12.0
                cfg.rewards["roulade_landing_composite"].params["gate_hi"] = _gate_hi_360
            if "roulade_upright_after_roll" in cfg.rewards:
                cfg.rewards["roulade_upright_after_roll"].weight = 8.0
                cfg.rewards["roulade_upright_after_roll"].params["gate_hi"] = _gate_hi_360
            if "roulade_height_after_roll" in cfg.rewards:
                cfg.rewards["roulade_height_after_roll"].weight = 4.0
                cfg.rewards["roulade_height_after_roll"].params["gate_hi"] = _gate_hi_360
            if "roulade_landing_sharp" in cfg.rewards:
                cfg.rewards["roulade_landing_sharp"].weight = 4.0
                cfg.rewards["roulade_landing_sharp"].params["gate_hi"] = _gate_hi_360
            if "roulade_rise_velocity" in cfg.rewards:
                cfg.rewards["roulade_rise_velocity"].weight = 3.0
                cfg.rewards["roulade_rise_velocity"].params["gate_lo"] = math.radians(150.0)
            if "roulade_airborne" in cfg.rewards:
                cfg.rewards["roulade_airborne"].weight = 3.0
                cfg.rewards["roulade_airborne"].params["gate_lo"] = math.radians(180.0)
            if "roulade_stand_tax" in cfg.rewards:
                cfg.rewards["roulade_stand_tax"].weight = 8.0
                cfg.rewards["roulade_stand_tax"].params.update(
                    {"gate_lo": math.radians(180.0), "gate_hi": _gate_hi_360})
        # ── Phase-B v14 (2026-10-19, B13 退化回滚) ──
        # B13 过夜诊断: frontier 卡 0.52π≈94°, final_z 0.067 (趴着),
        # rise 分 0.0058→0.0005 崩盘 —— stand_tax 权重 8 只罚不教, 把
        # B12b 冒头的起身苗头压没了。真正的洞: 趴下(90°)到起翻(150°)
        # 之间整段奖励真空, duck 不知道怎么从趴姿收头发力。修复:
        #   ① preroll_kick (weight 2): 20°-150° 窗口内 fold(收头贴地
        #      annuity 0.6) + kick(支撑前倾角速度 0.4) — 给"趴下→准备
        #      翻"补梯度;
        #   ② stand_tax 8→3 回落, 别再压制探索;
        #   ③ warm 回退到 B12b model_182474 (B13 ckpt 退化不用)。
        if _os.environ.get("DIVEWALL_PHASE_B14"):
            cfg.rewards["roulade_wall_cleared"].params["air_cross_gate"] = True
            cfg.rewards["roulade_wall_flight"].params.update(
                {"min_height": 0.08, "clearance": 0.0})
            cfg.rewards["roulade_wall_takeoff"].params.update(
                {"vx_gate": 0.6, "vz_gate": 0.25,
                 "pitch_min_deg": 10.0, "pitch_full_deg": 45.0,
                 "zone_near": 0.02, "zone_far": 0.15})
            _gate_hi_360 = math.radians(360.0)
            if "roulade_progress" in cfg.rewards:
                cfg.rewards["roulade_progress"].weight = 12.0
            if "roulade_landing_composite" in cfg.rewards:
                cfg.rewards["roulade_landing_composite"].weight = 12.0
                cfg.rewards["roulade_landing_composite"].params["gate_hi"] = _gate_hi_360
            if "roulade_upright_after_roll" in cfg.rewards:
                cfg.rewards["roulade_upright_after_roll"].weight = 8.0
                cfg.rewards["roulade_upright_after_roll"].params["gate_hi"] = _gate_hi_360
            if "roulade_height_after_roll" in cfg.rewards:
                cfg.rewards["roulade_height_after_roll"].weight = 4.0
                cfg.rewards["roulade_height_after_roll"].params["gate_hi"] = _gate_hi_360
            if "roulade_landing_sharp" in cfg.rewards:
                cfg.rewards["roulade_landing_sharp"].weight = 4.0
                cfg.rewards["roulade_landing_sharp"].params["gate_hi"] = _gate_hi_360
            if "roulade_rise_velocity" in cfg.rewards:
                cfg.rewards["roulade_rise_velocity"].weight = 3.0
                cfg.rewards["roulade_rise_velocity"].params["gate_lo"] = math.radians(150.0)
            if "roulade_airborne" in cfg.rewards:
                cfg.rewards["roulade_airborne"].weight = 3.0
                cfg.rewards["roulade_airborne"].params["gate_lo"] = math.radians(180.0)
            if "roulade_preroll_kick" in cfg.rewards:
                cfg.rewards["roulade_preroll_kick"].weight = 2.0
            if "roulade_stand_tax" in cfg.rewards:
                cfg.rewards["roulade_stand_tax"].weight = 3.0
                cfg.rewards["roulade_stand_tax"].params.update(
                    {"gate_lo": math.radians(180.0), "gate_hi": _gate_hi_360})
        # ── Phase-B v17 (2026-10-20, 方向2: 放宽允许斜向肩滚) ──
        # B13-B16 frontier 全卡 90-140°, 奖励侧手段 (preroll梯度/罚躺/
        # 空中结转/陡扑) 试遍未破 180°。B17 换假设: 不是不会滚, 是形貌
        # (扁喙圆身)翻不过头顶正上方——肩-头对角线滚物理上容易得多。
        # mdp 侧 DIVEWALL_PHASE_B17 已放宽 flatness gate (45°/75°) 并撤销
        # head-top 严判; 此处把三个"拉回矢状面"的罚项清零, 否则 gate 放
        # 了奖励端还在罚, 等于没放。其余=B14 全配置。
        if _os.environ.get("DIVEWALL_PHASE_B17"):
            cfg.rewards["roulade_wall_cleared"].params["air_cross_gate"] = True
            cfg.rewards["roulade_wall_flight"].params.update(
                {"min_height": 0.08, "clearance": 0.0})
            cfg.rewards["roulade_wall_takeoff"].params.update(
                {"vx_gate": 0.6, "vz_gate": 0.25,
                 "pitch_min_deg": 10.0, "pitch_full_deg": 45.0,
                 "zone_near": 0.02, "zone_far": 0.15})
            _gate_hi_360 = math.radians(360.0)
            if "roulade_progress" in cfg.rewards:
                cfg.rewards["roulade_progress"].weight = 12.0
            if "roulade_landing_composite" in cfg.rewards:
                cfg.rewards["roulade_landing_composite"].weight = 12.0
                cfg.rewards["roulade_landing_composite"].params["gate_hi"] = _gate_hi_360
            if "roulade_upright_after_roll" in cfg.rewards:
                cfg.rewards["roulade_upright_after_roll"].weight = 8.0
                cfg.rewards["roulade_upright_after_roll"].params["gate_hi"] = _gate_hi_360
            if "roulade_height_after_roll" in cfg.rewards:
                cfg.rewards["roulade_height_after_roll"].weight = 4.0
                cfg.rewards["roulade_height_after_roll"].params["gate_hi"] = _gate_hi_360
            if "roulade_landing_sharp" in cfg.rewards:
                cfg.rewards["roulade_landing_sharp"].weight = 4.0
                cfg.rewards["roulade_landing_sharp"].params["gate_hi"] = _gate_hi_360
            if "roulade_rise_velocity" in cfg.rewards:
                cfg.rewards["roulade_rise_velocity"].weight = 3.0
                cfg.rewards["roulade_rise_velocity"].params["gate_lo"] = math.radians(150.0)
            if "roulade_airborne" in cfg.rewards:
                cfg.rewards["roulade_airborne"].weight = 3.0
                cfg.rewards["roulade_airborne"].params["gate_lo"] = math.radians(180.0)
            if "roulade_preroll_kick" in cfg.rewards:
                cfg.rewards["roulade_preroll_kick"].weight = 2.0
            if "roulade_stand_tax" in cfg.rewards:
                cfg.rewards["roulade_stand_tax"].weight = 3.0
                cfg.rewards["roulade_stand_tax"].params.update(
                    {"gate_lo": math.radians(180.0), "gate_hi": _gate_hi_360})
            # 肩滚放行: 三个矢状面罚项清零 (sagittal/flatness/lateral_vel)
            if "roulade_sagittal" in cfg.rewards:
                cfg.rewards["roulade_sagittal"].weight = 0.0
            if "roulade_flatness" in cfg.rewards:
                cfg.rewards["roulade_flatness"].weight = 0.0
            if "roulade_lateral_vel" in cfg.rewards:
                cfg.rewards["roulade_lateral_vel"].weight = 0.0
        # ── Phase-B v18 (2026-10-21, 静态物理扫描后的双杠杆) ──
        # B17 证伪了 "gate 是卡点" (放宽后 frontier 仍卡 90-140°)。静态扫描
        # (scripts/probe_static_roll_physics.py, logs/static_roll_sweep.csv)
        # 结论: 滚通物理可行, 门速度≈3 rad/s; 180-220° 仰躺是稳定势阱,
        # 210° (趴地脸朝下) 是最难窗口——过顶后需持续发力, 而现有奖励只付
        # frontier 增量, 停滞时零梯度。B18 = B14 全配置 + 两项叠加:
        #   ① midroll spawn (DIVEWALL_B18_MIDROLL_FRAC, 默认 0.4): 部分 env
        #      直接 spawn 在墙后 roll 165-270° 后半圈 (扫描证明可滚通), 带
        #      2-5 rad/s 前滚初速, accumulator 从 spawn 角起算 — 反向课程
        #      给 210° 窗口喂 on-policy 数据。复用 postdive_spawn 机制。
        #   ② roulade_overhead_drive (weight 1.5): accum 180-240° 窗口内,
        #      接地状态下奖励保持前滚角速度 clip(ω/3, 0, 1) — 给最难窗口
        #      一个 rate 型梯度, 过顶不停手。past-wall 门控, 空中不计。
        if _os.environ.get("DIVEWALL_PHASE_B18"):
            cfg.rewards["roulade_wall_cleared"].params["air_cross_gate"] = True
            cfg.rewards["roulade_wall_flight"].params.update(
                {"min_height": 0.08, "clearance": 0.0})
            cfg.rewards["roulade_wall_takeoff"].params.update(
                {"vx_gate": 0.6, "vz_gate": 0.25,
                 "pitch_min_deg": 10.0, "pitch_full_deg": 45.0,
                 "zone_near": 0.02, "zone_far": 0.15})
            _gate_hi_360 = math.radians(360.0)
            if "roulade_progress" in cfg.rewards:
                cfg.rewards["roulade_progress"].weight = 12.0
            if "roulade_landing_composite" in cfg.rewards:
                cfg.rewards["roulade_landing_composite"].weight = 12.0
                cfg.rewards["roulade_landing_composite"].params["gate_hi"] = _gate_hi_360
            if "roulade_upright_after_roll" in cfg.rewards:
                cfg.rewards["roulade_upright_after_roll"].weight = 8.0
                cfg.rewards["roulade_upright_after_roll"].params["gate_hi"] = _gate_hi_360
            if "roulade_height_after_roll" in cfg.rewards:
                cfg.rewards["roulade_height_after_roll"].weight = 4.0
                cfg.rewards["roulade_height_after_roll"].params["gate_hi"] = _gate_hi_360
            if "roulade_landing_sharp" in cfg.rewards:
                cfg.rewards["roulade_landing_sharp"].weight = 4.0
                cfg.rewards["roulade_landing_sharp"].params["gate_hi"] = _gate_hi_360
            if "roulade_rise_velocity" in cfg.rewards:
                cfg.rewards["roulade_rise_velocity"].weight = 3.0
                cfg.rewards["roulade_rise_velocity"].params["gate_lo"] = math.radians(150.0)
            if "roulade_airborne" in cfg.rewards:
                cfg.rewards["roulade_airborne"].weight = 3.0
                cfg.rewards["roulade_airborne"].params["gate_lo"] = math.radians(180.0)
            if "roulade_preroll_kick" in cfg.rewards:
                cfg.rewards["roulade_preroll_kick"].weight = 2.0
            if "roulade_stand_tax" in cfg.rewards:
                cfg.rewards["roulade_stand_tax"].weight = 3.0
                cfg.rewards["roulade_stand_tax"].params.update(
                    {"gate_lo": math.radians(180.0), "gate_hi": _gate_hi_360})
            # ① 过顶续速: 210°±30° 窗口, 接地 + past-wall 才计
            cfg.rewards["roulade_overhead_drive"] = RewardTermCfg(
                func=microduck_mdp.roulade_overhead_drive,
                weight=1.5,
                params={"gate_lo": math.radians(180.0),
                        "gate_hi": math.radians(240.0),
                        "rate_norm": 3.0,
                        "wall_name": "wall"},
            )
            # ② midroll spawn: 后半圈 (165-270°) + 2-5 rad/s 前滚初速,
            #    墙后 0.15-0.40m 贴地。复用 postdive_spawn (机制同源:
            #    accumulator 从 spawn 角起算, head latch 预授, rise 时钟
            #    对 ≥260° 的 spawn 即刻启动)。事件在 reset_wall 之后插入。
            _b18_frac = float(_os.environ.get("DIVEWALL_B18_MIDROLL_FRAC", "0.4") or 0.4)
            cfg.events["midroll_spawn"] = EventTermCfg(
                func=microduck_mdp.postdive_spawn,
                mode="reset",
                params={
                    "prob": _b18_frac,
                    "pitch_min_deg": 165.0,
                    "pitch_max_deg": 270.0,
                    "z_min": 0.05, "z_max": 0.09,
                    "past_dist_range": (0.15, 0.40),
                    "omega_range": (2.0, 5.0),
                    "tuck_overrides": TUCK_OVERRIDES,
                },
            )
        # ── B18b 补丁 (2026-10-27 21:50, lean-B19 GIF verdict "小跳过墙、
        # 半翻滚顶地抬脚拿reward", 大人拍板从 B18 重调):
        #   ① takeoff 陡扑门前移 (pitch 10/45→20/65, w6→12): 小跳无前倾
        #      角, takeoff 分归零, 逼跑→大角度扑出;
        #   ② rise/airborne 完成度门 (rise gate 150→260, airborne 180→260):
        #      半翻滚 (accum<260°) 后的顶地抬脚打挺不再收 rise/airborne
        #      分, 起身梯度滚完 3/4 圈才解锁, 与 landing 栈同门。
        #   历史依据: B14 era 曾记 rise 180-260 为死项, 但那是老链;
        #   lean 链证明低门反而成了套利口。overhead_drive (180-240°)
        #   保留, 半翻窗口仍有"继续滚"的梯度。
        if _os.environ.get("DIVEWALL_PHASE_B18B"):
            cfg.rewards["roulade_wall_takeoff"].params.update(
                {"pitch_min_deg": 20.0, "pitch_full_deg": 65.0})
            cfg.rewards["roulade_wall_takeoff"].weight = 12.0
            if "roulade_rise_velocity" in cfg.rewards:
                cfg.rewards["roulade_rise_velocity"].params["gate_lo"] = math.radians(260.0)
            if "roulade_airborne" in cfg.rewards:
                cfg.rewards["roulade_airborne"].params["gate_lo"] = math.radians(260.0)
        # ── B18c 补丁 (2026-10-29, B18b GIF verdict "头顶地+脚蹬地划过墙,
        # 无翻滚": rise/airborne 260° 门已生效但过墙三件套不看翻滚, 作弊
        # 路径 takeoff(w12)+飞行jackpot+过墙年金照单全收). 两手抓:
        #   ① 飞行 jackpot 延迟支付 (roll_gate 150→260°+head latch):
        #      墙前飞行划过只 latch 不给钱, 翻滚真完成后一次性支付 —
        #      过墙年金 (air_cross_gate 接同一 latch) 随之自动锁死;
        #   ② roulade_headstand_penalty (w 2.0): 头触地+倒置(up_z<-0.707)
        #      +frontier<100° 每步罚 — 真翻滚经过倒置段时 accum 已>90°,
        #      不误伤; 静态倒立/蹬地滑行则持续流血。作弊口袋变坑。
        # takeoff 陡扑门维持 B18b (20/65°+w12) 不动 — 那是正确行为的梯度。
        if _os.environ.get("DIVEWALL_PHASE_B18C"):
            cfg.rewards["roulade_wall_flight"].params.update(
                {"roll_gate_lo_deg": 150.0, "roll_gate_hi_deg": 260.0})
            cfg.rewards["roulade_headstand_penalty"] = RewardTermCfg(
                func=microduck_mdp.roulade_headstand_penalty,
                weight=2.0,
                params={"pitch_min_deg": 135.0, "accum_max_deg": 100.0},
            )
        # ── B18D (2026-10-30, 已废弃): B14+续滚三件套路线, 8000 iter 训完
        # reward 35.7, GIF verdict 不符合预期, 大人 13:29 拍板删除。产物已移
        # ../trash_b18d_20261030/。教训: 无 midroll 的纯 reward 杠杆 (扩窗
        # overhead_drive + 延迟 jackpot + headstand 罚) 不足以带出第二阶段。
        # ── Phase-B v19 (2026-10-22, B18 验收"翻滚成立、跳前扑不足") ──
        # 小改动路线 (大人 09:45 拍板): B18 全配置 + 两项叠加:
        #   ① midroll spawn 0.4 → 0.22 (脚本 env 覆盖, 本块不改): B18 的
        #      40% midroll 稀释了前扑样本, 翻滚已成立可降比例还样本给
        #      takeoff/flight 段。
        #   ② roulade_flight_pitch (w 1.5): 空中 (无脚/无墙接触) + 墙前
        #      -0.40m 到墙后 +0.45m 窗口 + vx≥0.5 门, 只奖前倾姿态 ramp
        #      (15°→60°), 无任何旋转因子 — 补前扑梯度且规避 B15 空中
        #      刷旋转套利。
        if _os.environ.get("DIVEWALL_PHASE_B19"):
            cfg.rewards["roulade_flight_pitch"] = RewardTermCfg(
                func=microduck_mdp.roulade_flight_pitch,
                weight=1.5,
                params={
                    "pitch_min_deg": 15.0,
                    "pitch_full_deg": 60.0,
                    "zone_near": -0.40,
                    "zone_far": 0.45,
                    "vx_gate": 0.5,
                },
            )
        # ── Phase-B v20 (2026-10-22, B19 验收"全链成立: 跑→跳前扑→翻滚→跑",
        # 但"跳还不足、跳/扑衔接生硬") ──
        # B19 全配置 + 两项:
        #   ① 重启 B16 陡扑 takeoff 门: pitch_min 10→20°, pitch_full 45→65°
        #      — B16 当年不敢用是没翻滚兜底, 现有 B18/B19 翻滚能力保底,
        #      逼它带大前倾角扑出, 跳得更高更舒展;
        #   ② wall_takeoff weight 6.0 → 12.0: B19 末期该项仅 0.0005,
        #      "跳得足"在 reward 里几乎无存在感, 抬一档。
        if _os.environ.get("DIVEWALL_PHASE_B20"):
            cfg.rewards["roulade_wall_takeoff"].params.update(
                {"pitch_min_deg": 20.0, "pitch_full_deg": 65.0})
            cfg.rewards["roulade_wall_takeoff"].weight = 12.0
        # ── Phase-B v22 (2026-10-23, B21 验收"整体可以, 有时翻不过去;
        # 助跑要长, 翻过去后少跑") ──
        # B21 全配置 + 一项 (墙距由脚本 DIVEWALL_WALL_DIST 拉远, 不占 env flag):
        #   成功终止 — 过墙 0.35m + upright≥0.8 + 站高≥0.10 → 提前结束。
        #   翻滚后的长跑不再刷 track_linear_velocity; 助跑拉远后跑满速再跳。
        if _os.environ.get("DIVEWALL_PHASE_B22"):
            cfg.terminations["roulade_success"] = TerminationTermCfg(
                func=microduck_mdp.roulade_success_termination,
                params={
                    "past_margin": 0.35,
                    "upright_min": 0.80,
                    "height_min": 0.10,
                },
            )
        # ── Phase-B v23 (2026-10-23, B22 验收"翻滚太早, 没到墙就前扑") ──
        # 根因: actor 看不见墙 (本体感觉设计), 起跳时机是步数节奏; B21 在
        # 0.40-0.65m 短助跑练出的节奏, B22 拉到 0.90-1.30m 节奏没跟上。
        # B23: premature_dive 免罚区 0.30→0.45 (更早罚墙前扑); 助跑折中
        # 0.65-0.95 由脚本 WALL_DIST 控制; warm 回接 B21 (非 B22)。
        if _os.environ.get("DIVEWALL_PHASE_B23"):
            cfg.rewards["roulade_wall_premature_dive"].params.update(
                {"near_dist": 0.45})
            cfg.rewards["roulade_wall_premature_roll"].params.update(
                {"near_dist": 0.45})
        # ── Phase-B v16 (2026-10-19 晚, GIF 反馈"扑倒→蹬腿翻转"两截) ──
        # B15 空中结转失败已回滚 (accumulator 行为还原, 本块不含 B15 改动)。
        # 大人反馈: 动作是"扑倒死一下→蹬腿翻回来"两段拼接; 目标是
        # 跑→跳前扑→带旋转落地→顺势滚起来 一条龙。断点=落地瞬间旋转
        # 动量为零。B16 = B14 全配置 + 入场姿态更陡:
        #   takeoff pitch_min 10→20°, pitch_full 45→65° — 逼它带大前倾
        #   角度扑出, 落地时身体已翻过半, 蹬腿从"补救"变"顺势滚过"。
        if _os.environ.get("DIVEWALL_PHASE_B16"):
            cfg.rewards["roulade_wall_cleared"].params["air_cross_gate"] = True
            cfg.rewards["roulade_wall_flight"].params.update(
                {"min_height": 0.08, "clearance": 0.0})
            cfg.rewards["roulade_wall_takeoff"].params.update(
                {"vx_gate": 0.6, "vz_gate": 0.25,
                 "pitch_min_deg": 20.0, "pitch_full_deg": 65.0,
                 "zone_near": 0.02, "zone_far": 0.15})
            _gate_hi_360 = math.radians(360.0)
            if "roulade_progress" in cfg.rewards:
                cfg.rewards["roulade_progress"].weight = 12.0
            if "roulade_landing_composite" in cfg.rewards:
                cfg.rewards["roulade_landing_composite"].weight = 12.0
                cfg.rewards["roulade_landing_composite"].params["gate_hi"] = _gate_hi_360
            if "roulade_upright_after_roll" in cfg.rewards:
                cfg.rewards["roulade_upright_after_roll"].weight = 8.0
                cfg.rewards["roulade_upright_after_roll"].params["gate_hi"] = _gate_hi_360
            if "roulade_height_after_roll" in cfg.rewards:
                cfg.rewards["roulade_height_after_roll"].weight = 4.0
                cfg.rewards["roulade_height_after_roll"].params["gate_hi"] = _gate_hi_360
            if "roulade_landing_sharp" in cfg.rewards:
                cfg.rewards["roulade_landing_sharp"].weight = 4.0
                cfg.rewards["roulade_landing_sharp"].params["gate_hi"] = _gate_hi_360
            if "roulade_rise_velocity" in cfg.rewards:
                cfg.rewards["roulade_rise_velocity"].weight = 3.0
                cfg.rewards["roulade_rise_velocity"].params["gate_lo"] = math.radians(150.0)
            if "roulade_airborne" in cfg.rewards:
                cfg.rewards["roulade_airborne"].weight = 3.0
                cfg.rewards["roulade_airborne"].params["gate_lo"] = math.radians(180.0)
            if "roulade_preroll_kick" in cfg.rewards:
                cfg.rewards["roulade_preroll_kick"].weight = 2.0
            if "roulade_stand_tax" in cfg.rewards:
                cfg.rewards["roulade_stand_tax"].weight = 3.0
                cfg.rewards["roulade_stand_tax"].params.update(
                    {"gate_lo": math.radians(180.0), "gate_hi": _gate_hi_360})
        # ── Phase-B v15 (2026-10-19, 前扑旋转动量结转) ──
        # B14 诊断: head_latch=1 但 frontier 冻在 ~0.77π — 扑是平飞, 落地
        # 旋转全被卸掉, 趴姿重新发力拧不过头顶。B15 在 mdp 侧改
        # accumulator (DIVEWALL_PHASE_B15): 墙后空中前倾旋转计入 accum,
        # 墙前空中仍为 0。奖励权重与 B14 完全一致。
        if _os.environ.get("DIVEWALL_PHASE_B15"):
            cfg.rewards["roulade_wall_cleared"].params["air_cross_gate"] = True
            cfg.rewards["roulade_wall_flight"].params.update(
                {"min_height": 0.08, "clearance": 0.0})
            cfg.rewards["roulade_wall_takeoff"].params.update(
                {"vx_gate": 0.6, "vz_gate": 0.25,
                 "pitch_min_deg": 10.0, "pitch_full_deg": 45.0,
                 "zone_near": 0.02, "zone_far": 0.15})
            _gate_hi_360 = math.radians(360.0)
            if "roulade_progress" in cfg.rewards:
                cfg.rewards["roulade_progress"].weight = 12.0
            if "roulade_landing_composite" in cfg.rewards:
                cfg.rewards["roulade_landing_composite"].weight = 12.0
                cfg.rewards["roulade_landing_composite"].params["gate_hi"] = _gate_hi_360
            if "roulade_upright_after_roll" in cfg.rewards:
                cfg.rewards["roulade_upright_after_roll"].weight = 8.0
                cfg.rewards["roulade_upright_after_roll"].params["gate_hi"] = _gate_hi_360
            if "roulade_height_after_roll" in cfg.rewards:
                cfg.rewards["roulade_height_after_roll"].weight = 4.0
                cfg.rewards["roulade_height_after_roll"].params["gate_hi"] = _gate_hi_360
            if "roulade_landing_sharp" in cfg.rewards:
                cfg.rewards["roulade_landing_sharp"].weight = 4.0
                cfg.rewards["roulade_landing_sharp"].params["gate_hi"] = _gate_hi_360
            if "roulade_rise_velocity" in cfg.rewards:
                cfg.rewards["roulade_rise_velocity"].weight = 3.0
                cfg.rewards["roulade_rise_velocity"].params["gate_lo"] = math.radians(150.0)
            if "roulade_airborne" in cfg.rewards:
                cfg.rewards["roulade_airborne"].weight = 3.0
                cfg.rewards["roulade_airborne"].params["gate_lo"] = math.radians(180.0)
            if "roulade_preroll_kick" in cfg.rewards:
                cfg.rewards["roulade_preroll_kick"].weight = 2.0
            if "roulade_stand_tax" in cfg.rewards:
                cfg.rewards["roulade_stand_tax"].weight = 3.0
                cfg.rewards["roulade_stand_tax"].params.update(
                    {"gate_lo": math.radians(180.0), "gate_hi": _gate_hi_360})
    # (Curriculum v1 branch removed 2026-10-07 — superseded by the v3 ladder
    #  above; DIVEWALL_STAGE=4 or unset = full config.)
    # Run-4: wall-relative features for the CRITIC ONLY (teacher-student).
    # The actor stays blind — wall is training scaffolding; the critic needs
    # it to attribute the wall penalty/jackpot to the right states.
    cfg.observations["critic"].terms["wall_rel"] = ObservationTermCfg(
        func=microduck_mdp.wall_critic_obs, scale=1.0,
    )
    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckRouladeRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # normalizer MUST be baked into ONNX by export.py
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="microduck_roulade",
    run_name="microduck_roulade_run6_kipup",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=10_000,
)

# Dive variant gets its own runner cfg (distinct experiment_name per AGENTS.md)
# but inherits every hyperparameter from the roulade runner.
MicroduckRouladeDiveRlCfg = deepcopy(MicroduckRouladeRlCfg)
MicroduckRouladeDiveRlCfg.experiment_name = "microduck_roulade_dive"
MicroduckRouladeDiveRlCfg.run_name = "microduck_roulade_dive_run1"

# Dive-wall variant: same runner hyperparameters, own run name.
MicroduckRouladeDiveWallRlCfg = deepcopy(MicroduckRouladeRlCfg)
MicroduckRouladeDiveWallRlCfg.experiment_name = "microduck_roulade_dive"
MicroduckRouladeDiveWallRlCfg.run_name = "microduck_roulade_divewall_run15"
# Perf 2026-10-26: 每步 NaN 检查 = 每步强制 GPU 同步; 环境侧已有 nan_state
# termination 兜底, 关闭 runner 层检查 (见 docs/perf/2026-10-26-*.md).
MicroduckRouladeDiveWallRlCfg.check_for_nan = False
