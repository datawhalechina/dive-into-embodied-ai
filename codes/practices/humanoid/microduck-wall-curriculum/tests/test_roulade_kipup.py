"""Run-6 kip-up reward regression tests (CPU, fake env — no sim needed).

Locks in the run-6 reward semantics:
  • stand_tax rise-ramp: no tax before the frontier opens the gate, ramped tax
    after, full tax ≥ ramp_steps after the crossing, always ≤ 0 (sign rule).
  • airborne (腾空) credit: pays only feet-off × gate-open × below max_height —
    hop-farming upright and pre-gate flight earn nothing.
"""

import math

import torch

from mjlab_microduck.tasks import mdp


class _Data:
    pass


class _FakeAsset:
    def __init__(self):
        d = _Data()
        d.root_link_ang_vel_b = torch.zeros(1, 3)
        d.root_link_pos_w = torch.tensor([[0.0, 0.0, 0.05]])
        d.root_link_lin_vel_w = torch.zeros(1, 3)
        d.root_link_quat_w = torch.zeros(1, 4)
        d.root_link_quat_w[:, 0] = 1.0  # identity — sagittal flat, upright
        d.body_link_quat_w = torch.zeros(1, 1, 4)
        d.body_link_quat_w[:, 0, 0] = 1.0
        self.data = d

    def find_bodies(self, pattern):
        return [0], ["jaw_soft"]


class _FakeTerrain:
    env_origins = torch.zeros(1, 3)


class _FakeScene:
    def __init__(self, asset, sensors):
        self._asset = asset
        self.sensors = sensors
        self.terrain = _FakeTerrain()

    def __getitem__(self, name):
        return self._asset


def _feet_sensor(contact: bool):
    class _S:
        data = _Data()

    s = _S()
    s.data.found = torch.tensor([[1.0 if contact else 0.0]])
    return s


class _FakeEnv:
    def __init__(self, frontier_deg=0.0, z=0.05, feet_contact=True, latched=True, open_step=None, step=1000):
        self.device = "cpu"
        self.num_envs = 1
        self.step_dt = 0.02
        self.common_step_counter = step
        self._asset = _FakeAsset()
        self._asset.data.root_link_pos_w[0, 2] = z
        self.scene = _FakeScene(self._asset, {"feet_ground_contact": _feet_sensor(feet_contact)})
        self._roulade_accum = torch.zeros(1)
        self._roulade_max = torch.full((1,), math.radians(frontier_deg))
        self._roulade_paid = torch.zeros(1)
        self._roulade_head_latch = torch.tensor([latched])
        self._roulade_open_step = torch.full(
            (1,), mdp._ROULADE_OPEN_NEVER if open_step is None else float(open_step)
        )
        self._roulade_last_update_step = -1


GATE_LO = math.radians(260.0)
GATE_HI = math.radians(330.0)


def test_stand_tax_is_zero_before_gate_opens():
    env = _FakeEnv(frontier_deg=200.0)
    out = mdp.roulade_stand_tax(env, target_height=0.115, gate_lo=GATE_LO, gate_hi=GATE_HI)
    assert float(out[0]) == 0.0


def test_stand_tax_ramp_is_zero_right_after_crossing():
    # just opened (open_step == current step): the ramp shelters the launch
    env = _FakeEnv(frontier_deg=350.0, open_step=1000, step=1000)
    out = mdp.roulade_stand_tax(env, target_height=0.115, gate_lo=GATE_LO, gate_hi=GATE_HI)
    assert float(out[0]) == 0.0


def test_stand_tax_ramp_is_partial_then_full():
    shortfall = 0.115 - 0.05
    half = _FakeEnv(frontier_deg=350.0, open_step=962, step=1000)  # 38/75 ≈ 0.51
    out_half = mdp.roulade_stand_tax(half, target_height=0.115, gate_lo=GATE_LO, gate_hi=GATE_HI, ramp_steps=75.0)
    full = _FakeEnv(frontier_deg=350.0, open_step=850, step=1000)  # 150/75 → saturated
    out_full = mdp.roulade_stand_tax(full, target_height=0.115, gate_lo=GATE_LO, gate_hi=GATE_HI, ramp_steps=75.0)
    assert out_full[0] < out_half[0] < 0.0
    assert abs(float(out_full[0]) + shortfall) < 1e-6
    assert abs(float(out_half[0]) + shortfall * (38.0 / 75.0)) < 1e-6


def test_stand_tax_never_positive():
    envs = [
        _FakeEnv(frontier_deg=f, z=z, open_step=os_)
        for f, z, os_ in [(350.0, 0.05, 900), (270.0, 0.11, 990), (360.0, 0.115, 999), (100.0, 0.01, None)]
    ]
    for env in envs:
        out = mdp.roulade_stand_tax(env, target_height=0.115, gate_lo=GATE_LO, gate_hi=GATE_HI)
        assert float(out[0]) <= 0.0


def test_airborne_pays_for_feet_off_after_gate():
    env = _FakeEnv(frontier_deg=360.0, z=0.05, feet_contact=False)
    out = mdp.roulade_airborne(env, max_height=0.115, gate_lo=GATE_LO, gate_hi=math.radians(360.0))
    assert abs(float(out[0]) - 1.0) < 1e-6  # gate saturated at gate_hi, airborne, below height


def test_airborne_gate_is_partial_before_saturation():
    env = _FakeEnv(frontier_deg=350.0, z=0.05, feet_contact=False)
    out = mdp.roulade_airborne(env, max_height=0.115, gate_lo=GATE_LO, gate_hi=math.radians(360.0))
    t = (350.0 - 260.0) / 100.0
    expected = t * t * (3.0 - 2.0 * t)
    assert abs(float(out[0]) - expected) < 1e-6


def test_airborne_grounded_pays_nothing():
    env = _FakeEnv(frontier_deg=350.0, z=0.05, feet_contact=True)
    out = mdp.roulade_airborne(env, max_height=0.115, gate_lo=GATE_LO, gate_hi=math.radians(360.0))
    assert float(out[0]) == 0.0


def test_airborne_above_max_height_pays_nothing_hop_farm_guard():
    env = _FakeEnv(frontier_deg=350.0, z=0.13, feet_contact=False)
    out = mdp.roulade_airborne(env, max_height=0.115, gate_lo=GATE_LO, gate_hi=math.radians(360.0))
    assert float(out[0]) == 0.0


def test_airborne_before_gate_pays_nothing_roll_stays_support_gated():
    env = _FakeEnv(frontier_deg=200.0, z=0.05, feet_contact=False)
    out = mdp.roulade_airborne(env, max_height=0.115, gate_lo=GATE_LO, gate_hi=math.radians(360.0))
    assert float(out[0]) == 0.0


def test_airborne_requires_head_latch():
    env = _FakeEnv(frontier_deg=350.0, z=0.05, feet_contact=False, latched=False)
    out = mdp.roulade_airborne(env, max_height=0.115, gate_lo=GATE_LO, gate_hi=math.radians(360.0))
    assert float(out[0]) == 0.0


def test_rise_open_ref_matches_landing_gate_lo():
    # the rise clock reference and the cfg landing gate must stay in sync
    assert abs(mdp._RISE_OPEN_REF - GATE_LO) < 1e-9
