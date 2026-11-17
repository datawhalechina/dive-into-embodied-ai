"""Experimental high-cadence steering; default walking controller is unchanged.

Frozen recipe and finite-condition results: reports/optimization-2026-11-16.json.
Only writes four driven joint targets. Never writes base state or external forces.
"""
import json
import math
from pathlib import Path

import numpy as np

from gait_controller import CONFIG


class CadenceController:
    def __init__(self, mode="stand", amplitude=0.10, config=None):
        self.config = CONFIG if config is None else config
        self.signs = np.array(self.config["actuator_signs"], dtype=float)
        self.amplitudes = np.array(self.config["amplitude_rad"])
        self.biases = np.array(self.config["bias_rad"])
        self.phases = np.array(self.config["phase_rad"])
        self.startup_bias = np.array(self.config.get("startup_bias_rad", [0., 0., 0., 0.]), dtype=float)
        self.startup_ramp = self.config.get("startup_ramp_seconds", .15)
        if self.startup_bias.shape != (4,) or not np.isfinite(self.startup_bias).all():
            raise ValueError("startup_bias_rad must contain four finite angles")
        if not math.isfinite(self.startup_ramp) or self.startup_ramp <= 0:
            raise ValueError("startup_ramp_seconds must be finite and positive")
        self.mode = mode
        self.amplitude = amplitude
        self._last_mode = None
        self._last_time = -1.0
        self._start = 0.0
        self._from = np.zeros(4)
        self._tick = -1
        self._target = np.zeros(4)
        self._start_x = 0.

    def __call__(self, data):
        config = self.config
        if self.mode != self._last_mode or data.time < self._last_time:
            self._initializing = self._last_mode is None or data.time < self._last_time
            self._start = float(data.time)
            self._from = data.ctrl.copy()
            self._last_mode = self.mode
            self._tick = -1
            self._target = np.zeros(4)
            self._start_x = float(data.qpos[0])
        self._last_time = float(data.time)
        elapsed = float(data.time) - self._start
        if self.mode == "walk":
            # Match the 100 Hz action updates used in the search, independently
            # of the rendering frame rate and the 1 kHz physics step.
            tick = int(np.floor((elapsed - config["settle_seconds"]) * config["control_hz"] + 1e-8))
            if tick != self._tick:
                self._tick = tick
                t = max(0., tick / config["control_hz"])
                ramp = min(1., t / config["ramp_seconds"])
                rot = data.body("body").xmat.reshape(3, 3)
                yaw = math.atan2(rot[1, 0], rot[0, 0])
                # Read simulated heading and lateral position. Correct direction
                # solely by changing left/right stride amplitude.
                heading_error = yaw
                if "heading_target_rad" in config:
                    delta = yaw - config["heading_target_rad"]
                    heading_error = math.atan2(math.sin(delta), math.cos(delta))
                error = heading_error + config["lateral_kp"] * (data.qpos[0] - self._start_x)
                steer = np.clip(config["heading_kp"] * error + config["heading_kd"] * data.qvel[5],
                                -config["steering_limit"], config["steering_limit"])
                profile = config.get("steering_profile", "amplitude")
                phase = 2 * np.pi * config["frequency_hz"] * t + self.phases
                if profile == "phase":
                    phase = phase + steer * self.signs
                if config.get("waveform") == "cosine_duty":
                    cycle = (phase / (2 * np.pi)) % 1.
                    duty = config["sweep_fraction"]
                    if config.get("endpoint_dwell") and any(config["endpoint_dwell"]):
                        high, low = config["endpoint_dwell"]
                        free = 1. - high - low
                        down = free * duty
                        up = free * (1. - duty)
                        a = np.clip((cycle-high)/down, 0., 1.)
                        b = np.clip((cycle-high-down-low)/up, 0., 1.)
                        phase = np.where(cycle < high+down, np.pi*a, np.pi+np.pi*b)
                    else:
                        phase = np.where(cycle < duty, np.pi * cycle / duty,
                                         np.pi + np.pi * (cycle - duty) / (1 - duty))
                    waveform = np.cos(phase)
                else:
                    h = config["harmonic"]
                    waveform = (np.sin(phase) + h * np.sin(2 * phase)) / (1 + abs(h))
                startup_ramp = min(1., elapsed / self.startup_ramp)
                startup = self.startup_bias * (startup_ramp * startup_ramp * (3 - 2 * startup_ramp))
                if profile == "amplitude":
                    # Original expression retained exactly for control parity.
                    walk = self.biases + self.amplitudes * (1 + steer * self.signs) * waveform
                elif profile == "front_amplitude":
                    walk = self.biases + self.amplitudes * (1 + steer * np.array([1., -1., 0., 0.])) * waveform
                elif profile == "rear_amplitude":
                    walk = self.biases + self.amplitudes * (1 + steer * np.array([0., 0., -1., 1.])) * waveform
                elif profile == "bias":
                    walk = self.biases + self.amplitudes * waveform + steer * self.signs
                elif profile == "phase":
                    walk = self.biases + self.amplitudes * waveform
                else:
                    raise ValueError(profile)
                target = np.clip(self.signs * ((1 - ramp) * startup + ramp * walk), -1.2, 1.2)
                if "max_target_speed_rad_s" in config:
                    max_delta = config["max_target_speed_rad_s"] / config["control_hz"]
                    self._target += np.clip(target - self._target, -max_delta, max_delta)
                else:
                    self._target = target
            target = self._target
        elif self.mode == "motion":
            ramp = np.clip((elapsed - 1.0) / 2.0, 0.0, 1.0)
            target = self.signs * self.amplitude * ramp * np.sin(2 * np.pi * .55 * max(0., elapsed - 1.))
        elif self.mode == "stand":
            target = np.zeros(4)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        # Smooth a user-requested stop or mode change over 350 ms.
        blend = np.clip(elapsed / .35, 0., 1.)
        blend = blend * blend * (3 - 2 * blend)
        if self._initializing and self.mode == "walk" and "startup_bias_rad" in config:
            # Startup already has its own smooth ramp and target slew limit.
            blend = 1.
        data.ctrl[:] = (1 - blend) * self._from + blend * target
