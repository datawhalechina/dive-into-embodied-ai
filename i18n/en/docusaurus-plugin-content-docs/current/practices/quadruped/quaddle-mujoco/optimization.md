---
title: "Quaddle optimization and bottlenecks: beyond faster stepping"
sidebar_position: 4
displayed_sidebar: practicesQuadrupedSidebar
description: "Replay a 3.34 Hz steering improvement and distinguish joint cadence, aerial running, terrain completion, and model limits."
---

# Quaddle optimization and bottlenecks: beyond faster stepping

**The legs can cycle faster without moving forward quickly.** The follow-up study on 2026-11-16 produced a simulation candidate at about 3.34 Hz. On the same six perturbed conditions, the old controller passed 2/6 and the new one passed 6/6. This did not solve the original rough terrain, continuous stair ascent, or aerial-running task. Earlier scene results remain in the [multi-scenario experiments](./experiments).

This is four-joint feedback-control optimization: no new PPO training, default-gait replacement, dynamics change, or hardware validation. Each fixed-condition replay contains 20 seconds of motion plus 2 seconds of startup; 6/6 does not establish a statistical 100% success rate.

## Watch the motion and steering improvement \{#cadence}

<video controls playsInline preload="none" src={require('@site/docs/practices/quadruped/quaddle-mujoco/figs/optimization/cadence-optimized.mp4').default} poster={require('@site/docs/practices/quadruped/quaddle-mujoco/figs/optimization/cadence-optimized.webp').default} style={{width: '100%', maxWidth: 960}} />

This is an actual 22-second replay at normal speed, including startup. Hz means full back-and-forth cycles of one driven leg joint per second, not the total footfall count across four legs or the 100 Hz control-update rate.

| Nominal metric | Old 3.34 Hz candidate | New candidate |
| --- | ---: | ---: |
| Average forward speed | 0.07025 m/s | 0.06953 m/s |
| Maximum heading deviation | 17.39° | 7.62° |
| Actual driven-joint peak speed | 8.929 rad/s | 8.929 rad/s |
| Actual dominant motion frequency | About 3.34 Hz | About 3.34 Hz |

Each new-candidate leg has 57 distinct angle peaks during seconds 5–22. The FFT peak is 3.352941 Hz and the peak-interval estimate is about 3.333333 Hz, consistent with the 3.342608 Hz command at the sampling resolution. Actual joint-angle P95–P5 spans 0.416–0.486 rad, retaining substantial motion.

### What changed in steering \{#controller}

Frequency, base amplitudes, and front/rear phase remain fixed. Steering now uses a small left/right neutral-angle difference, with the lateral-position term removed from the heading error. Only four position targets are written; the controller does not move the base state or apply propulsive external forces.

```text
steering_profile = bias
heading_kp = 0.1
heading_kd = 0.03
steering_limit = 0.12 rad
lateral_kp = 0
frequency = 3.342608 Hz
```

The original 8 rad/s target-slew limit and 10 rad/s actual-joint check remain. Disabling lateral-position feedback alone still produced about 179.91° deviation under low friction, so deleting one term does not explain the whole improvement. Higher gains are not automatically better: a rear-amplitude candidate passed screening but reached 61.68° deviation and 10.703 rad/s under higher friction.

### Freeze parameters before validation \{#validation}

Select on six development conditions, then freeze one candidate for the six conditions below. The old controller is evaluated on exactly the same conditions. The new candidate passed all six development conditions and all six frozen-candidate checks.

| Condition | Old maximum deviation | New maximum deviation | Old / new cadence checks |
| --- | ---: | ---: | --- |
| Heading −1.5°, friction ×0.95 | 11.92° | 8.19° | Pass / pass |
| Heading +1.5°, friction ×1.05 | 119.09° | 6.50° | Fail / pass |
| Lateral position −7.5 mm | 16.89° | 7.98° | Pass / pass |
| Lateral position +7.5 mm | 52.27° | 6.96° | Fail / pass |
| Friction ×0.95 | 169.87° | 6.70° | Fail / pass |
| Friction ×1.05 | 179.27° | 7.80° | Fail / pass |

New-case speeds are 0.06579–0.07140 m/s; the highest actual joint speed is 9.090 rad/s. Posture, contact, linkage closure, torque, and force-accounting checks pass.

![Matched heading, speed, and actual joint-motion comparison](@site/docs/practices/quadruped/quaddle-mujoco/figs/optimization/cadence-comparison.webp)

This improves heading holding, not precise path tracking. Lateral drift still reaches about 7.2 cm; in the first condition it increases from about 5.7 cm to 7.2 cm. Every metric does not improve simultaneously.

## What remains unfinished \{#bottlenecks}

| Scene / objective | Supported progress | Current bottleneck |
| --- | --- | --- |
| High-cadence slow walking | About 3.34 Hz; matched checks improve from 2/6 to 6/6 | Only the listed short runs are validated; 4–6 Hz joint tracking still causes heading errors and is not a usable walking mode |
| Single 2 mm step | Blocked-foot retraction repairs one failure with the edge shifted +15 mm: deviation 85.37°→20.53° | Other edge locations still fail; no blanket capability below 2 mm |
| Three 2 mm steps | Nominal completion and some perturbation improvements | The frozen recovery candidate passes only 4/10 mixed flat/stair/rough conditions and regresses on an old passing case; not adopted globally |
| Original rough terrain, ±3 mm | Adjacent jumps up to about 5.759 mm and contact/heading issues identified | All three tested original maps still fail; easier graded terrain does not replace the original task |
| Three 4 mm steps upward | Original failure evidence retained | Propulsion and heading unresolved; optimization paused; mechanism infeasibility not established |
| Original aerial-running task | Nominal predictive control combines speed limiting and flight: 0.06011 m/s, 9.938 rad/s, 27 intervals | Still below 0.1 m/s; lower friction can cause overspeed; no complete running pass |
| Walk→run→decelerate→turn→stop | Walk and turn segments pass in the first PPO experiment | Run, deceleration, and stopping fail; further optimization paused |

Earlier flat, slope, stair-descent, and graded-rough results retain their own limited scope. **The high-cadence configuration has not been validated on those terrains and does not replace their separate controllers.**

Running-specific PPO used 131,072 steps initially and another 131,072 in contrasts without a passing running result. The combination experiment used another 32,768 steps and did not pass overall. Recent cadence gains come from feedback control, not a claim that PPO learned these actions. This PR includes records and the cadence replay entry point, not those checkpoints or a complete PPO retraining workflow.

### Stepping and aerial running use separate criteria \{#criteria}

Cadence requires a command of at least 3.3 Hz, actual frequency error at most 15% for every leg, each leg's actual angle P95–P5 at least 0.12 rad, forward progress at least 0.1 m in 20 seconds, and all original physical/heading checks. It does not require flight or 0.1 m/s.

Original `run_attempt` still requires at least 0.1 m/s, at least three completed terrain-contact-free intervals of at least 10 ms each, and its other original checks. The script saves both `cadence_pass` and original `result.passed`; passing the new objective never overwrites a running failure.

### Missing model evidence \{#model-limits}

Each leg has one actuator and passive linkages, so independent knee commands or arbitrary foot trajectories are unavailable. Springs are uncalibrated torsional approximations; servo, friction, and contact parameters also lack hardware calibration. The 10 rad/s threshold is a project acceptance rule, not a measured motor maximum. Existing failures do not establish a hardware-design defect, and simulation success does not guarantee hardware success.

Separate next steps are longer cadence/command-sequence tests, pre-edge foot placement and support timing for stairs, and joint propulsion/speed-limit optimization for aerial running. Calibrate the model and interfaces before hardware transfer, rather than only extending the same training recipe.

## Replay the comparison \{#reproduce}

Install the Python dependencies using the [introductory page](./), and ensure LFS meshes are real files. From `codes/practices/quadruped/quaddle-mujoco`:

```bash
# Frozen old/new configurations, nominal plus six perturbed cases, 22 s each
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python evaluate_cadence.py --workers 3 --output outputs/cadence-review
# Record only the optimized nominal case; use a new output directory
MUJOCO_GL=egl .venv/bin/python evaluate_cadence.py --controller optimized --suite nominal --video --output outputs/cadence-video
```

Outputs include original checks, separate cadence results, joint samples, and frequency measurements. Existing individual-case outputs are not overwritten. `cadence_controller.py` and `reference/cadence-*.json` are experimental entry points; Python and browser defaults remain unchanged.

Source `reports/optimization-2026-11-16.json` records the research and metrics for all 73 runs. Environment: Python 3.12, MuJoCo 3.16.0; source commit `b0e5ec7`; model SHA256 `10d1990f12c2dc79b739b67203422ff8a0090a01e424dacf780b9a03d6015b83`. Deterministic contact simulation is sensitive to versions and numerical changes; compare full checks and trajectories when reproducing.

## Diagnose a failure yourself \{#exercise}

1. Compare `cadence_pass` with original `result.passed`. Explain how one trajectory can pass stepping and fail running.
2. In a separate output directory, change only initial heading or friction. Inspect heading, actual frequency, and lateral drift.
3. Use joint-angle traces to check whether a 4 Hz command becomes 4 Hz motion before judging forward walking. Measure command frequency, actual frequency, and translation separately.
