"""Replay frozen high-cadence configurations without changing running acceptance."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from cadence_controller import CadenceController
import scenario_matrix as matrix

ROOT = Path(__file__).resolve().parent
LEGS = ['left_front', 'right_front', 'right_back', 'left_back']
# Keep all original physical checks, including heading; use a separate task goal.
TASK_CHECKS = {'running_speed', 'repeated_flight', 'forward_speed', 'task_goal'}
HELDOUT = [
    {'initial_yaw_deg': -1.5, 'friction_scale': .95},
    {'initial_yaw_deg': 1.5, 'friction_scale': 1.05},
    {'initial_x_offset': -.0075}, {'initial_x_offset': .0075},
    {'friction_scale': .95}, {'friction_scale': 1.05},
]


def measure_frequency(samples, commanded_hz):
    values = np.asarray(samples)
    values = values[values[:, 0] >= 5.]
    times = values[:, 0]
    measured = []
    for index, leg in enumerate(LEGS):
        position = values[:, index + 1]
        position = position - position.mean()
        frequencies = np.fft.rfftfreq(len(position), .01)
        power = np.abs(np.fft.rfft(position * np.hanning(len(position)))) ** 2
        band = (frequencies >= .5) & (frequencies <= 12.)
        dominant = float(frequencies[band][np.argmax(power[band])])
        peaks, _ = find_peaks(position, prominence=.03,
                              distance=max(1, int(.5 / commanded_hz / .01)))
        interval_hz = (float(1 / np.median(np.diff(times[peaks])))
                       if len(peaks) > 2 else None)
        measured.append(dict(
            leg=leg, dominant_frequency_hz=dominant, peak_count=len(peaks),
            peak_interval_frequency_hz=interval_hz,
            excursion_p95_minus_p05_rad=float(np.quantile(position, .95) -
                                            np.quantile(position, .05))))
    return measured


def evaluate(job):
    kind, condition, perturbation, destination, video = job
    output = Path(destination) / f'{kind}_{condition}'
    if output.exists():
        raise FileExistsError(f'Preserve previous evidence; choose a new --output: {output}')
    config = json.loads((ROOT / 'reference' / f'cadence-{kind}.json').read_text())
    samples = []

    def observer(model, data, controller, step):
        if step % 10 == 0:
            position = [float(data.qpos[model.jnt_qposadr[model.joint(s + '_thigh').id]])
                        for s in LEGS]
            samples.append([float(data.time)] + position + list(map(float, data.ctrl)))

    if video:
        original_recorder = matrix.Recorder

        class CadenceRecorder(original_recorder):
            def frame(self, *args, **kwargs):
                self.controller_label = f'{kind} cadence / {config["frequency_hz"]:.2f} Hz'
                self.note_override = 'Actual simulation | Slow forward cadence | No aerial-running claim'
                self.show_goal = False
                return super().frame(*args, **kwargs)

        matrix.Recorder = CadenceRecorder
    try:
        result = matrix.run_case(
            'flat', 'run_attempt', seconds=20., output=output, video=video,
            config_override=config, observer=observer,
            controller_factory=lambda model, cfg: CadenceController('walk', config=cfg),
            **perturbation)
    finally:
        if video:
            matrix.Recorder = original_recorder
    with (output / 'joint_samples.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time_s'] + [s + '_q_rad' for s in LEGS] +
                        [s + '_target_rad' for s in LEGS])
        writer.writerows(samples)
    measured = measure_frequency(samples, config['frequency_hz'])
    checks = dict(
        original_physical_checks=all(v for k, v in result['checks'].items()
                                     if k not in TASK_CHECKS),
        slow_forward_progress=result['forward_m'] >= .1,
        command_at_least_3_3_hz=config['frequency_hz'] >= 3.3,
        all_legs_frequency_tracking=all(
            abs(x['dominant_frequency_hz'] / config['frequency_hz'] - 1) <= .15
            for x in measured),
        all_legs_excursion_at_least_0_12_rad=all(
            x['excursion_p95_minus_p05_rad'] >= .12 for x in measured))
    record = dict(controller=kind, condition=condition, perturbation=perturbation,
                  result=result, measured=measured, cadence_checks=checks,
                  cadence_pass=all(checks.values()),
                  model_sha256=hashlib.sha256((ROOT / 'models/scene.xml').read_bytes()).hexdigest())
    (output / 'summary.json').write_text(json.dumps(record, indent=2) + '\n')
    print(kind, condition, 'cadence:', record['cadence_pass'],
          'original running:', result['passed'], flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', choices=['baseline', 'optimized', 'both'], default='both')
    parser.add_argument('--suite', choices=['nominal', 'heldout', 'all'], default='all')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--video', action='store_true', help='Record nominal cases only; EGL may be needed')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/cadence_evaluation')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    kinds = ['baseline', 'optimized'] if args.controller == 'both' else [args.controller]
    cases = []
    if args.suite in ['nominal', 'all']:
        cases.append(('nominal', {}))
    if args.suite in ['heldout', 'all']:
        cases.extend((f'heldout_{i}', p) for i, p in enumerate(HELDOUT))
    jobs = [(kind, label, perturbation, str(args.output), args.video and label == 'nominal')
            for kind in kinds for label, perturbation in cases]
    with ProcessPoolExecutor(args.workers) as pool:
        records = list(pool.map(evaluate, jobs))
    (args.output / 'summary.json').write_text(json.dumps(dict(
        records=records, note='Cadence success is separate from original running success; finite simulation conditions.'),
        indent=2) + '\n')


if __name__ == '__main__':
    main()
