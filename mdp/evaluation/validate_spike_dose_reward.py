"""Offline reward validation; never changes a checkpoint or runs a policy."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ..spike_dose_reward import DEFAULT_CONFIG, PathAdvance, SpikeDoseConfig, SpikeDoseReward
from ..tests.test_spike_dose_reward import response_trial


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay(directory, reference, config):
    telemetry = np.genfromtxt(directory / 'telemetry.csv', names=True, delimiter=',')
    with np.load(directory / 'derived_signals.npz') as data:
        time, xyz, force, rate = (data[key].copy() for key in
                                 ('time_s', 'xyz_mm', 'force_n', 'raw_rate_n_mm_s'))
    np.testing.assert_allclose(np.diff(time), .008, atol=1.e-7, rtol=1.e-4)
    # Reward follows the 125 Hz controller clock, as does removal integration.
    # ROS simulation timestamps can repeat across two control samples.
    np.testing.assert_allclose(time, telemetry['elapsed_s'] - telemetry['elapsed_s'][0], atol=1.e-7)
    np.testing.assert_allclose(force, telemetry['normal_force_n'], atol=1.e-7)
    displacement = np.diff(xyz.astype(float), axis=0, prepend=xyz[:1])
    displacement[:, 2] = 0.
    tracker, project = SpikeDoseReward(config), PathAdvance(reference)
    rows, reference_time = [], None
    progress = 100. * telemetry['cursor_mm'] / telemetry['path_length_mm']
    for i in range(len(time)):
        sample = tracker.update(float(force[i]), float(rate[i]) * .008,
                                project.project(displacement[i], telemetry['cursor_mm'][i]),
                                contact=bool(force[i] >= 1.5 and telemetry['fault'][i] == 0.),
                                shield=bool(telemetry['shield'][i]))
        if tracker.reference_rate is not None and reference_time is None:
            reference_time = float(time[i] - time[0])
        rows.append((sample.event, sample.error, sample.reward))
    rows = np.asarray(rows)
    events = rows[:, 0].astype(bool)
    changes = np.diff(np.r_[False, events, False].astype(int))
    intervals = []
    for first, stop in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        intervals.append({'start_active_s': float(time[first] - time[0]),
                          'end_active_s': float(time[stop - 1] - time[0] + .008),
                          'start_progress_pct': float(progress[first]),
                          'end_progress_pct': float(progress[stop - 1]),
                          'peak_abs_error': float(abs(rows[first:stop, 1]).max()),
                          'reward': float(rows[first:stop, 2].sum())})
    regions = {}
    for label, low, high in [('quiet_reference', 5., 80.), ('large_spikes', 85., 90.),
                              ('terminal_spikes', 98.8, 100.)]:
        selected = (progress >= low) & (progress <= high)
        region_events = selected & events
        regions[label] = {'progress_percent': [low, high], 'seconds': float(selected.sum() * .008),
                         'event_seconds': float(region_events.sum() * .008),
                         'event_fraction': float(events[selected].mean()),
                         'reward': float(rows[selected, 2].sum()),
                         'peak_abs_error': float(abs(rows[region_events, 1]).max()) if region_events.any() else 0.}
    return {'metrics': tracker.metrics(), 'reference_first_established_active_s': reference_time,
            'regions': regions, 'events': intervals,
            'source_sha256': {name: sha(directory / name) for name in ['telemetry.csv', 'derived_signals.npz']}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = SpikeDoseConfig.load(DEFAULT_CONFIG)
    reference_file = args.archive / 'contact_reference_path.txt'
    reference = np.loadtxt(reference_file)[:, :3]
    synthetic = {}
    for force in (17., 23.):
        synthetic[str(force)] = {name: response_trial(force, **kwargs)[0] for name, kwargs in
                                [('fast', {}), ('delayed_320ms', {'extra_delay_s': .32}),
                                 ('hold', {'hold': True}), ('wrong_direction', {'reverse': True})]}
        values = synthetic[str(force)]
        assert values['fast']['reward'] > values['delayed_320ms']['reward'] > values['hold']['reward'] > values['wrong_direction']['reward']
    _, speed, _ = response_trial()
    response_seconds = {str(fraction): float(np.flatnonzero(speed >= 6. + .9 * fraction)[0] * .008)
                        for fraction in (.1, .5, .9, .95)}
    task = Path(__file__).parents[2]
    result = {'scope': 'offline reward behavior and old GUI log replay; no new trained-policy result',
              'config': config.metadata(), 'synthetic': synthetic,
              'synthetic_fraction_of_speed_change_reached_s': response_seconds,
              'implementation_sha256': {name: sha(task / name) for name in (
                  'mdp/spike_dose_reward.py', 'mdp/isaac_latest_force_env.py',
                  'mdp/agents/train_latest_force_ppo.py', 'mdp/config/spike_dose_reward.json',
                  'mdp/tests/test_spike_dose_reward.py', 'mdp/evaluation/validate_spike_dose_reward.py')},
              'synthetic_force_model': 'prescribed 20->17/23 N step; Q/K=1000*force; unchanged 80ms policy/action filter and feed limits',
              'replay_scope': 'all force-control-active samples; stable moving contact establishes the transient reference',
              'replay_comparison_limit': 'each policy has its own pre-event dose reference; costs are diagnostics, not absolute quality ranking',
              'reference_sha256': sha(reference_file),
              'replay': {mode: replay(args.archive / mode, reference, config) for mode in ('mode5', 'mode6')}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'synthetic_reward': {
        force: {name: metric['reward'] for name, metric in cases.items()} for force, cases in synthetic.items()},
        'replay': {mode: {'metrics': row['metrics'], 'regions': row['regions']}
                   for mode, row in result['replay'].items()}}, indent=2))


if __name__ == '__main__':
    main()
