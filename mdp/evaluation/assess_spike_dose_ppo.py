"""Score measured Isaac traces with the frozen causal force-transient reward."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from ..spike_dose_reward import PathAdvance, SpikeDoseConfig, SpikeDoseReward


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def assess(directory, config):
    directory = Path(directory)
    source = json.loads((directory / 'summary.json').read_text())
    processing = json.loads((directory / 'processing_summary.json').read_text())
    if not processing['duplicate_mode6_verified']:
        raise ValueError('Duplicate Mode 6 validation failed')
    with h5py.File(directory / 'path.h5') as h:
        project = PathAdvance(h['position'][:, :3])
    report = {'seed': source['seed'], 'config': config.metadata(), 'candidates': {},
              'scope': 'measured headless Isaac; initial approach contact included in contact CV; transient loss uses active control',
              'nominal_repetitions_are_not_statistical_significance': True}
    for name, metrics in source['candidates'].items():
        trace_path = directory / f'{name}_trace.npz'
        with np.load(trace_path) as archive:
            trace = dict(archive)
        time = trace['time_s']
        xyz = trace['tcp_pose'][:, :3].astype(float)
        delta = np.diff(xyz, axis=0, prepend=xyz[:1])
        delta[0] = 2. * (xyz[0] - trace['deposition_midpoint_mm'][0])
        delta[:, 2] = 0.
        force, rate, cursor = (trace[key] for key in ('normal_force_n', 'raw_mrr_n_mm_s', 'cursor_mm'))
        np.testing.assert_allclose(np.diff(time), .008, atol=1.e-9)
        tracker = SpikeDoseReward(config)
        values, first_reference = [], None
        for i in range(len(time)):
            result = tracker.update(float(force[i]), max(0., float(rate[i])) * .008,
                project.project(delta[i], float(cursor[i])),
                contact=bool(force[i] >= 1.5 and trace['safety_fault_reason'][i] == 0),
                shield=bool(trace['safety_shield_active'][i]))
            if first_reference is None and tracker.reference_rate is not None:
                first_reference = float(time[i])
            values.append((result.event, result.error, result.reward))
        values = np.array(values)
        event = values[:, 0].astype(bool)
        progress = cursor / source['path_length_mm'] * 100.
        edge = np.diff(np.r_[False, event, False].astype(int))
        intervals = [{'start_s': float(time[a]), 'end_s': float(time[b-1] + .008),
                      'start_progress_percent': float(progress[a]), 'end_progress_percent': float(progress[b-1]),
                      'peak_abs_budget_error': float(abs(values[a:b, 1]).max())}
                     for a, b in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1))]
        spike = tracker.metrics()
        spike['reference_established_active_s'] = first_reference
        spike['intervals'] = intervals
        np.savez_compressed(directory / f'{name}_spike_dose.npz', time_s=time,
                            progress_percent=progress, event=event, budget_error=values[:, 1], reward=values[:, 2])
        p = processing['candidates'][name]
        if 'contact_inclusive_cv' not in metrics:
            raise ValueError('Run probe with --include-precontrol-contact')
        report['candidates'][name] = {
            'contact_cv': metrics['contact_inclusive_cv'], 'active_cv': metrics['spatial_cv'],
            'processing_cv': p['spatial_cv'], 'contact_volume_over_k': metrics['contact_inclusive_volume_over_k'],
            'processing_volume_over_k': p['roi_volume_over_k'],
            'time_s': metrics['contact_inclusive_time_s'], 'active_time_s': metrics['time_s'],
            'precontrol_contact_samples': metrics['precontrol_contact_samples'],
            'processing_force_rmse_n': p['processing_force_rmse_n'], 'force_peak_n': p['force_peak_n'],
            'completed': p['completed'], 'shield_fraction': metrics['shield_fraction'],
            'fault_fraction': p['fault_fraction'], 'existing_quality_eligible': p['eligible'],
            'spike': spike, 'actor_sha256': source['checkpoint_sha256'][name],
            'trace_sha256': hashlib.sha256(trace_path.read_bytes()).hexdigest()}
    save(directory / 'spike_summary.json', report)
    return report


def compare(candidate, parent, mode6):
    """Preserve at least ~90% of the prior CV gain while selecting transient response.

    The 1% relative CV tolerance is explicit, not a significance claim. Each
    parent used here is run in the SAME environment slot and seed as candidate.
    """
    parent_error = parent['spike']['event_abs_error_seconds']
    candidate_error = candidate['spike']['event_abs_error_seconds']
    gains = {'spike_abs_error_improvement_percent':
             (100. * (1. - candidate_error / parent_error) if parent_error > 1.e-9 else None),
             'contact_cv_change_percent': 100. * (candidate['contact_cv'] / parent['contact_cv'] - 1.),
             'contact_cv_improvement_vs_mode6_percent': 100. * (1. - candidate['contact_cv'] / mode6['contact_cv']),
             'processing_cv_change_percent': 100. * (candidate['processing_cv'] / parent['processing_cv'] - 1.),
             'contact_volume_ratio_vs_parent': candidate['contact_volume_over_k'] / parent['contact_volume_over_k'],
             'processing_volume_ratio_vs_parent': candidate['processing_volume_over_k'] / parent['processing_volume_over_k'],
             'time_change_percent': 100. * (candidate['time_s'] / parent['time_s'] - 1.)}
    reference_time = candidate['spike']['reference_established_active_s']
    checks = {
        'completion_and_original_constraints': bool(candidate['existing_quality_eligible'] and parent['existing_quality_eligible']),
        'contact_cv_within_1pct_of_parent': bool(candidate['contact_cv'] <= parent['contact_cv'] * 1.01),
        'processing_cv_within_1pct_of_parent': bool(candidate['processing_cv'] <= parent['processing_cv'] * 1.01),
        'parent_volume_retained': bool(min(gains['contact_volume_ratio_vs_parent'], gains['processing_volume_ratio_vs_parent']) >= .9),
        'contact_and_faults': bool(candidate['fault_fraction'] == 0. and candidate['shield_fraction'] <= parent['shield_fraction'] + 1.e-6),
        'reference_established': bool(reference_time is not None and reference_time <= 10.),
    }
    quality_ok = all(checks.values())
    improvement = gains['spike_abs_error_improvement_percent']
    return {**gains, 'quality_checks': checks, 'quality_eligible': quality_ok,
            'accepted': bool(quality_ok and improvement is not None and improvement >= 5.),
            'acceptance': '>=5% transient absolute-error integral reduction; <=1% relative contact/processing CV degradation; volume >=90% of parent',
            'automatic_deployment': False}
