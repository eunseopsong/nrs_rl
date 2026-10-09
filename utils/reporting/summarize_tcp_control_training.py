#!/usr/bin/env python3
"""Report the frozen joint-control experiment, including its controller ablation."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT as _REPO_ROOT
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import h5py


MODES = ('constant', 'old_d14', 'calibrated_only', 'learned')
LABELS = ('Original constant', 'Old D14', 'New MDK only', 'New MDK + policy')
COLORS = ('#43799e', '#d59444', '#8065a9', '#348975')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def values(metrics, dt):
    surface = metrics['preston_surface']
    return {
        'spatial_depth_cv': surface['spatial_depth_cv'],
        'roi_volume_over_k': surface['roi_volume_over_k'],
        'full_footprint_volume_over_k': surface['tcp_full_footprint_volume_over_k'],
        'uncovered_fraction': surface['zero_depth_fraction'],
        'raw_rate_cv': metrics['processing_rate_cv'],
        'processing_time_s': metrics['processing_samples'] * dt,
        'cross_track_rmse_mm': metrics['cross_track_rmse_mm'],
        'endpoint_error_mm': metrics['endpoint_tangential_error_mm'],
        'force_rmse_n': metrics['force_rmse_n'],
        'force_mean_n': metrics['force_mean_n'],
        'force_peak_n': metrics['force_peak_n'],
        'measured_path_length_mm': metrics['measured_path_length_mm'],
        'shield_fraction': metrics['shield_fraction'],
        'fault_fraction': metrics['fault_fraction'],
    }


def distribution(samples):
    return {'mean': float(np.mean(samples)), 'min': float(np.min(samples)),
            'max': float(np.max(samples))}


def compare(candidate, baseline):
    fields = ('spatial_depth_cv', 'raw_rate_cv', 'cross_track_rmse_mm',
              'endpoint_error_mm', 'force_rmse_n', 'processing_time_s')
    out = {f'{key}_reduction_percent': 100 * (1 - candidate[key] / baseline[key])
           for key in fields}
    out['roi_volume_increase_percent'] = 100 * (candidate['roi_volume_over_k'] /
                                                baseline['roi_volume_over_k'] - 1)
    out['uncovered_area_reduction_percentage_points'] = 100 * (
        baseline['uncovered_fraction'] - candidate['uncovered_fraction'])
    return out


def summarize(root):
    root = root.resolve()
    state = json.loads((root / 'result.json').read_text())
    protocol = json.loads((root / 'protocol.json').read_text())
    seeds = state['validation_seeds']
    if (state['phase'] != 'complete' or len(seeds) != 3 or
            sorted(row['seed'] for row in state['validation']) != sorted(seeds)):
        raise ValueError('All three frozen independent evaluations must finish')
    if state['gates'] != protocol['gates'] or state['reward'] != protocol['reward']:
        raise ValueError('The frozen reward or acceptance limits changed')
    if state['validation_passed'] != all(v['passed'] for v in state['validation']):
        raise ValueError('Validation decision does not match the individual results')
    checkpoint = Path(state['checkpoint'])
    actor_hash = sha256(checkpoint)
    reference = _REPO_ROOT / (
        'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/datasets/cmd_continue9D_flat.h5')
    if sha256(reference) != state['reference_path_sha256']:
        raise ValueError('Pinned reference changed after training')
    with h5py.File(reference) as data:
        reference_pose, reference_force = data['position'][:], data['force'][:]
    rows, paired, hashes = [], [], {}
    for seed in seeds:
        source = root / f'held_out_{seed}/summary.json'
        hashes[str(source.relative_to(root))] = sha256(source)
        batch = json.loads(source.read_text())
        if any(any(value != draws[0] for value in draws)
               for draws in batch['randomization'].values()):
            raise ValueError('Sensor scale, bias or delay differs between paired arms')
        with h5py.File(source.parent / 'path.h5') as data:
            if (not np.array_equal(data['position'][:], reference_pose) or
                    not np.array_equal(data['force'][:], reference_force)):
                raise ValueError('Held-out trial did not cover the entire pinned reference')
        if (batch['candidate_checkpoint_sha256'] != actor_hash or
                batch['source_path_sha256'] != state['reference_path_sha256'] or
                batch['target_normal_force_n'] != 20 or batch['rotation_depth_model'] is not None):
            raise ValueError('Held-out checkpoint, reference, force or removal model changed')
        group = {}
        roi = None
        for name, metrics in batch['candidates'].items():
            current_roi = tuple(metrics['preston_surface'][k] for k in ('roi_cells', 'roi_area_mm2'))
            if roi is not None and current_roi != roi:
                raise ValueError('ROI differs between comparison arms')
            roi = current_roi
            group[name] = values(metrics, batch['dt_s'])
            rows.append({'seed': seed, 'policy': name, 'completed': metrics['completed'],
                         'damping_ns_m': metrics['tangential_mdk'][0],
                         'stiffness_n_m': metrics['tangential_mdk'][1], **group[name]})
        group['constant'] = {k: float(np.mean([group[n][k] for n in ('constant_a', 'constant_b')]))
                             for k in group['constant_a']}
        paired.append(group)
    with (root / 'validation_comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    statistics = {}
    for mode in MODES:
        # Keep the full replicate range: averaging duplicate baseline peaks
        # before taking the maximum would hide isolated force excursions.
        members = ('constant_a', 'constant_b') if mode == 'constant' else (mode,)
        statistics[mode] = {key: distribution([p[m][key] for p in paired for m in members])
                            for key in paired[0][mode]}
    comparisons = {}
    for candidate, baseline in [('learned', 'constant'), ('learned', 'old_d14'),
                                ('learned', 'calibrated_only'), ('calibrated_only', 'constant')]:
        per_seed = [compare(p[candidate], p[baseline]) for p in paired]
        comparisons[f'{candidate}_vs_{baseline}'] = {
            key: distribution([p[key] for p in per_seed]) for key in per_seed[0]}
    contrasts = [c for result in state['validation'] for c in result['comparisons']]
    report = {
        'phase': 'complete', 'algorithm': state['algorithm'],
        'training_seeds': state['training_seeds'], 'validation_seeds': seeds,
        'development_seeds': state.get('development_seeds_from_previous_validation', []),
        'adaptive_velocity_enabled': state.get('adaptive_velocity_enabled', True),
        'selected_family': state.get('selected_family', 'joint_cem'),
        'reference_rows': len(reference_pose),
        'reference_path_length_mm': batch['path_length_mm'],
        'checkpoint': str(checkpoint), 'checkpoint_sha256': actor_hash,
        'learned_parameters': dict(zip(state['parameter_names'], state['best_parameters'])),
        'rotation_contribution': False, 'spindle_rpm': None, 'normal_force_target_n': 20.,
        'frozen_gates_passed': state['validation_passed'], 'frozen_gates': state['gates'],
        'disposition': state['disposition'], 'statistics': statistics, 'paired_comparisons': comparisons,
        'spatial_cv_gain_percent_range_vs_both_baselines': [
            100 * min(c['spatial_cv_gain'] for c in contrasts),
            100 * max(c['spatial_cv_gain'] for c in contrasts)],
        'roi_volume_gain_percent_range_vs_both_baselines': [
            100 * min(c['roi_volume_gain'] for c in contrasts),
            100 * max(c['roi_volume_gain'] for c in contrasts)],
        'validation_rejections': {str(v['seed']): v['rejections'] for v in state['validation']},
        'calibration_only_rejections': {str(v['seed']): v['calibration_only']['rejections']
                                        for v in state['validation']},
        'input_sha256': hashes,
        'attribution': ('The full-stack gain includes tangential controller changes. Only learned_vs_calibrated_only isolates the feed policy.'
            if state.get('adaptive_velocity_enabled', True) else
            'Constant feed was selected. Improvement comes from learned tangential MDK; learned and calibrated_only are independent replicates of the same controller.'),
        'limits': [
            'Uncalibrated F*TCP-distance surface estimate; no physical material-removal measurement.',
            'Three held-out sensor/delay conditions on one 697.53 mm flat reference path.',
            'Mean and range are descriptive, not confidence intervals or a generalization claim.',
            'Completion uses the reference cursor; actual endpoint settling is not simulated.',
            'Headless simulator tracking differs from the ROS/Isaac GUI plant.',
            'ROS clock corrections have separate integration tests; no GUI machining rerun is included.',
            'This checkpoint requires its matching tangential MDK configuration.',
            'No results from this experiment\'s listed final validation seeds were used to tune it; prior development seeds are listed separately.',
        ],
    }
    (root / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    plot(root, report, seeds[0])
    print(json.dumps({k: report[k] for k in ('frozen_gates_passed', 'learned_parameters',
        'spatial_cv_gain_percent_range_vs_both_baselines',
        'roi_volume_gain_percent_range_vs_both_baselines', 'paired_comparisons')}, indent=2))


def plot(root, report, representative_seed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    stats = report['statistics']
    labels = LABELS if report['adaptive_velocity_enabled'] else (
        *LABELS[:3], 'Selected constant feed')
    metrics = [('spatial_depth_cv', 'Spatial depth CV'),
               ('roi_volume_over_k', 'ROI volume / K (uncalibrated)'),
               ('raw_rate_cv', 'Raw removal-rate CV'),
               ('processing_time_s', 'Processing time [s]'),
               ('cross_track_rmse_mm', 'Cross-track RMS [mm]'),
               ('endpoint_error_mm', 'Endpoint error [mm]'),
               ('force_rmse_n', 'Force RMS error [N]'),
               ('force_mean_n', 'Mean force [N]'),
               ('force_peak_n', 'Peak force [N]')]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), layout='constrained')
    for ax, (field, title) in zip(axes.flat, metrics):
        mean = np.array([stats[m][field]['mean'] for m in MODES])
        low = np.array([stats[m][field]['min'] for m in MODES])
        high = np.array([stats[m][field]['max'] for m in MODES])
        ax.bar(labels, mean, color=COLORS, yerr=np.maximum(0., np.stack([mean-low, high-mean])), capsize=4)
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=25, labelsize=9)
        ax.grid(axis='y', alpha=.2)
        ax.set_ylim(0, high.max()*1.23)
        for i, value in enumerate(mean):
            ax.annotate(f'{value:.3g}', (i, high[i]), ha='center',
                        xytext=(0, 6), textcoords='offset points', fontsize=9)
    verdict = 'PASS' if report['frozen_gates_passed'] else 'FAIL'
    fig.suptitle(f'RPM excluded | Frozen simulator validation: {verdict}\n'
                 '3 held-out seeds: mean and full trial range (two constant replicates) | MDK-only ablation')
    fig.savefig(root / 'validation_comparison.png', dpi=160)
    plt.close(fig)
    # Fix the representative seed by protocol order, never by performance.
    maps = []
    for name in ('constant_a', 'old_d14', 'calibrated_only', 'learned'):
        with np.load(root / f'held_out_{representative_seed}/{name}_preston_surface.npz') as data:
            maps.append(dict(data))
    maximum = max(float(m['depth_over_k'][m['roi']].max()) for m in maps)
    fig, axes = plt.subplots(1, 4, figsize=(15, 5), layout='constrained')
    for ax, data, title in zip(axes, maps, labels):
        view = ax.imshow(np.where(data['roi'], data['depth_over_k'], np.nan), origin='lower',
                         vmin=0, vmax=maximum, extent=[data['u_mm'][0], data['u_mm'][-1],
                                                       data['v_mm'][0], data['v_mm'][-1]])
        ax.set(title=title, xlabel='u [mm]', ylabel='v [mm]')
    fig.colorbar(view, ax=axes, label='Depth / K (uncalibrated)')
    fig.suptitle(f'Identical reference ROI and color scale; zeros included | seed {representative_seed}')
    fig.savefig(root / 'validation_depth_maps.png', dpi=160)
    plt.close(fig)
    reference_file = _REPO_ROOT / (
        'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/datasets/cmd_continue9D_flat.h5')
    with h5py.File(reference_file) as data:
        reference = data['position'][:, :3]
    fig, axes = plt.subplots(1, 4, figsize=(15, 5), layout='constrained')
    for ax, data, mode, name, title, color in zip(
            axes, maps, MODES, ('constant_a', 'old_d14', 'calibrated_only', 'learned'), labels, COLORS):
        with np.load(root / f'held_out_{representative_seed}/{name}_trace.npz') as trace:
            xyz = trace['tcp_pose'][trace['polishing_active'] > 0, :3]
        target = (reference - data['origin_mm']) @ data['basis'].T
        actual = (xyz - data['origin_mm']) @ data['basis'].T
        ax.plot(target[:, 0], target[:, 1], '--', color='#888888', lw=1., label='Reference')
        ax.plot(actual[:, 0], actual[:, 1], color=color, lw=1., label='Measured TCP')
        ax.plot(actual[-1, 0], actual[-1, 1], 'o', color=color, ms=4)
        ax.set(title=title, xlabel='u [mm]', ylabel='v [mm]')
        ax.set_aspect('equal')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=9)
    fig.suptitle(f'Measured path coverage at the same 20 N target | held-out seed {representative_seed}')
    fig.savefig(root / 'validation_paths.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('training_directory', type=Path)
    summarize(parser.parse_args().training_directory)
