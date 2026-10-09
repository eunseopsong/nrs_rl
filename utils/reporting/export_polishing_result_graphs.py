#!/usr/bin/env python3
"""Export saved TCP-removal validation traces in the polishing_results layout.

No Isaac session or controller is started. Every CV is computed from the raw
processing samples. Spatial rate maps, temporal rate means, temporal variation
and accumulated removal are explicitly distinguished.
"""
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
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter


ROOT = _REPO_ROOT
DEFAULT_TRAINING = ROOT / 'logs/experiments/tcp_removal_retrain_20261005/calibrated_control/force_stability/training'
REFERENCE_LAYOUT = ROOT / 'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/logs/polishing_results/20260909_123708_flat'
REFERENCE_PATH = ROOT / 'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/datasets/cmd_continue9D_flat.h5'
MODES = ('constant_a', 'constant_b', 'old_d14', 'calibrated_only', 'learned')
LABELS = {'constant_a': 'Original constant A', 'constant_b': 'Original constant B',
          'old_d14': 'Old Mode 5 (D14)', 'calibrated_only': 'New MDK + constant',
          'learned': 'New MDK + policy'}
COLORS = {'constant_a': '#666666', 'constant_b': '#999999', 'old_d14': '#D6604D',
          'calibrated_only': '#8065a9', 'learned': '#2166AC'}
RATE_LABEL = 'Preston rate [N m/s]'
SPATIAL_SIGMA_MM = 8.
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'figure.titlesize': 11,
    'axes.linewidth': .8, 'savefig.dpi': 220})


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def savefig(fig, path, *, footer=None):
    fig.tight_layout(rect=(0, .065 if footer else 0, 1, .95))
    if footer:
        fig.text(.5, .012, footer, ha='center', va='bottom', fontsize=7)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def decorate(ax, duration=None):
    ax.grid(alpha=.25, linewidth=.5)
    ax.tick_params(direction='in')
    if duration is not None:
        ax.set_xlim(0, duration)


def load_trials(training):
    state = json.loads((training / 'result.json').read_text())
    report = json.loads((training / 'report.json').read_text())
    if state['phase'] != 'complete' or not state['validation_passed']:
        raise ValueError('This exporter requires the completed final validation')
    if digest(REFERENCE_PATH) != state['reference_path_sha256']:
        raise ValueError('Reference trajectory changed')
    trials, sources = [], {}
    for episode, seed in enumerate(state['validation_seeds'], 1):
        batch_path = training / f'held_out_{seed}/summary.json'
        batch = json.loads(batch_path.read_text())
        sources[str(batch_path)] = digest(batch_path)
        if batch['rotation_depth_model'] is not None:
            raise ValueError('Expected RPM-free removal data')
        if batch['candidate_checkpoint_sha256'] != report['checkpoint_sha256']:
            raise ValueError('Held-out checkpoint differs from the final report')
        for mode in MODES:
            path = batch_path.parent / f'{mode}_trace.npz'
            with np.load(path, allow_pickle=False) as data:
                trace = {key: data[key].copy() for key in data.files}
            active = trace['polishing_active'] > 0
            indices = np.flatnonzero(active)
            if not len(indices) or indices[-1] - indices[0] + 1 != len(indices):
                raise ValueError('Expected one contiguous processing interval')
            time = trace['time_s'][active]
            time = time - time[0]
            rate = trace['raw_mrr_n_mm_s'][active].astype(float)
            dt = batch['dt_s']
            if (not np.allclose(np.diff(time), dt, atol=1e-9) or
                    not np.isfinite(rate).all() or (rate < 0).any() or rate.mean() <= 0):
                raise ValueError('Invalid raw processing timestamps or removal rate')
            cv = float(rate.std(ddof=0) / rate.mean())
            metrics = batch['candidates'][mode]
            if (len(rate) != metrics['processing_samples'] or
                    not np.isclose(cv, metrics['processing_rate_cv'], rtol=1e-10)):
                raise ValueError('Raw-sample CV does not reproduce the frozen evaluation')
            trials.append({'episode': episode, 'seed': seed, 'mode': mode,
                'label': LABELS[mode], 'path': path, 'trace': trace, 'active': active,
                'time': time, 'rate': rate, 'cv': cv, 'dt': dt,
                'duration': len(rate) * dt, 'metrics': metrics})
            sources[str(path)] = digest(path)
    for name in ('result.json', 'report.json', 'protocol.json'):
        sources[str(training / name)] = digest(training / name)
    return state, report, trials, sources


def spatial_rate(trial, extent, bins=220):
    """Legacy-style normalized smoothing of occupied cell means, not dose."""
    xy = trial['trace']['tcp_pose'][trial['active'], :2]
    count, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins,
        range=[extent[:2], extent[2:]])
    total, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins,
        range=[extent[:2], extent[2:]], weights=trial['rate'] * .001)
    occupied = count.T > 0
    mean = np.divide(total.T, count.T, out=np.zeros_like(total.T), where=occupied)
    cell_x, cell_y = (extent[1]-extent[0])/bins, (extent[3]-extent[2])/bins
    sigma = (SPATIAL_SIGMA_MM/cell_y, SPATIAL_SIGMA_MM/cell_x)
    support = gaussian_filter(occupied.astype(float), sigma=sigma)
    smoothed = np.divide(gaussian_filter(mean, sigma=sigma), support,
        out=np.zeros_like(mean), where=support > 1e-9)
    distance = distance_transform_edt(~occupied, sampling=(cell_y, cell_x))
    opacity = np.clip((np.exp(-.5*(distance/SPATIAL_SIGMA_MM)**2)-.05)/.95, 0, 1)
    opacity[support <= 1e-9] = 0
    return np.ma.array(smoothed, mask=opacity == 0), opacity


def draw_map(ax, trial, extent, upper, title):
    grid, opacity = trial['spatial_map']
    ax.set_facecolor(plt.get_cmap('viridis')(0.))
    image = ax.imshow(grid.filled(0), origin='lower', extent=extent,
        cmap='viridis', vmin=0, vmax=upper, alpha=opacity, interpolation='bilinear')
    ax.set(title=title, xlabel='X [mm]', ylabel='Y [mm]', aspect='equal')
    ax.text(.018, .982, f"Raw temporal CV = {trial['cv']:.3f}\n"
        f"Mean rate = {trial['rate'].mean()*.001:.4f} N m/s",
        transform=ax.transAxes, va='top', fontsize=8,
        bbox={'facecolor': 'white', 'alpha': .87, 'edgecolor': '.4', 'boxstyle': 'round,pad=.3'})
    return image


def temporal_bins(trial, width, end):
    count_bins = int(np.ceil(end / width))
    ids = np.floor((trial['time'] + 1e-10) / width).astype(int)
    count = np.bincount(ids, minlength=count_bins)
    total = np.bincount(ids, weights=trial['rate'], minlength=count_bins)
    means = np.divide(total, count, out=np.full(count_bins, np.nan), where=count > 0)
    relative = (trial['rate'] - trial['rate'].mean()) / trial['rate'].mean()
    square_sum = np.bincount(ids, weights=relative**2, minlength=count_bins)
    rms = np.sqrt(np.divide(square_sum, count, out=np.full(count_bins, np.nan), where=count > 0))
    reconstructed_cv = float(np.sqrt(np.nansum(rms**2 * count) / count.sum()))
    if count.sum() != len(trial['rate']) or not np.isclose(reconstructed_cv, trial['cv'], rtol=1e-10):
        raise ValueError('Time-window heatmap does not conserve the raw CV')
    return {'mean': means, 'relative_rms': rms, 'count': count,
            'cv_reconstructed': reconstructed_cv}


def temporal_heatmap(trials, output, end, rate_upper, fluctuation_upper, prefix):
    # This window divides 8 ms exactly: 16 samples / column, with a partial tail.
    for width, key, suffix, title, colorbar, upper in (
        (.128, 'mean', 'rate_heatmap', 'Instantaneous removal rate over time',
         'Mean rate / 128 ms [N m/s]', rate_upper),
        (1., 'relative_rms', 'fluctuation_heatmap', 'Temporal fluctuation of instantaneous removal rate',
         'Relative RMS deviation / 1 s', fluctuation_upper)):
        arrays = [temporal_bins(t, width, end) for t in trials]
        matrix = np.stack([a[key] for a in arrays]) * (.001 if key == 'mean' else 1.)
        fig, ax = plt.subplots(figsize=(12, max(3.3, .38*len(trials)+1.7)))
        cmap = plt.get_cmap('viridis' if key == 'mean' else 'magma').copy()
        cmap.set_bad('#e0e0e0')
        image = ax.imshow(matrix, origin='upper', aspect='auto', cmap=cmap,
            interpolation='nearest', vmin=0, vmax=upper,
            extent=(0, matrix.shape[1]*width, len(trials)-.5, -.5))
        ax.set_yticks(np.arange(len(trials)), [
            f"{t['seed']} | {t['label']} | CV {t['cv']:.3f}" for t in trials])
        for i in range(1, len(trials)):
            if trials[i]['seed'] != trials[i-1]['seed']:
                ax.axhline(i-.5, color='white', lw=2)
        ax.set(xlabel='Elapsed simulation time from processing start [s]', title=title)
        fig.colorbar(image, ax=ax, label=colorbar, pad=.025, fraction=.025)
        footer = ('Raw 125 Hz samples; 128 ms means are for display. CV labels use ALL raw samples. '
                  'Grey: no observations; zeros retained.' if key == 'mean' else
                  'Each cell = RMS[(rate - full-run mean) / full-run mean] over 1 s. '
                  'Higher = more variation. Count-weighted RMS across cells equals raw CV.')
        fig.tight_layout(rect=(0, .065, 1, .97))
        fig.text(.5, .012, footer+'\nRPM excluded; simulation proxy, not measured material removal.',
                 ha='center', va='bottom', fontsize=7)
        filename = f'{prefix}_{suffix}.png'
        fig.savefig(output / filename, dpi=220)
        plt.close(fig)
        rows = []
        for trial, values in zip(trials, arrays):
            for index, count in enumerate(values['count']):
                if count:
                    rows.append({'seed': trial['seed'], 'mode': trial['mode'],
                        'window_start_s': index*width,
                        'window_end_s': min((index+1)*width, trial['duration']),
                        'samples': int(count), 'mean_rate_n_mm_s': float(values['mean'][index]),
                        'relative_rms_from_run_mean': float(values['relative_rms'][index]),
                        'raw_run_cv': trial['cv']})
        write_csv(output / filename.replace('.png', '.csv'), rows)


def signal_plot(trial, output, filename, target_force, title):
    data, active = trial['trace'], trial['active']
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 4.4), sharex=True)
    axes[0].plot(trial['time'], data['normal_force_n'][active], color='#B2182B', lw=.8)
    axes[0].axhline(target_force, color='.35', ls='--', lw=.7, label=f'Target {target_force:g} N')
    axes[0].set(title='Measured normal force', ylabel='Force [N]')
    axes[0].legend(loc='upper right')
    axes[1].plot(trial['time'], data['measured_speed_mm_s'][active]*.001,
                 color='#1B7837', lw=.8, label='Measured TCP sliding speed')
    axes[1].plot(trial['time'], data['commanded_speed_mm_s'][active]*.001,
                 color='.3', ls='--', lw=.7, label='Applied speed command')
    axes[1].set(title='Sliding velocity', xlabel='Processing time [s]', ylabel='Velocity [m/s]')
    axes[1].legend(loc='upper right')
    for ax in axes:
        decorate(ax, trial['duration'])
        ax.margins(y=.12)
    fig.suptitle(title)
    savefig(fig, output / filename)


def episode_plots(group, output, reference, extent, map_upper, raw_upper, temporal_upper, fluctuation_upper, end):
    old, new, constant = (group[k] for k in ('old_d14', 'learned', 'constant_a'))
    seed = new['seed']
    fig, ax = plt.subplots(figsize=(5.4, 4.5))
    image = draw_map(ax, new, extent, map_upper, 'Spatial Preston Rate: '+new['label'])
    fig.colorbar(image, ax=ax, label='Mean '+RATE_LABEL, fraction=.048, pad=.035)
    savefig(fig, output/'01_preston_rate_heatmap.png',
        footer=f'Seed {seed} | Mean of instantaneous rates projected onto XY, spatial smoothing 8 mm.\n'
               'Not accumulated removal. Temporal variation is shown separately in 10/11.')

    fig, ax = plt.subplots(figsize=(6.8, 3.3))
    ax.plot(new['time'], new['rate']*.001, color=COLORS['learned'], lw=.8,
            label=f"{new['label']} (raw CV={new['cv']:.3f})")
    ax.axhline(new['rate'].mean()*.001, color='.3', ls='--', lw=.8, label='Full-window mean')
    ax.set(title='Preston Rate Profile: '+new['label'], xlabel='Processing time [s]', ylabel=RATE_LABEL,
           ylim=(0, raw_upper*1.05))
    decorate(ax, end)
    ax.legend(loc='upper right')
    savefig(fig, output/'02_adaptive_preston_rate_profile.png',
            footer='Raw 125 Hz signal; complete processing interval, including zero rates and entry transients.')
    signal_plot(new, output, '03_adaptive_force_velocity_signals.png', 20., f"{new['label']} | seed {seed}")

    xyz = new['trace']['tcp_pose'][new['active'], :3]
    fig = plt.figure(figsize=(6.5, 5.4))
    ax = fig.add_subplot(111, projection='3d')
    segments = np.stack([xyz[:-1], xyz[1:]], axis=1)
    line = Line3DCollection(segments, cmap='viridis', norm=Normalize(0, raw_upper), lw=1.6)
    line.set_array(.5*(new['rate'][:-1]+new['rate'][1:])*.001)
    ax.add_collection3d(line)
    ax.plot(*reference[:, :3].T, color='.3', ls='--', lw=.8, label='Reference path')
    ax.scatter(*xyz[0], color='#1B7837', s=24, label='Processing start')
    ax.scatter(*xyz[-1], color='#B2182B', s=24, label='Processing end')
    both = np.vstack([xyz, reference[:, :3]])
    center = (both.min(0)+both.max(0))*.5
    radius = max(np.ptp(both, axis=0).max()*.55, 1.)
    ax.set(xlim=(center[0]-radius,center[0]+radius), ylim=(center[1]-radius,center[1]+radius),
           zlim=(center[2]-radius,center[2]+radius), xlabel='X [mm]', ylabel='Y [mm]',
           zlabel='Z [mm]', title='Processing-Window End-Effector Path')
    ax.set_box_aspect((1,1,1))
    ax.view_init(elev=28, azim=-58)
    ax.legend(loc='upper left')
    fig.colorbar(line, ax=ax, label=RATE_LABEL, pad=.1, shrink=.65)
    savefig(fig, output/'04_contact_path_3d.png')

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.8), sharex=True, sharey=True)
    for ax, trial in zip(axes, (old, new)):
        image = draw_map(ax, trial, extent, map_upper, trial['label'])
    fig.subplots_adjust(left=.08, right=.88, bottom=.19, top=.85, wspace=.2)
    fig.colorbar(image, cax=fig.add_axes([.90,.24,.02,.51]), label='Mean '+RATE_LABEL)
    fig.suptitle(f'Spatial Projection of Preston Rate | seed {seed}')
    fig.text(.5,.025,'Independent simulated rollouts; shared XY extent and physical color scale.\n'
             'Labels report raw temporal CV; spatial smoothing does not enter the CV.', ha='center', fontsize=7)
    fig.savefig(output/'05_preston_rate_comparison.png', dpi=220)
    plt.close(fig)

    signal_plot(constant, output, '06_constant_force_velocity_signals.png', 20.,
                f'Independent original constant-speed rollout A | seed {seed}')
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for trial in (constant, old, new):
        ax.plot(trial['time'], trial['rate']*.001, lw=.65, alpha=.8,
                color=COLORS[trial['mode']], label=f"{trial['label']} (CV={trial['cv']:.3f})")
    ax.set(title='Temporal Preston Rate Profile', xlabel='Processing time [s]',
           ylabel=RATE_LABEL, ylim=(0, raw_upper*1.05))
    decorate(ax, end)
    ax.legend(loc='upper right')
    savefig(fig, output/'07_preston_rate_profile_comparison.png',
        footer='Independent simulated rollouts. Raw 125 Hz data; no temporal smoothing or peak clipping.')

    shutil.copy2(new['path'], output/'08_control_trace.npz')
    comparison_dir = output/'comparison_traces'
    comparison_dir.mkdir()
    for mode in MODES[:-1]:
        shutil.copy2(group[mode]['path'], comparison_dir/f'{mode}_trace.npz')
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.6), sharex=True)
    data, active = new['trace'], new['active']
    for key, label, color, style in (
        ('measured_speed_mm_s','Measured TCP','#1B7837','-'),
        ('commanded_speed_mm_s','Applied command','#2166AC','-'),
        ('requested_speed_mm_s','Requested speed','.35','--')):
        axes[0].plot(new['time'], data[key][active], label=label, color=color, ls=style, lw=.8)
    axes[0].set_ylabel('Speed [mm/s]')
    axes[0].legend(loc='upper right')
    axes[1].plot(new['time'], data['command_acceleration_mm_s2'][active], lw=.8)
    axes[1].set_ylabel('Command acceleration\n[mm/s²]')
    axes[2].plot(new['time'], data['command_jerk_mm_s3'][active], lw=.8)
    axes[2].set(ylabel='Command jerk\n[mm/s³]', xlabel='Processing time [s]')
    for ax in axes:
        decorate(ax, end)
    fig.suptitle(f'Raw Commands and Measured TCP | seed {seed}')
    savefig(fig, output/'09_command_tracking.png')
    temporal_heatmap(list(group.values()), output, end, temporal_upper, fluctuation_upper, '10_temporal')
    # Keep the two added plot families consecutive after the historical 01--09.
    for suffix in ('.png', '.csv'):
        (output/f'10_temporal_fluctuation_heatmap{suffix}').rename(output/f'11_temporal_fluctuation_heatmap{suffix}')


def metric_row(trial):
    metrics, trace, active = trial['metrics'], trial['trace'], trial['active']
    surface = metrics['preston_surface']
    return {
        'episode': trial['episode'], 'seed': trial['seed'],
        'data_role': 'held_out_validation', 'mode': trial['mode'],
        'processing_samples': len(trial['rate']), 'duration_s': trial['duration'],
        'mean_rate_n_mm_s': float(trial['rate'].mean()),
        'std_rate_n_mm_s': float(trial['rate'].std(ddof=0)),
        'raw_rate_cv': trial['cv'],
        'zero_rate_fraction': float(np.mean(trial['rate'] == 0)),
        'force_mean_n': metrics['force_mean_n'], 'force_rmse_n': metrics['force_rmse_n'],
        'force_peak_n': metrics['force_peak_n'],
        'mean_sliding_velocity_mm_s': float(trace['measured_speed_mm_s'][active].mean()),
        'measured_path_length_mm': metrics['measured_path_length_mm'],
        'spatial_depth_cv': surface['spatial_depth_cv'],
        'roi_volume_over_k': surface['roi_volume_over_k'],
        'full_footprint_volume_over_k': surface['tcp_full_footprint_volume_over_k'],
        'uncovered_fraction': surface['zero_depth_fraction'],
        'cross_track_rmse_mm': metrics['cross_track_rmse_mm'],
        'endpoint_tangential_error_mm': metrics['endpoint_tangential_error_mm'],
        'shield_fraction': metrics['shield_fraction'], 'fault_fraction': metrics['fault_fraction']}


def summaries(output, trials, state, report):
    rows = [metric_row(t) for t in trials]
    write_csv(output/'00_comparison_summary.csv', rows)
    validation = {v['seed']: v for v in state['validation']}
    new_rows = []
    for trial, row in zip(trials, rows):
        if trial['mode'] != 'learned':
            continue
        row = dict(row, validation_score=validation[trial['seed']]['score'],
                   frozen_gates_passed=validation[trial['seed']]['passed'])
        new_rows.append(row)
        text = [f"Episode {trial['episode']} | held-out validation seed {trial['seed']}",
                'Frozen new MDK + velocity policy; these are validation runs, not training iterations.',
                f"Checkpoint SHA-256: {report['checkpoint_sha256']}",
                'RPM excluded; F * measured TCP sliding velocity proxy. No material calibration.',
                'Graph rate unit: N m/s. CSV raw rate unit: N mm/s (divide by 1000 for graphs).',
                'Spatial depth and volume are uncalibrated Preston estimates; /K retained.', '',
                *[f'{key}: {value}' for key, value in row.items()], '', 'Same-seed comparisons:']
        for base in (t for t in trials if t['seed'] == trial['seed'] and t['mode'] != 'learned'):
            text.append(f"{base['label']}: raw CV={base['cv']:.9f}; "
                        f"new CV reduction={100*(1-trial['cv']/base['cv']):.4f}%")
        text += ['', '08_control_trace.npz is an exact copy of the saved learned_trace.npz.',
                 'comparison_traces/ contains the four independent comparison traces.',
                 f"Source: {trial['path']}"]
        (output/f"ep{trial['episode']}"/'00_summary.txt').write_text('\n'.join(text)+'\n')
    write_csv(output/'00_episode_summary.csv', new_rows)
    paired = report['paired_comparisons']['learned_vs_old_d14']
    lines = [f'Run summary: {output.name}', 'Export of completed training and held-out validation.',
        f"Algorithm: {state['algorithm']}", f"Selected family: {state['selected_family']}",
        f"Checkpoint SHA-256: {report['checkpoint_sha256']}",
        'ep1=seed409, ep2=seed410, ep3=seed411 (held-out conditions; not training epochs).',
        'Each episode includes all five independently recorded controller variants.',
        'RPM excluded; contact-force target 20 N; raw period 0.008 s / 125 Hz.',
        'The 20260909_123708_flat folder supplies the figure layout only, not comparison data.', '',
        'Mean across the three validation conditions:',
        'mode | raw temporal CV | spatial depth CV | ROI volume/K | force RMSE [N]']
    for mode, stats in report['statistics'].items():
        lines.append(f"{mode} | {stats['raw_rate_cv']['mean']:.6f} | "
            f"{stats['spatial_depth_cv']['mean']:.6f} | {stats['roi_volume_over_k']['mean']:.3f} | "
            f"{stats['force_rmse_n']['mean']:.6f}")
    lines += ['', 'Paired mean improvements versus old Mode 5:',
        f"Raw temporal CV reduction: {paired['raw_rate_cv_reduction_percent']['mean']:.4f}%",
        f"Spatial depth CV reduction: {paired['spatial_depth_cv_reduction_percent']['mean']:.4f}%",
        f"ROI volume/K increase: {paired['roi_volume_increase_percent']['mean']:.4f}%", '',
        'Most improvement comes from MDK calibration. Additional adaptive-policy removal gain',
        'versus the calibrated constant controller is not demonstrated by these three trials.',
        'Simulator proxy results, not a new ROS machining run or calibrated material-removal measurement.',
        'reward_logs records actual candidate-search scores; no PPO reward time series is fabricated.']
    (output/'00_episode_summary.txt').write_text('\n'.join(lines)+'\n')


def reward_plots(state, output, sources):
    """Plot the saved candidate scores, without inventing per-step PPO rewards."""
    output.mkdir()
    component_rows, candidate_rows = [], []
    for generation in state['generations']:
        for family, assessment in generation['ranking'].items():
            replicate = min(assessment['replicates'], key=lambda row: row['score'])
            base = replicate['base_assessment']
            # All four final-search candidates pass the frozen gates. With no
            # penalty, the worst weighted comparison exactly determines score.
            if not base['passed']:
                raise ValueError('Reward decomposition requires unpenalized final-search candidates')
            weights = state['reward']['base']
            comparison = min(base['comparisons'], key=lambda c: sum(weights[k]*c[k] for k in weights))
            components = {key: weights[key]*comparison[key] for key in weights}
            if not np.isclose(sum(components.values()), base['score'], atol=1e-12):
                raise ValueError('Base reward decomposition differs from saved score')
            for label, weight, gain in (
                ('extra_force_vs_original', 'extra_worst_force_gain_vs_original', 'force_gain_vs_original'),
                ('force_vs_calibrated', 'worst_force_gain_vs_calibrated', 'force_gain_vs_calibrated'),
                ('rate_vs_calibrated', 'worst_rate_gain_vs_calibrated', 'raw_rate_gain_vs_calibrated')):
                components[label] = state['reward'][weight]*replicate[gain]
            if not np.isclose(sum(components.values()), assessment['score'], atol=1e-12):
                raise ValueError('Reward components do not reproduce the family score')
            row = {'generation': generation['generation'], 'seed': generation['seed'],
                'family': family, 'worst_replicate': replicate['name'],
                'eligible': assessment['eligible'], 'selected': family == generation['winner'],
                'total_score': assessment['score'], **components}
            component_rows.append(row)
            for rep in assessment['replicates']:
                candidate_rows.append({'generation': generation['generation'], 'seed': generation['seed'],
                    'family': family, 'replicate': rep['name'], 'score': rep['score'],
                    'eligible': rep['eligible'], 'base_score': rep['base_assessment']['score'],
                    'force_gain_vs_original': rep['force_gain_vs_original'],
                    'force_gain_vs_calibrated': rep['force_gain_vs_calibrated'],
                    'raw_rate_gain_vs_calibrated': rep['raw_rate_gain_vs_calibrated']})
    write_csv(output/'00_reward_components.csv', component_rows)
    write_csv(output/'00_candidate_scores.csv', candidate_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw={'width_ratios': [1, 1.5]})
    x = np.arange(len(component_rows))
    ticks = [r['family'].replace('_', '\n') for r in component_rows]
    scores = [r['total_score'] for r in component_rows]
    bars = axes[0].bar(x, scores, color=['#2166AC' if r['selected'] else '#b9c7d4' for r in component_rows])
    axes[0].bar_label(bars, fmt='%.4f', padding=4, fontsize=9)
    axes[0].set(title='Saved family scores', ylabel='Search reward', ylim=(0, max(scores)*1.17))
    positive, negative = np.zeros(len(x)), np.zeros(len(x))
    colors = plt.get_cmap('tab20').colors
    for index, component in enumerate(components):
        values = np.array([r[component] for r in component_rows])
        axes[1].bar(x, values, bottom=np.where(values >= 0, positive, negative),
            color=colors[index*2 % len(colors)], label=component.replace('_', ' '))
        positive += np.maximum(values, 0)
        negative += np.minimum(values, 0)
    axes[1].axhline(0, color='.35', lw=.7)
    axes[1].plot(x, scores, 'k_', ms=15, label='total score')
    axes[1].set(title='Weighted reward components', ylabel='Contribution to search reward')
    axes[1].legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=7)
    for ax in axes:
        ax.set_xticks(x, ticks)
        ax.grid(axis='y', alpha=.25)
        ax.set_axisbelow(True)
    fig.suptitle('Final velocity-policy search | training seed 319 | direct_quarter selected')
    savefig(fig, output/'00_reward_components.png', footer=
        'Actual bounded-parameter candidate search, not PPO training steps. '
        'Worst replicate / comparison determines each family score. Validation seeds were held out.')

    # Preserve the preceding two CEM generations as a separate training plot;
    # their scores include penalties and must not share a claimed reward scale
    # with the subsequent force-stability search above.
    previous_path = Path(state['previous_result'])
    if digest(previous_path) != state['previous_result_sha256']:
        raise ValueError('Preceding calibration training result changed')
    sources[str(previous_path)] = digest(previous_path)
    previous = json.loads(previous_path.read_text())
    fig, axes = plt.subplots(1, len(previous['generations']), figsize=(12, 4.7), squeeze=False)
    search_rows = []
    for ax, generation in zip(axes[0], previous['generations']):
        ranking = generation['ranking']
        names, scores = list(ranking), [r['score'] for r in ranking.values()]
        for name, assessment in ranking.items():
            search_rows.append({'generation': generation['generation'], 'seed': generation['seed'],
                'candidate': name, 'score': assessment['score'], 'passed': assessment['passed'],
                'selected_for_next_stage': name == generation['winner'],
                'rejections': ' | '.join(assessment['rejections'])})
        xx = np.arange(len(names))
        ax.scatter(xx, scores, c=['#2166AC' if name == generation['winner'] else '.55' for name in names], s=35)
        winner = names.index(generation['winner'])
        ax.scatter([winner], [scores[winner]], marker='*', s=150, color='#2166AC', label='Selected')
        ax.set_xticks(xx, names, rotation=50, ha='right')
        ax.set(title=f"CEM generation {generation['generation']} | seed {generation['seed']}", ylabel='Saved score (penalties included)')
        ax.grid(axis='y', alpha=.25)
        ax.legend(loc='best')
    savefig(fig, output/'01_calibration_search.png', footer=
        'Different training seeds and separate y-axis scales. All saved candidates shown. '
        'Generation 0 includes a baseline fault penalty; it is not a learning-curve discontinuity.')
    write_csv(output/'01_calibration_search.csv', search_rows)


def export(args):
    training = args.training.resolve()
    state, report, trials, sources = load_trials(training)
    checkpoint = Path(report['checkpoint'])
    if digest(checkpoint) != report['checkpoint_sha256']:
        raise ValueError('Final checkpoint changed since validation')
    sources[str(checkpoint)] = digest(checkpoint)
    sources[str(REFERENCE_PATH)] = digest(REFERENCE_PATH)
    with h5py.File(REFERENCE_PATH) as file:
        reference = file['position'][:]
    xy = np.vstack([reference[:, :2]] + [t['trace']['tcp_pose'][t['active'], :2] for t in trials])
    lower, upper = xy.min(axis=0)-20, xy.max(axis=0)+20
    extent = [float(lower[0]), float(upper[0]), float(lower[1]), float(upper[1])]
    for trial in trials:
        trial['spatial_map'] = spatial_rate(trial, extent)
    end = max(t['duration'] for t in trials)
    map_upper = float(max(t['spatial_map'][0].max() for t in trials))
    raw_upper = float(max(t['rate'].max() for t in trials)*.001)
    temporal_upper = float(max(np.nanmax(temporal_bins(t, .128, end)['mean']) for t in trials)*.001)
    fluctuation_upper = float(max(np.nanmax(temporal_bins(t, 1., end)['relative_rms']) for t in trials))
    timestamp = datetime.now(ZoneInfo('Asia/Seoul'))
    output = (args.output or ROOT/'logs/polishing_results'/timestamp.strftime('%Y%m%d_%H%M')).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for episode, seed in enumerate(state['validation_seeds'], 1):
        folder = output/f'ep{episode}'
        folder.mkdir()
        group = {t['mode']: t for t in trials if t['seed'] == seed}
        episode_plots(group, folder, reference, extent, map_upper, raw_upper,
                      temporal_upper, fluctuation_upper, end)
        print(f'Exported ep{episode}: held-out seed {seed}', flush=True)
    temporal_heatmap(trials, output, end, temporal_upper, fluctuation_upper, '00_temporal')
    summaries(output, trials, state, report)
    reward_plots(state, output/'reward_logs', sources)
    for source_name, target_name in (
        ('validation_comparison.png', '00_validation_overview.png'),
        ('report.json', '00_frozen_validation_report.json'),
        ('protocol.json', '00_frozen_validation_protocol.json')):
        source = training/source_name
        sources[str(source)] = digest(source)
        shutil.copy2(source, output/target_name)
    notes = f'''시각화 안내 — {output.name}

20260909_123708_flat의 ep*/01~09 파일 구성을 유지했습니다.
이번 ep1~ep3는 학습에 사용하지 않은 검증 seed 409, 410, 411입니다.
학습 epoch 번호가 아닙니다. 학습 후보별 실제 점수는 reward_logs에 따로 저장했습니다.
기존 폴더는 그래프 형식만 참고했습니다. 비교 곡선은 이번 검증의 독립된 기록입니다.

01: XY 위치별 순간 Preston 제거율 평균. 경로를 따라 반복 방문한 값의 평균입니다.
    기존 스타일의 공간 평활화(가우시안 sigma 8 mm)와 경로 외곽 투명도를 적용했습니다.
    누적 깊이, 누적 제거량 또는 시간별 변동을 나타내는 지도가 아닙니다.
02: 신규 모델 순간 제거율 원시 시계열 (125 Hz).
03: 신규 모델 접촉력/실측 TCP sliding velocity/적용 속도 명령.
04: 신규 모델 가공 구간의 3D 경로. 색은 해당 시각의 순간 제거율.
05: 기존 Mode 5와 신규 Mode 5의 공간 투영 비교; XY 범위와 색 범위 공통.
06: 원래 정속 제어 A의 독립된 접촉력/속도 기록.
07: 정속 A, 기존 Mode 5, 신규 Mode 5의 원시 시간 신호 비교.
08: 신규 모델 원본 trace.npz의 바이트 단위 동일 복사본.
09: 속도 요청/적용/실측 및 명령 가속도/jerk.
10: 시간별 순간 제거율 히트맵. 128 ms(16개 원시 샘플) 평균을 한 칸으로 표시.
11: 시간별 제거율 변동 히트맵. 1초 구간별 상대 RMS 편차.
    cell = sqrt(mean(((r(t) - 전체 가공 구간 평균) / 전체 가공 구간 평균)^2))
    밝을수록 해당 시간대가 그 주행 전체 평균에서 많이 벗어납니다.
    구간 내 표준편차/구간 평균으로 계산하는 local CV와 다릅니다.
    각 구간 샘플 수로 가중한 cell의 RMS는 전체 원시 제거율 CV와 정확히 같습니다.
    정상 상태에서 일정하게 높거나 낮아진 구간도 변동으로 포함됩니다.

모든 지표는 8 ms 원시 가공 샘플 전체에서 계산했습니다. 가공 시작/정지 중 0값과
초기 과도 응답을 삭제하지 않았습니다. 128 ms 평균은 히트맵 표시용으로만 사용했습니다.
CSV에는 각 시간 구간의 샘플 수와 원시 단위의 평균/상대 RMS를 저장했습니다.
각 주행 가공 시작을 t=0으로 맞췄고, 회색은 종료 이후 관측값이 없는 구간입니다.
색 범위는 15개 비교 주행 전체에서 공통이며, 포화 절단을 하지 않았습니다.
raw_mrr_n_mm_s는 컨트롤러가 저장한 접촉 판정 적용 값을 사용했습니다.
힘×속도로 다시 계산해 접촉 판정을 바꾸거나 누락하지 않았습니다.

모든 rate 그래프 단위: N m/s. 원시 NPZ/CSV의 rate 단위: N mm/s.
raw N mm/s를 1000으로 나눠 그래프의 N m/s로 표시합니다.
스핀들 RPM 기여는 제외했습니다. 접촉력 목표 20 N, 패드 직경 30 mm.
실제 재료 제거 계수가 보정되지 않은 시뮬레이션 proxy이므로 mm³/s가 아닙니다.
누적 공간 깊이 CV와 ROI volume/K는 기존 frozen 평가의 값을 그대로 저장했습니다.
이 export는 로봇/학습을 실행하지 않으며 새 ROS GUI 가공 측정도 포함하지 않습니다.

reward_logs/00: seed 319의 실제 최종 정책 후보 점수와 보상 성분.
각 family의 최저 점수 replicate와 최저 비교 점수 기준으로 성분 합을 검증했습니다.
reward_logs/01: 앞선 MDK CEM 탐색 2세대의 실제 후보 점수(패널별 독립 y축).
generation 0의 기준 제어 fault 페널티도 그대로 포함하므로 seed가 다른 세대 사이를
PPO 학습 곡선처럼 해석하면 안 됩니다. 에피소드별 PPO return은 만들지 않았습니다.

원본 데이터: {training}
형식 참고: {REFERENCE_LAYOUT}
재생성: python scripts/export_polishing_result_graphs.py --training {training}
'''
    (output/'00_visualization_notes.txt').write_text(notes)
    checks = []
    for trial in trials:
        folder = output/f"ep{trial['episode']}"
        copied = folder/'08_control_trace.npz' if trial['mode'] == 'learned' else folder/'comparison_traces'/f"{trial['mode']}_trace.npz"
        if digest(copied) != sources[str(trial['path'])]:
            raise ValueError('Exported trace differs from source')
        check = {'seed': trial['seed'], 'mode': trial['mode'], 'raw_samples': len(trial['rate']),
                 'raw_cv': trial['cv'], 'trace_copy_sha256_verified': True}
        for width in (.128, 1.):
            binned = temporal_bins(trial, width, end)
            check[f'{width:g}s_samples'] = int(binned['count'].sum())
            check[f'{width:g}s_reconstructed_cv'] = binned['cv_reconstructed']
        checks.append(check)
    for path, expected in sources.items():
        if digest(path) != expected:
            raise ValueError(f'Input changed during export: {path}')
    required = ('00_summary.txt', '01_preston_rate_heatmap.png', '02_adaptive_preston_rate_profile.png',
        '03_adaptive_force_velocity_signals.png', '04_contact_path_3d.png', '05_preston_rate_comparison.png',
        '06_constant_force_velocity_signals.png', '07_preston_rate_profile_comparison.png',
        '08_control_trace.npz', '09_command_tracking.png', '10_temporal_rate_heatmap.png',
        '11_temporal_fluctuation_heatmap.png')
    for episode in range(1, len(state['validation_seeds'])+1):
        for filename in required:
            if not (output/f'ep{episode}'/filename).is_file():
                raise ValueError(f'Missing graph-layout artifact: {filename}')
    manifest = {'created_at': timestamp.isoformat(), 'training_source': str(training),
        'reference_layout': str(REFERENCE_LAYOUT), 'export_script': str(Path(__file__).resolve()),
        'export_script_sha256': digest(__file__), 'checkpoint_sha256': report['checkpoint_sha256'],
        'episode_mapping': {f'ep{i}': seed for i, seed in enumerate(state['validation_seeds'], 1)},
        'data_role': 'held_out_simulation_validation', 'rotation_contribution': False,
        'robot_or_training_started': False, 'material_removal_calibrated': False,
        'display_rate_unit': 'N m/s', 'raw_rate_unit': 'N mm/s',
        'temporal_display_window_s': .128, 'fluctuation_window_s': 1.,
        'fluctuation_definition': 'RMS((raw_rate - full_run_mean) / full_run_mean) in each window',
        'shared_scales': {'xy_extent_mm': extent, 'spatial_rate_max_n_m_s': map_upper,
            'raw_rate_max_n_m_s': raw_upper, 'temporal_mean_rate_max_n_m_s': temporal_upper,
            'relative_rms_max': fluctuation_upper, 'time_end_s': end},
        'checks_passed': True, 'checks': checks, 'input_sha256': sources,
        'plot_files': sorted(str(p.relative_to(output)) for p in output.rglob('*.png'))}
    (output/'00_export_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(output), 'plots': len(manifest['plot_files']),
        'trials': len(trials), 'raw_samples': sum(len(t['rate']) for t in trials),
        'checks_passed': True}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, default=DEFAULT_TRAINING)
    parser.add_argument('--output', type=Path, help='New output directory; default: logs/polishing_results/YYYYMMDD_HHMM (KST)')
    export(parser.parse_args())


if __name__ == '__main__':
    main()
