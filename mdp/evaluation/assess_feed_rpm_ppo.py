"""Select only against predeclared RPM criteria; preserve all failed candidates."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import shlex

import numpy as np


def load(path):
    return json.loads(Path(path).read_text())


def measured_results(directory):
    rows = load(directory / 'results.json')['slots']
    target = load(directory / 'protocol.json')['rate_objective']['target_rate_over_k']
    for slot, row in rows.items():
        trace = np.load(directory / f'slot{slot}_trace.npz')
        error = trace['removal_rate_over_k'] / target - 1.
        index = int(error.argmax())
        entry = trace['time_s'] <= 3.
        row.update(maximum_positive_rate_error=float(max(0., error[index])),
            cursor_mm=float(trace['cursor_mm'][-1]),
            peak_positive_error_time_s=float(trace['time_s'][index]),
            entry_maximum_positive_rate_error=float(max(0., error[entry].max())),
            seconds_above_110pct_rate=float((error > .1).sum() * .008))
    return rows


def comparison(parent, candidate, mode6):
    def reduction(old, new):
        return 100. * (1. - new / old) if old else None
    result = {
        'spike_rate_rmse_reduction_pct': reduction(parent['spike_relative_rmse'], candidate['spike_relative_rmse']),
        'temporal_rate_cv_reduction_pct': reduction(parent['temporal_rate_cv'], candidate['temporal_rate_cv']),
        'spatial_depth_cv_reduction_vs_parent_pct': reduction(parent['contact_cv'], candidate['contact_cv']),
        'spatial_depth_cv_reduction_vs_mode6_pct': reduction(mode6['contact_cv'], candidate['contact_cv']),
        'contact_time_increase_vs_parent_pct': 100. * (candidate['contact_seconds'] / parent['contact_seconds'] - 1.),
        'estimated_volume_increase_vs_parent_pct': 100. * (candidate['contact_volume_over_k'] / parent['contact_volume_over_k'] - 1.),
        'late_window_rate_rmse_reductions_pct': {key: reduction(
            parent['windows'][key]['rate_relative_rmse'], candidate['windows'][key]['rate_relative_rmse'])
            for key in ('path85_90pct', 'path98p5_100pct')}}
    if 'maximum_positive_rate_error' in parent:
        result['maximum_positive_rate_error_reduction_pct'] = reduction(
            parent['maximum_positive_rate_error'], candidate['maximum_positive_rate_error'])
    return result


def eligible(parent, candidate, rules):
    checks = {
        'completed': candidate['completed'],
        'fault_free': candidate['fault_samples'] == rules['fault_samples'],
        'fixed_mean_rate_retained': rules['mean_rate_ratio_min'] <= candidate['mean_rate_ratio'] <= rules['mean_rate_ratio_max'],
        'spike_rate_error_improved': candidate['spike_relative_rmse'] <= parent['spike_relative_rmse'] * rules['spike_rate_rmse_ratio_max'],
        'whole_contact_temporal_cv_improved': candidate['temporal_rate_cv'] <= parent['temporal_rate_cv'] * rules['temporal_rate_cv_ratio_max'],
        'spatial_depth_cv_retained': candidate['contact_cv'] <= parent['contact_cv'] * rules['full_contact_spatial_cv_ratio_max'],
        'both_late_windows_improved': all(candidate['windows'][key]['rate_relative_rmse'] <
            parent['windows'][key]['rate_relative_rmse'] for key in ('path85_90pct', 'path98p5_100pct'))}
    if 'maximum_positive_rate_error_max' in rules:
        checks['maximum_positive_rate_spike_bounded'] = (
            candidate['maximum_positive_rate_error'] <= rules['maximum_positive_rate_error_max'])
    return checks


def assess(study):
    plan = load(study / 'plan.json')
    baseline = measured_results(study / 'baseline_5211')
    candidates = []
    for update in (0, 3, 6):
        path = study / f'candidate_{update:04d}_5211/results.json'
        run = measured_results(path.parent)
        for slot in ('0', '1', '2'):
            # All unchanged controllers must reproduce their own fixed slots.
            for key in ('contact_cv', 'contact_seconds', 'mean_rate_over_k', 'temporal_rate_cv'):
                np.testing.assert_allclose(run[slot][key], baseline[slot][key], rtol=1.e-8, atol=1.e-9)
        row = run['3']
        checks = eligible(baseline['3'], row, plan['selection_criteria'])
        candidates.append({'update': update, 'trained_by_ppo': update != 0,
            'directory': str(path.parent), 'metrics': row,
            'comparison': comparison(baseline['3'], row, baseline['0']),
            'checks': checks, 'eligible': all(checks.values()) and update != 0})
    qualified = [row for row in candidates if row['eligible']]
    best = min(qualified or [r for r in candidates if r['trained_by_ppo']],
               key=lambda row: row['metrics']['spike_relative_rmse'])
    result = {'criteria_frozen_before_training': plan['selection_criteria'],
        'matched_parent_slot': 3, 'candidate_slot': 3, 'baseline': baseline,
        'candidates': candidates, 'selected_update': best['update'],
        'passed_selection': bool(qualified), 'selected': best,
        'physical_spindle_actuated': False, 'automatic_runtime_deployment': False}
    heldout = study / f'heldout_{best["update"]:04d}_5212/results.json'
    if heldout.exists():
        base = measured_results(study / 'baseline_5212')
        new = measured_results(heldout.parent)
        checks = eligible(base['3'], new['3'], plan['selection_criteria'])
        result['heldout'] = {'seed': 5212, 'parent': base['3'], 'candidate': new['3'],
            'mode6': base['0'], 'checks': checks, 'passed': all(checks.values()),
            'comparison': comparison(base['3'], new['3'], base['0'])}
    (study / 'assessment.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def plot_contact_traces(runs, traces, graph, update, execution_label='Isaac headless'):
    """Show the four path signals after the initial three seconds."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    entry_seconds = 3.
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .22})
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    colors = ['#B2182B', '#1B7837', '#762A83', '#2166AC']
    light = ['#D6604D', '#AAAAAA', '#AF8DC3', '#92C5DE']
    labels = ['Measured normal force [N]', 'Scheduled path feed [mm/s]',
              'Virtual spindle RPM', 'Estimated removal rate / K\n(uncalibrated, per s)']
    fields = ['normal_force_n', 'commanded_speed_mm_s', 'rpm', 'removal_rate_over_k']
    policy_labels = ['Constant feed velocity, RPM', 'Adaptive feed velocity',
                     'Adaptive feed velocity, RPM']
    display = {'excluded_initial_contact_seconds': entry_seconds,
               'initial_contact_panels': False, 'signal_panels': 4, 'panels': {}}
    for row, field in enumerate(fields):
        ax = axes[row]
        limits = []
        for i, trace in enumerate(traces):
            mask = trace['time_s'] > entry_seconds
            x = trace['progress_pct']
            y = trace[field]
            color = light[row] if i == 0 else '#666666' if i == 1 else colors[row]
            ax.plot(x[mask], y[mask], color=color, ls='--' if i == 0 else ':' if i == 1 else '-',
                    linewidth=1.05 if i < 2 else 1.25, label=policy_labels[i], alpha=.95)
            limits.extend([float(y[mask].min()), float(y[mask].max())])
        low, high = min(limits), max(limits)
        margin = max((high - low) * .03, abs(high) * .0005, .01)
        ax.set_ylim(low - margin, high + margin)
        ax.set_xlim(0., 100.)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        display['panels'][field] = {'data_min': low, 'data_max': high,
                                   'y_min': low - margin, 'y_max': high + margin}
        ax.set_ylabel(labels[row])
    axes[-1].set_xlabel('Same path progress [%]')
    axes[0].legend(loc='lower left', ncol=3, fontsize=9)
    fig.suptitle(f'Checkpoint {update} | {execution_label} | TCP 146 mm | 20 N force target | nominal 1000 RPM, variable virtual spindle\n'
                 'Path after first 3 s | raw 125 Hz data', y=.98)
    fig.tight_layout(rect=(0, 0, 1, .95), h_pad=1.1)
    fig.savefig(graph, dpi=190)
    plt.close(fig)
    return display


def plot_temporal_removal_rate(runs, traces, graph, update, target_rate,
                               execution_label='Isaac headless evaluation'):
    """Compare raw full-contact rates with a shared y-axis zoom and full-contact CVs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = ['1. constant feed velocity, RPM', '2. adaptive feed velocity',
              '3. adaptive feed velocity, RPM']
    colors = ['#D6604D', '#666666', '#2166AC']
    styles = ['--', ':', '-']
    statistics = []
    for label, run, trace in zip(labels, runs, traces):
        rate = np.asarray(trace['removal_rate_over_k'])
        time = np.asarray(trace['time_s'])
        if not np.isfinite(rate).all() or rate.mean() <= 0. or len(rate) != len(time):
            raise ValueError('Finite, time-aligned removal rates with a positive mean are required.')
        np.testing.assert_allclose(np.diff(time), .008, rtol=1.e-8, atol=1.e-10)
        cv_pct = float(100. * rate.std() / rate.mean())
        np.testing.assert_allclose(cv_pct, 100. * run[3]['temporal_rate_cv'], rtol=1.e-7)
        statistics.append({'policy': label, 'samples': len(rate),
                           'mean_rate_over_k': float(rate.mean()),
                           'temporal_rate_cv_pct': cv_pct,
                           'contact_seconds': run[3]['contact_seconds']})
    reference_cv = statistics[0]['temporal_rate_cv_pct']
    feed_only_cv = statistics[1]['temporal_rate_cv_pct']
    for i, row in enumerate(statistics):
        row['cv_reduction_vs_constant_pct'] = (
            0. if i == 0 else 100. * (1. - row['temporal_rate_cv_pct'] / reference_cv))
        row['cv_reduction_vs_feed_only_pct'] = (
            0. if i == 1 else 100. * (1. - row['temporal_rate_cv_pct'] / feed_only_cv))

    style = {'font.family': 'DejaVu Sans', 'font.size': 11,
             'axes.spines.top': False, 'axes.spines.right': False,
             'axes.grid': True, 'grid.alpha': .20, 'figure.facecolor': 'white'}
    display = {'metric': 'temporal CV of instantaneous estimated removal rate',
               'cv_interval': 'all measured contact including entry; pre-contact excluded',
               'smoothing': 'none', 'target_rate_over_k': target_rate,
               'statistics': statistics, 'panels': {}}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(14, 9.2))
        grid = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.25],
                               left=.09, right=.98, bottom=.14, top=.79, hspace=.38)
        rate_axis = fig.add_subplot(grid[0])
        cv_axis = fig.add_subplot(grid[1])
        time_limit = float(np.ceil(max(row['contact_seconds'] for row in statistics) / 5.) * 5.)
        extrema = [target_rate]
        for i, trace in enumerate(traces):
            time = trace['time_s']
            rate = trace['removal_rate_over_k']
            rate_axis.plot(time, rate, color=colors[i], ls=styles[i],
                           linewidth=1.05 if i < 2 else 1.35, alpha=.95, label=labels[i])
            extrema.extend([float(rate.min()), float(rate.max())])
        rate_axis.axhline(target_rate, color='.25', ls=(0, (4, 3)), lw=.8, alpha=.6,
                         label='Fixed removal-rate target')
        low, high = 17000., 25000.
        rate_axis.set_ylim(low, high)
        rate_axis.set_xlim(0., time_limit)
        rate_axis.set_yticks(np.arange(low, high + 1., 2000.))
        rate_axis.ticklabel_format(axis='y', style='plain', useOffset=False)
        rate_axis.set_ylabel('Removal rate / K\n(uncalibrated, per s)')
        display['panels']['full_contact_fluctuation'] = {
            'y_limits': [low, high], 'x_limits_s': [0., time_limit],
            'raw_data_limits': [min(extrema), max(extrema)],
            'y_axis_zoom_only': True, 'all_contact_samples_included_in_cv': True}
        rate_axis.set_title('Removal-rate fluctuation after first measured contact', loc='left', pad=9)
        rate_axis.set_xlabel('Time since first measured contact [simulation s]', labelpad=8)

        cvs = [row['temporal_rate_cv_pct'] for row in statistics]
        cv_axis.bar(np.arange(3), cvs, width=.46, color=colors, alpha=.9)
        cv_axis.set_ylim(0., max(cvs) * 1.5)
        cv_axis.set_xlim(-.55, 2.55)
        cv_axis.set_xticks(np.arange(3), [label.replace('feed velocity', 'feed\nvelocity') for label in labels])
        cv_axis.set_ylabel('Temporal rate CV [%]')
        cv_axis.set_title('Temporal CV over the same full-contact interval', loc='left', pad=10)
        cv_axis.grid(axis='x', visible=False)
        cv_axis.set_axisbelow(True)
        for i, row in enumerate(statistics):
            cv_axis.text(i, cvs[i] + .07, f'{cvs[i]:.3f}%', ha='center', va='bottom',
                         fontsize=15, fontweight='bold', color=colors[i])
            comparison_text = ('Reference' if i == 0 else
                f'{row["cv_reduction_vs_constant_pct"]:.2f}% lower than 1')
            if i == 2:
                comparison_text += f'\n{row["cv_reduction_vs_feed_only_pct"]:.2f}% lower than 2'
            cv_axis.text(i, .96, comparison_text, transform=cv_axis.get_xaxis_transform(),
                         ha='center', va='top', fontsize=11)

        fig.suptitle('Instantaneous Removal Rate over Time', fontsize=23, y=.98)
        fig.text(.5, .94,
                 f'Checkpoint {update} | {execution_label} | TCP 146 mm | 20 N target | virtual RPM',
                 ha='center', va='center', fontsize=12, color='.25')
        handles, legend_labels = rate_axis.get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(.5, .913),
                   ncol=2, frameon=True, fontsize=11)
        fig.text(.5, .025,
            'Y-axis zoom: 17,000-25,000. CV = 100 x std(rate) / mean(rate) over all contact samples. '
            'Each curve ends at its own completion time.\n'
            'These are temporal removal-rate CVs; accumulated spatial depth CV is reported in PNG 2.',
            ha='center', va='bottom', fontsize=10, color='.25')
        fig.savefig(graph, dpi=190, facecolor='white')
        plt.close(fig)
    return display


def plot_contact_depths(runs, depths, heatmap, update, execution_label=None):
    """Display relative depth deviations whose RMS is the reported CV."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MaxNLocator

    deviations = []
    statistics = []
    reference_cv = runs[0][3]['contact_cv']
    for run, d in zip(runs, depths):
        values = d['depth_over_k'][d['roi']]
        mean = float(values.mean())
        if not np.isfinite(values).all() or mean <= 0.:
            raise ValueError('Finite depths with a positive ROI mean are required.')
        deviation = 100. * (d['depth_over_k'] / mean - 1.)
        # Normalizing by the mean is exactly the relative spread measured by CV.
        cv = float(np.sqrt(np.mean(np.square(deviation[d['roi']]))) / 100.)
        np.testing.assert_allclose(cv, run[3]['contact_cv'], rtol=1.e-8)
        deviations.append(np.ma.masked_where(~d['roi'], deviation))
        statistics.append({'roi_mean_depth_over_k': mean, 'raw_depth_cv': cv,
                           'cv_reduction_vs_reference_pct': 100. * (1. - cv / reference_cv)})
    maximum_deviation = max(float(np.abs(d).max()) for d in deviations)
    limit = max(10., float(np.ceil(maximum_deviation / 10.) * 10.))
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#E7E7E7')
    norm = Normalize(vmin=-limit, vmax=limit)
    titles = ['1. constant feed velocity, RPM', '2. adaptive feed velocity',
              '3. adaptive feed velocity, RPM']
    style = {'font.family': 'DejaVu Sans', 'font.size': 18,
             'axes.titlesize': 22, 'axes.labelsize': 22,
             'xtick.labelsize': 18, 'ytick.labelsize': 18,
             'axes.spines.top': True, 'axes.spines.right': True,
             'axes.grid': False, 'figure.facecolor': 'white'}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(22.5, 8.8), sharex=True, sharey=True)
        fig.subplots_adjust(left=.055, right=.91, bottom=.17, top=.79, wspace=.14)
        for i, (ax, run, d, title, field) in enumerate(zip(axes, runs, depths, titles, deviations)):
            im = ax.imshow(field, origin='lower',
                extent=[d['u_mm'][0], d['u_mm'][-1], d['v_mm'][0], d['v_mm'][-1]],
                cmap=cmap, norm=norm, interpolation='bilinear', interpolation_stage='data',
                aspect='equal')
            ax.set_facecolor('#E7E7E7')
            ax.set_title(title, pad=16)
            ax.set_xlabel('Plane u [mm]', labelpad=10)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.tick_params(direction='in', length=7, width=1.4, pad=8)
            info = ('Reference' if i == 0 else
                    f'Depth CV reduction = {statistics[i]["cv_reduction_vs_reference_pct"]:.2f}%')
            ax.text(.018, .982, f'{info}\nTime = {run[3]["contact_seconds"]:.2f} s',
                transform=ax.transAxes, va='top', ha='left', fontsize=19,
                linespacing=1.15, color='black',
                bbox={'boxstyle': 'round,pad=0.25', 'facecolor': 'white',
                      'edgecolor': '.35', 'linewidth': 1.5, 'alpha': .88})
            for spine in ax.spines.values():
                spine.set_linewidth(1.6)
                spine.set_color('.25')
        axes[0].set_ylabel('Plane v [mm]', labelpad=10)
        position = axes[-1].get_position()
        cax = fig.add_axes([.93, position.y0, .017, position.height])
        tick_step = max(10., float(np.ceil(limit / 4. / 10.) * 10.))
        tick_max = float(np.floor(limit / tick_step) * tick_step)
        ticks = np.arange(-tick_max, tick_max + tick_step / 2., tick_step)
        colorbar = fig.colorbar(im, cax=cax, ticks=ticks)
        colorbar.set_label('Deviation from each ROI mean depth [%]', labelpad=14)
        colorbar.ax.tick_params(direction='in', length=7, width=1.4, pad=8)
        colorbar.outline.set_linewidth(1.5)
        fig.suptitle('Spatial Uniformity of Removal Depth', fontsize=34, y=.97)
        fig.text(.5, .885,
                 f'Checkpoint {update} comparison' + (f' | {execution_label}' if execution_label else '')
                 + ' | spatial depth CV reductions relative to 1 | full contact, including entry',
                 ha='center', va='center', fontsize=17, color='.25')
        fig.text(.5, .045,
            'White = near own mean; blue = below; red = above | same ROI and linear color scale | CV = RMS(relative depth deviation)',
            ha='center', va='center', fontsize=13, color='.25')
        fig.savefig(heatmap, dpi=190, facecolor='white', bbox_inches='tight', pad_inches=.16)
        plt.close(fig)
    return {'quantity': '100 * (depth / mean_depth_in_same_ROI - 1)',
            'shared_color_limits_pct': [-limit, limit], 'color_normalization': 'linear',
            'colormap': 'RdBu_r', 'outside_roi': 'gray background, excluded from CV',
            'image_interpolation': 'bilinear', 'cv_calculation': 'unchanged raw full-ROI depth',
            'all_extrema_within_color_scale': True, 'panels': statistics}


def plot_and_archive(study, result):
    plan = load(study / 'plan.json')
    archive = Path(plan['archive'])
    archive.mkdir(parents=True, exist_ok=True)
    selected = result['selected']
    update = selected['update']
    candidate_dir = Path(selected['directory'])
    baseline_dir = study / 'baseline_5211'
    runs = [('Mode 6: fixed feed/RPM', baseline_dir, 0, result['baseline']['0']),
            ('Mode 5: feed only', baseline_dir, 3, result['baseline']['3']),
            ('Feed + RPM PPO', candidate_dir, 3, selected['metrics'])]
    copied = []
    traces = []
    depths = []
    for name, folder, slot, metrics in runs:
        label = ('mode6' if slot == 0 else 'rpm_ppo' if folder == candidate_dir else 'parent_mode5')
        for kind in ('trace', 'depth'):
            src = folder / f'slot{slot}_{kind}.npz'
            dest = archive / f'{label}_{kind}.npz'
            shutil.copy2(src, dest)
            copied.append(dest)
        traces.append(dict(np.load(folder / f'slot{slot}_trace.npz')))
        depths.append(dict(np.load(folder / f'slot{slot}_depth.npz')))
    table = archive / 'comparison.csv'
    with table.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['policy', 'contact_seconds', 'spike_rate_relative_rmse_pct',
            'whole_contact_rate_cv_pct', 'spatial_depth_cv_pct', 'mean_rate_over_target_pct',
            'maximum_positive_rate_error_pct', 'seconds_above_110pct_rate', 'rpm_min', 'rpm_max',
            'roi_volume_over_k'])
        for name, _, _, m in runs:
            writer.writerow([name, m['contact_seconds'], 100. * m['spike_relative_rmse'],
                100. * m['temporal_rate_cv'], 100. * m['contact_cv'], 100. * m['mean_rate_ratio'],
                100. * m['maximum_positive_rate_error'], m['seconds_above_110pct_rate'],
                m['rpm_min'], m['rpm_max'], m['contact_volume_over_k']])
    copied.append(table)
    table = archive / 'checkpoint_comparison.csv'
    with table.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['update', 'selected', 'eligible', 'spike_error_reduction_pct',
            'rate_cv_reduction_pct', 'maximum_positive_rate_error_pct'])
        for row in result['candidates']:
            writer.writerow([row['update'], row['update'] == update, row['eligible'],
                row['comparison']['spike_rate_rmse_reduction_pct'],
                row['comparison']['temporal_rate_cv_reduction_pct'],
                100. * row['metrics']['maximum_positive_rate_error']])
    copied.append(table)
    graph = archive / '01_force_feed_rpm_removal.png'
    plot_contact_traces(runs, traces, graph, update)
    copied.append(graph)
    heatmap = archive / '02_contact_depth_heatmaps.png'
    plot_contact_depths(runs, depths, heatmap, update)
    copied.append(heatmap)
    rate_graph = archive / '03_temporal_removal_rate.png'
    target_rate = load(candidate_dir / 'protocol.json')['rate_objective']['target_rate_over_k']
    plot_temporal_removal_rate(runs, traces, rate_graph, update, target_rate)
    copied.append(rate_graph)
    for src, dest in [(study / 'assessment.json', archive / 'comparison.json'),
                      (study / 'plan.json', archive / 'experiment_plan.json'),
                      (study / 'input_manifest.json', archive / 'input_manifest.json'),
                      (study / 'orchestration_revision.json', archive / 'orchestration_revision.json'),
                      (study / 'trainer_cli_revision.json', archive / 'trainer_cli_revision.json'),
                      (study / 'training/training_complete.json', archive / 'training_complete.json'),
                      (study / 'training/updates.jsonl', archive / 'training_updates.jsonl'),
                      (study / 'training/episodes.jsonl', archive / 'training_episodes.jsonl'),
                      (candidate_dir / 'protocol.json', archive / 'evaluation_protocol.json'),
                      (candidate_dir / 'live_robot_geometry_audit.json', archive / 'runtime_geometry.json'),
                      (study / 'training/protocol.json', archive / 'training_protocol.json'),
                      (study / f'training/checkpoints/ppo_{update:04d}.pt', archive / 'actor.pt'),
                      (study / f'training/checkpoints/training_{update:04d}.pt', archive / 'ppo_training_state.pt')]:
        shutil.copy2(src, dest)
        copied.append(dest)
    source_output = archive / 'source_snapshot'
    source_output.mkdir(exist_ok=True)
    for src in sorted((study / 'training/source_snapshot').glob('*.py')):
        dest = source_output / src.name
        shutil.copy2(src, dest)
        copied.append(dest)
    report = archive / 'results.txt'
    comp = selected['comparison']
    report.write_text('\n'.join([
        f'Feed+RPM PPO: update {update}; selection passed={result["passed_selection"]}',
        f'Heldout passed={result.get("heldout", {}).get("passed", "not run")}',
        f'제거율 스파이크 오차 감소율: {comp["spike_rate_rmse_reduction_pct"]:.4f}% (기존 Mode 5 대비)',
        f'전체 접촉 순간 제거율 CV 감소율: {comp["temporal_rate_cv_reduction_pct"]:.4f}% (기존 Mode 5 대비)',
        f'전체 접촉 순간 제거율 CV: {100.*result["baseline"]["3"]["temporal_rate_cv"]:.6f}% -> {100.*selected["metrics"]["temporal_rate_cv"]:.6f}%',
        f'최대 초과 제거율: {100.*selected["metrics"]["maximum_positive_rate_error"]:.4f}% (목표 대비)',
        f'목표의 110% 초과 제거율 지속 시간: {selected["metrics"]["seconds_above_110pct_rate"]:.4f}s',
        f'공간 제거 깊이 CV 감소율: {comp["spatial_depth_cv_reduction_vs_mode6_pct"]:.4f}% (Mode 6 대비)',
        f'공간 제거 깊이 CV 감소율: {comp["spatial_depth_cv_reduction_vs_parent_pct"]:.4f}% (기존 Mode 5 대비)',
        f'공간 제거 깊이 CV: {100.*selected["metrics"]["contact_cv"]:.6f}%',
        f'평균 제거율 / 고정 목표: {100.*selected["metrics"]["mean_rate_ratio"]:.6f}%',
        f'접촉부터 완료까지: 기존 {result["baseline"]["3"]["contact_seconds"]:.3f}s -> 새 정책 {selected["metrics"]["contact_seconds"]:.3f}s',
        f'사용 RPM 범위: {selected["metrics"]["rpm_min"]:.3f} ~ {selected["metrics"]["rpm_max"]:.3f}',
        '스파이크는 힘 이벤트 구간의 고정 목표 제거율 대비 상대 RMSE로 평가. 힘 진동 자체의 제거율이 아님.',
        '실제 Isaac 측정 힘/TCP + 가상 RPM 응답/Preston 모델. 실제 회전 접촉 동역학 검증 아님.',
        '최초 접촉 1.5N부터 종료까지 초반 정지 접촉 포함; 최초 접촉 이전 제외.',
        '첫 접촉 순간 RPM 응답 전의 저제거 과도응답은 남아 있으며 모든 전체 구간 지표에 포함했다.',
        'RPM 기본 1000; 하드웨어 한계 없음. 수치 탐색 범위 49.787~20085.537RPM.',
        '모터 응답 가정: 16ms 1차 지연, 200000 RPM/s 변화율. policy 16ms, physics 8ms.',
        '두 seed 모두 동일한 명목 센서/장면 조건의 반복 실행이며, 조건 변경에 대한 일반화 시험이 아님.',
        'K 미보정 상대 제거량. 기존 ROS 정책 자동 배포 없음.',
        f'전체 실험 로그: {study}']) + '\n')
    copied.append(report)
    invocation = load(study / 'training_invocation.json')['command']
    def argument(name):
        return invocation[invocation.index(name) + 1]
    evaluation = [invocation[0], '-m',
        'nrs_rl.tasks.manager_based.nrs_rl.mdp.evaluation.probe_feed_rpm_ppo',
        '--device', 'cpu', '--seed', '5212', '--parent',
        load(candidate_dir / 'protocol.json')['parent'], '--candidate', str(archive / 'actor.pt'),
        '--native-dir', argument('--native-dir'), '--control-snapshot', argument('--control-snapshot'),
        '--gui-scene', argument('--gui-scene')]
    script = archive / 'run_gui_evaluation.sh'
    script.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
        'if [ "$#" -lt 1 ]; then\n'
        '  printf "%s\\n" "Usage: $0 NEW_OUTPUT_DIRECTORY [--headless]"\n'
        '  exit 2\nfi\nrpm_evaluation_output="$1"\nshift\n'
        'cd /home/eunseop/nrs_rl\nexec ' + shlex.join(evaluation) +
        ' --output "$rpm_evaluation_output" "$@"\n')
    script.chmod(0o755)
    copied.append(script)
    manifest = [{'file': str(p.relative_to(archive)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in copied]
    (archive / 'artifact_manifest.json').write_text(json.dumps(manifest, indent=2))
    return archive


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--archive', action='store_true')
    args = p.parse_args()
    result = assess(args.study)
    print(json.dumps({'selected_update': result['selected_update'], 'passed': result['passed_selection'],
        'candidates': [{k: row[k] for k in ('update', 'eligible', 'checks', 'comparison')}
                       for row in result['candidates']]}, indent=2))
    if args.archive:
        print('ARCHIVE', plot_and_archive(args.study, result))
