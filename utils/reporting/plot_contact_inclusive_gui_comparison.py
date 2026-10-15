"""Replot a saved GUI pair from first measured contact through completion.

Keep the original audited results intact. Restore the raw contact samples before
Force control, add their Preston deposit to the saved full surface, and use the
whole reference ROI. No first-two-second or first-18-mm exclusion is applied.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator
import numpy as np

DT = .008
CONTACT_N = 1.5
# Match utils/visualization.py and the 20260909_123708_flat reference figures.
COLORS = {5: '#2166AC', 6: '#D6604D'}
SIGNAL_COLORS = (
    {5: '#B2182B', 6: '#D6604D'},  # Measured force.
    {5: '#1B7837', 6: '#4D4D4D'},  # Feed; constant reference in gray.
    COLORS,                       # Estimated removal rate.
)
REFERENCE_COLOR = '#4D4D4D'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def npz(path):
    with np.load(path) as data:
        return dict(data)


def columns(rows, keys):
    return np.column_stack([rows[key] for key in keys])


def measured_normal_force(raw):
    # Controller pose orientation is a spatial rotation vector, not Euler RPY.
    rot = columns(raw, ('actual_tcp_rx', 'actual_tcp_ry', 'actual_tcp_rz'))
    theta = np.linalg.norm(rot, axis=1)
    axis = rot / np.maximum(theta[:, None], 1.e-30)
    ez = np.broadcast_to([0., 0., 1.], rot.shape)
    normal = (ez * np.cos(theta)[:, None]
              + np.cross(axis, ez) * np.sin(theta)[:, None]
              + axis * axis[:, 2, None] * (1. - np.cos(theta))[:, None])
    return np.abs(np.sum(normal * columns(raw, ('fx', 'fy', 'fz')), axis=1))


def recover_prefix(directory):
    """Archive only the newly included raw rows; retain all existing CSVs."""
    prefix_path = directory / 'precontrol_contact.csv'
    metadata_path = directory / 'contact_window.json'
    if prefix_path.exists():
        metadata = json.loads(metadata_path.read_text())
        assert sha(prefix_path) == metadata['prefix_csv_sha256']
        return np.atleast_1d(np.genfromtxt(prefix_path, delimiter=',', names=True)), metadata
    source = Path(json.loads((directory / 'audit.json').read_text())['run_directory']) / 'raw.csv'
    # A logger killed after the run may leave an incomplete final idle row.
    # The selected interval must be complete and ends at the first done tick.
    lines = source.read_bytes().rsplit(b'\n', 1)[0].decode().splitlines(keepends=True)
    raw = np.genfromtxt(io.StringIO(''.join(lines)), delimiter=',', names=True)
    active = int(np.flatnonzero(raw['active'] == 1.)[0])
    done = int(np.flatnonzero(raw['done'] == 1.)[0])
    force = measured_normal_force(raw[:done + 1])
    first = int(np.flatnonzero(force >= CONTACT_N)[0])
    assert 0 < first < active < done
    assert np.all(force[:first] < CONTACT_N)
    np.testing.assert_array_equal(np.diff(raw['control_step'][first:done + 1]), 1.)
    archived = np.genfromtxt(directory / 'control_active.csv', delimiter=',', names=True)
    np.testing.assert_array_equal(raw[active:done + 1], archived)
    prefix_path.write_text(lines[0] + ''.join(lines[first + 1:active + 1]))
    metadata = {
        'source_raw_csv': str(source), 'source_raw_csv_sha256': sha(source),
        'prefix_csv_sha256': sha(prefix_path), 'contact_threshold_n': CONTACT_N,
        'first_contact_raw_row_zero_based': first, 'first_active_raw_row_zero_based': active,
        'first_completed_raw_row_zero_based': done,
        'excluded_precontact_rows': first, 'included_precontrol_contact_rows': active - first,
        'precontrol_contact_tick_duration_s': (active - first) * DT,
        'first_contact_force_n': float(force[first]),
        'last_excluded_force_n': float(force[first - 1]),
        'first_contact_simulation_time_s': float(raw['time'][first]),
        'force_control_start_simulation_time_s': float(raw['time'][active]),
        'completion_simulation_time_s': float(raw['time'][done]),
        'selected_control_steps_contiguous': True,
        'active_rows_match_original_archive_exactly': True,
        'force_basis': 'abs(tool Z axis from logged rotation vector dot logged base-frame force)',
        'raw_contact_flag_not_used': 'Scheduler flag is unset before Force control despite measured contact.',
    }
    dump(metadata_path, metadata)
    return raw[first:active], metadata


def prepare(mode, output, surface_class, reference):
    directory = output / f'mode{mode}'
    prefix, window = recover_prefix(directory)
    raw = np.genfromtxt(directory / 'control_active.csv', delimiter=',', names=True)
    telemetry = np.genfromtxt(directory / 'telemetry.csv', delimiter=',', names=True)
    original = npz(directory / 'derived_signals.npz')
    full_grid = npz(directory / 'surface.npz')
    # Check the reconstructed force convention against independently logged
    # atomic telemetry before using it for the pre-control contact interval.
    error = float(np.max(np.abs(measured_normal_force(raw) - original['force_n'])))
    assert error < 1.e-9
    prefix_force = measured_normal_force(prefix)
    assert prefix_force[0] >= CONTACT_N
    surface = surface_class(reference, cell_size_mm=.5, velocity_model='rotating_disk', spindle_rpm=1000.)
    for key, actual in [('roi', surface.roi), ('u_mm', surface.u), ('v_mm', surface.v),
                        ('origin_mm', surface.origin), ('basis', surface.basis)]:
        np.testing.assert_array_equal(actual, full_grid[key])
    xyz = columns(prefix, ('actual_tcp_x', 'actual_tcp_y', 'actual_tcp_z')).astype(np.float32)
    displacement = np.diff(xyz, axis=0, prepend=xyz[:1])
    displacement -= (displacement @ surface.normal)[:, None] * surface.normal
    velocity = columns(prefix, ('actual_vx', 'actual_vy', 'actual_vz'))
    velocity -= (velocity @ surface.normal)[:, None] * surface.normal
    rates = np.zeros(len(prefix))
    for i in range(len(prefix)):
        before = surface.integrated_volume
        surface.deposit(xyz[i] - .5 * displacement[i], velocity[i], prefix_force[i], DT,
                        contact=bool(prefix_force[i] >= CONTACT_N))
        rates[i] = (surface.integrated_volume - before) / DT
    prefix_volume = float(surface.depth[surface.roi].sum() * surface.area)
    # Original active ticks, including their initial 2 seconds, remain exact.
    # This avoids changing any already audited active-force integration.
    surface.depth += full_grid['depth_over_k']
    combined = np.concatenate([prefix, raw])
    surface.processing_time = len(combined) * DT
    metrics = surface.metrics()
    surface.save(directory / 'contact_surface.npz')
    cursor = np.r_[prefix['trajectory_cursor'], telemetry['cursor_mm']]
    path_length = float(telemetry['path_length_mm'][-1])
    np.testing.assert_allclose(telemetry['path_length_mm'], path_length, atol=1.e-9, rtol=0)
    assert cursor[0] == 0. and np.all(np.diff(cursor) >= 0.)
    assert abs(cursor[-1] - path_length) < 1.e-8
    force = np.r_[prefix_force, original['force_n']]
    rate = np.r_[rates, original['raw_rate_n_mm_s']]
    signals = {
        'time_since_first_contact_s': np.arange(len(combined)) * DT,
        'simulation_time_s': combined['time'],
        'path_progress_percent': 100. * cursor / path_length,
        'cursor_mm': cursor, 'force_n': force,
        # Before controller activation the scheduler feed is zero; the raw
        # action_scale default of 1 must not become a fictitious 6 mm/s feed.
        'scheduled_feed_mm_s': combined['base_progress_rate'] * combined['action_scale'],
        'force_control_active': combined['active'],
        'raw_rate_n_mm_s': rate,
    }
    assert all(np.isfinite(v).all() for v in signals.values())
    np.testing.assert_array_equal(force[len(prefix):], original['force_n'])
    np.testing.assert_array_equal(rate[len(prefix):], original['raw_rate_n_mm_s'])
    np.savez_compressed(directory / 'contact_signals.npz', **signals)
    metrics.update({
        'force_mean_n': float(force.mean()),
        'raw_rate_cv': float(rate.std() / rate.mean()),
        'samples': len(combined), 'added_precontrol_contact_samples': len(prefix),
        'added_precontrol_roi_volume_over_k': prefix_volume,
        'active_force_reconstruction_max_error_n': error,
        'precontact_samples_included': 0,
    })
    # The rate integrates the full grid, while ROI volume excludes off-ROI cells.
    np.testing.assert_allclose(rate.sum() * DT, metrics['grid_volume_over_k'], rtol=1.e-11)
    return {'mode': mode, 'metrics': metrics, 'signals': signals, 'window': window,
            'grid': npz(directory / 'contact_surface.npz'), 'path_length_mm': path_length,
            'entry_uv': ((xyz[0] - surface.origin) @ surface.basis.T).tolist()}


def display_rate(rate):
    # Trailing 16-tick display mean, with shorter warm-up at the first sample.
    # Include the very first and last samples on both horizontal axes.
    indices = np.unique(np.r_[np.arange(0, len(rate), 16), len(rate) - 1])
    start = np.maximum(indices - 15, 0)
    sums = np.r_[0., np.cumsum(rate)]
    values = (sums[indices + 1] - sums[start]) / (indices + 1 - start)
    return indices, values


def plots(runs, output):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.grid': True, 'grid.color': '.75', 'grid.linewidth': .45,
                         'grid.alpha': .45, 'axes.linewidth': .8, 'axes.edgecolor': '.25'})
    style = {
        'reference_directory': 'logs/polishing_results/20260909_123708_flat',
        'palette_source': 'utils/visualization.py PAPER_*_COLOR constants',
        'signal_colors': {key: colors for key, colors in zip(
            ('force', 'feed', 'removal_rate'), SIGNAL_COLORS)},
        'mode_line_styles': {5: 'solid', 6: 'dashed'},
        'heatmap_colormap': 'viridis',
        'signal_y_axis_rule': 'Both modes combined, visible data minimum/maximum plus 4% range margin; independent per panel.',
        'main_panel_selection': 'Path progress > 0%; stationary initial contact is shown in full in the right-hand entry panels.',
        'all_contact_samples_displayed_across_panels': True,
        'signal_panels': {},
    }
    def save(fig, filename):
        fig.savefig(output / filename, dpi=180, bbox_inches='tight')
        plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(14, 8), layout='constrained',
                             gridspec_kw={'width_ratios': [3.2, 1.35]}, sharex='col')
    visible_values = {(row, column): [] for row in range(3) for column in range(2)}
    for run in runs:
        s = run['signals']
        indices, rate = display_rate(s['raw_rate_n_mm_s'])
        label = f'Mode {run["mode"]}'
        # Every contact tick must be visible in at least one of the two panels.
        assert np.all((s['path_progress_percent'] > 0.) | (s['time_since_first_contact_s'] <= 3.))
        for column, xkey in enumerate(('path_progress_percent', 'time_since_first_contact_s')):
            x = s[xkey]
            for row, (xx, yy) in enumerate(((x, s['force_n']), (x, s['scheduled_feed_mm_s']),
                                             (x[indices], rate))):
                selected = (xx > 0.) if column == 0 else ((xx >= 0.) & (xx <= 3.))
                axes[row, column].plot(xx[selected], yy[selected], color=SIGNAL_COLORS[row][run['mode']],
                                       lw=1.15 if run['mode'] == 5 else 1.05,
                                       ls='-' if run['mode'] == 5 else '--', label=label)
                visible_values[row, column].append(yy[selected])
    for (row, column), series in visible_values.items():
        values = np.concatenate(series)
        low, high = float(values.min()), float(values.max())
        pad = .04 * (high - low if high > low else max(abs(high), 1.))
        limits = (low - pad, high + pad)
        axes[row, column].set_ylim(*limits)
        axes[row, column].yaxis.set_major_locator(MaxNLocator(nbins=5))
        style['signal_panels'][f'row{row}_col{column}'] = {
            'data_min': low, 'data_max': high, 'ylim': list(limits)}
    labels = ['Measured normal force [N]', 'Scheduled path feed [mm/s]',
              'Estimated removal rate / K\n128 ms trailing display mean']
    for row, label in enumerate(labels):
        axes[row, 0].set_ylabel(label)
        axes[row, 0].set_xlim(0, 100)
        axes[row, 0].set_xticks(np.arange(0, 101, 20))
        axes[row, 1].set_xlim(0, 3)
        axes[row, 1].set_xticks([0, 1, 2, 3])
        for start in sorted({r['window']['precontrol_contact_tick_duration_s'] for r in runs}):
            axes[row, 1].axvline(start, color=REFERENCE_COLOR, lw=.8, ls=':')
        axes[row, 0].legend(loc='lower right', ncol=2, fontsize=9, framealpha=.9)
    axes[0, 0].axhline(20., color=REFERENCE_COLOR, lw=.8, ls=':', label='20 N control target')
    start = runs[0]['window']['precontrol_contact_tick_duration_s']
    axes[0, 1].plot([start, 3.], [20., 20.], color=REFERENCE_COLOR, lw=.8, ls=':')
    axes[0, 0].legend(loc='lower right', ncol=3, fontsize=9)
    axes[0, 0].set_title('Path traversal: same position | stationary entry shown at right')
    axes[0, 1].set_title('Initial contact: first 3 s')
    axes[0, 1].text(start + .04, .04, 'Force control starts', rotation=90,
                    transform=axes[0, 1].get_xaxis_transform(), color=REFERENCE_COLOR, fontsize=9)
    axes[2, 0].set_xlabel(f'Path progress [%] (cursor / {runs[0]["path_length_mm"]:.2f} mm)')
    axes[2, 1].set_xlabel('Time since first contact [control s]')
    durations = ' | '.join(f'Mode {r["mode"]}: {r["metrics"]["processing_time_s"]:.3f} s' for r in runs)
    fig.suptitle('Isaac GUI | TCP 146 mm | first measured contact (1.5 N) to completion\n'
                 f'{durations} | virtual 1000 RPM | entry included, pre-contact excluded', fontsize=12)
    save(fig, '04_contact_force_feed_removal_progress.png')

    for normalized, filename in [(False, '05_contact_inclusive_depth_maps.png'),
                                 (True, '06_contact_inclusive_normalized_depth_maps.png')]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.7), layout='constrained')
        maps = []
        for run in runs:
            grid = run['grid']
            values = grid['depth_over_k']
            denominator = values[grid['roi']].mean() if normalized else 1.
            maps.append(np.where(grid['roi'], values / denominator, np.nan))
        vmax = max(float(np.nanmax(values)) for values in maps)
        for ax, run, data in zip(axes, runs, maps):
            grid = run['grid']
            half = float(grid['cell_area_mm2']) ** .5 / 2.
            extent = [grid['u_mm'][0] - half, grid['u_mm'][-1] + half,
                      grid['v_mm'][0] - half, grid['v_mm'][-1] + half]
            im = ax.imshow(data, extent=extent, origin='lower', cmap='viridis', vmin=0, vmax=vmax)
            ax.add_patch(Circle(run['entry_uv'], 15., fill=False, edgecolor='white', lw=.9, ls='--'))
            ax.annotate('First contact', xy=run['entry_uv'], xytext=(12, 22),
                        textcoords='offset points', color='black', fontsize=9,
                        bbox={'facecolor': 'white', 'alpha': .85, 'edgecolor': 'none'},
                        arrowprops={'arrowstyle': '->', 'color': 'black', 'lw': .8})
            ax.set(title=f'Mode {run["mode"]} | full-contact CV {run["metrics"]["spatial_depth_cv"]:.4f}',
                   xlabel='Plane u [mm]', ylabel='Plane v [mm]')
            ax.grid(False)
        fig.colorbar(im, ax=axes, shrink=.88,
                     label='Depth / own full-ROI mean' if normalized else 'Estimated depth / K (uncalibrated)')
        fig.suptitle('First contact to completion | initial contact and full starting footprint included\n'
                     'Same full reference ROI and color scale | zero-depth cells retained', fontsize=12)
        save(fig, filename)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), layout='constrained')
    for ax, (key, title) in zip(axes, [('spatial_depth_cv', 'Final spatial depth CV'),
                                      ('roi_volume_over_k', 'Estimated removal in full ROI'),
                                      ('processing_time_s', 'Time from first contact')]):
        baseline = runs[0]['metrics'][key]
        values = [100. * run['metrics'][key] / baseline for run in runs]
        bars = ax.bar([f'Mode {r["mode"]}' for r in runs], values, color=[COLORS[r['mode']] for r in runs])
        ax.bar_label(bars, fmt='%.2f%%', padding=3)
        ax.set(title=title, ylabel='Relative to Mode 6 [%]', ylim=(0, max(values) * 1.2))
        ax.axhline(100., color='.5', lw=.8, ls='--')
    fig.suptitle('First contact to completion | initial contact included, pre-contact excluded')
    save(fig, '07_contact_inclusive_comparison.png')
    return style


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    output = args.archive.resolve()
    model_relative = Path('model_based/runtime/fixed_force_12p59_source/nrs_fixed_force_12p59/preston_surface.py')
    model = next(p / model_relative for p in Path(__file__).resolve().parents if (p / model_relative).is_file())
    spec = importlib.util.spec_from_file_location('contact_preston_surface', model)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest_path = output / 'archive_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    # Verify all previously archived files before adding a new interpretation.
    for relative, expected in manifest['files_sha256'].items():
        assert sha(output / relative) == expected, f'Archive hash mismatch: {relative}'
    reference_path = output / 'contact_reference_path.txt'
    if not reference_path.exists():
        deployment = json.loads((output / 'mode5/deployment.json').read_text())
        shutil.copy2(deployment['trajectory_file'], reference_path)
    reference = np.loadtxt(reference_path)[:, :3]
    runs = [prepare(mode, output, module.PlanarPrestonSurface, reference) for mode in (6, 5)]
    for key in ('roi', 'u_mm', 'v_mm', 'origin_mm', 'basis'):
        np.testing.assert_array_equal(runs[0]['grid'][key], runs[1]['grid'][key])
    assert runs[0]['path_length_mm'] == runs[1]['path_length_mm']
    change = {key: 100. * (runs[1]['metrics'][key] / runs[0]['metrics'][key] - 1.)
              for key in ('spatial_depth_cv', 'roi_volume_over_k', 'processing_time_s')}
    report = {
        'created_at': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        'selection': 'First measured TCP-normal force >= 1.5 N through first trajectory-done tick.',
        'contact_threshold_n': CONTACT_N, 'precontact_samples_included': 0,
        'initial_two_seconds_included': True, 'initial_reference_footprint_included': True,
        'spatial_roi': 'Full reference sweep, identical grid/ROI for both modes, including zero-depth cells.',
        'horizontal_axis': '100 * scheduled path cursor / reference path length; not elapsed-time normalization.',
        'entry_zoom': 'First 3 control seconds from first contact; pre-control contact is stationary in path progress.',
        'clock': 'Contiguous raw controller steps at 125 Hz; simulation timestamps retained separately.',
        'rate_display': 'Trailing 16-tick (128 ms) mean with shorter warm-up; first/last samples included.',
        'heatmap_integration': 'Saved full active surface plus pre-control contact deposits at 8 ms; no entry subtraction.',
        'prefix_velocity': 'Logged TCP linear velocity projected onto surface plane; midpoint float32 poses.',
        'virtual_spindle_rpm': 1000., 'physical_depth_validated': False,
        'spindle_assumption': 'Same virtual 1000 RPM model also applied during measured contact before Force activation.',
        'source_hashes': {'surface_model': sha(model), 'reference_path': sha(reference_path),
                          'analysis_script': sha(__file__)},
        'source_model_path': str(model), 'path_length_mm': runs[0]['path_length_mm'],
        'runs': {f'mode{r["mode"]}': {'metrics': r['metrics'], 'window': r['window']} for r in runs},
        'change_percent_vs_mode6': change,
        'spatial_cv_improvement_percent_vs_mode6': -change['spatial_depth_cv'],
    }
    report['plot_style'] = plots(runs, output)
    dump(output / 'contact_inclusive_comparison.json', report)
    with (output / 'contact_inclusive_comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['metric', 'mode6', 'mode5', 'change_percent_vs_mode6'])
        for key, percent in change.items():
            writer.writerow([key, runs[0]['metrics'][key], runs[1]['metrics'][key], percent])
    notes = [
        '최초 접촉부터 완주까지 비교. 접촉 전 데이터 제외, 초반 접촉/안정화 구간 및 시작부 전체 ROI 포함.',
        '최초 접촉 기준: 기록된 TCP 수직 힘 절댓값 >= 1.5 N (기존 제거량 모델과 동일).',
        '추가한 접촉 구간: ' + ', '.join(
            f'Mode {r["mode"]} {r["metrics"]["added_precontrol_contact_samples"]}개 '
            f'({r["window"]["precontrol_contact_tick_duration_s"]:.3f}초)' for r in runs) + '.',
        '주 그래프: 같은 경로 위치 기준 진행률 0~100%. 우측: 접촉 시작 후 3초 확대.',
        '히트맵: 동일 전체 ROI/색상 범위. 처음 2초/18 mm 제외 규칙 적용하지 않음.',
        '제거량: 접촉 중 가상 1000 RPM을 적용한 미보정 Preston 추정치. 실측 제거량이 아님.',
        f'제거 균일도 CV 개선: {-change["spatial_depth_cv"]:.4f}% (Mode 6 대비).',
        f'전체 ROI 추정 제거량 변화: {change["roi_volume_over_k"]:+.4f}%.',
        f'최초 접촉부터 완주까지 제어 시간 변화: {change["processing_time_s"]:+.4f}%.',
    ]
    for run in runs:
        m = run['metrics']
        notes.append(f'Mode {run["mode"]}: CV={m["spatial_depth_cv"]:.8f}, V/K={m["roi_volume_over_k"]:.6f}, '
                     f'시간={m["processing_time_s"]:.3f}s, 표본={m["samples"]}.')
    (output / 'contact_inclusive_comparison.txt').write_text('\n'.join(notes) + '\n')
    snapshot = output / 'contact_inclusive_analysis.py'
    if snapshot.resolve() != Path(__file__).resolve():
        shutil.copy2(__file__, snapshot)
    manifest['contact_inclusive_revision'] = {
        'report': 'contact_inclusive_comparison.json',
        'signals_png': '04_contact_force_feed_removal_progress.png',
        'heatmap_pngs': ['05_contact_inclusive_depth_maps.png', '06_contact_inclusive_normalized_depth_maps.png'],
        'cv_improvement_percent_vs_mode6': -change['spatial_depth_cv'],
        'previous_reports_preserved': True,
    }
    manifest.setdefault('original_processing_cv_improvement_percent_vs_mode6',
                        manifest['cv_improvement_percent_vs_mode6'])
    manifest.setdefault('original_heatmap_pngs', manifest['heatmap_pngs'])
    manifest['primary_report'] = 'contact_inclusive_comparison.json'
    manifest['cv_improvement_percent_vs_mode6'] = -change['spatial_depth_cv']
    manifest['heatmap_pngs'] = manifest['contact_inclusive_revision']['heatmap_pngs']
    manifest['files_sha256'] = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*'))
                               if p.is_file() and p != manifest_path}
    dump(manifest_path, manifest)
    print('\n'.join(notes))
    print(f'Archive: {output}')


if __name__ == '__main__':
    main()
