"""Audit completed fixed-force GUI trials without touching the ROS runtime."""
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
import importlib.util
import io
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = _REPO_ROOT


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def read_npz(path):
    with np.load(path) as stream:
        return dict(stream)


def load_run(path, mode, output, measurement):
    sessions = list(path.glob('removal/session_*/summary.json'))
    assert len(sessions) == 1, 'Expected one measurement session per launch'
    session = sessions[0].parent
    destination = output / f'mode{mode}'
    destination.mkdir()
    deployment = json.loads((path / 'deployment.json').read_text())
    summary = json.loads(sessions[0].read_text())
    telemetry = np.genfromtxt(session / 'telemetry.csv', delimiter=',', names=True)
    policy = np.genfromtxt(path / 'policy.csv', delimiter=',', names=True)
    assert deployment['force_control_mode'] == mode
    assert deployment['policy_controls_force'] is False
    assert deployment['policy_output_applied'] is (mode == 5)
    assert summary['valid_completed_run'] and not summary['interrupted']
    assert len(telemetry) == len(policy) == summary['samples']
    assert all(np.isfinite(telemetry[k]).all() for k in telemetry.dtype.names)
    assert all(np.isfinite(policy[k]).all() for k in policy.dtype.names)
    assert all(np.array_equal(telemetry[k], policy[k]) for k in telemetry.dtype.names)
    np.testing.assert_array_equal(telemetry['sequence'], np.arange(1, len(telemetry) + 1))
    assert len(np.unique(telemetry['epoch'])) == 1
    np.testing.assert_allclose(np.diff(telemetry['elapsed_s']), .008, atol=1e-8, rtol=0)
    assert np.all(np.diff(telemetry['simulation_time_s']) >= 0)
    assert np.all(telemetry['target_force_n'] == 20.)
    assert not np.any(telemetry['reverse_mm']) and not np.any(telemetry['return_allowed'])
    assert not np.any(telemetry['fault']) and not np.any(policy['missing_samples'])
    assert telemetry['done'][-1] == 1 and not np.any(telemetry['done'][:-1])
    np.testing.assert_allclose(telemetry['cursor_mm'][-1], telemetry['path_length_mm'][-1], atol=1e-5)
    if mode == 5:
        np.testing.assert_array_equal(policy['feed_action'], policy['actor_feed_action'])
    else:
        assert np.all(policy['feed_action'] == 0.)

    # The ROS node may continue appending idle rows. Read only complete lines,
    # and retain the first active tick through the first completed tick.
    snapshot = (path / 'raw.csv').read_bytes().rsplit(b'\n', 1)[0] + b'\n'
    lines = snapshot.decode().splitlines(keepends=True)
    names = next(csv.reader([lines[0]]))
    active_index, done_index = names.index('active'), names.index('done')
    selected = [lines[0]]
    started = False
    for line in lines[1:]:
        values = next(csv.reader([line]))
        assert len(values) == len(names), 'Malformed complete raw CSV row'
        started = started or float(values[active_index]) == 1.
        if started:
            selected.append(line)
            if float(values[done_index]) == 1.:
                break
    active_text = ''.join(selected)
    raw = np.genfromtxt(io.StringIO(active_text), delimiter=',', names=True)
    assert len(raw) == len(telemetry) and raw['done'][-1] == 1.
    assert np.all(raw['force_control_mode'] == mode) and np.all(raw['target_fz'] == 20.)
    np.testing.assert_allclose(raw['time'], telemetry['simulation_time_s'], rtol=0, atol=1e-9)
    for axis in 'xyz':
        np.testing.assert_allclose(raw[f'actual_tcp_{axis}'], telemetry[f'tcp_{axis}_mm'], atol=1e-9)
    if mode == 6:
        assert not np.any(raw['raw_ppo_action'])
    (destination / 'control_active.csv').write_text(active_text)

    matching_logs = []
    for source in (Path.home() / '.ros/log').glob('singleArm_cmd_*.log'):
        text = source.read_text()
        if str(session) in text:
            matching_logs.append((source, text))
    assert len(matching_logs) == 1, 'Expected one matching command log'
    command_path, command_log = matching_logs[0]
    assert f'Mode {mode} trajectory completed.' in command_log
    assert '/polishing_removal_node/end response: success=true' in command_log
    assert 'home returned; Idling' in command_log
    home = [float(x) for x in re.search(r'TxtLoad home saved \(mm/rad\): ([^\n]+)', command_log).group(1).split()]
    (destination / 'command.log').write_text(command_log)

    sources = [path / 'deployment.json', path / 'policy.csv', session / 'telemetry.csv',
               session / 'summary.json', session / 'surface.npz', session / 'processing_surface.npz',
               session / 'derived_signals.npz']
    before = {str(p): sha(p) for p in sources}
    for source in sources:
        shutil.copy2(source, destination / source.name)
    assert before == {str(p): sha(p) for p in sources}, 'Recording changed during audit'

    # Reintegrate all raw controller snapshots; no previously computed metric
    # is used as input to this replay.
    reference = np.loadtxt(deployment['trajectory_file'])[:, :3]
    rows = np.column_stack([telemetry[k] for k in measurement.FIELDS])
    replay = measurement.summarize(rows, reference, destination / 'replay', make_plots=False)
    assert replay == summary, 'Recomputed summary differs from saved summary'
    grid_errors = {}
    for stem in ('surface', 'processing_surface', 'derived_signals'):
        actual = read_npz(destination / 'replay' / f'{stem}.npz')
        expected = read_npz(destination / f'{stem}.npz')
        assert actual.keys() == expected.keys()
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key])
        grid_errors[stem] = 0.

    signals = read_npz(destination / 'derived_signals.npz')
    keep = signals['time_s'] >= 2.
    diagnostics = {
        'run_directory': str(path), 'session_directory': str(session),
        'command_log': str(command_path), 'valid_completed_processing_run': True,
        'home_return_completed': True, 'saved_home_mm_rad': home,
        'samples': len(telemetry), 'missing_control_ticks': 0, 'policy_missing_samples': 0,
        'hard_fault_samples': 0, 'reverse_mm': 0., 'target_force_min_max_n': [20., 20.],
        'requested_feed_min_max_mm_s': [float((6. + 6.*policy['feed_action']).min()),
                                       float((6. + 6.*policy['feed_action']).max())],
        'cursor_mm': float(telemetry['cursor_mm'][-1]),
        'whole_active_duration_s': len(telemetry)*.008,
        'force_entry_pose_mm': [float(telemetry[f'tcp_{a}_mm'][0]) for a in 'xyz'],
        'measured_force_mean_after_entry_n': float(signals['force_n'][keep].mean()),
        'action_age_max_s': float(raw['ppo_action_age_s'].max()),
        'action_age_over_250ms_samples': int(np.count_nonzero(raw['ppo_action_age_s'] > .25)),
        'worker_latency_p99_ms': float(np.percentile(policy['latency_ms'], 99)),
        'raw_active_window_matches_telemetry': True,
        'replayed_summary_matches': True, 'replayed_arrays_max_errors': grid_errors,
        'source_hashes': before,
    }
    dump(destination / 'audit.json', diagnostics)
    print(f'Mode {mode}: {len(telemetry)} samples, full measurement replay matches.', flush=True)
    return {'mode': mode, 'summary': summary, 'deployment': deployment, 'diagnostics': diagnostics,
            'signals': signals, 'raw': raw, 'policy': policy,
            'surface': read_npz(destination / 'surface.npz'),
            'processing_surface': read_npz(destination / 'processing_surface.npz')}


def comparison(a, b):
    result = {}
    for label, section in [('processing', 'processing_metrics'), ('full', 'metrics')]:
        left, right = a['summary'][section], b['summary'][section]
        fields = ['spatial_depth_cv', 'roi_volume_over_k', 'processing_time_s']
        if label == 'processing':
            fields.append('raw_rate_cv')
        result[label] = {
            'mode5': {k: left[k] for k in fields}, 'mode6': {k: right[k] for k in fields},
            'change_percent_vs_mode6': {k: 100.*(left[k]/right[k]-1.) for k in fields},
            'spatial_cv_improvement_percent_vs_mode6': 100.*(1.-left['spatial_depth_cv']/right['spatial_depth_cv']),
            'removal_retention_percent_vs_mode6': 100.*left['roi_volume_over_k']/right['roi_volume_over_k'],
            'volume_at_least_half_mode6': left['roi_volume_over_k'] >= .5*right['roi_volume_over_k'],
        }
    return result


def plots(a, b, output, report):
    plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .2})
    colors = {5: '#2563a8', 6: '#d16c37'}
    def save(fig, filename):
        fig.savefig(output / filename, dpi=170, bbox_inches='tight')
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), layout='constrained')
    keys = [('spatial_depth_cv', 'Final spatial depth CV'),
            ('roi_volume_over_k', 'Removal in processing ROI'),
            ('processing_time_s', 'Processing time')]
    for ax, (key, title) in zip(axes, keys):
        ratio = 100.*report['comparison']['processing']['mode5'][key]/report['comparison']['processing']['mode6'][key]
        bars = ax.bar(['Mode 6', 'Mode 5'], [100., ratio], color=[colors[6], colors[5]])
        ax.bar_label(bars, fmt='%.2f%%', padding=3)
        ax.set(title=title, ylabel='Relative to Mode 6 [%]', ylim=(0, max(100., ratio)*1.2))
        ax.axhline(100, color='gray', lw=.8, ls='--')
    fig.suptitle('Isaac GUI | fixed 20 N, virtual 1000 RPM | first 2 s / 18 mm footprint excluded')
    save(fig, '00_mode6_relative_comparison.png')

    for normalized, filename in [(False, '01_processing_depth_maps.png'), (True, '02_normalized_depth_maps.png')]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout='constrained')
        maps = []
        for run in (b, a):
            grid = run['processing_surface']
            values = grid['depth_over_k']
            maps.append(np.where(grid['roi'], values/(values[grid['roi']].mean() if normalized else 1.), np.nan))
        vmax = max(float(np.nanmax(v)) for v in maps)
        for ax, run, data in zip(axes, (b, a), maps):
            grid = run['processing_surface']
            cell = float(grid['cell_area_mm2'])**.5
            extent = [grid['u_mm'][0]-.5*cell, grid['u_mm'][-1]+.5*cell,
                      grid['v_mm'][0]-.5*cell, grid['v_mm'][-1]+.5*cell]
            im = ax.imshow(data, extent=extent, origin='lower', cmap='viridis', vmin=0, vmax=vmax)
            cv = run['summary']['processing_metrics']['spatial_depth_cv']
            ax.set(title=f'Mode {run["mode"]} | CV {cv:.4f}', xlabel='Plane u [mm]', ylabel='Plane v [mm]')
            ax.grid(False)
            fig.colorbar(im, ax=ax, label='Depth / own ROI mean' if normalized else 'Depth / K (uncalibrated)')
        fig.suptitle('Final spatial removal | same ROI / color scale | includes zero-depth cells')
        save(fig, filename)

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, layout='constrained')
    for run in (b, a):
        signals, color = run['signals'], colors[run['mode']]
        time = signals['time_s']
        axes[0].plot(time, signals['force_n'], color=color, lw=.55, label=f'Mode {run["mode"]}')
        axes[1].plot(time, run['raw']['action_scale']*6., color=color, lw=.8, label=f'Mode {run["mode"]}')
        # Only the display is averaged; all summary metrics use every raw tick.
        indices = np.arange(0, len(time), 16)
        rate = signals['raw_rate_n_mm_s']
        axes[2].plot(time[indices], [rate[i:i+16].mean() for i in indices], color=color, lw=.8, label=f'Mode {run["mode"]}')
    for ax, label in zip(axes, ['Measured normal force [N]', 'Scheduled path feed [mm/s]', 'Removal rate / K\n128 ms display mean']):
        ax.axvspan(0, 2, color='gray', alpha=.12)
        ax.set_ylabel(label)
        ax.legend(loc='upper right')
    axes[0].axhline(20., color='black', lw=.7, ls='--')
    axes[-1].set_xlabel('Active control time [simulation s]')
    fig.suptitle('Isaac GUI | force target fixed 20 N in both modes | virtual 1000 RPM')
    save(fig, '03_force_feed_removal_rate.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode5', type=Path, required=True)
    parser.add_argument('--mode6', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    timestamp = datetime.now(ZoneInfo('Asia/Seoul'))
    output = args.output or ROOT/'logs/polishing_results'/timestamp.strftime('%Y%m%d_%H%M')
    output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('fixed_gui_measurement', ROOT/'deployment/fixed_force_12p59/paired_measurement.py')
    measurement = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(measurement)
    a = load_run(args.mode5.resolve(), 5, output, measurement)
    b = load_run(args.mode6.resolve(), 6, output, measurement)
    # These are all configuration differences allowed by the two launch modes.
    differences = {k for k in a['deployment'] if a['deployment'][k] != b['deployment'][k]}
    allowed = {'force_control_mode', 'policy_output_applied', 'applied_action_override',
               'inference', 'run', 'removal_directory'}
    assert differences <= allowed, f'Deployment settings differ: {differences-allowed}'
    assert a['deployment'].keys() == b['deployment'].keys()
    assert a['summary']['model'] == b['summary']['model']
    assert a['summary']['entry_protocol'] == b['summary']['entry_protocol']
    for gridname in ('surface', 'processing_surface'):
        for key in ('roi', 'u_mm', 'v_mm', 'cell_area_mm2', 'origin_mm', 'basis'):
            np.testing.assert_array_equal(a[gridname][key], b[gridname][key])
    pinned = a['deployment']['pinned_files']
    assert all(Path(p).is_file() and sha(p) == expected for p, expected in pinned.items())
    home_delta = np.array(a['diagnostics']['saved_home_mm_rad']) - np.array(b['diagnostics']['saved_home_mm_rad'])
    entry_delta = np.array(a['diagnostics']['force_entry_pose_mm']) - np.array(b['diagnostics']['force_entry_pose_mm'])
    report = {
        'created_at': timestamp.isoformat(), 'checkpoint_sha256': a['deployment']['checkpoint_sha256'],
        'comparison_baseline': 'Mode 6 GUI at 20 N / 6 mm/s / virtual 1000 RPM',
        'primary_mode5_selection': 'Latest completed Mode 5 run, selected before reading Mode 6 results',
        'comparison': comparison(a, b), 'entry_protocol': a['summary']['entry_protocol'],
        'matching': {'configuration_and_pinned_files_match': True, 'pinned_files_verified': len(pinned),
            'measurement_model_and_grid_match': True, 'target_force_fixed_20n_both_modes': True,
            'mode6_learned_actions_not_applied': True, 'home_position_delta_mm': home_delta[:3].tolist(),
            'home_position_distance_mm': float(np.linalg.norm(home_delta[:3])),
            'force_entry_position_distance_mm': float(np.linalg.norm(entry_delta)),
            'exact_initial_state_match_verified': False},
        'runs': {f'mode{r["mode"]}': r['diagnostics'] for r in (a, b)},
        'limits': ['One primary GUI run per mode; repeated paired trials are not yet available.',
                   'Saved home positions differ; exact scene/physics reset equality is not logged.',
                   'Removal is the uncalibrated rotating-disk Preston estimate, not physically measured volume.',
                   'Spatial final-depth CV and temporal instantaneous-rate CV are separate metrics.'],
    }
    dump(output/'comparison.json', report)
    with (output/'comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['window', 'metric', 'mode6', 'mode5', 'change_percent_vs_mode6'])
        for window, values in report['comparison'].items():
            for key, change in values['change_percent_vs_mode6'].items():
                writer.writerow([window, key, values['mode6'][key], values['mode5'][key], change])
    plots(a, b, output, report)
    p, f = report['comparison']['processing'], report['comparison']['full']
    notes = [
        'Isaac GUI 고정 힘 모델 비교: 모든 변화율은 이번 Mode 6 실증 대비.',
        f'Mode 5: {args.mode5.resolve()}', f'Mode 6: {args.mode6.resolve()}',
        '양쪽 경로 완주, 제거량 저장, 홈 복귀 확인. 제어/정책 누락 및 fault 0.',
        '목표 힘 20 N 고정, 가상 회전 1000 RPM, 동일 경로/제어기/0.5 mm 격자.',
        '주 지표: 초반 2초의 제거량과 기준 첫 18 mm가 덮는 영역 제외, 깊이 0인 셀 포함.',
        f'가공 구간 공간 CV 개선: {p["spatial_cv_improvement_percent_vs_mode6"]:.5f}%',
        f'가공 구간 제거량 유지: {p["removal_retention_percent_vs_mode6"]:.5f}%',
        f'가공 구간 시간 변화: {p["change_percent_vs_mode6"]["processing_time_s"]:+.5f}%',
        f'전체 경로 제거량 유지: {f["removal_retention_percent_vs_mode6"]:.5f}%',
        f'시간축 순간 제거율 CV 변화: {p["change_percent_vs_mode6"]["raw_rate_cv"]:+.5f}%',
        f'저장된 홈 위치 차이: {report["matching"]["home_position_distance_mm"]:.6f} mm; 완전한 초기 상태 일치는 확인하지 못함.',
        '각 모드 1회 주 비교. 제거량은 미교정 회전 공구 모델 추정치이며 실제 재료 측정값이 아님.',
        '모든 원시 제어 샘플을 재적분해 저장된 전체/가공 구간 깊이와 요약값의 일치를 검증함.',
    ]
    (output/'comparison.txt').write_text('\n'.join(notes)+'\n')
    shutil.copy2(__file__, output/'analysis_script.py')
    dump(output/'artifact_manifest.json', {'files': {str(p.relative_to(output)): sha(p)
         for p in sorted(output.rglob('*')) if p.is_file()}})
    print(json.dumps({'output': str(output), 'comparison': report['comparison'], 'matching': report['matching']}, indent=2))


if __name__ == '__main__':
    main()
