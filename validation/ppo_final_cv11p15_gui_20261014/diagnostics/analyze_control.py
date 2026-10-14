"""Supplement the frozen paired GUI audit with control and failed-run diagnostics."""
from pathlib import Path
import hashlib
import io
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PAIR = HERE.parent
FAILED = Path('/home/eunseop/evaluation/rl_no_speed_modulation/ppo_final11p15_20261014_134219_424942')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def stats(values):
    values = np.asarray(values)
    return {key: float(value) for key, value in {
        'min': values.min(), 'mean': values.mean(), 'std': values.std(),
        'max': values.max(), 'p05': np.percentile(values, 5),
        'median': np.median(values), 'p95': np.percentile(values, 95),
        'p99': np.percentile(values, 99),
    }.items()}


def held_wrench(raw):
    vector = np.column_stack([raw[k] for k in ('fx', 'fy', 'fz', 'tx', 'ty', 'tz')])
    starts = np.r_[0, np.flatnonzero(np.any(np.diff(vector, axis=0) != 0., axis=1)) + 1]
    durations = (np.r_[starts[1:], len(raw)] - starts) * .008
    return {
        'unchanged_adjacent_rows_percent': float(100 * (len(raw) - len(starts)) / len(raw)),
        'hold_duration_control_s': stats(durations),
        'recorded_vector_segments': int(len(starts)),
        'longest_hold_start_s': float(starts[durations.argmax()] * .008),
        'interpretation': 'Exact repeated base-frame six-axis wrench values seen by the controller; not a direct measurement of sensor message age.',
    }


manifest = json.loads((PAIR / 'artifact_manifest.json').read_text())
assert all(sha(PAIR / name) == digest for name, digest in manifest['files'].items())
report = json.loads((PAIR / 'comparison.json').read_text())
runs, arrays = {}, {}
inputs = [PAIR / 'comparison.json', PAIR / 'artifact_manifest.json']
for mode in (5, 6):
    folder = PAIR / f'mode{mode}'
    p = np.genfromtxt(folder / 'policy.csv', delimiter=',', names=True)
    r = np.genfromtxt(folder / 'control_active.csv', delimiter=',', names=True)
    summary = json.loads((folder / 'summary.json').read_text())
    time = (p['sequence'] - 1) * .008
    keep = time >= 2.
    force = np.abs(p['normal_force_n'])
    assert len(r) == len(p)
    np.testing.assert_allclose(force[keep].mean(), summary['processing_metrics']['force_mean_n'], rtol=0, atol=1e-12)
    assert int(keep.sum()) * .008 == summary['processing_metrics']['processing_time_s']
    xyz = np.column_stack([p[f'tcp_{axis}_mm'] for axis in 'xyz'])
    changed = np.r_[True, np.any(np.diff(xyz, axis=0) != 0., axis=1)]
    duplicate_clock = np.flatnonzero(np.diff(p['simulation_time_s']) == 0.) + 1
    age = r['ppo_action_age_s']
    d = {
        'processing_window': 'Control index time >= 2 s; same sample mask as paired_measurement.py',
        'force_processing_n': stats(force[keep]),
        'force_processing_rmse_n': float(np.sqrt(np.mean((force[keep] - 20.) ** 2))),
        'force_outside_18_22_processing_percent': float(100 * np.mean((force[keep] < 18.) | (force[keep] > 22.))),
        'force_peak_whole_n': float(force.max()),
        'force_peak_control_time_s': float(time[force.argmax()]),
        'tracking_error_processing_mm': stats(p['tracking_error_mm'][keep]),
        'requested_feed_processing_mm_s': stats((6. + 6. * p['feed_action'])[keep]),
        'scheduled_feed_processing_mm_s': stats((6. * r['action_scale'])[keep]),
        'recorded_tcp_speed_processing_mm_s': stats(p['actual_speed_mm_s'][keep]),
        'recorded_zero_tcp_speed_processing_percent': float(100 * np.mean(p['actual_speed_mm_s'][keep] == 0.)),
        'unchanged_tcp_rows_percent': float(100 * np.mean(~changed)),
        'observed_tcp_change_interval_control_s': stats(np.diff(time[changed])),
        'shield_whole_s': float(.008 * np.count_nonzero(p['shield'])),
        'shield_processing_s': float(.008 * np.count_nonzero(p['shield'][keep])),
        'fault_rows': int(np.count_nonzero(p['fault'])),
        'progression_blocked_rows': int(np.count_nonzero(r['ppo_progression_allowed'] == 0.)),
        'progression_blocked_after_entry_rows': int(np.count_nonzero((r['ppo_progression_allowed'] == 0.) & keep)),
        'action_not_fresh_rows': int(np.count_nonzero(r['ppo_action_fresh'] == 0.)),
        'action_not_fresh_after_received_rows': int(np.count_nonzero((r['ppo_action_fresh'] == 0.) & (age >= 0.))),
        'action_age_received_s': stats(age[age >= 0.]),
        'action_age_over_250ms_rows': int(np.count_nonzero(age > .25)),
        'inference_latency_on_decision_rows_ms': stats(p['latency_ms'][p['decision'] == 1.]),
        'duplicate_sim_clock_rows': int(len(duplicate_clock)),
        'duplicate_sim_clock_control_times_s': time[duplicate_clock].tolist(),
        'sim_clock_duration_s': float(p['simulation_time_s'][-1] - p['simulation_time_s'][0] + .008),
        'control_tick_duration_s': float(len(p) * .008),
        'zero_depth_processing_roi_percent': float(100 * summary['processing_metrics']['zero_depth_fraction']),
        'base_wrench_hold': held_wrench(r),
    }
    runs[f'mode{mode}'] = d
    arrays[mode] = (p, r, time, keep)
    inputs += [folder / name for name in ('policy.csv', 'control_active.csv', 'summary.json')]

failure = json.loads((FAILED / 'policy_failure.json').read_text())
failed_session = next(FAILED.glob('removal/session_*/telemetry.csv')).parent
failed_summary = json.loads((failed_session / 'summary.json').read_text())
failed = np.genfromtxt(failed_session / 'telemetry.csv', delimiter=',', names=True)
failed_raw_path = FAILED / 'raw.csv'
failed_raw_bytes = failed_raw_path.read_bytes().rsplit(b'\n', 1)[0] + b'\n'
failed_raw = np.genfromtxt(io.StringIO(failed_raw_bytes.decode()), delimiter=',', names=True)
failed_start = np.flatnonzero(failed_raw['active'] == 1.)[0]
failed_raw = failed_raw[failed_start:failed_start + len(failed)]
np.testing.assert_allclose(failed_raw['time'], failed['simulation_time_s'], rtol=0, atol=1e-9)
failed_cmd = Path('/home/eunseop/.ros/log/singleArm_cmd_144910_1791952941821.log')
failed_motion = Path('/home/eunseop/.ros/log/singleArm_motion_144908_1791952941780.log')
home = [float(v) for v in re.search(r'TxtLoad home saved \(mm/rad\): ([^\n]+)', failed_cmd.read_text()).group(1).split()]
success_home = report['runs']['mode6']['saved_home_mm_rad']
aborted = {
    'run': str(FAILED), 'excluded_from_completed_pair': True,
    'valid_completed_run': failed_summary['valid_completed_run'],
    'recorded_control_duration_s': float(len(failed) * .008),
    'first_fault': failure,
    'max_recorded_force_n': float(np.max(np.abs(failed['normal_force_n']))),
    'minimum_tcp_z_mm': float(failed['tcp_z_mm'].min()),
    'base_wrench_hold': held_wrench(failed_raw),
    'saved_home_matches_completed_mode6': home == success_home,
    'first_tcp_pose_matches_completed_mode6': bool(np.array_equal(
        np.array([failed[f'tcp_{a}_mm'][0] for a in 'xyz']),
        np.array([arrays[6][0][f'tcp_{a}_mm'][0] for a in 'xyz']))),
    'direct_stop_reason': 'Normal force 238.655 N exceeded the controller hard force stop of 100 N; fault bit 1 latched, then policy exited and launch shut down.',
    'root_cause': 'Not established. Aborted controller wrench samples are held up to 392 ms (completed runs: 16/24 ms), consistent with delayed input refresh. Sensor message source stamps and receipt times were not recorded, so the location and cause of the delay cannot be established. Saved home and first TCP match the successful Mode 6 at logged precision.',
}
assert not failed_summary['valid_completed_run']
inputs += [FAILED / 'policy_failure.json', failed_session / 'summary.json', failed_session / 'telemetry.csv', failed_raw_path, failed_cmd, failed_motion]
p = report['comparison']['processing']
headless = json.loads((PAIR / 'mode5/deployment.json').read_text())['nominal_headless_cv_improvement_percent']
derived = {
    'spatial_cv_improvement_percent': p['spatial_cv_improvement_percent_vs_mode6'],
    'processing_extra_time_s': p['mode5']['processing_time_s'] - p['mode6']['processing_time_s'],
    'processing_volume_per_time_change_percent': 100 * ((p['mode5']['roi_volume_over_k'] / p['mode5']['processing_time_s']) / (p['mode6']['roi_volume_over_k'] / p['mode6']['processing_time_s']) - 1),
    'force_rmse_change_percent': 100 * (runs['mode5']['force_processing_rmse_n'] / runs['mode6']['force_processing_rmse_n'] - 1),
    'nominal_headless_cv_improvement_percent': headless,
    'gui_minus_headless_improvement_percentage_points': p['spatial_cv_improvement_percent_vs_mode6'] - headless,
    'initialization_equivalence_verified': False,
}
result = {
    'completed_pair': str(PAIR), 'runs': runs, 'comparison': derived, 'aborted_mode6': aborted,
    'interpretation_limits': [
        'One completed trial per mode; no repeated-trial confidence interval.',
        'Removal is an uncalibrated fixed-footprint rotating Preston estimate (1000 RPM), not measured material volume.',
        'Approximately 80% of adjacent TCP snapshots repeat. Apparent 8 ms finite-difference speed spikes do not by themselves establish physical acceleration.',
        'Joint-state message receipt times and source stamps are not in this CSV; observed TCP change interval is not a measurement of publisher frequency.',
        'Mode 6 contains one duplicated simulation timestamp. The official estimator uses control-index 8 ms time.',
        'Pinned steady v3 configuration is identical, but its applied correction topic was not saved in these CSV files.',
    ],
    'input_sha256': {str(f): sha(f) for f in inputs},
}
write_json(HERE / 'control_diagnostics.json', result)

colors = {5: '#2563a8', 6: '#d16c37'}
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .2})
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, layout='constrained')
for mode, (policy, raw, time, keep) in arrays.items():
    progress = 100 * policy['cursor_mm'] / policy['path_length_mm']
    c = colors[mode]
    axes[0].plot(progress[keep], policy['normal_force_n'][keep], lw=.6, color=c, label=f'Mode {mode}')
    axes[1].plot(progress[keep], 6 * raw['action_scale'][keep], lw=.8, color=c, label=f'Mode {mode}')
    axes[2].plot(progress[keep], policy['tracking_error_mm'][keep], lw=.65, color=c, label=f'Mode {mode}')
for ax, title in zip(axes, ('Normal force [N]', 'Scheduled feed [mm/s]', '3D reference tracking error [mm]')):
    ax.set_ylabel(title)
    ax.legend(loc='upper right')
axes[0].axhline(20, color='black', ls='--', lw=.8)
axes[0].axhspan(18, 22, color='gray', alpha=.1)
axes[2].axhline(10, color='black', ls='--', lw=.8, label='Tracking stop threshold')
axes[-1].set_xlabel('Reference path progress [%]')
fig.suptitle('Completed Isaac GUI pair | control time >= 2 s | fixed 20 N')
fig.savefig(HERE / 'control_quality_vs_progress.png', dpi=170)
plt.close(fig)

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout='constrained')
for mode, (policy, raw, time, keep) in arrays.items():
    entry = time <= 3.
    axes[0].plot(time[entry], policy['normal_force_n'][entry], color=colors[mode], label=f'Mode {mode}, completed')
    axes[1].plot(time[entry], policy['tcp_z_mm'][entry], color=colors[mode])
tf = (failed['sequence'] - 1) * .008
axes[0].plot(tf, failed['normal_force_n'], color='#9b2226', label='Mode 6, aborted 13:42')
axes[1].plot(tf, failed['tcp_z_mm'], color='#9b2226')
axes[0].axhline(100, color='black', ls='--', lw=.8, label='Hard force stop: 100 N')
axes[0].axhline(20, color='gray', ls=':', lw=.8)
axes[0].legend(loc='upper left')
axes[0].set_ylabel('Normal force [N]')
axes[1].set(ylabel='Recorded TCP Z [mm]', xlabel='Force control time [s]')
fig.suptitle('Entry transient | aborted Mode 6 excluded from performance comparison')
fig.savefig(HERE / 'entry_and_aborted_mode6.png', dpi=170)
plt.close(fig)

notes = [
    '최신 PPO GUI 비교 진단 (2026-10-14, 완주한 Mode 5/6 각 1회)',
    f'가공 공간 CV 개선: {derived["spatial_cv_improvement_percent"]:.5f}%',
    f'가공 제거량 증가: {p["change_percent_vs_mode6"]["roi_volume_over_k"]:.5f}%; 시간 증가: {p["change_percent_vs_mode6"]["processing_time_s"]:.5f}% ({derived["processing_extra_time_s"]:.3f} s)',
    f'가공 ROI 제거량/시간 변화: {derived["processing_volume_per_time_change_percent"]:+.5f}%. 제거량 증가는 긴 가공 시간과 함께 해석해야 함.',
    f'힘 RMSE (초기 2초 제외): Mode 5 {runs["mode5"]["force_processing_rmse_n"]:.6f} N, Mode 6 {runs["mode6"]["force_processing_rmse_n"]:.6f} N.',
    f'헤드리스 검증 개선율 {headless:.5f}% 대비 이번 GUI 개선율은 {abs(derived["gui_minus_headless_improvement_percentage_points"]):.5f}%p 작음. 한 쌍 결과로 원인을 확정할 수 없음.',
    '완주한 두 기록: 경로 697.533 mm, 목표 20 N, fault/제어 순번 누락 0, 원시 재적분 일치, 저장 및 홈 복귀 확인.',
    '초기 접촉 최대 힘은 Mode 5 69.87 N, Mode 6 69.23 N. 2초 이후 shield 활성 시간은 모두 0.',
    '13:42 Mode 6 중단: elapsed_s=2.344 s, Fn=238.655 N에서 100 N hard stop 초과로 fault=1. 성능 비교에서 제외.',
    '중단 회차의 제어기 F/T 입력이 최대 392 ms 동안 동일 값 유지(완주 Mode 5/6 최대 16/24 ms). 힘 입력 갱신 지연이 의심되지만 메시지 시각을 기록하지 않아 지연 위치와 근본 원인은 미확정.',
    '중단 기록과 완주 Mode 6의 저장 홈 및 첫 TCP 위치는 로그 정밀도에서 같음. 물리 상태 및 센서 지연을 기록하지 않아 근본 원인은 미확정.',
    'TCP 위치가 약 80%의 연속 제어 행에서 반복됨. 8 ms 차분 속도의 0 및 약 152 mm/s 피크를 실제 급가감속으로 단정할 수 없음.',
    'Mode 6 simulation timestamp 중복 1회(8 ms)가 있으나 제어 순번 누락은 없음. 공식 제거량은 고정 제어 tick 시간으로 계산됨.',
    '제거량은 미교정 가상 회전 Preston 모델 추정값. 각 모드 1회이며 완전한 초기 물리 상태 일치는 확인되지 않음.',
]
(HERE / 'analysis.txt').write_text('\n'.join(notes) + '\n')
write_json(HERE / 'artifact_manifest.json', {'files': {str(f.relative_to(HERE)): sha(f) for f in sorted(HERE.iterdir()) if f.is_file() and f.name != 'artifact_manifest.json'}})
print(json.dumps({'output': str(HERE), 'comparison': derived, 'force_rmse_n': {mode: d['force_processing_rmse_n'] for mode, d in runs.items()}, 'aborted_force_n': failure['telemetry']['normal_force_n']}, indent=2))
