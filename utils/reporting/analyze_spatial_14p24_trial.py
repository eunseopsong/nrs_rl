#!/usr/bin/env python3
"""Reproduce the user-selected physics_515 GUI trial against recorded Mode 6.

Uses the previously declared entry exclusion and fixed spatial ROI, retains
zero-depth cells, and checks the current >=50% removal requirement.
"""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np
import torch

from nrs_rl.tasks.manager_based.nrs_rl.utils.reporting.compare_isaac_tcp_runs import ROOT, close, csv_rows, dump, load_run, sha
from nrs_rl.tasks.manager_based.nrs_rl.utils.reporting.compare_isaac_spatial_runs import plots, processing


def policy_audit(run, output):
    source = Path(run['audit']['run']) / 'policy.csv'
    p = np.genfromtxt(source, delimiter=',', names=True)
    if any(not np.isfinite(p[k]).all() for k in p.dtype.names):
        raise ValueError('Nonfinite policy log')
    close(p['sequence'], np.arange(1, len(p) + 1), 'policy sequence')
    if len(np.unique(p['epoch'])) != 1 or np.any(p['fault']) or p['done'][-1] != 1:
        raise ValueError('Incomplete or faulted policy run')
    close(np.diff(p['elapsed_s']), .008, 'policy cadence')
    close(p['cursor_mm'][-1], p['path_length_mm'][-1], 'policy completion')
    close(p['return_action'], -1., 'disabled return action')
    close(p['reverse_mm'], 0., 'no reverse path')
    if p['missing_samples'].max() != 0:
        raise ValueError('Missing policy observations')
    dep = run['deployment']
    expected = 'e809395fc49fef510129f55e0a4ae086e6757bb04af12b89c6ca252c069a1b01'
    if dep['checkpoint_sha256'] != expected or sha(dep['checkpoint']) != expected:
        raise ValueError('Not the selected physics_515 actor')
    bounds = dep['force_bounds_n']
    if np.any(p['target_force_n'] < bounds[0]-1.e-8) or np.any(p['target_force_n'] > bounds[1]+1.e-8):
        raise ValueError('Force target outside configured bounds')
    slew = float(np.max(np.abs(np.diff(p['target_force_n']))) / .008)
    if slew > dep['force_slew_n_s'] + 1.e-8:
        raise ValueError('Force target slew exceeded')
    for path, digest in dep['pinned_files'].items():
        if sha(path) != digest:
            raise ValueError('Deployment file changed: '+path)
    torch.set_num_threads(1)
    actor = torch.jit.load(dep['checkpoint'], map_location='cpu').eval()
    decisions = p['decision'] > 0
    close(decisions, (p['sequence']-1) % 10 == 0, 'decision cadence')
    obs = np.column_stack([p[k] for k in p.dtype.names if k.startswith('obs_')]).astype(np.float32)
    actions = np.column_stack([p[k] for k in ('feed_action', 'force_action', 'return_action')])
    with torch.inference_mode():
        reproduced = actor(torch.from_numpy(obs[decisions])).numpy()
    if not np.allclose(reproduced, actions[decisions], atol=1.e-6, rtol=0):
        raise ValueError('Recorded actions differ from checkpoint inference')
    held_indices = np.maximum.accumulate(np.where(decisions, np.arange(len(p)), 0))
    close(actions, actions[held_indices], 'held actions')
    keep = (p['elapsed_s'] >= 2.) & (p['done'] == 0)
    error = p['normal_force_n'][keep] - p['target_force_n'][keep]
    run['policy'] = p
    shutil.copy2(source, output/run['name']/'policy.csv')
    return {'checkpoint_sha256': expected, 'samples': len(p), 'decisions': int(decisions.sum()),
            'missing_samples': 0, 'fault_samples': 0, 'path_completed': True,
            'checkpoint_decision_replay_max_error': float(np.abs(reproduced-actions[decisions]).max()),
            'all_pinned_files_unchanged': True, 'pinned_file_count': len(dep['pinned_files']),
            'force_target_min_max_n': [float(p['target_force_n'].min()), float(p['target_force_n'].max())],
            'force_target_max_slew_n_s': slew,
            'processing_force_target_mean_n': float(p['target_force_n'][keep].mean()),
            'processing_force_measured_mean_n': float(p['normal_force_n'][keep].mean()),
            'processing_force_tracking_rmse_n': float(np.sqrt(np.mean(error**2))),
            'processing_force_tracking_bias_n': float(error.mean()),
            'requested_feed_min_max_mm_s': [float((6+3*p['feed_action']).min()), float((6+3*p['feed_action']).max())],
            'reverse_mm': float(p['reverse_mm'][-1]),
            'worker_latency_median_p99_max_ms': np.quantile(p['latency_ms'], [.5,.99,1.]).tolist(),
            'policy_sha256': sha(source)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode5', type=Path, required=True)
    parser.add_argument('--mode6', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    out = args.output or ROOT/'logs/polishing_results'/datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M')
    out.mkdir(parents=True, exist_ok=False)
    a = load_run(args.mode5.resolve(), 5, out)
    b = load_run(args.mode6.resolve(), 6, out)
    for key in ('trajectory_file', 'tangential_controller', 'backend', 'removal_model'):
        if a['deployment'][key] != b['deployment'][key]:
            raise ValueError('Comparison setting mismatch: '+key)
    if a['summary']['model'] != b['summary']['model']:
        raise ValueError('Removal model mismatch')
    close(a['audit']['saved_home_mm_rad'], b['audit']['saved_home_mm_rad'], 'initial home')
    for key in ('roi','u_mm','v_mm','cell_area_mm2','origin_mm','basis'):
        if not np.array_equal(a['surface'][key], b['surface'][key]):
            raise ValueError('Removal grid mismatch: '+key)
    if a['audit']['reference_txt_sha256'] != b['audit']['reference_txt_sha256']:
        raise ValueError('Reference mismatch')
    for run in (a,b):
        processing(run, out)
        run['diagnostics']['note'] = ('Primary metrics: fixed entry-separated ROI; full-window values retained. '
            'Legacy RMSE fields relative to nominal 20 N do not measure variable-target force tracking.')
        keep = run['t'] >= 2.
        integral = float(np.dot(run['rate'][keep], run['dt'][keep]))
        run['processing_metrics']['tcp_full_footprint_volume_over_k'] = integral
        run['processing_metrics']['distance_weighted_force_n'] = integral/run['processing_metrics']['distance_mm']
    if not np.array_equal(a['processing_roi'], b['processing_roi']):
        raise ValueError('Processing ROI mismatch')
    policy = policy_audit(a, out)
    fields = ['spatial_depth_cv','roi_volume_over_k','raw_rate_cv','processing_time_s','zero_depth_fraction','force_mean_n','mean_speed_mm_s']
    changes = {window: {k:100*(a[key][k]/b[key][k]-1) for k in fields}
               for window,key in [('full','metrics'),('processing','processing_metrics')]}
    ratios = {'processing_roi': a['processing_metrics']['roi_volume_over_k']/b['processing_metrics']['roi_volume_over_k'],
              'full_roi': a['metrics']['roi_volume_over_k']/b['metrics']['roi_volume_over_k'],
              'whole_grid': a['metrics']['grid_volume_over_k']/b['metrics']['grid_volume_over_k']}
    checks = {'completed_without_fault_and_returned_home': True,
              'spatial_cv_better_than_mode6': changes['processing']['spatial_depth_cv'] < 0,
              **{k+'_removal_at_least_50_percent': v >= .5 for k,v in ratios.items()}}
    historical_path = Path(a['deployment']['physics_validation'])
    historical = json.loads(historical_path.read_text())
    shutil.copy2(historical_path, out/'headless_validation.json')
    report = {'created_at': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
              'comparison_baseline': 'Mode 6', 'experiment': 'Isaac GUI physics_515 20261008_225006',
              'one_run_per_mode': True, 'matched_path_model_grid_home_tangent_mdk': True,
              'entry_protocol': {'seconds': 2., 'reference_footprint_mm': 18.,
                  'entry_deposit_subtracted': True, 'zero_depth_cells_retained': True,
                  'processing_roi_cells': int(a['processing_roi'].sum()),
                  'full_roi_cells': int(a['surface']['roi'].sum()), 'entry_peak_is_gate': False},
              'changes_percent_vs_mode6': changes, 'removal_ratios_vs_mode6': ratios,
              'acceptance_checks': checks, 'accepted': all(checks.values()), 'policy': policy,
              'runs': {r['name']: {k:r[k] for k in ('metrics','processing_metrics','diagnostics','audit')} for r in (a,b)},
              'headless_mode6_comparisons': historical['heldouts'],
              'limits': ['One historical Mode 6 and one new Mode 5 run; no repeatability or causal isolation claim.',
                  'Removal is uncalibrated nonrotating Preston V/K, not measured physical material volume.',
                  'Comparison includes force/feed policy and tangent tracking compensation.',
                  'ROS repeated position samples affect finite-difference instantaneous rate and travelled distance.',
                  'GUI/headless differences cannot be assigned to a unique physical or timing cause from this pair.']}
    dump(out/'comparison.json', report)
    csv_rows(out/'comparison.csv', [{'mode': r['name'], 'window': window, **{k:r[key][k] for k in fields}}
             for r in (a,b) for window,key in [('full','metrics'),('processing','processing_metrics')]])
    gain = -changes['processing']['spatial_depth_cv']
    text = [
        '14.24% physics_515 Isaac GUI 실증 분석 (모든 변화율은 Mode 6 대비)',
        f'Mode 5: {args.mode5.resolve()}', f'Mode 6: {args.mode6.resolve()}',
        '경로 완료, 제거량 저장, 홈 복귀 확인. 정확한 actor 해시와 모든 의사결정 출력 재현 확인.',
        '평가: 첫 2초 적분 제외 + 기준 첫 18 mm 공구영역 제외; 0 깊이 셀 포함, 공통 0.5 mm 격자.',
        f'주요 공간 CV: {b["processing_metrics"]["spatial_depth_cv"]:.6f} -> {a["processing_metrics"]["spatial_depth_cv"]:.6f}; {gain:.4f}% 개선.',
        f'가공 ROI 제거량: {ratios["processing_roi"]*100:.4f}% 유지 ({(1-ratios["processing_roi"])*100:.4f}% 감소).',
        f'전체 ROI 제거량: {ratios["full_roi"]*100:.4f}% 유지 ({(1-ratios["full_roi"])*100:.4f}% 감소).',
        f'전체 격자 제거량: {ratios["whole_grid"]*100:.4f}% 유지.',
        f'가공 시간: {b["processing_metrics"]["processing_time_s"]:.3f} -> {a["processing_metrics"]["processing_time_s"]:.3f} s.',
        f'시간 제거율 CV: {b["processing_metrics"]["raw_rate_cv"]:.6f} -> {a["processing_metrics"]["raw_rate_cv"]:.6f} (공간 CV와 별개).',
        f'실제 평균 힘: {b["processing_metrics"]["force_mean_n"]:.4f} -> {a["processing_metrics"]["force_mean_n"]:.4f} N.',
        f'Mode 5 가변 목표 추종 RMSE: {policy["processing_force_tracking_rmse_n"]:.4f} N; 목표 평균 {policy["processing_force_target_mean_n"]:.4f} N.',
        '', '판정: 균일도 개선은 확인되지만 제거량 50% 이상 유지 조건은 미달. headless 14.24% 개선은 GUI에서 재현되지 않음.',
        '원인 해석: 정책이 낮은 목표 힘과 가변 속도를 실제 적용했고, 측정 힘은 가변 목표를 추종함. 정책 미적용이 아님.',
        '기존 학습 검증은 Mode 6 가공량의 약 50.92~50.96%만 유지해 전이 오차를 견딜 여유가 작았음.',
        'GUI와 headless의 샘플 유지/이동거리/접촉 응답 차이는 존재하나, 이 한 쌍만으로 성능 차이의 단일 원인을 확정할 수 없음.',
        '다음 개선안: 이번 GUI의 위치별 힘-제거량 응답으로 보정하고, 학습 시 Mode 6 대비 제거량 55~60% 유지 여유를 둔 후보를 검증.',
        '초기 오버슛은 선택 기준에서 제외. 기존 체크포인트/배포 설정은 변경하지 않음.',
        '제거량은 미교정 Preston V/K 추정치이며 실측 재료 체적(mm³)이 아님. 각 1회 비교.',
    ]
    (out/'REPORT.txt').write_text('\n'.join(text)+'\n')
    plots([a,b], out)
    shutil.copy2(__file__, out/'analyze_spatial_14p24_trial.py')
    for name in ('compare_isaac_tcp_runs.py','compare_isaac_spatial_runs.py'):
        shutil.copy2(ROOT/'scripts'/name, out/name)
    dump(out/'artifact_manifest.json', {'files': {str(f.relative_to(out)): sha(f)
         for f in sorted(out.rglob('*')) if f.is_file()}})
    print(json.dumps({'output':str(out),'changes_percent_vs_mode6':changes,
          'removal_ratios':ratios,'accepted':report['accepted'],'policy':policy},indent=2))


if __name__ == '__main__':
    main()
