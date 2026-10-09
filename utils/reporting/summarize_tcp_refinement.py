#!/usr/bin/env python3
"""Audit and plot refinement results, including unsuccessful candidates."""
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

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import nrs_rl.tasks.manager_based.nrs_rl.utils.reporting.export_polishing_result_graphs as plots
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_tcp_refinement import REFERENCE, evaluate_candidate, promotion_decision, sha, WEIGHTS
from nrs_rl.tasks.manager_based.nrs_rl.utils.reporting.summarize_tcp_control_training import values, distribution, compare

MODES = ('constant_a', 'constant_b', 'old_d14', 'previous', 'calibrated_only', 'learned')
LABELS = {'constant_a': 'Original constant A', 'constant_b': 'Original constant B',
          'old_d14': 'Original Mode 5 (D14)', 'previous': 'Previous validated model',
          'calibrated_only': 'Refined MDK + constant', 'learned': 'Refined MDK + policy'}
COLORS = {'constant': '#666666', 'old_d14': '#D6604D', 'previous': '#8e66aa',
          'calibrated_only': '#b1a161', 'learned': '#2166AC'}


def audit(root):
    state = json.loads((root/'result.json').read_text())
    protocol = json.loads((root/'protocol.json').read_text())
    if state['phase'] != 'complete':
        raise ValueError('Wait for the complete frozen experiment')
    if 'tracking_compensation' in state:
        LABELS['learned'] = 'Learned tracking + existing policy'
        LABELS['calibrated_only'] = 'Learned tracking + constant feed'
    for key in ('reward', 'original_frozen_gates', 'promotion_gates', 'training_seeds', 'validation_seeds'):
        if state[key] != protocol[key]:
            raise ValueError('Frozen protocol changed: '+key)
    if set(state['training_seeds']+[state['screen_seed']]) & set(state['validation_seeds']):
        raise ValueError('Training and validation conditions overlap')
    if sha(state['checkpoint']) != state['checkpoint_sha256'] or sha(REFERENCE) != state['reference_path_sha256']:
        raise ValueError('Checkpoint or reference changed')
    with h5py.File(REFERENCE) as file:
        reference, force = file['position'][:], file['force'][:]
    trials, table, paired, reconstructed_validation, input_hashes = [], [], [], [], {}
    for episode, seed in enumerate(state['validation_seeds'], 1):
        folder = root/f'held_out_{seed}'
        batch = json.loads((folder/'summary.json').read_text())
        with h5py.File(folder/'path.h5') as file:
            if not np.array_equal(file['position'][:], reference) or not np.array_equal(file['force'][:], force):
                raise ValueError('Validation path is not the complete pinned reference')
        if (batch['rotation_depth_model'] is not None or batch['candidate_checkpoint_sha256'] != state['checkpoint_sha256']
                or batch['comparison_checkpoint_sha256']['previous'] != state['previous_checkpoint_sha256']):
            raise ValueError('Validation policy identity or RPM model changed')
        if 'tracking_compensation' in state:
            expected_gains = {name: (state['tracking_compensation']['gain'] if name in ('learned', 'calibrated_only') else 0.)
                              for name in MODES}
            if batch['tracking_compensation']['gains'] != expected_gains:
                raise ValueError('Validation tracking compensation differs from the frozen controller')
        if any(any(v != a[0] for v in a) for a in batch['randomization'].values()):
            raise ValueError('Paired conditions differ')
        group = {}
        for mode in MODES:
            path = folder/f'{mode}_trace.npz'
            with np.load(path) as file:
                trace = {key: file[key].copy() for key in file.files}
            active = trace['polishing_active'] > 0
            rate = trace['raw_mrr_n_mm_s'][active].astype(float)
            time = trace['time_s'][active]
            metrics = batch['candidates'][mode]
            cv = float(rate.std(ddof=0)/rate.mean())
            if (not np.isfinite(rate).all() or not np.allclose(np.diff(time), .008, atol=1e-9)
                    or len(rate) != metrics['processing_samples']
                    or not np.isclose(cv, metrics['processing_rate_cv'], rtol=1e-10)):
                raise ValueError('Raw trace does not reproduce saved metrics')
            trial = {'episode': episode, 'seed': seed, 'mode': mode, 'label': LABELS[mode],
                'path': path, 'trace': trace, 'active': active, 'time': time-time[0],
                'rate': rate, 'cv': cv, 'dt': .008, 'duration': len(rate)*.008, 'metrics': metrics}
            trials.append(trial)
            group[mode] = values(metrics, .008)
            table.append({'episode': episode, 'seed': seed, 'mode': mode,
                          'completed': metrics['completed'], **group[mode]})
            input_hashes[str(path)] = sha(path)
        group['constant'] = {key: float(np.mean([group[n][key] for n in ('constant_a', 'constant_b')]))
                             for key in group['constant_a']}
        paired.append(group)
        reconstructed_validation.append({'seed': seed, 'assessment': evaluate_candidate(
            batch['candidates']['learned'], batch['candidates'], batch['path_length_mm'], full_path=True)})
        input_hashes[str(folder/'summary.json')] = sha(folder/'summary.json')
    decision = promotion_decision(reconstructed_validation)
    if not state['training_candidate_eligible']:
        decision['passed'] = False
        decision['rejections'].append('No candidate eligible on both training conditions')
    if decision != state['promotion'] or decision['passed'] != state['validation_passed']:
        raise ValueError('Stored promotion decision does not reproduce from raw comparisons')
    stats = {}
    for mode in ('constant', 'old_d14', 'previous', 'calibrated_only', 'learned'):
        members = ('constant_a', 'constant_b') if mode == 'constant' else (mode,)
        stats[mode] = {key: distribution([p[m][key] for p in paired for m in members]) for key in paired[0][mode]}
    contrasts = {}
    for base in ('constant', 'old_d14', 'previous', 'calibrated_only'):
        rows = [compare(group['learned'], group[base]) for group in paired]
        contrasts[f'learned_vs_{base}'] = {key: distribution([r[key] for r in rows]) for key in rows[0]}
    adaptive = any(abs(x) > 1e-12 for x in state['best_parameters'][2:])
    report = {'phase': 'complete', 'algorithm': state['algorithm'], 'checkpoint': state['checkpoint'],
        'checkpoint_sha256': state['checkpoint_sha256'], 'previous_checkpoint': state['previous_checkpoint'],
        'previous_checkpoint_sha256': state['previous_checkpoint_sha256'], 'validation_seeds': state['validation_seeds'],
        'training_seeds': state['training_seeds'], 'selected_family': state['selected_family'],
        'parameters': dict(zip(state['parameter_names'], state['best_parameters'])),
        'frozen_gates_passed': state['validation_passed'], 'adaptive_velocity_enabled': adaptive,
        'promotion': decision, 'statistics': stats,
        'paired_comparisons': contrasts, 'input_sha256': input_hashes,
        'rotation_contribution': False, 'spindle_rpm': None, 'physical_depth_validated': False,
        'reference_path_length_mm': batch['path_length_mm'], 'reference_rows': len(reference),
        'attribution': ('MDK and feed were optimized together; only the calibrated_only ablation isolates feed-policy benefit.'
                        if adaptive else 'Zero feedback selected: this is an MDK calibration gain, not an adaptive-policy benefit.'),
        'limits': ['Raw rate is an uncalibrated F*TCP-velocity proxy; volume is /K, not material mm³.',
                   'Three new sensor/delay conditions on one flat path; descriptive means/ranges only.',
                   'Headless physics validation, not a ROS GUI or physical machining run.',
                   'A rejected refinement must not replace the previous validated checkpoint.']}
    if 'tracking_compensation' in state:
        report['tracking_compensation'] = state['tracking_compensation']
        report['runtime_integration_validated'] = state['runtime_integration_validated']
        report['parameters']['tracking_gain'] = state['tracking_compensation']['gain']
        report['attribution'] = ('Only the bounded tangential tracking-feedback gain was trained. '
            'Velocity policy coefficients, MDK, force target, servo and physics are unchanged. '
            'learned_vs_previous isolates the tracking-compensation effect.')
        report['limits'].append('Experimental tracking backend requires a matching new controller runtime; the legacy ROS worker rejects it.')
    return state, report, trials, table, paired, reference


def overview(report, trials, output):
    modes = tuple(report['statistics'])
    labels = ['Original\nconstant', 'Original\nMode 5', 'Previous\nvalidated', 'Refined\nMDK only', 'Refined\nMDK + policy']
    if not report['adaptive_velocity_enabled']:
        labels[-1] = 'Refined MDK\n(selected constant)'
    if 'tracking_compensation' in report:
        labels[-2:] = ['Tracking\n+ constant', 'Tracking\n+ policy']
    fields = [('roi_volume_over_k', 'ROI removal / K', False), ('spatial_depth_cv', 'Spatial depth CV', True),
              ('raw_rate_cv', 'Raw temporal rate CV', True), ('force_rmse_n', 'Force RMSE [N]', True),
              ('cross_track_rmse_mm', 'Cross-track RMSE [mm]', True), ('processing_time_s', 'Processing time [s]', True)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.6))
    for ax, (key, title, lower) in zip(axes.flat, fields):
        data = [report['statistics'][mode][key] for mode in modes]
        means = np.array([d['mean'] for d in data])
        errors = np.array([[m-d['min'], d['max']-m] for m, d in zip(means, data)]).T
        if errors.min() < -1e-9:
            raise ValueError('A reported mean falls outside the sample range')
        errors = np.maximum(errors, 0.)  # Equal-value means can round by ~1e-14.
        bars = ax.bar(range(len(modes)), means, color=[COLORS[mode] for mode in modes],
                      yerr=errors, capsize=3, error_kw={'lw': .8})
        ax.bar_label(bars, fmt='%.3f' if means.max()<10 else '%.1f', fontsize=7, padding=5)
        ax.set_xticks(range(len(modes)), labels, fontsize=7)
        ax.set_title(title+(' (lower is better)' if lower else ' (higher is better)'))
        ax.set_ylim(0, max(d['max'] for d in data)*1.2)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    result = 'PASSED frozen promotion gates' if report['frozen_gates_passed'] else 'NOT PROMOTED: retain previous model'
    seeds = ' / '.join(str(seed) for seed in report['validation_seeds'])
    fig.suptitle('New held-out conditions '+seeds+' | '+result)
    plots.savefig(fig, output/'00_validation_overview.png', footer=
        'Mean and min/max across held-out conditions; original constant includes both replicates. '
        'RPM excluded; uncalibrated simulation proxy.')
    # Show the previous checkpoint beside the new controller and its constant-feed
    # ablation; original Mode 5 remains in the historical 01--09 plots.
    for episode in range(1, 4):
        group = {t['mode']: t for t in trials if t['episode'] == episode}
        fig, axes = plt.subplots(2, 1, figsize=(10, 5.7), sharex=True)
        for mode in ('previous', 'calibrated_only', 'learned'):
            t = group[mode]
            axes[0].plot(t['time'], t['rate']*.001, color=COLORS[mode], alpha=.75, lw=.65,
                label=f"{t['label']} (raw CV={t['cv']:.3f})")
            axes[1].plot(t['time'], t['trace']['tracking_error_mm'][t['active']],
                         color=COLORS[mode], alpha=.75, lw=.7, label=t['label'])
        axes[0].set(ylabel='Raw Preston rate [N m/s]', ylim=(0, None))
        axes[0].legend(loc='upper right')
        axes[1].set(ylabel='Reference tracking error [mm]', xlabel='Processing time [s]')
        for ax in axes:
            plots.decorate(ax)
        fig.suptitle(f"Previous vs refined controller | held-out seed {group['learned']['seed']}")
        ablation = ('Tracking compensation is enabled in both new traces; the velocity policy is unchanged.'
                    if 'tracking_compensation' in report else
                    'The MDK-only trace separates controller calibration from policy effect.')
        plots.savefig(fig, output/f'ep{episode}/12_previous_vs_refined.png', footer=
            'All raw samples including zeros and transients. '+ablation)


def rewards(state, output):
    output.mkdir()
    rows = []
    for group in state['generations']:
        for name, record in group['ranking'].items():
            rows.append({'seed': group['seed'], 'candidate': name, 'score': record['score'],
                'eligible': record['eligible'], 'constraint_penalty': record['constraint_penalty'],
                **record['weighted_reward'], 'rejections': ' | '.join(record['rejections'])})
    plots.write_csv(output/'00_reward_components.csv', rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, group in zip(axes, state['generations']):
        ranking = group['ranking']
        names = list(ranking)
        colors = ['#2166AC' if name == state['selected_family'] else '.65' for name in names]
        bars = ax.bar(range(len(names)), [ranking[n]['score'] for n in names], color=colors)
        ax.bar_label(bars, fmt='%.3f', padding=4, fontsize=8)
        ax.set_xticks(range(len(names)), names, rotation=35, ha='right')
        ax.set(title=f"Full-path training seed {group['seed']}", ylabel='Search reward relative to previous model')
        ax.grid(axis='y', alpha=.2)
    plots.savefig(fig, output/'00_reward_components.png', footer=
        'Bounded parameter search. Candidates must pass both training conditions; selection maximizes the worst-condition reward.')


def summarize(args):
    root = args.training.resolve()
    state, report, trials, table, paired, reference = audit(root)
    if not report['adaptive_velocity_enabled']:
        LABELS['learned'] = 'Refined MDK + selected constant feed'
        for trial in trials:
            trial['label'] = LABELS[trial['mode']]
    root.joinpath('report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    plots.write_csv(root/'validation_comparison.csv', table)
    now = datetime.now(ZoneInfo('Asia/Seoul'))
    output = (args.output or plots.ROOT/'logs/polishing_results'/now.strftime('%Y%m%d_%H%M')).resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Reuse only plotting functions; frozen source traces/reports are untouched.
    plots.MODES = MODES
    plots.LABELS.update(LABELS)
    plots.COLORS.update(COLORS)
    xy = np.vstack([reference[:, :2]]+[t['trace']['tcp_pose'][t['active'], :2] for t in trials])
    lo, hi = xy.min(0)-20, xy.max(0)+20
    extent = [float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])]
    end = max(t['duration'] for t in trials)
    for t in trials:
        t['spatial_map'] = plots.spatial_rate(t, extent)
    spatial_max = float(max(t['spatial_map'][0].max() for t in trials))
    raw_max = float(max(t['rate'].max() for t in trials)*.001)
    temporal_max = float(max(np.nanmax(plots.temporal_bins(t, .128, end)['mean']) for t in trials)*.001)
    fluctuation_max = float(max(np.nanmax(plots.temporal_bins(t, 1., end)['relative_rms']) for t in trials))
    for episode, seed in enumerate(state['validation_seeds'], 1):
        folder = output/f'ep{episode}'
        folder.mkdir()
        group = {t['mode']: t for t in trials if t['seed'] == seed}
        plots.episode_plots(group, folder, reference, extent, spatial_max, raw_max, temporal_max, fluctuation_max, end)
        lines = [f'ep{episode}: independent validation seed {seed}',
                 f'Checkpoint: {state["checkpoint_sha256"]}',
                 f'Overall promotion passed: {state["validation_passed"]}',
                 'RPM excluded; raw rate N mm/s, graph rate N m/s; volume/K is uncalibrated.',
                 json.dumps({mode: plots.metric_row(t) for mode, t in group.items()}, indent=2)]
        (folder/'00_summary.txt').write_text('\n'.join(lines)+'\n')
    plots.temporal_heatmap(trials, output, end, temporal_max, fluctuation_max, '00_temporal')
    overview(report, trials, output)
    rewards(state, output/'reward_logs')
    plots.write_csv(output/'00_comparison_summary.csv', table)
    plots.write_csv(output/'00_episode_summary.csv', [r for r in table if r['mode'] == 'learned'])
    for name in ('report.json', 'result.json', 'protocol.json'):
        shutil.copy2(root/name, output/('00_frozen_'+name))
    contrast = report['paired_comparisons']['learned_vs_previous']
    lines = [f'추가 재학습 결과 — {now.isoformat()}',
        f"판정: {'새 후보 개선 기준 통과' if state['validation_passed'] else '새 후보 채택하지 않음; 직전 검증 모델 유지'}",
        f"후보: {state['selected_family']}; 파라미터: {report['parameters']}",
        '접촉력 목표 20 N, RPM 제외, 동일한 전체 경로 및 고정 ROI.',
        '직전 검증 모델도 새 조건에서 체크포인트로 다시 실행해 비교했습니다.',
        ', '.join(f'ep{i}={seed}' for i, seed in enumerate(state['validation_seeds'], 1))+': 검증 조건.',
        '학습 조건 '+ '/'.join(str(seed) for seed in [state['screen_seed'], *state['training_seeds']])+'과 분리했습니다.', '',
        '직전 모델 대비 새 조건 3개에서의 평균 변화:',
        f"ROI 제거량/K 증가: {contrast['roi_volume_increase_percent']['mean']:.4f}%",
        f"공간 깊이 CV 감소: {contrast['spatial_depth_cv_reduction_percent']['mean']:.4f}%",
        f"순간 제거율 CV 감소: {contrast['raw_rate_cv_reduction_percent']['mean']:.4f}%",
        f"힘 RMS 오차 감소: {contrast['force_rmse_n_reduction_percent']['mean']:.4f}%",
        f"가공 시간 감소: {contrast['processing_time_s_reduction_percent']['mean']:.4f}%", '',
        '판정 근거: '+json.dumps(state['promotion'], ensure_ascii=False), '',
        '01~09: 기존 20260909_123708_flat의 그래프 구성.',
        '10: 128 ms 평균 순간 제거율. CV는 표시 평균이 아닌 원시 125 Hz 값으로 계산.',
        '11: 1초별 RMS((rate - 해당 주행 전체 평균) / 전체 평균). 밝을수록 편차가 큼.',
        ('12: 직전 모델/추종 보정 정속/추종 보정+기존 정책의 원시 제거율 및 추종 오차 직접 비교.'
         if 'tracking_compensation' in state else
         '12: 직전 모델/새 MDK 정속/새 정책의 원시 제거율 및 추종 오차 직접 비교.'),
        '모든 시간 구간, 0값, 진입 과도 응답을 포함했습니다. 회색은 관측 종료 후 빈 구간.',
        '색 범위는 전체 비교에 공통입니다. 공간 평활화(01/05)와 시간별 변동(11)은 별도 지표입니다.',
        'reward_logs: 전체 경로 학습 후보 점수와 실제 보상 성분; PPO 학습 곡선이 아닙니다.',
        '시뮬레이션 Preston proxy이며 실제 재료 제거 체적(mm³) 보정이나 ROS GUI 재실행은 포함하지 않습니다.',
        f'원본: {root}']
    if 'tracking_compensation' in state:
        lines += ['', '이번 추가 학습 대상은 접선 추종 오차 보정 게인입니다. 기존 속도 정책 계수와 MDK는 고정했습니다.',
                  '보정 설정: '+json.dumps(state['tracking_compensation']),
                  '기존 ROS worker는 이 실험용 backend를 거부합니다. 맞는 추종 보정 제어부가 필요합니다.']
    (output/'00_episode_summary.txt').write_text('\n'.join(lines)+'\n')
    (output/'00_visualization_notes.txt').write_text('\n'.join(lines[-11:])+'\n')
    (root.parent/'REPORT.txt').write_text('\n'.join(lines)+'\n\n'+json.dumps(report['statistics'], indent=2)+'\n')
    for t in trials:
        folder = output/f"ep{t['episode']}"
        copy = folder/'08_control_trace.npz' if t['mode'] == 'learned' else folder/'comparison_traces'/f"{t['mode']}_trace.npz"
        if sha(copy) != sha(t['path']):
            raise ValueError('Exported trace differs from source')
    manifest = {'training_source': str(root), 'created_at': now.isoformat(), 'checks_passed': True,
        'validation_passed': state['validation_passed'], 'checkpoint_sha256': state['checkpoint_sha256'],
        'episode_mapping': {f'ep{i}': s for i, s in enumerate(state['validation_seeds'], 1)},
        'input_sha256': report['input_sha256'], 'trials': len(trials), 'raw_samples': sum(len(t['rate']) for t in trials),
        'trace_copies_sha256_verified': len(trials), 'raw_metrics_reproduce_report': True,
        'plot_files': sorted(str(p.relative_to(output)) for p in output.rglob('*.png'))}
    (output/'00_export_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    (root/'postprocessing.json').write_text(json.dumps({'output': str(output), 'checks_passed': True,
        'report': str(root/'report.json'), 'plots': len(manifest['plot_files'])}, indent=2)+'\n')
    print(json.dumps({'graphs': str(output), 'promotion_passed': state['validation_passed'],
                      'comparison': contrast, 'plots': len(manifest['plot_files'])}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    summarize(parser.parse_args())
