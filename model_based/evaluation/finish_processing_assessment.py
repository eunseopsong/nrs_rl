#!/usr/bin/env python3
"""Freeze training-only selection, then assess fixed post-entry holdouts.

This supplements the original all-episode reports; it never rewrites them.
Selection must agree with the actor actually run in independent validation.
"""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from nrs_rl.tasks.manager_based.nrs_rl.utils.assess_processing_uniformity import evaluate
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.spatial_uniformity import ROOT, assess


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def wait_for(path):
    while not path.exists():
        time.sleep(5)


def process_study(root, refined):
    training_seed = 1701 if refined else 1501
    heldout_seeds = [1801, 1802] if refined else [1601, 1602]
    wait_for(root/f'train_{training_seed}/summary.json')
    training = evaluate(root/f'train_{training_seed}')
    names = ['learned', 'alternative_1'] if refined else ['learned', 'force_only', 'alternative_1', 'alternative_2']
    references = ['constant'] if refined else ['constant', 'constant_b']
    ranking = {}
    for name in names:
        m = training['candidates'][name]
        checks = [assess(m, training['candidates'][b]) for b in references]
        ranking[name] = {'score': min(c['score'] for c in checks), 'checks': checks}
        if refined:
            ranking[name]['matched_tracking_comparison'] = assess(m, training['candidates']['matched_constant'])
    selected = max(names, key=lambda n: ranking[n]['score'])
    selection = {'selected': selected, 'training_seed': training_seed, 'ranking': ranking,
                 'independent_scores_used_for_selection': False,
                 'selected_at': datetime.now().isoformat(),
                 'evaluation': 'Fixed entry separation described in processing_evaluation_protocol.json'}
    selection_path = root/'processing_selection.json'
    if selection_path.exists():
        previous_selection = json.loads(selection_path.read_text())
        if previous_selection.get('ranking') == ranking and previous_selection.get('selected') == selected:
            selection = previous_selection
    save(selection_path, selection)
    print('PROCESSING_SELECTION', root.name, selected, flush=True)
    # This assertion precedes opening any holdout summaries here.
    while True:
        state = json.loads((root/'status.json').read_text())
        if state.get('selected') is not None:
            break
        time.sleep(5)
    if selected != state['selected']:
        raise RuntimeError(f"{root.name}: processing winner {selected} differs from executed actor {state['selected']}; new holdouts required")
    frozen_hash = state['checkpoint_sha256']
    rows = []
    for seed in heldout_seeds:
        directory = root/f'held_out_{seed}'
        wait_for(directory/'summary.json')
        original = json.loads((directory/'summary.json').read_text())
        assert original['checkpoint_sha256'][selected] == frozen_hash
        data = evaluate(directory)
        m = data['candidates'][selected]
        checks = [assess(m, data['candidates'][b]) for b in references]
        row = {'seed': seed, 'metrics': m, 'checks': checks,
               'previous_policy_comparison': assess(m, data['candidates']['previous']),
               'passed': all(c['passed'] and c['gains']['spatial_cv_gain'] >= .03 for c in checks),
               'full_episode_metrics': original['candidates'][selected]}
        if refined:
            row['matched_tracking_comparison'] = assess(m, data['candidates']['matched_constant'])
            row['passed'] = row['passed'] and row['matched_tracking_comparison']['gains']['spatial_cv_gain'] >= .02
        rows.append(row)
        print('PROCESSING_HELDOUT', root.name, seed, row['passed'], checks[0]['gains'], flush=True)
    result = {'phase': 'complete', 'selected': selected, 'checkpoint_sha256': frozen_hash,
              'training_seed': training_seed, 'heldout_seeds': heldout_seeds, 'heldouts': rows,
              'entry_seconds': 2., 'entry_force_peak_is_selection_gate': False,
              'processing_roi_cells': training['processing_roi_cells'],
              'full_roi_cells': training['full_roi_cells'],
              'validation_passed': all(r['passed'] for r in rows), 'deployed': False,
              'minimum_spatial_gain': .03, 'minimum_gain_over_matched_tracking': .02 if refined else None,
              'independent_scores_used_for_selection': False,
              'selection': selection,
              'interpretation': 'Uncalibrated removal model in headless Isaac; same flat path, independent seeds; not real material validation'}
    save(root/'processing_result.json', result)
    return result


def plots(export, directory, selected):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/nrs_processing_assessment_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['constant', 'previous']
    if (directory/'matched_constant_trace.npz').exists():
        names.append('matched_constant')
    names.append(selected)
    labels = {'constant': 'Fixed 20 N, 6 mm/s', 'previous': 'Previous feed policy',
              'matched_constant': 'Tracking correction only', selected: 'New force/feed policy'}
    arrays = []
    for name in names:
        with np.load(directory/f'{name}_processing_surface.npz') as data:
            arrays.append({k: data[k] for k in data.files})
    vmax = max(float(np.max(a['depth_over_k'][a['roi']])) for a in arrays)
    fig, axes = plt.subplots(1, len(names), figsize=(5*len(names), 5), constrained_layout=True)
    for ax, name, data in zip(axes, names, arrays):
        values = data['depth_over_k'][data['roi']]
        im = ax.imshow(np.where(data['roi'], data['depth_over_k'], np.nan), origin='lower',
                       extent=[data['u_mm'][0], data['u_mm'][-1], data['v_mm'][0], data['v_mm'][-1]],
                       vmin=0., vmax=vmax, cmap='viridis')
        ax.set_title(f"{labels[name]}\nSpatial CV {values.std()/values.mean():.4f}")
        ax.set_xlabel('Plane u (mm)');ax.set_ylabel('Plane v (mm)')
    fig.colorbar(im, ax=axes, label='Depth / K (uncalibrated)')
    fig.suptitle('Processing depth: fixed entry footprint excluded; unprocessed cells retained')
    fig.savefig(export/'04_processing_depth.png', dpi=150);plt.close(fig)
    fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True, constrained_layout=True)
    heatmaps = []
    maximum_time = 0.
    for name in names:
        with np.load(directory/f'{name}_trace.npz') as trace:
            t = trace['time_s'];maximum_time = max(maximum_time, float(t[-1]))
            for ax, key in zip(axes, ['normal_force_n', 'measured_speed_mm_s', 'raw_mrr_n_mm_s', 'target_force_n']):
                ax.plot(t, trace[key], lw=.5, alpha=.8, label=labels[name])
            heatmaps.append((t.copy(), trace['raw_mrr_n_mm_s'].copy()))
    for ax, label in zip(axes, ['Contact force (N)', 'Actual sliding speed (mm/s)', 'Instantaneous removal / K (N mm/s)', 'Target force (N)']):
        ax.axvspan(0., 2., color='grey', alpha=.25)
        ax.set_ylabel(label);ax.grid(alpha=.2);ax.set_xlim(0., maximum_time)
    axes[0].legend(ncol=2);axes[-1].set_xlabel('Time since active processing began (s)')
    fig.suptitle('125 Hz measured signals; shaded first 2 s reported separately')
    fig.savefig(export/'05_entry_separated_signals.png', dpi=150);plt.close(fig)
    grid = np.arange(0., maximum_time+.008, .008)
    heat = np.full((len(names), len(grid)), np.nan)
    for i, (t, rate) in enumerate(heatmaps):
        indices = np.rint(t/.008).astype(int)
        heat[i, indices] = rate
    fig, ax = plt.subplots(figsize=(15, 4), constrained_layout=True)
    image = ax.imshow(heat, aspect='auto', origin='upper', interpolation='nearest',
                      extent=[0., grid[-1], len(names)-.5, -.5],
                      cmap='viridis', vmin=0., vmax=float(np.nanmax(heat)))
    for boundary in np.arange(.5, len(names)-.5, 1.):
        ax.axhline(boundary, color='white', lw=.4)
    ax.axvline(2., color='white', ls='--', lw=1.)
    ax.set_yticks(np.arange(len(names)), [labels[n] for n in names])
    ax.set_xlabel('Active processing time (s)')
    ax.set_title('Instantaneous removal rate over time; equal scale, no smoothing')
    fig.colorbar(image, ax=ax, label='Removal rate / K (N mm/s)')
    fig.savefig(export/'06_removal_rate_time_heatmap.png', dpi=160);plt.close(fig)


def main(args):
    study = args.study.resolve()
    roots = [study/'physics', study/'refined_physics']
    results = [process_study(root, refined=bool(i)) for i, root in enumerate(roots)]
    # The all-episode exporter and this postprocessor often finish within
    # the same minute. Reuse its completed export, with distinct filenames.
    wait_for(roots[1]/'exports.json')
    export = Path(json.loads((roots[1]/'exports.json').read_text())['directory'])
    save(export/'processing_results.json', {root.name: result for root, result in zip(roots, results)})
    shutil.copyfile(study/'processing_evaluation_protocol.json', export/'processing_evaluation_protocol.json')
    rows = []
    report = ['비회전·단일 연속 가공의 힘/속도 정책 학습 및 Isaac 검증',
              '평가 변경 근거: 사용자의 요청에 따라 첫 2초 힘 오버슛을 모델 선정에서 제외.',
              '주 평가: 첫 2초 누적 제거량을 뺀 뒤 공통 고정 진입 footprint 영역을 제외한 공간 CV.',
              '진입 영역은 기준 경로 첫 18 mm (2초 × 최대 지령 9 mm/s)의 공구 footprint. 후보별로 바꾸지 않는다.',
              '이후 정지/복귀/미가공 셀은 모두 유지. 전체 경로의 결과는 기존 보고서에 보존.',
              '접촉력 및 속도는 학습 대상; 체류시간은 움직임의 결과로 계산. 정지 자체에 제거량을 부여하지 않음.',
              '필요할 때 경로 도중 제한된 국소 복귀 허용. 전체 경로를 별도로 두 번째 가공하지 않음.',
              '']
    for root, result in zip(roots, results):
        report += [f"{root.name}: 선정 {result['selected']}, 주 평가 검증 통과 {result['validation_passed']}",
                   f"고정 평가 영역 {result['processing_roi_cells']} / 전체 {result['full_roi_cells']} cells"]
        for item in result['heldouts']:
            gain = item['checks'][0]['gains'];m = item['metrics']
            report.append(f"seed {item['seed']}: 공간 CV {m['spatial_cv']:.6f}, 개선 {100*gain['spatial_cv_gain']:.3f}%, "
                          f"ROI 제거량 {100*(gain['volume_ratio']-1):+.3f}%, 시간 {100*(gain['time_ratio']-1):+.3f}%, "
                          f"순간 제거율 CV {100*(gain['raw_cv_ratio']-1):+.3f}%, 복귀 {m['return_events']}회")
            report.append('제약 위반: '+json.dumps({k: v for k, v in item['checks'][0]['violations'].items() if v > 1e-8}, ensure_ascii=False))
            if 'matched_tracking_comparison' in item:
                g = item['matched_tracking_comparison']['gains']
                report.append(f"동일 추종 보정·고정 힘/속도 대비: 정책의 추가 공간 CV 개선 {100*g['spatial_cv_gain']:.3f}%, "
                              f"ROI 제거량 {100*(g['volume_ratio']-1):+.3f}%, 시간 {100*(g['time_ratio']-1):+.3f}%, "
                              f"순간 제거율 CV {100*(g['raw_cv_ratio']-1):+.3f}%")
            directory = root/f"held_out_{item['seed']}"
            summary = json.loads((directory/'processing_summary.json').read_text())
            full = json.loads((directory/'summary.json').read_text())
            for name, metrics in summary['candidates'].items():
                row = {'study': root.name, 'seed': item['seed'], 'policy': name}
                row.update({k: metrics[k] for k in ('spatial_cv', 'rate_cv', 'roi_volume_over_k', 'time_s',
                    'force_mean_n', 'force_peak_n', 'force_tracking_rmse_n', 'zero_fraction',
                    'entry_force_peak_n', 'entry_force_tracking_rmse_n', 'return_events')})
                row.update({'full_'+k: full['candidates'][name][k] for k in ('spatial_cv', 'rate_cv', 'roi_volume_over_k', 'time_s', 'force_peak_n', 'zero_fraction')})
                rows.append(row)
        report += ['ckpt: '+str(root/'candidate/checkpoints/actor.pt'), 'SHA256: '+result['checkpoint_sha256'], '']
    report += ['알고리즘: 물리식 기반 대리 모델의 CEM 정책 파라미터 탐색 후 Isaac 학습 시드로 선정; PPO 신경망 재학습 아님.',
               '통과 조건의 제거량/미가공 면적 기준은 원래 고정 힘·속도 비교군이다. 추종 보정 전용 비교군 대비로는 추가 공간 CV 개선 2%를 요구했다.',
               '검증 시드는 선정에 사용하지 않음. 동일 flat 경로에 한정된 결과; 다른 경로 및 실물 검증은 미완료.',
               '제거 깊이와 제거량은 교정 전 h/K, V/K. 실제 재료 제거량 개선을 확정하지 않음.',
               '국소 복귀 전달 시험: 기준 cursor 약 6.62 mm 후퇴에 실제 TCP 후퇴 약 0.96 mm. 복귀 효과에 모델 오차 존재.',
               '새 16D 관측 / 3-action 정책은 기존 ROS Mode 5와 직접 호환되지 않음. 현재 로봇 설정이나 배포 ckpt를 바꾸지 않음.',
               '초반 피크의 실환경 재현 여부는 아직 확인하지 않았고, 하드웨어 보호 및 제어기 fault guard는 유지.',
               '원본 전체 구간 보고서: '+str(roots[0]/'REPORT.txt')+', '+str(roots[1]/'REPORT.txt')]
    with (export/'processing_comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (export/'PROCESSING_REPORT.txt').write_text('\n'.join(report)+'\n')
    for root, result in zip(roots, results):
        subdirectory = export/root.name;subdirectory.mkdir(exist_ok=True)
        plots(subdirectory, root/f"held_out_{result['heldout_seeds'][0]}", result['selected'])
        save(root/'processing_exports.json', {'directory': str(export)})
    for relative in ('scripts/assess_processing_uniformity.py', 'scripts/finish_processing_assessment.py',
                     'scripts/test_processing_assessment.py'):
        target = export/'source_snapshot'/relative;target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, target)
    save(export/'manifest.json', {str(p.relative_to(export)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in export.rglob('*') if p.is_file() and p.name != 'manifest.json'})
    save(study/'processing_exports.json', {'directory': str(export)})
    print('PROCESSING_REPORT_COMPLETE', str(export), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    main(parser.parse_args())
