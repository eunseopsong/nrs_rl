#!/usr/bin/env python3
"""Select on one Isaac seed, freeze, then evaluate two independent seeds."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.spatial_uniformity import ROOT, SpatialUniformityActor, assess


def save(path, obj):
    temporary = path.with_suffix('.next.json')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temporary.replace(path)


def run_batch(training, root, name, seed, names, tracking=None):
    command = [sys.executable, str(ROOT/'scripts/probe_spatial_uniformity.py'), '--headless', '--device', 'cpu',
               '--training-dir', str(training), '--output', str(root/name), '--seed', str(seed),
               '--names', ','.join(names)]
    if tracking is not None:
        path = root/f'{name}_tracking.json'
        save(path, tracking)
        command += ['--tracking-parameters', str(path)]
    save(root/f'{name}_command.json', command)
    with (root/f'{name}.log').open('w') as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True,
                       env={**os.environ, 'MPLCONFIGDIR': '/tmp/nrs_spatial_uniformity_mpl', 'MPLBACKEND': 'Agg'})
    return json.loads((root/name/'summary.json').read_text())


def figures(export, evaluation, selected):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/nrs_spatial_uniformity_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['constant', 'previous', selected]
    arrays = [np.load(evaluation/f'{name}_surface.npz') for name in names]
    vmax = max(np.quantile(a['depth_over_k'][a['roi']], .995) for a in arrays)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, name, a in zip(axes, names, arrays):
        data = np.where(a['roi'], a['depth_over_k'], np.nan)
        im = ax.imshow(data, origin='lower', extent=[a['u_mm'][0], a['u_mm'][-1], a['v_mm'][0], a['v_mm'][-1]],
                       vmin=0., vmax=vmax, cmap='viridis')
        values = a['depth_over_k'][a['roi']]
        ax.set_title(f'{name}\nspatial CV={values.std()/values.mean():.4f}')
        ax.set_xlabel('Plane u (mm)');ax.set_ylabel('Plane v (mm)')
    fig.colorbar(im, ax=axes, label='Depth / K (uncalibrated)')
    fig.savefig(export/'01_spatial_depth.png', dpi=160);plt.close(fig)
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True, constrained_layout=True)
    for name in names:
        with np.load(evaluation/f'{name}_trace.npz') as t:
            time_s = t['time_s']
            axes[0].plot(time_s, t['normal_force_n'], lw=.5, label=name)
            axes[1].plot(time_s, t['measured_speed_mm_s'], lw=.4)
            axes[2].plot(time_s, t['raw_mrr_n_mm_s'], lw=.4)
            axes[3].plot(time_s, t['cursor_mm'], lw=1.)
    for ax, label in zip(axes, ['Contact force (N)', 'Actual speed (mm/s)', 'Removal rate / K (N mm/s)', 'Path cursor (mm)']):
        ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0].legend();axes[-1].set_xlabel('Active process time (s)')
    fig.savefig(export/'02_process_signals.png', dpi=160);plt.close(fig)
    with np.load(evaluation/f'{selected}_residence.npz') as t:
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        im = ax.imshow(np.where(t['roi'], t['residence_s'], np.nan), origin='lower',
                       extent=[t['u_mm'][0], t['u_mm'][-1], t['v_mm'][0], t['v_mm'][-1]], cmap='magma')
        ax.set_title('Cumulative contact residence; motion and pauses included')
        ax.set_xlabel('Plane u (mm)');ax.set_ylabel('Plane v (mm)')
        fig.colorbar(im, ax=ax, label='Time (s)')
        fig.savefig(export/'03_contact_residence.png', dpi=160);plt.close(fig)


def main(args):
    training, root = args.training_dir.resolve(), args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    candidates = json.loads((training/'candidates.json').read_text())
    force_only = list(candidates['learned'])
    force_only[3:8] = [0.]*5
    extra = {'spatial_policy.json': ''}
    torch.jit.load(str(training/'checkpoints/learned.pt'), _extra_files=extra)
    contract = json.loads(extra['spatial_policy.json'])
    contract['parameters'] = force_only
    contract['ablation'] = 'learned_force_with_constant_feed_and_no_return'
    torch.jit.save(torch.jit.script(SpatialUniformityActor(force_only)), str(training/'checkpoints/force_only.pt'),
                   _extra_files={'spatial_policy.json': json.dumps(contract)})
    candidates['force_only'] = force_only
    save(root/'candidate_parameters.json', candidates)
    state = {'phase': 'physics_training_selection', 'selected': None, 'deployed': False,
             'algorithm': 'surrogate_CEM_then_physics_candidate_selection',
             'training_seed': 1501, 'heldout_seeds': [1601, 1602],
             'validation_used_for_selection': False, 'minimum_heldout_spatial_gain': .03,
             'runtime_contract': contract, 'heldouts': []}
    save(root/'status.json', state)
    source_files = ['scripts/spatial_uniformity.py', 'scripts/spatial_uniformity_action.py',
                    'scripts/probe_spatial_uniformity.py', 'scripts/finish_spatial_uniformity.py',
                    'scripts/test_spatial_uniformity.py',
                    'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/mdp/action.py']
    manifest = {}
    for rel in source_files:
        target = root/'source_snapshot'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/rel, target)
        manifest[rel] = hashlib.sha256(target.read_bytes()).hexdigest()
    save(root/'source_manifest.json', manifest)
    names = ['constant', 'constant_b', 'previous', 'learned', 'force_only', 'alternative_1', 'alternative_2']
    screen = run_batch(training, root, 'train_1501', 1501, names)
    learned_names = ['learned', 'force_only', 'alternative_1', 'alternative_2']
    ranking = {}
    for name in learned_names:
        metrics = screen['candidates'][name]
        checks = [assess(metrics, screen['candidates'][b]) for b in ('constant', 'constant_b')]
        ranking[name] = {'score': min(c['score'] for c in checks), 'checks': checks}
    selected = max(ranking, key=lambda n: ranking[n]['score'])
    selected_file = training/'checkpoints'/f'{selected}.pt'
    # Freeze before opening either holdout. Keep original candidate files.
    bundle = root/'candidate'
    (bundle/'checkpoints').mkdir(parents=True)
    shutil.copyfile(selected_file, bundle/'checkpoints/actor.pt')
    extra = {'spatial_policy.json': ''}
    torch.jit.load(str(selected_file), _extra_files=extra)
    save(bundle/'spatial_policy.json', json.loads(extra['spatial_policy.json']))
    state.update(phase='independent_physics_validation', selected=selected, selection=ranking,
                 runtime_contract=json.loads(extra['spatial_policy.json']),
                 checkpoint_sha256=hashlib.sha256(selected_file.read_bytes()).hexdigest())
    save(root/'status.json', state)
    print('SPATIAL_PHYSICS_SELECTED', selected, json.dumps(ranking), flush=True)
    no_return = list(candidates[selected]);no_return[7] = 0.
    no_return_contract = json.loads(extra['spatial_policy.json'])
    no_return_contract['parameters'] = no_return
    no_return_contract['ablation'] = 'selected_policy_without_local_return'
    torch.jit.save(torch.jit.script(SpatialUniformityActor(no_return)), str(training/'checkpoints/selected_no_return.pt'),
                   _extra_files={'spatial_policy.json': json.dumps(no_return_contract)})
    eval_names = ['constant', 'constant_b', 'previous', selected, 'selected_no_return']
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {pool.submit(run_batch, training, root, f'held_out_{seed}', seed, eval_names): seed for seed in state['heldout_seeds']}
        for future in as_completed(jobs):
            data = future.result()
            m = data['candidates'][selected]
            checks = [assess(m, data['candidates'][b]) for b in ('constant', 'constant_b')]
            result = {'seed': jobs[future], 'checks': checks,
                      'previous_policy_comparison': assess(m, data['candidates']['previous']),
                      'metrics': m, 'no_return_metrics': data['candidates']['selected_no_return'],
                      'passed': all(c['passed'] and c['gains']['spatial_cv_gain'] >= .03 for c in checks)}
            state['heldouts'].append(result)
            state['heldouts'].sort(key=lambda x: x['seed'])
            save(root/'status.json', state)
            print('SPATIAL_HELDOUT', json.dumps(result), flush=True)
    passed = all(x['passed'] for x in state['heldouts'])
    state.update(phase='complete', validation_passed=passed, elapsed_s=time.monotonic()-started,
                 disposition='isolated_isaac_candidate_requires_runtime_integration' if passed else 'not_accepted_for_deployment')
    save(root/'result.json', state);save(root/'status.json', state)
    export = ROOT/'logs/polishing_results'/datetime.now().strftime('%Y%m%d_%H%M')
    export.mkdir(parents=True, exist_ok=False)
    save(export/'spatial_uniformity_result.json', state)
    rows = []
    for seed in state['heldout_seeds']:
        summary = json.loads((root/f'held_out_{seed}/summary.json').read_text())
        for name, m in summary['candidates'].items():
            row = {'seed': seed, 'policy': name}
            row.update({k: m[k] for k in ('spatial_cv', 'rate_cv', 'roi_volume_over_k', 'time_s', 'force_mean_n',
                'force_peak_n', 'force_tracking_rmse_n', 'distance_mm', 'return_events', 'reverse_mm', 'completed', 'fault_fraction')})
            rows.append(row)
    with (export/'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    figures(export, root/'held_out_1601', selected)
    gain = [x['checks'][0]['gains']['spatial_cv_gain']*100 for x in state['heldouts']]
    volume = [(x['checks'][0]['gains']['volume_ratio']-1)*100 for x in state['heldouts']]
    duration = [(x['checks'][0]['gains']['time_ratio']-1)*100 for x in state['heldouts']]
    report = [
        '비회전 가공량 균일도 학습 및 독립 Isaac 검증',
        f'선정 정책: {selected}', f'독립 검증 통과: {passed}',
        f'공간 CV 개선율: {min(gain):.3f} ~ {max(gain):.3f}%',
        f'ROI 제거량 변화: {min(volume):+.3f} ~ {max(volume):+.3f}%',
        f'가공 시간 변화: {min(duration):+.3f} ~ {max(duration):+.3f}%',
        '각 수치는 동일 MDK/20 N/6 mm/s 기준과 비교. 순간 제거율 CV는 comparison.csv에 별도 기록.',
        '학습: 비회전 Preston/힘 지연 대리 모델의 CEM 정책 파라미터 탐색, 이후 Isaac 학습 시드로 후보 선택.',
        '검증: 별도 Isaac 시드 1601/1602. 검증 결과로 후보를 다시 고르지 않음.',
        '속도/접촉력/국소 복귀 action을 지원하며 체류시간을 누적 관측. 정지 체류에 추가 제거 보상 없음.',
        '국소 복귀 사용 횟수: '+str([x['metrics']['return_events'] for x in state['heldouts']]),
        '단위는 교정되지 않은 h/K 및 V/K. 실제 재료 제거 실험 결과가 아님.',
        '경로 완료 판정은 기준 cursor. 로봇의 최종점 정착은 별도 검증 필요.',
        'ROS 배포 없음. 기존 14D/1-action Mode 5에 새 16D/3-action ckpt를 그대로 넣으면 안 됨.',
        '체크포인트: '+str(bundle/'checkpoints/actor.pt'),
        'SHA256: '+state['checkpoint_sha256'],
        '상세 로그: '+str(root),
    ]
    (export/'REPORT.txt').write_text('\n'.join(report)+'\n')
    shutil.copyfile(export/'REPORT.txt', root/'REPORT.txt')
    save(root/'exports.json', {'directory': str(export)})
    for directory in (export, bundle):
        hashes = {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in directory.rglob('*') if p.is_file() and p.name != 'manifest.json'}
        save(directory/'manifest.json', hashes)
    print('SPATIAL_COMPLETE', json.dumps({'passed': passed, 'export': str(export)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args)
