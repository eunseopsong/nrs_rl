#!/usr/bin/env python3
"""Physics selection/holdouts for capped force plus tracking compensation."""
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
from pathlib import Path
import shutil
import time
import torch
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.spatial_uniformity import ROOT, assess
from nrs_rl.tasks.manager_based.nrs_rl.model_based.evaluation.finish_spatial_uniformity import run_batch, save, figures


def main(args):
    training, root = args.training_dir.resolve(), args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    gain = .6
    # Resolve the shared pretraining helper's default physics seed metadata
    # before any physics selection or holdout is launched.
    exports = {}
    for path in (training/'checkpoints').glob('*.pt'):
        extra = {'spatial_policy.json': ''}
        actor = torch.jit.load(str(path), _extra_files=extra)
        contract = json.loads(extra['spatial_policy.json'])
        contract['physics_training_seed'] = 1701
        contract['physics_holdout_seeds'] = [1801, 1802]
        contract['tracking_compensation']['gain'] = 0. if path.stem == 'constant' else gain
        torch.jit.save(actor, str(path), _extra_files={'spatial_policy.json': json.dumps(contract)})
        exports[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    extra = {'spatial_policy.json': ''}
    actor = torch.jit.load(str(training/'checkpoints/constant.pt'), _extra_files=extra)
    matched = json.loads(extra['spatial_policy.json']);matched['tracking_compensation']['gain'] = gain
    matched['ablation'] = 'tracking_compensation_only_constant_20N_6mm_s'
    torch.jit.save(actor, str(training/'checkpoints/matched_constant.pt'),
                   _extra_files={'spatial_policy.json': json.dumps(matched)})
    exports['matched_constant.pt'] = hashlib.sha256((training/'checkpoints/matched_constant.pt').read_bytes()).hexdigest()
    save(root/'prephysics_checkpoint_manifest.json', exports)
    protocol = json.loads((training/'protocol.json').read_text())
    protocol.update(physics_training_seed=1701, physics_holdout_seeds=[1801, 1802])
    save(training/'protocol.json', protocol)
    state = {'phase': 'physics_training_selection', 'selected': None, 'deployed': False,
             'training_seed': 1701, 'heldout_seeds': [1801, 1802],
             'validation_used_for_selection': False, 'minimum_heldout_spatial_gain': .03,
             'minimum_gain_over_matched_tracking': .02, 'tracking_gain': gain, 'heldouts': [],
             'refinement_basis': 'Original physics TRAINING seed 1501: entry/end coverage loss and larger endpoint lag.'}
    save(root/'status.json', state)
    files = ['scripts/spatial_uniformity.py', 'scripts/spatial_uniformity_action.py',
             'scripts/train_spatial_uniformity.py', 'scripts/train_spatial_refinement.py',
             'scripts/finish_spatial_uniformity.py', 'scripts/finish_spatial_refinement.py',
             'scripts/probe_spatial_uniformity.py', 'scripts/tcp_tracking_compensation.py',
             'scripts/test_spatial_uniformity.py',
             'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/mdp/action.py']
    manifest = {}
    for rel in files:
        dest = root/'source_snapshot'/rel;dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/rel, dest);manifest[rel] = hashlib.sha256(dest.read_bytes()).hexdigest()
    save(root/'source_manifest.json', manifest)
    names = ['constant', 'previous', 'matched_constant', 'learned', 'alternative_1']
    tracking = {name: (0. if name in ('constant', 'previous') else gain) for name in names}
    screen = run_batch(training, root, 'train_1701', 1701, names, tracking)
    ranking = {}
    for name in ('learned', 'alternative_1'):
        m = screen['candidates'][name]
        original = assess(m, screen['candidates']['constant'])
        matched_comparison = assess(m, screen['candidates']['matched_constant'])
        # Coverage gates refer to the original operating condition. Report
        # matched-controller coverage as well; do not count tracking gains
        # as if they came from the learned force/feed policy.
        ranking[name] = {'score': original['score'], 'original': original, 'matched': matched_comparison}
    selected = max(ranking, key=lambda name: ranking[name]['score'])
    checkpoint = training/'checkpoints'/f'{selected}.pt'
    bundle = root/'candidate';(bundle/'checkpoints').mkdir(parents=True)
    shutil.copyfile(checkpoint, bundle/'checkpoints/actor.pt')
    extra = {'spatial_policy.json': ''};torch.jit.load(str(checkpoint), _extra_files=extra)
    contract = json.loads(extra['spatial_policy.json']);save(bundle/'spatial_policy.json', contract)
    state.update(phase='independent_physics_validation', selected=selected, selection=ranking,
                 runtime_contract=contract, checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    save(root/'status.json', state)
    print('SPATIAL_REFINED_SELECTED', selected, json.dumps(ranking), flush=True)
    eval_names = ['constant', 'previous', 'matched_constant', selected]
    gains = {name: (0. if name in ('constant', 'previous') else gain) for name in eval_names}
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {pool.submit(run_batch, training, root, f'held_out_{seed}', seed, eval_names, gains): seed
                for seed in state['heldout_seeds']}
        for future in as_completed(jobs):
            data = future.result();m = data['candidates'][selected]
            original = assess(m, data['candidates']['constant'])
            matched_comparison = assess(m, data['candidates']['matched_constant'])
            item = {'seed': jobs[future], 'checks': [original], 'metrics': m,
                    'previous_policy_comparison': assess(m, data['candidates']['previous']),
                    'matched_tracking_comparison': matched_comparison,
                    'passed': original['passed'] and original['gains']['spatial_cv_gain'] >= .03
                              and matched_comparison['gains']['spatial_cv_gain'] >= .02}
            state['heldouts'].append(item);state['heldouts'].sort(key=lambda x: x['seed'])
            save(root/'status.json', state)
            print('SPATIAL_REFINED_HELDOUT', json.dumps(item), flush=True)
    passed = all(row['passed'] for row in state['heldouts'])
    state.update(phase='complete', validation_passed=passed, elapsed_s=time.monotonic()-started,
                 disposition='isolated_isaac_candidate_requires_runtime_integration' if passed else 'not_accepted_for_deployment')
    save(root/'result.json', state);save(root/'status.json', state)
    export = ROOT/'logs/polishing_results'/datetime.now().strftime('%Y%m%d_%H%M')
    export.mkdir(parents=True, exist_ok=False);save(export/'spatial_uniformity_result.json', state)
    rows = []
    for seed in state['heldout_seeds']:
        summary = json.loads((root/f'held_out_{seed}/summary.json').read_text())
        for name, m in summary['candidates'].items():
            rows.append({'seed': seed, 'policy': name, **{k: m[k] for k in ('spatial_cv', 'rate_cv',
                'roi_volume_over_k', 'time_s', 'force_mean_n', 'force_peak_n', 'force_tracking_rmse_n',
                'zero_fraction', 'distance_mm', 'return_events', 'reverse_mm', 'completed', 'fault_fraction')}})
    with (export/'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    figures(export, root/'held_out_1801', selected)
    report = ['비회전 가공량 균일도 재학습 및 독립 Isaac 검증', f'선정 정책: {selected}', f'검증 통과: {passed}',
              '수정: 목표 힘 상한 24 N, 제한된 접선 추종 보정 gain 0.6, 0.08 s 필터, 최대 4 mm.',
              '별도의 tracking-only 비교군으로 제어기 변경과 정책 효과를 구분한다.',
              '국소 복귀는 선택 가능. 정지 체류에 제거량을 부여하지 않는다. 별도의 전체 두 번째 패스 없음.']
    for row in state['heldouts']:
        g=row['checks'][0]['gains'];mg=row['matched_tracking_comparison']['gains']
        report.append(f"seed {row['seed']}: 공간 CV 개선 {100*g['spatial_cv_gain']:.3f}%, "
                      f"tracking-only 대비 {100*mg['spatial_cv_gain']:.3f}%, 제거량 {100*(g['volume_ratio']-1):+.3f}%, "
                      f"시간 {100*(g['time_ratio']-1):+.3f}%, 순간 CV {100*(g['raw_cv_ratio']-1):+.3f}%, "
                      f"복귀 횟수 {row['metrics']['return_events']}, 통과 {row['passed']}")
    report += ['학습: 비회전 제거량 대리 모델에서 CEM 정책 파라미터 탐색, 이후 별도 Isaac 학습 시드 1701로 후보 선택.',
               '검증 시드 1801/1802는 후보 선택에 사용하지 않았다. 동일 flat 경로에서 검증한 결과이다.',
               '깊이와 제거량은 교정되지 않은 h/K, V/K. 실제 재료 제거 실증 결과가 아니다.',
               '기준 cursor 도달이 완료 조건이며 실제 끝점 정착은 별도 검증 필요.',
               '새 16D/3-action ckpt는 기존 ROS Mode 5에 그대로 연동되지 않는다. ROS 배포는 하지 않았다.',
               '체크포인트: '+str(bundle/'checkpoints/actor.pt'), 'SHA256: '+state['checkpoint_sha256']]
    (export/'REPORT.txt').write_text('\n'.join(report)+'\n');shutil.copyfile(export/'REPORT.txt', root/'REPORT.txt')
    save(root/'exports.json', {'directory': str(export)})
    for directory in (export, bundle):
        save(directory/'manifest.json', {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in directory.rglob('*') if p.is_file() and p.name != 'manifest.json'})
    print('SPATIAL_REFINED_COMPLETE', json.dumps({'passed': passed, 'export': str(export)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
