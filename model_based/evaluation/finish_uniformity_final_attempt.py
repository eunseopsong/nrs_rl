#!/usr/bin/env python3
"""Validate one final search, then fix either the new policy or the saved one."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.spatial_uniformity import ROOT
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_uniformity_priority import save, sha
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_uniformity_final_attempt import (
    SAVED, SAVED_SHA, NEW, SELECTION_SEED, HOLDOUTS, MINIMUM_GAIN, MINIMUM_ADVANTAGE, check_preserved,
)
from nrs_rl.tasks.manager_based.nrs_rl.model_based.evaluation.finish_uniformity_volume50 import run_constrained, figures
from nrs_rl.tasks.manager_based.nrs_rl.model_based.evaluation.finish_uniformity_refinement import audit_batch
from nrs_rl.tasks.manager_based.nrs_rl.model_based.evaluation.uniformity_volume_guard import assess_volume_guard

SELECTION_GROUPS = {'selection_a_2501': ['constant'] + NEW[:3],
                    'selection_b_2501': ['constant'] + NEW[3:]}


def final_guard(candidate, mode6, saved):
    guard = assess_volume_guard(candidate, mode6, saved, minimum_gain=MINIMUM_GAIN)
    gains = [guard['mode6_spatial_cv_gain'], guard['saved_mode6_spatial_cv_gain']]
    guard['checks']['paired_margin_exceeds_noise_allowance'] = bool(
        all(v is not None for v in gains) and gains[0] - gains[1] >= MINIMUM_ADVANTAGE)
    guard['minimum_mode6_normalized_advantage'] = MINIMUM_ADVANTAGE
    guard['threshold_kind'] = 'predeclared practical replacement threshold, not a statistical significance test'
    guard['passed'] = all(guard['checks'].values())
    return guard


def manifest(directory, name='manifest.json'):
    data = {str(p.relative_to(directory)): sha(p) for p in directory.rglob('*') if p.is_file() and p.name != name}
    save(directory / name, data)
    for rel, digest in data.items():
        assert sha(directory / rel) == digest


def fix_model(root, result):
    """Pin an immutable checkpoint only after the entire attempt is audited."""
    assert result['phase'] == 'complete'
    assert json.loads((root / 'verification.json').read_text())['passed']
    pointer = SAVED.parent / 'CURRENT.json'
    assert json.loads(pointer.read_text())['sha256'] == SAVED_SHA, 'Current model changed during validation'
    check_preserved()
    accepted = result['accepted']
    target = SAVED
    if accepted:
        assert all(item['passed'] for item in result['heldouts'])
        maximum = max(item['guard']['mode6_spatial_cv_gain'] for item in result['heldouts']) * 100.
        stamp = datetime.now().strftime('%Y%m%d_%H%M')
        target = SAVED.parent / f'mode6_{maximum:.2f}_volume50_fixed_{stamp}'.replace('.', 'p')
        target.mkdir(exist_ok=False)
        shutil.copytree(root / 'candidate', target / 'candidate')
        for name in ['result.json', 'verification.json', 'predeclared_protocol.json', 'REPORT.txt']:
            shutil.copyfile(root / name, target / name)
        save(target / 'SAVE.json', {'source': str(root), 'sha256': result['checkpoint_sha256'],
            'reason': 'Final-attempt replacement threshold passed on both independent runs',
            'previous_save': str(SAVED), 'minimum_mode6_volume_ratio': .5,
            'created_at_utc': datetime.now(timezone.utc).isoformat()})
        manifest(target, 'save_manifest.json')
    checkpoint = target / 'candidate/checkpoints/actor.pt'
    digest = sha(checkpoint)
    assert digest == (result['checkpoint_sha256'] if accepted else SAVED_SHA)
    decision = {'status': 'fixed', 'save_directory': str(target), 'checkpoint': str(checkpoint),
        'sha256': digest, 'fixed_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_attempt': str(root / 'result.json'), 'comparison_baseline': 'Mode 6 only',
        'minimum_mode6_volume_ratio': .5, 'replacement_mode6_cv_gain_threshold': MINIMUM_GAIN,
        'new_candidate_accepted': accepted, 'previous_save': str(SAVED),
        'reason': ('New candidate passed the predeclared substantial-improvement threshold' if accepted else
                   'Further tuning did not pass the replacement threshold; user-requested 13.45% save fixed'),
        'contract': str(target / 'candidate/spatial_policy.json'),
        'validation': str(target / 'candidate/validation.json'),
        'purpose': 'Final selected model for subsequent integration; does not change ROS/GUI deployment',
        'deployed': False, 'current_GUI_runtime_compatible': False}
    save(root / 'fix_decision.json', decision)
    save(SAVED.parent / 'FIXED.json', decision)
    save(pointer, decision)
    return decision


def main(args):
    training = args.training_dir.resolve()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.monotonic()
    check_preserved()
    state = {'phase': 'physics_selection', 'selection_seed': SELECTION_SEED, 'selection_groups': SELECTION_GROUPS,
        'heldout_seeds': HOLDOUTS, 'minimum_mode6_volume_ratio_each_run': .5,
        'volume_constraint_scopes': ['processing_roi', 'full_roi'],
        'minimum_mode6_cv_gain_each_holdout': MINIMUM_GAIN,
        'minimum_mode6_normalized_advantage_over_save': MINIMUM_ADVANTAGE,
        'report_baseline': 'Mode 6 only', 'preserved_save': str(SAVED), 'preserved_sha256': SAVED_SHA,
        'training_dir': str(training), 'candidate_names': ['constant', 'saved'] + NEW,
        'deployed': False, 'new_gui_experiment_executed': False, 'holdouts_used_for_selection': False,
        'selection_rule': 'Best paired Mode 6 spatial CV gain among volume-feasible candidates; per-batch Mode 6',
        'acceptance_rule': 'Both heldouts: >=50% removal in both ROIs, >=14.45% CV gain vs Mode 6, paired margin >=0.005 Mode 6 CV',
        'threshold_kind': 'Practical replacement threshold chosen before this attempt, not statistical significance',
        'fallback': 'Fix preserved 13.45% checkpoint; stop tuning in this attempt', 'heldouts': [],
        'rejection_policy': 'Discard unselected/rejected experimental actors, retain logs and prior saves'}
    save(root / 'status.json', state)
    save(root / 'predeclared_protocol.json', state)
    sources = ['scripts/' + name for name in [
        'train_uniformity_final_attempt.py', 'finish_uniformity_final_attempt.py', 'test_uniformity_final_attempt.py',
        'train_uniformity_volume50.py', 'finish_uniformity_volume50.py', 'train_uniformity_volume50_refine.py',
        'train_uniformity_refinement.py', 'finish_uniformity_refinement.py', 'train_uniformity_priority.py',
        'finish_uniformity_priority.py', 'uniformity_volume_guard.py', 'uniformity_priority_metrics.py',
        'assess_processing_uniformity.py', 'probe_spatial_uniformity.py', 'spatial_uniformity.py',
        'spatial_uniformity_action.py', 'tcp_tracking_compensation.py']]
    sources += ['source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/mdp/action.py',
                'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/utils/preston_surface.py']
    hashes = {}
    for rel in sources:
        dest = root / 'source_snapshot' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dest)
        hashes[rel] = sha(dest)
    save(root / 'source_manifest.json', hashes)
    checkpoints = {name: sha(training / 'checkpoints' / f'{name}.pt') for name in state['candidate_names']}
    save(root / 'checkpoint_manifest.json', checkpoints)
    screens = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {pool.submit(run_constrained, training, root, label, SELECTION_SEED, names): label
                for label, names in SELECTION_GROUPS.items()}
        for job in as_completed(jobs):
            screens[jobs[job]] = job.result()
            print('FINAL_SELECTION_GROUP', jobs[job], flush=True)
    selection = {name: {'batch': label, 'metrics': m} for label, screen in screens.items()
                 for name, m in screen['candidates'].items() if name != 'constant'}
    save(root / 'selection_metrics.json', selection)
    feasible = [name for name, item in selection.items() if item['metrics']['volume_constraint']['passed']]
    eligible = [name for name, item in selection.items() if item['metrics']['eligible']]
    if not eligible:
        raise RuntimeError('No valid completed new candidate; inspect controller faults before finalizing')
    # Still validate the best completed candidate if all failed volume selection;
    # this cannot pass acceptance merely by averaging it with another run.
    selected = max(feasible or eligible,
                   key=lambda name: selection[name]['metrics']['volume_constraint']['mode6_spatial_cv_gain'])
    bundle = root / 'candidate'
    (bundle / 'checkpoints').mkdir(parents=True)
    shutil.copyfile(training / 'checkpoints' / f'{selected}.pt', bundle / 'checkpoints/actor.pt')
    extra = {'spatial_policy.json': ''}
    torch.jit.load(str(bundle / 'checkpoints/actor.pt'), _extra_files=extra)
    contract = json.loads(extra['spatial_policy.json'])
    save(bundle / 'spatial_policy.json', contract)
    state.update(phase='independent_physics_validation', selected=selected, checkpoint_sha256=checkpoints[selected],
        runtime_contract=contract, selected_before_holdouts_at=datetime.now(timezone.utc).isoformat(),
        selection_metrics=selection[selected], selection_volume_feasible=bool(feasible))
    save(bundle / 'selection.json', state)
    save(root / 'status.json', state)
    print('FINAL_SELECTED', selected, json.dumps(selection[selected]['metrics']['volume_constraint']), flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {pool.submit(run_constrained, training, root, f'held_out_{seed}', seed,
                            ['constant', 'saved', selected]): seed for seed in HOLDOUTS}
        for job in as_completed(jobs):
            metrics = job.result()['candidates']
            guard = final_guard(metrics[selected], metrics['constant'], metrics['saved'])
            state['heldouts'].append({'seed': jobs[job], 'candidate_metrics': metrics[selected],
                'preserved_metrics': metrics['saved'], 'mode6_metrics': metrics['constant'],
                'guard': guard, 'passed': guard['passed']})
            state['heldouts'].sort(key=lambda item: item['seed'])
            save(root / 'status.json', state)
            print('FINAL_HOLDOUT', jobs[job], json.dumps(guard), flush=True)
    accepted = bool(feasible) and all(item['passed'] for item in state['heldouts'])
    state.update(phase='artifact_verification', accepted=accepted)
    save(root / 'status.json', state)
    for name, digest in checkpoints.items():
        assert sha(training / 'checkpoints' / f'{name}.pt') == digest
    for rel, digest in hashes.items():
        assert sha(ROOT / rel) == digest, rel
    for file, digest in json.loads((training / 'source_data_manifest.json').read_text()).items():
        assert sha(file) == digest
    check_preserved()
    audits, full_checks = [], []
    labels = list(SELECTION_GROUPS) + [f'held_out_{seed}' for seed in HOLDOUTS]
    for label in labels:
        audits.extend(audit_batch(root / label))
        metrics = json.loads((root / label / 'processing_summary.json').read_text())['candidates']
        for name, values in metrics.items():
            with np.load(root / label / f'{name}_surface.npz') as data:
                volume = float(data['depth_over_k'][data['roi']].sum() * data['cell_area_mm2'])
                assert np.isclose(volume, values['full_roi_volume_over_k'], rtol=1.e-10)
            full_checks.append({'batch': label, 'policy': name, 'passed': True})
    save(root / 'verification.json', {'passed': True, 'raw_reintegration': audits, 'full_volume_checks': full_checks,
        'preserved_checkpoint_unchanged': True, 'checkpoint_source_and_training_data_hashes_unchanged': True})
    discarded = []
    for name in NEW:
        if accepted and name == selected:
            continue
        checkpoint = training / 'checkpoints' / f'{name}.pt'
        assert checkpoint.is_relative_to(training / 'checkpoints') and sha(checkpoint) == checkpoints[name]
        checkpoint.unlink()
        discarded.append({'candidate': name, 'sha256': checkpoints[name],
            'reason': 'not selected' if name != selected else 'did not meet final replacement threshold'})
    if not accepted:
        (bundle / 'checkpoints/actor.pt').unlink()
    save(root / 'discarded_checkpoints.json', discarded)
    export = ROOT / 'logs/polishing_results' / datetime.now().strftime('%Y%m%d_%H%M')
    export.mkdir(parents=True, exist_ok=False)
    state.update(phase='complete', accepted=accepted, elapsed_s=time.monotonic() - started,
        disposition='fix_new_model' if accepted else 'fix_preserved_13p45_model',
        retained_checkpoint=str(bundle / 'checkpoints/actor.pt' if accepted else SAVED / 'candidate/checkpoints/actor.pt'),
        discarded_checkpoints=discarded, export_directory=str(export))
    for dest in [root / 'result.json', root / 'status.json', export / 'result.json']:
        save(dest, state)
    rows = []
    for label in labels:
        summary = json.loads((root / label / 'processing_summary.json').read_text())
        save(export / f'{label}_summary.json', summary)
        baseline = summary['candidates']['constant']
        for name, m in summary['candidates'].items():
            rows.append({'batch': label, 'seed': summary['seed'],
                'role': 'selection' if label.startswith('selection') else 'holdout', 'policy': name,
                'spatial_cv': m['spatial_cv'], 'mode6_cv_gain_percent': 100. * (1. - m['spatial_cv'] / baseline['spatial_cv']),
                'mode6_processing_volume_change_percent': 100. * (m['roi_volume_over_k'] / baseline['roi_volume_over_k'] - 1.),
                'mode6_full_volume_change_percent': 100. * (m['full_roi_volume_over_k'] / baseline['full_roi_volume_over_k'] - 1.),
                'completed': m['completed'], 'fault_fraction': m['fault_fraction'],
                'volume_floor_passed': m['volume_constraint']['passed']})
    with (export / 'mode6_comparison.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figures(export, root, selected, accepted, holdouts=HOLDOUTS, selection_dirs=list(SELECTION_GROUPS),
        scatter_offsets={'recalibrated_515': (-90, -35), 'residual_515': (20, 35), 'slow_515': (-90, 15),
                         'flatfeed_515': (35, -35), 'physics_515': (20, -15), 'dense_515': (-90, -15)})
    shutil.copyfile(training / 'learned_profiles.png', export / '05_learned_profiles.png')
    validation = {'validation_status': 'passed_independent_headless_Isaac' if accepted else 'rejected_final_threshold',
        'checkpoint_sha256': checkpoints[selected], 'comparison_baseline': 'Mode 6 only',
        'minimum_mode6_volume_ratio': .5, 'minimum_mode6_cv_gain': MINIMUM_GAIN,
        'heldouts': [{'seed': item['seed'], **item['guard']} for item in state['heldouts']],
        'raw_reintegration_runs': len(audits), 'deployed': False, 'current_GUI_runtime_compatible': False,
        'training_method': 'model-based constrained profile/residual/feed optimization, not PPO',
        'source_result': str(root / 'result.json'),
        'embedded_metadata_status_note': 'Actor frozen before independent validation; this file records the outcome'}
    save(bundle / 'validation.json', validation)
    save(export / 'model_validation.json', validation)
    report = ['최종 추가 학습 및 모델 확정', '모든 수치의 비교 기준: 같은 실행 조건의 Mode 6',
        '보존 세이브본: ' + str(SAVED), '선정 후보: ' + selected, '새 후보 채택: ' + str(accepted),
        '학습: 이전 selection 2401의 요청 힘 응답 식별, 0.5 mm 격자, 잔차 보정, 힘/속도/체류시간 교대 최적화.',
        '후보 6개, 힘 프로파일 절점 193/289개, 학습 제거량 하한 51.5%. 물리 모델 기반 최적화이며 PPO가 아님.',
        '이전 heldout 2111/2112/2211/2212/2311/2312/2411/2412는 학습에서 제외.',
        '선정 seed 2501 후 모델 고정. 독립 검증 seed 2511/2512; 각 실행에 Mode 6 및 보존 모델 동시 비교.',
        '교체 기준: 검증 각각 Mode 6 대비 공간 CV 14.45% 이상 개선, 가공 구간/전체 제거량 모두 50% 이상 유지.',
        '동일 실행 보존 모델에 대한 추가 차이는 Mode 6 CV로 정규화해 0.005 이상이어야 함.',
        '기준 미달 시 기존 13.45% 모델 확정. 이는 사전 운용 기준이며 통계적 유의성 검정이 아님.',
        '비회전, 단일 전진 경로, 복귀 없음. 초기 2초 및 첫 18 mm 고정 footprint 제외 후 공간 CV 평가.',
        '100 N hard stop 유지, force slew 5 N/s. 초반 과도 응답은 별도 균일도 목적함수가 아님.',
        '가공량은 미교정 V/K 모델 추정치. GUI 실증/ROS 실행 설정 변경은 수행하지 않음.']
    for item in state['heldouts']:
        guard = item['guard']
        report.append(f"seed {item['seed']}: 새 후보 Mode 6 대비 CV 개선 {100. * guard['mode6_spatial_cv_gain']:.3f}%; "
            f"가공 구간 제거량 변화 {100. * (guard['mode6_volume_ratio'] - 1.):+.3f}%; "
            f"전체 제거량 변화 {100. * (guard['mode6_full_volume_ratio'] - 1.):+.3f}%; 교체 기준 통과 {item['passed']}")
        saved = item['preserved_metrics']
        baseline = item['mode6_metrics']
        report.append(f"seed {item['seed']}: 보존 모델 Mode 6 대비 CV 개선 {100. * guard['saved_mode6_spatial_cv_gain']:.3f}%; "
            f"가공 구간 제거량 변화 {100. * (saved['roi_volume_over_k'] / baseline['roi_volume_over_k'] - 1.):+.3f}%; "
            f"전체 제거량 변화 {100. * (saved['full_roi_volume_over_k'] / baseline['full_roi_volume_over_k'] - 1.):+.3f}%")
    report += ['원본 로그 재적분 검산: ' + str(len(audits)) + '개 실행 통과.',
               '확정 대상 체크포인트: ' + state['retained_checkpoint'], '상세 로그: ' + str(root)]
    for dest in [root / 'REPORT.txt', export / 'REPORT.txt']:
        dest.write_text('\n'.join(report) + '\n')
    manifest(bundle)
    decision = fix_model(root, state)
    for name in ['verification.json', 'discarded_checkpoints.json', 'fix_decision.json']:
        shutil.copyfile(root / name, export / name)
    manifest(export)
    save(root / 'exports.json', {'directory': str(export)})
    print('FINAL_ATTEMPT_COMPLETE', json.dumps({'new_candidate_accepted': accepted, 'fixed_model': decision,
                                              'export': str(export)}), flush=True)


if __name__ == '__main__':
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/nrs_uniformity_final_mpl')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
