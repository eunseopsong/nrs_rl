"""Resume saved PPO, select in Isaac, and lock a candidate before new heldouts.

Code entry points stay in tasks/manager_based. Logs and immutable checkpoints
are study data. Never modifies the protected save or the GUI deployment.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from nrs_rl.tasks.manager_based.nrs_rl.mdp.agents.fixed_force_ppo import check_ppo_or_baseline, sha
from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT, TASK_ROOT
from nrs_rl.tasks.manager_based.nrs_rl.utils.removal_metrics import save
from nrs_rl.tasks.manager_based.nrs_rl.utils.source_snapshot import snapshot_sources
from nrs_rl.tasks.manager_based.nrs_rl.utils.assess_fixed_force_velocity import evaluate
from nrs_rl.tasks.manager_based.nrs_rl.mdp.evaluation.audit_ppo_training import audit

VARIANTS = [(3201, .98, 1.e-4), (3202, .995, 1.e-4), (3203, .998, 1.e-4)]
SELECTION_SEED = 3241
HELDOUT_SEEDS = [3251, 3252]


def read(path):
    return json.loads(Path(path).read_text())


def normalize_plan(plan=None):
    """Lock independent seed groups and per-run settings before any learning."""
    if plan is None:
        plan = {'variants': [{'seed': s, 'gae_lambda': g, 'learning_rate': lr}
                             for s, g, lr in VARIANTS],
                'selection_seed': SELECTION_SEED, 'heldout_seeds': HELDOUT_SEEDS}
    defaults = {'updates': 192, 'eval_interval': 16, 'num_envs': 64,
                'rollout_steps': 512, 'entropy_coefficient': .0001}
    result = {**plan, 'variants': [{**defaults, **run} for run in plan['variants']]}
    seeds = [run['seed'] for run in result['variants']]
    validation_seeds = [result['selection_seed'], *result['heldout_seeds']]
    if not seeds or len(result['heldout_seeds']) < 2:
        raise ValueError('Need training candidates and at least two independent heldouts')
    if any(type(seed) is not int or seed <= 0 for seed in seeds + validation_seeds):
        raise ValueError('Seeds must be positive integers')
    if len(set(seeds + validation_seeds)) != len(seeds + validation_seeds):
        raise ValueError('Training, selection and heldout seeds must be distinct')
    for run in result['variants']:
        for key in ['seed', 'updates', 'eval_interval', 'num_envs', 'rollout_steps']:
            if type(run[key]) is not int or run[key] <= 0:
                raise ValueError('Invalid positive integer: ' + key)
        if not 0. < run['gae_lambda'] <= 1. or not 0. < run['learning_rate'] < .01:
            raise ValueError('Invalid GAE lambda or learning rate')
        if not math.isfinite(run['entropy_coefficient']) or run['entropy_coefficient'] < 0.:
            raise ValueError('Invalid entropy coefficient')
    return result


def preserve(root):
    protocol = read(root / 'protocol.json')
    record = read(protocol['protected_save_record'])
    assert record['checkpoint_sha256'] == protocol['protected_actor_sha256'] == sha(record['checkpoint'])
    assert record['training_state_sha256'] == sha(record['training_state'])
    if 'protected_training_state_sha256' in protocol:
        assert record['training_state_sha256'] == protocol['protected_training_state_sha256']
        assert sha(protocol['protected_save_record']) == protocol['protected_save_record_sha256']
    assert sha(ROOT / 'deployment/FIXED_MODEL.json') == protocol['gui_manifest_sha256']
    locked = read(ROOT / 'deployment/FIXED_MODEL.json')
    assert sha(locked['checkpoint']) == locked['checkpoint_sha256']
    return record


def initialize(root, plan=None):
    if (root / 'protocol.json').exists():
        if plan is not None:
            stored = normalize_plan(read(root / 'protocol.json'))
            requested = normalize_plan(plan)
            for key in ['variants', 'selection_seed', 'heldout_seeds']:
                assert stored[key] == requested[key], 'Cannot change an initialized study: ' + key
        return preserve(root)
    plan = normalize_plan(plan)
    root.mkdir(parents=True, exist_ok=False)
    record_file = ROOT / 'logs/experiments/uniformity_saves/PPO_FIXED.json'
    record = read(record_file)
    assert sha(record['checkpoint']) == record['checkpoint_sha256']
    assert record['algorithm'] == 'PPO'
    # Pin the immutable per-save record, not a mutable registry pointer.
    immutable_record = Path(record['save_directory']) / 'SAVE.json'
    assert read(immutable_record) == record
    assert sha(record['training_state']) == record['training_state_sha256']
    save(root / 'protocol.json', {
        'algorithm': 'PPO', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'protected_save_record': str(immutable_record), 'protected_actor_sha256': record['checkpoint_sha256'],
        'protected_save_record_sha256': sha(immutable_record),
        'protected_training_state_sha256': record['training_state_sha256'],
        'gui_manifest_sha256': sha(ROOT / 'deployment/FIXED_MODEL.json'),
        'variants': plan['variants'],
        'selection_seed': plan['selection_seed'], 'heldout_seeds': plan['heldout_seeds'],
        'selection_rule': 'highest eligible CV improvement versus paired Mode 6',
        'acceptance_rule': 'each heldout: CV improvement versus Mode 6 exceeds saved PPO by max(0.5 percentage point, 2x Mode 6 repeat variation)',
        'all_volume_scopes_floor_vs_mode6': .5,
        'force_target_n': 20., 'virtual_spindle_rpm': 1000.,
        'entry_seconds': 2., 'entry_reference_footprint_mm': 18.,
        'training_environment': 'approximate rotary-removal / digital feed response',
        'validation_environment': 'independent Isaac robot and contact dynamics',
        'automatic_save_replacement': False, 'gui_deployment': False,
    })
    return preserve(root)


def train(root, plan=None):
    record = initialize(root, plan)
    protocol = read(root / 'protocol.json')
    variants = normalize_plan(protocol)['variants']
    save(root / 'status.json', {'phase': 'continued_ppo_training', 'saved_actor_sha256': record['checkpoint_sha256']})

    def one(run):
        seed, gae, lr = run['seed'], run['gae_lambda'], run['learning_rate']
        output = root / f'training_seed{seed}'
        if (output / 'training_complete.json').exists():
            return read(output / 'training_complete.json')
        if output.exists():
            raise RuntimeError('Incomplete run retained, use a new study: ' + str(output))
        command = [sys.executable, '-m', 'nrs_rl.tasks.manager_based.nrs_rl.mdp.agents.train_fixed_force_ppo',
                   '--output', str(output), '--resume', record['training_state'],
                   '--seed', str(seed), '--updates', str(run['updates']), '--eval-interval', str(run['eval_interval']),
                   '--gae-lambda', str(gae), '--learning-rate', str(lr),
                   '--num-envs', str(run['num_envs']), '--rollout-steps', str(run['rollout_steps']),
                   '--entropy-coefficient', str(run['entropy_coefficient']), '--device', 'cpu',
                   '--isaac-selection-seed', str(protocol['selection_seed']),
                   '--isaac-holdout-seeds', *map(str, protocol['heldout_seeds'])]
        save(root / f'training_seed{seed}_command.json', command)
        with (root / f'training_seed{seed}.log').open('w') as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True,
                           env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
        result = read(output / 'training_complete.json')
        print('TRAINING_COMPLETE', seed, json.dumps(result), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=len(variants)) as pool:
        futures = [pool.submit(one, run) for run in variants]
        results = [future.result() for future in futures]
    proof = audit(root, seeds=tuple(run['seed'] for run in variants))
    save(root / 'ppo_training_audit.json', proof)
    preserve(root)
    save(root / 'status.json', {'phase': 'training_complete', 'runs': results, 'training_audit_passed': proof['passed']})


def batch(root, training, label, names, seed):
    output = root / 'physics' / label
    command = [sys.executable, str(TASK_ROOT / 'mdp/evaluation/probe_fixed_force_ppo.py'),
               '--headless', '--device', 'cpu', '--training-dir', str(training),
               '--output', str(output), '--names', ','.join(names), '--seed', str(seed),
               '--max-steps', '42000', '--hard-force-stop-n', '100', '--uniformity-only']
    save(root / 'physics' / f'{label}_command.json', command)
    if not (output / 'summary.json').exists():
        if output.exists():
            raise RuntimeError('Incomplete Isaac rollout retained: ' + str(output))
        with (root / 'physics' / f'{label}.log').open('w') as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True,
                           env={**os.environ, 'MPLCONFIGDIR': '/tmp/nrs_ppo_refinement_mpl',
                                'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    summary = read(output / 'summary.json')
    assert summary['seed'] == seed and summary['names'] == names
    for values in summary['randomization'].values():
        assert all(v == values[0] for v in values)
    for name in names:
        source = 'constant' if name == 'constant_b' else name
        assert sha(training / 'checkpoints' / f'{source}.pt') == summary['checkpoint_sha256'][name]
    assert summary['checkpoint_sha256']['ppo_saved'] == read(root / 'protocol.json')['protected_actor_sha256']
    report = evaluate(output, audit=True)
    print('ISAAC_BATCH', label, json.dumps({k: {
        'eligible': v['eligible'], 'cv_improvement_percent_vs_mode6': 100.*v['mode6_cv_gain'],
        'volume_retention_percent_vs_mode6': {a: 100.*b for a, b in v['mode6_volume_ratios'].items()}}
        for k, v in report['candidates'].items()}), flush=True)
    return report


def physics(root):
    torch.set_num_threads(1)
    record = preserve(root)
    protocol = read(root / 'protocol.json')
    selection_seed, heldout_seeds = protocol['selection_seed'], protocol['heldout_seeds']
    assert read(root / 'ppo_training_audit.json')['passed']
    if not (root / 'package_source_snapshot').exists():
        snapshot_sources(root)
    training = root / 'physics_candidates'
    (training / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (root / 'physics').mkdir(exist_ok=True)
    # Identical Mode 6 export to the previous study, no new profile fitting.
    constant = ROOT / 'logs/experiments/ppo_fixed_force_20261009/physics_candidates/checkpoints/constant.pt'
    shutil.copy2(constant, training / 'checkpoints/constant.pt')
    shutil.copy2(record['checkpoint'], training / 'checkpoints/ppo_saved.pt')
    candidates = {}
    for run in protocol['variants']:
        seed = run['seed']
        selected = read(root / f'training_seed{seed}/selection_surrogate.json')
        name = f'ppo_seed{seed}'
        candidates[name] = selected
        shutil.copy2(selected['actor'], training / 'checkpoints' / f'{name}.pt')
    save(root / 'status.json', {'phase': 'isaac_selection', 'seed': selection_seed})
    selection = batch(root, training, f'selection_{selection_seed}',
                      ['constant', 'ppo_saved', *candidates], selection_seed)
    eligible = [(selection['candidates'][name]['mode6_cv_gain'], name) for name in candidates
                if selection['candidates'][name]['eligible']]
    if not eligible:
        raise RuntimeError('No eligible PPO candidate; protected save retained')
    gain, name = max(eligible)
    checkpoint = training / 'checkpoints/selected.pt'
    source = training / 'checkpoints' / f'{name}.pt'
    if checkpoint.exists():
        assert sha(checkpoint) == sha(source)
    else:
        shutil.copy2(source, checkpoint)
    selection_file = root / 'selection.json'
    timestamp = (read(selection_file)['selected_before_holdouts_at'] if selection_file.exists()
                 else datetime.now(timezone.utc).isoformat())
    picked = {'selected': name, 'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
              'selected_before_holdouts_at': timestamp, 'selection_report': selection,
              'training_selection': candidates[name], 'selection_cv_improvement_percent_vs_mode6': 100.*gain}
    save(selection_file, picked)
    save(root / 'status.json', {'phase': 'independent_isaac_holdouts', **picked})
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {seed: pool.submit(batch, root, training, f'heldout_{seed}',
                                    ['constant', 'constant_b', 'ppo_saved', 'selected'], seed)
                   for seed in heldout_seeds}
        heldouts = {str(seed): future.result() for seed, future in futures.items()}
    checks = []
    for seed, report in heldouts.items():
        metrics = report['candidates']
        noise = abs(metrics['constant_b']['mode6_cv_gain'])
        threshold = max(.005, 2.*noise)
        difference = metrics['selected']['mode6_cv_gain'] - metrics['ppo_saved']['mode6_cv_gain']
        passed = metrics['selected']['eligible'] and metrics['ppo_saved']['eligible'] and difference >= threshold
        report['acceptance'] = {'passed': bool(passed),
            'additional_mode6_normalized_cv_gain': difference, 'minimum_additional_mode6_normalized_gain': threshold,
            'mode6_repeat_cv_variation': noise, 'practical_threshold_not_statistical_significance': True}
        checks.append(passed)
    preserve(root)
    assert sha(checkpoint) == picked['checkpoint_sha256']
    result = {'phase': 'complete', 'accepted': bool(all(checks)), 'candidate_is_ppo': True,
              'algorithm': 'PPO', **picked, 'heldouts': heldouts, 'protected_save_preserved': True,
              'gui_trial_executed': False, 'gui_deployment_changed': False}
    save(root / 'result.json', result)
    save(root / 'status.json', result)
    package(root)
    print('REFINEMENT_COMPLETE', json.dumps({'accepted': result['accepted'], 'selected': name,
        'heldout_cv_gains_vs_mode6_percent': {k: 100.*v['candidates']['selected']['mode6_cv_gain'] for k, v in heldouts.items()}}), flush=True)


def package(root):
    result = read(root / 'result.json')
    source = Path(result['checkpoint'])
    candidate = root / 'candidate'
    candidate.mkdir(exist_ok=True)
    (candidate / 'checkpoints').mkdir(exist_ok=True)
    shutil.copy2(source, candidate / 'checkpoints/actor.pt')
    training = root / result['selected'].replace('ppo_', 'training_')
    update = result['training_selection']['update']
    shutil.copy2(training / f'checkpoints/training_{update:04d}.pt', candidate / 'checkpoints/ppo_training_state.pt')
    extra = {'fixed_force_policy.json': ''}
    actor = torch.jit.load(str(source), _extra_files=extra)
    contract = json.loads(extra['fixed_force_policy.json'])
    check_ppo_or_baseline(contract)
    save(candidate / 'fixed_force_policy.json', contract)
    means = {}
    for name in ['constant', 'ppo_saved', 'selected']:
        rows = [r['candidates'][name] for r in result['heldouts'].values()]
        means[name] = {'mode6_cv_improvement_percent': float(np.mean([100.*r['mode6_cv_gain'] for r in rows])),
            'spatial_cv': float(np.mean([r['spatial_cv'] for r in rows])),
            'mode6_processing_volume_retention_percent': float(np.mean([100.*r['mode6_volume_ratios']['roi_volume_over_k'] for r in rows])),
            'mode6_full_volume_retention_percent': float(np.mean([100.*r['mode6_volume_ratios']['full_roi_volume_over_k'] for r in rows])),
            'mode6_whole_grid_volume_retention_percent': float(np.mean([100.*r['mode6_volume_ratios']['whole_grid_volume_over_k'] for r in rows])),
            'time_s': float(np.mean([r['time_s'] for r in rows]))}
    report = {'accepted': result['accepted'], 'algorithm': 'PPO', 'selected': result['selected'],
        'checkpoint': str(candidate / 'checkpoints/actor.pt'), 'checkpoint_sha256': sha(source),
        'ppo_training_state': str(candidate / 'checkpoints/ppo_training_state.pt'),
        'mean_heldout_metrics': means['selected'], 'same_batch_comparison': means,
        'per_seed': {s: r['candidates'] for s, r in result['heldouts'].items()},
        'all_comparisons_relative_to_matched_mode6': True,
        'entry_seconds': 2., 'entry_reference_footprint_mm': 18., 'force_target_n': 20.,
        'virtual_spindle_rpm': 1000., 'action_dimension': 1, 'protected_save_replaced': False,
        'training_environment': 'rotary-removal and digital feed-response simulator, not Isaac',
        'validation_environment': 'two independent full-path Isaac robot/contact batches',
        'gui_trial_executed': False, 'numerical_audit_passed': False,
        'limitations': ['Only this reference path; two heldout seeds do not establish statistical significance.',
                       'Depth and volume are uncalibrated h/K and V/K using assumed 1000 RPM.']}
    save(root / 'final_report.json', report)
    save(candidate / 'validation.json', report)
    save(candidate / 'manifest.json', {str(p.relative_to(candidate)): sha(p) for p in candidate.rglob('*')
                                       if p.is_file() and p.name != 'manifest.json'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=['train', 'physics', 'all'], default='all')
    parser.add_argument('--wait-training', action='store_true')
    parser.add_argument('--plan', type=Path, help='JSON training variants and independent evaluation seeds')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase in ['train', 'all']:
        train(root, read(args.plan) if args.plan else None)
    if args.phase in ['physics', 'all']:
        if args.wait_training:
            deadline = time.monotonic() + 1800.
            while not (root / 'ppo_training_audit.json').exists():
                if time.monotonic() > deadline:
                    raise RuntimeError('Training readiness deadline exceeded')
                time.sleep(2.)
        physics(root)
