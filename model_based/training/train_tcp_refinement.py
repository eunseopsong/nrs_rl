#!/usr/bin/env python3
"""RPM-free refinement against the frozen Oct-06 controller/policy.

Screen structured MDK/feed candidates on a path prefix, update a bounded CEM
population, rank it on two full-path training conditions, then freeze one
checkpoint before three fresh full-path validation conditions. No robot starts.
"""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_tcp_control import ROOT, BASE, OLD, REFERENCE, GATES, assess, save
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_tcp_control import SCHEDULER_CONTRACT, _build_agent, _validate_scheduler_contract
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.tcp_removal_actor import SignedTcpRemovalActor

PRIOR = ROOT/'logs/experiments/tcp_removal_retrain_20261005/calibrated_control/force_stability/training'
NAMES = ('tangent_damping_ns_m', 'tangent_stiffness_n_m', 'inverse_force_gain',
         'rate_gain', 'corner_slowdown', 'tracking_slowdown', 'nominal_bias', 'force_slope_damping')
LOW = np.array([500., 8000., -.45, 0., 0., 0., -.02, 0.])
HIGH = np.array([6000., 16000., .25, .15, .50, .40, .08, .15])
ZERO = [6000., 2000., 0., 0., 0., 0., 0., 0.]
WEIGHTS = {'spatial_cv_gain': 2.5, 'raw_rate_cv_gain': 2., 'roi_volume_gain': .75,
           'coverage_gain': .75, 'cross_track_gain': .5, 'force_rmse_gain': .75,
           'time_gain': .25, 'endpoint_gain': .25}
PROMOTION = {'mean_spatial_cv_ratio_max': .98, 'mean_raw_rate_cv_ratio_max': .97,
             'mean_roi_volume_ratio_min': 1.002, 'per_seed_spatial_cv_ratio_max': 1.01,
             'per_seed_raw_rate_cv_ratio_max': 1.03, 'per_seed_roi_volume_ratio_min': .99,
             'per_seed_time_ratio_max': 1.05, 'per_seed_force_rmse_ratio_max': 1.10}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gains(candidate, previous):
    a, b = candidate['preston_surface'], previous['preston_surface']
    return {'spatial_cv_gain': 1-a['spatial_depth_cv']/b['spatial_depth_cv'],
            'raw_rate_cv_gain': 1-candidate['processing_rate_cv']/previous['processing_rate_cv'],
            'roi_volume_gain': a['roi_volume_over_k']/b['roi_volume_over_k']-1,
            'coverage_gain': b['zero_depth_fraction']-a['zero_depth_fraction'],
            'cross_track_gain': 1-candidate['cross_track_rmse_mm']/previous['cross_track_rmse_mm'],
            'force_rmse_gain': 1-candidate['force_rmse_n']/previous['force_rmse_n'],
            'time_gain': 1-candidate['processing_samples']/previous['processing_samples'],
            'endpoint_gain': 1-candidate['endpoint_tangential_error_mm']/previous['endpoint_tangential_error_mm']}


def evaluate_candidate(candidate, metrics, length, *, full_path):
    base = assess(candidate, [metrics['constant_a'], metrics['constant_b']], length,
                  validation=full_path, check_endpoint=full_path)
    relative = gains(candidate, metrics['previous'])
    rejections = list(base['rejections'])
    limits = {'spatial_cv_gain': -.01, 'raw_rate_cv_gain': -.03, 'roi_volume_gain': -.01,
              'time_gain': -.05, 'force_rmse_gain': -.10}
    penalty = 0.
    for name, minimum in limits.items():
        violation = max(0., minimum-relative[name])
        penalty += violation
        if violation > 1e-12:
            rejections.append('previous: '+name)
    if not metrics['previous']['completed'] or any(metrics['previous'][k] for k in ('shield_fraction', 'fault_fraction')):
        rejections.append('previous: incomplete_or_safety_event')
        penalty += 10.
    # The original score already penalizes physical/safety constraint violations.
    original_gain_score = min(sum({'spatial_cv_gain': 2., 'roi_volume_gain': 1., 'coverage_gain': 2.,
        'raw_rate_cv_gain': .25, 'cross_track_gain': .2, 'force_rmse_gain': .1, 'time_gain': .1}[k]*c[k]
        for k in c) for c in base['comparisons'])
    original_penalty = min(0., base['score']-original_gain_score)
    weighted = {key: WEIGHTS[key]*relative[key] for key in WEIGHTS}
    if not full_path:
        weighted['endpoint_gain'] = 0.
    score = sum(weighted.values()) + original_penalty - 10.*penalty
    if not np.isfinite(score):
        raise ValueError('Nonfinite candidate score')
    return {'score': score, 'eligible': not rejections, 'rejections': rejections,
            'gains_vs_previous': relative, 'weighted_reward': weighted,
            'constraint_penalty': original_penalty-10.*penalty, 'original_assessment': base}


def promotion_decision(validation):
    """Fixed aggregate improvement plus per-condition non-regression gates."""
    reasons = []
    for row in validation:
        reasons += [f"seed {row['seed']}: {reason}" for reason in row['assessment']['rejections']]
    means = {key: float(np.mean([r['assessment']['gains_vs_previous'][key] for r in validation])) for key in WEIGHTS}
    if 1-means['spatial_cv_gain'] > PROMOTION['mean_spatial_cv_ratio_max']:
        reasons.append('Mean spatial CV reduction below 2% versus previous')
    if 1-means['raw_rate_cv_gain'] > PROMOTION['mean_raw_rate_cv_ratio_max']:
        reasons.append('Mean raw rate CV reduction below 3% versus previous')
    if 1+means['roi_volume_gain'] < PROMOTION['mean_roi_volume_ratio_min']:
        reasons.append('Mean ROI volume increase below 0.2% versus previous')
    return {'passed': not reasons, 'rejections': reasons, 'paired_mean_gains_vs_previous': means}


def run_batch(root, name, population, seed, previous_checkpoint, *, length=698., checkpoint=None, old=False):
    """Every comparison uses the same randomization, path and raw-rate metric."""
    policies = {key: value[2:] for key, value in population.items()}
    controllers = {key: value[:2] for key, value in population.items()}
    save(root/f'{name}_policies.json', policies)
    save(root/f'{name}_mdk.json', controllers)
    save(root/f'{name}_checkpoints.json', {'previous': str(previous_checkpoint)})
    command = [sys.executable, str(ROOT/'scripts/probe_tcp_control.py'), '--headless', '--device', 'cpu',
        '--tcp-removal', '--signed-force-feedback', '--parameters-file', str(root/f'{name}_policies.json'),
        '--tangent-parameters', str(root/f'{name}_mdk.json'), '--checkpoint-map', str(root/f'{name}_checkpoints.json'),
        '--path-length-mm', str(length), '--max-steps', '26000', '--seed', str(seed),
        '--surface-cell-mm', '.5', '--output', str(root/name)]
    if checkpoint:
        command += ['--candidate-checkpoint', str(checkpoint)]
    if old:
        command += ['--include-checkpoint', str(OLD)]
    save(root/f'{name}_command.json', command)
    with (root/f'{name}.log').open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env={**os.environ, 'MPLCONFIGDIR': '/tmp/nrs_refinement_mpl',
            'MPLBACKEND': 'Agg'}, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{name} failed with exit {result.returncode}; see {root/name}.log')
    summary = json.loads((root/name/'summary.json').read_text())
    if (summary['rotation_depth_model'] is not None or summary['target_normal_force_n'] != 20.
            or summary['comparison_checkpoint_sha256']['previous'] != sha(previous_checkpoint)
            or summary['source_path_sha256'] != sha(REFERENCE)):
        raise ValueError('Comparison/reference/force contract changed during physics evaluation')
    if any(any(v != values[0] for v in values) for values in summary['randomization'].values()):
        raise ValueError('Paired measurement/latency conditions differ')
    return summary


def snapshot(root, previous):
    relative = ['scripts/train_tcp_refinement.py', 'scripts/probe_tcp_control.py',
        'scripts/train_tcp_control.py', 'scripts/skrl/tcp_removal_actor.py', 'scripts/skrl/ppo_policy_node.py',
        'scripts/calibrated_mode3.cpp', 'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/mdp/action.py',
        'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/nrs_rl_env_cfg.py',
        'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/utils/preston_surface.py',
        'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/utils/velocity_policy.py']
    hashes = {}
    for name in relative:
        destination = root/'source_snapshot'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, destination)
        hashes[str(ROOT/name)] = sha(destination)
    hashes[str(REFERENCE)] = sha(REFERENCE)
    native = ROOT/'logs/experiments/tcp_removal_retrain_20261005/calibrated_control/native'
    for path in native.glob('*.so'):
        hashes[str(path)] = sha(path)
    for name in ('checkpoints/actor.pt', 'params/velocity_policy.json', 'result.json', 'report.json'):
        hashes[str(previous/name)] = sha(previous/name)
    save(root/'source_manifest.json', hashes)
    return hashes


def freeze_actor(root, parameters, state):
    tangent = dict(zip(('damping_ns_m', 'stiffness_n_m'), parameters[:2]))
    contract = {**SCHEDULER_CONTRACT, 'schema_version': 3, 'observation_size': 14,
        'robot': 'ur10_cb3', 'control_period_s': .008, 'policy_action_repeat': 1, 'policy_period_s': .008,
        'policy_backend': 'torchscript_tcp_removal', 'policy_family': 'signed_force_stability',
        'objective': 'preston_tcp_spatial_removal', 'rotation_contribution': False, 'spindle_rpm': None,
        'tool_diameter_mm': 30., 'physics_tool_diameter_mm': 30.,
        'training_algorithm': state['algorithm'], 'learned_parameters': parameters[2:],
        'tangential_controller': tangent, 'requires_matching_controller_configuration': True,
        'reference_path_sha256': state['reference_path_sha256'],
        'adaptive_velocity_enabled': any(abs(x) > 1e-12 for x in parameters[2:]),
        'validation_status': 'candidate; consult result.json before integration'}
    checkpoint = root/'checkpoints/actor.pt'
    checkpoint.parent.mkdir()
    (root/'params').mkdir()
    save(root/'params/velocity_policy.json', contract)
    save(root/'params/controller.json', tangent)
    actor = SignedTcpRemovalActor(parameters[2:]).eval()
    torch.jit.save(torch.jit.script(actor), str(checkpoint),
                   _extra_files={'velocity_policy.json': json.dumps(contract, sort_keys=True)})
    agent, _ = _build_agent(str(checkpoint))
    _validate_scheduler_contract(agent.velocity_contract, tangent)
    generator = torch.Generator().manual_seed(6006)
    observations = torch.randn(2048, 14, generator=generator)
    observations[:, 10] = 1.
    observations[:, 11] = 0.
    with torch.inference_mode():
        error = float((actor(observations)-agent.act(observations)[-1]['mean_actions']).abs().max())
        observations[:, 11] = 1.
        masked = float(agent.act(observations)[-1]['mean_actions'].abs().max())
    if error > 1e-7 or masked != 0:
        raise ValueError('Checkpoint export changed inference or shield handling')
    save(root/'export_validation.json', {'passed': True, 'observations': 2048,
        'max_action_error': error, 'shielded_action_max': masked, 'checkpoint_sha256': sha(checkpoint)})
    return checkpoint


def train(args):
    torch.set_num_threads(1)
    previous = args.previous.resolve()
    old_state = json.loads((previous/'result.json').read_text())
    contract = json.loads((previous/'params/velocity_policy.json').read_text())
    previous_parameters = [contract['tangential_controller'][k] for k in ('damping_ns_m', 'stiffness_n_m')] + contract['learned_parameters']
    if old_state['phase'] != 'complete' or not old_state['validation_passed']:
        raise ValueError('Previous model must have completed validation')
    checkpoint_previous = previous/'checkpoints/actor.pt'
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    state = {'phase': 'structured_screen', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'algorithm': 'bounded_mdk_feed_cem_refinement_with_two_full_path_training_seeds',
        'parameter_names': NAMES, 'bounds': [LOW.tolist(), HIGH.tolist()],
        'reward': WEIGHTS, 'original_frozen_gates': GATES, 'promotion_gates': PROMOTION,
        'screen_seed': 519, 'screen_path_mm': 240., 'training_seeds': [529, 530],
        'prefix_screen_excludes_artificial_endpoint': True,
        'reused_screen_root': str(args.reuse_screen_root.resolve()) if args.reuse_screen_root else None,
        'validation_seeds': [609, 610, 611], 'selection_uses_validation': False,
        'previous_validation_is_now_development': [409, 410, 411],
        'previous_checkpoint': str(checkpoint_previous), 'previous_checkpoint_sha256': sha(checkpoint_previous),
        'previous_parameters': previous_parameters, 'reference_path_sha256': sha(REFERENCE),
        'normal_force_target_n': 20., 'normal_axis_controller_unchanged': True,
        'rotation_contribution': False, 'spindle_rpm': None,
        'production_checkpoint_replaced': False, 'physical_depth_validated': False,
        'candidate_selection': 'Maximum worst-condition reward among candidates eligible on both full training paths; rejected candidates cannot be promoted.',
        'screening': [], 'generations': [], 'validation': []}
    save(root/'protocol.json', state)
    source_hashes = snapshot(root, previous)

    def update(**values):
        state.update(values, updated_at_utc=datetime.now(timezone.utc).isoformat())
        save(root/'status.json', state)
        print('TCP_REFINEMENT_STATUS', state['phase'], state.get('current_batch', ''), flush=True)

    def make(damping, stiffness, feedback=None):
        return [damping, stiffness] + (previous_parameters[2:] if feedback is None else feedback)

    proposed = {
        'mdk_fast': make(750., 16000.), 'mdk_tight': make(1500., 16000.),
        'mdk_mid': make(2500., 16000.), 'mdk_damped': make(4500., 16000.),
        'corner': make(*previous_parameters[:2], [-.1375, .013, .30, 0., .035, 0.]),
        'tracking': make(*previous_parameters[:2], [-.1375, .013, 0., .30, .035, 0.]),
        'combined': make(1500., 16000., [-.1375, .013, .25, .20, .05, 0.]),
        'force_damped': make(1500., 16000., [-.25, .03, .15, .10, .03, .10]),
        'constant_mdk': make(1500., 16000., [0.]*6)}
    screen_jobs = []
    proposed_names = list(proposed)
    for group_index in range(2):
        names = proposed_names[group_index::2]
        population = {'constant_a': ZERO, 'constant_b': ZERO, 'previous': previous_parameters,
                      **{name: proposed[name] for name in names}}
        screen_jobs.append((f'screen_{group_index}', population))
    update(current_batch='screen_0 + screen_1')
    rankings = {}
    try:
        def screen(name, population):
            if args.reuse_screen_root is None:
                return run_batch(root, name, population, 519, checkpoint_previous, length=240.)
            source = args.reuse_screen_root.resolve()/name
            summary = json.loads((source/'summary.json').read_text())
            if (set(summary['candidates']) != set(population) or summary['seed'] != 519
                    or summary['source_path_sha256'] != state['reference_path_sha256']
                    or summary['comparison_checkpoint_sha256']['previous'] != state['previous_checkpoint_sha256']
                    or summary['rotation_depth_model'] is not None):
                raise ValueError('Cached prefix experiment differs from the frozen screening protocol')
            for key, values in population.items():
                if (summary['candidates'][key]['parameters'] != values[2:]
                        or summary['candidates'][key]['tangential_mdk'] != values[:2]):
                    raise ValueError('Cached screening policy parameters differ')
            shutil.copytree(source, root/name)
            return summary
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(screen, name, population):
                       (name, population) for name, population in screen_jobs}
            for future in as_completed(futures):
                name, population = futures[future]
                summary = future.result()
                candidates = summary['candidates']
                ranking = {key: evaluate_candidate(candidates[key], candidates, summary['path_length_mm'], full_path=False)
                           for key in population if key not in ('constant_a', 'constant_b', 'previous')}
                rankings.update(ranking)
                state['screening'].append({'batch': name, 'seed': 519, 'population': population, 'ranking': ranking})
                update()
        order = sorted(rankings, key=lambda k: (rankings[k]['eligible'], rankings[k]['score']), reverse=True)
        elites = np.array([proposed[name] for name in order[:3]])
        normalized = (elites-LOW)/(HIGH-LOW)
        center, spread = normalized.mean(0), np.maximum(normalized.std(0), .075)
        rng = np.random.default_rng(2026100602)
        full_population = {'constant_a': ZERO, 'constant_b': ZERO, 'previous': previous_parameters,
                           **{name: proposed[name] for name in order[:3]}}
        for index, sample in enumerate(rng.normal(center, spread, (3, len(LOW)))):
            full_population[f'cem_{index}'] = (LOW+np.clip(sample, 0., 1.)*(HIGH-LOW)).tolist()
        update(phase='full_path_training', current_batch='train_529 + train_530', full_population=full_population)
        train_rows = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(run_batch, root, f'train_{seed}', full_population, seed, checkpoint_previous): seed
                       for seed in state['training_seeds']}
            for future in as_completed(futures):
                seed = futures[future]
                summary = future.result()
                metrics = summary['candidates']
                ranking = {key: evaluate_candidate(metrics[key], metrics, summary['path_length_mm'], full_path=True)
                           for key in full_population if key not in ('constant_a', 'constant_b', 'previous')}
                train_rows.append({'seed': seed, 'ranking': ranking})
                train_rows.sort(key=lambda row: row['seed'])
                update(generations=train_rows)
        aggregate = {}
        for name in train_rows[0]['ranking']:
            rows = [r['ranking'][name] for r in train_rows]
            aggregate[name] = {'eligible': all(r['eligible'] for r in rows),
                               'worst_score': min(r['score'] for r in rows),
                               'mean_score': float(np.mean([r['score'] for r in rows]))}
        eligible = [name for name, row in aggregate.items() if row['eligible']]
        pool = eligible or list(aggregate)
        winner = max(pool, key=lambda name: aggregate[name]['worst_score'])
        chosen = full_population[winner]
        update(phase='checkpoint_freeze', current_batch='', selected_family=winner,
               best_parameters=chosen, training_candidate_eligible=bool(eligible), aggregate_ranking=aggregate)
        checkpoint = freeze_actor(root, chosen, state)
        update(phase='independent_validation', current_batch='held_out_609 + held_out_610 + held_out_611',
               checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint))
        population = {'constant_a': ZERO, 'constant_b': ZERO, 'old_d14': ZERO,
            'previous': previous_parameters, 'calibrated_only': chosen[:2]+[0.]*6, 'learned': chosen}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(run_batch, root, f'held_out_{seed}', population, seed,
                checkpoint_previous, checkpoint=checkpoint, old=True): seed for seed in state['validation_seeds']}
            for future in as_completed(futures):
                seed = futures[future]
                summary = future.result()
                metrics = summary['candidates']
                row = {'seed': seed, 'assessment': evaluate_candidate(metrics['learned'], metrics,
                        summary['path_length_mm'], full_path=True),
                       'calibration_only': evaluate_candidate(metrics['calibrated_only'], metrics,
                        summary['path_length_mm'], full_path=True),
                       'gains_vs_calibration': gains(metrics['learned'], metrics['calibrated_only']),
                       'gains_vs_old_d14': gains(metrics['learned'], metrics['old_d14'])}
                state['validation'].append(row)
                state['validation'].sort(key=lambda row: row['seed'])
                update()
        decision = promotion_decision(state['validation'])
        if not state['training_candidate_eligible']:
            decision['passed'] = False
            decision['rejections'].append('No candidate eligible on both training conditions')
        if sha(checkpoint) != state['checkpoint_sha256']:
            raise ValueError('Frozen candidate changed during validation')
        for path, expected in source_hashes.items():
            if sha(path) != expected:
                raise ValueError(f'Experiment input changed during training: {path}')
        update(phase='complete', current_batch='', validation_passed=decision['passed'], promotion=decision,
               disposition='improved_simulator_candidate' if decision['passed'] else 'retain_previous_validated_candidate')
        save(root/'result.json', state)
        print('TCP_REFINEMENT_COMPLETE', decision['passed'], str(root/'result.json'), flush=True)
    except Exception as error:
        update(phase='failed', error=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, default=PRIOR)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-screen-root', type=Path, help='Reuse matching completed training prefix rollouts only; never validation data')
    train(parser.parse_args())
