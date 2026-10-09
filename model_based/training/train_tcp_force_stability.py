#!/usr/bin/env python3
"""Refine force/feed coupling after the joint controller study's force-RMS failure.

Previous validation seeds 309--311 are now development evidence. This study
freezes its choice on training seed 319, then uses fresh seeds 409--411 once.
The learned tangential MDK is fixed; both feedback signs and zero feedback are
eligible. A zero-feedback winner must be reported as constant-feed calibration.
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
from pathlib import Path
import shutil

import numpy as np
import torch

from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_tcp_control import ROOT, BASE, GATES, REWARD, batch, assess, save
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_tcp_control import SCHEDULER_CONTRACT, _build_agent, _validate_scheduler_contract
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.tcp_removal_actor import SignedTcpRemovalActor


def rank_family(metrics, names, original, calibrated, path_length, *, constant=False):
    rows = []
    for name in names:
        candidate = metrics[name]
        result = assess(candidate, original, path_length, validation=True)
        force_gain = min(1-candidate['force_rmse_n']/b['force_rmse_n'] for b in original)
        versus_calibrated_force = 0. if constant else min(
            1-candidate['force_rmse_n']/b['force_rmse_n'] for b in calibrated)
        versus_calibrated_rate = 0. if constant else min(
            1-candidate['processing_rate_cv']/b['processing_rate_cv'] for b in calibrated)
        eligible = (result['passed'] and versus_calibrated_force >= -.02
                    and versus_calibrated_rate >= -.02)
        score = (result['score'] + 1.5*force_gain + .5*versus_calibrated_force
                 + .25*versus_calibrated_rate)
        rows.append({'name': name, 'base_assessment': result, 'score': score, 'eligible': eligible,
                     'force_gain_vs_original': force_gain,
                     'force_gain_vs_calibrated': versus_calibrated_force,
                     'raw_rate_gain_vs_calibrated': versus_calibrated_rate})
    return {'score': min(r['score'] for r in rows), 'eligible': all(r['eligible'] for r in rows),
            'replicates': rows}


def train(args):
    torch.set_num_threads(1)
    prior = args.previous.resolve()
    previous = json.loads((prior/'result.json').read_text())
    if previous['phase'] != 'complete':
        raise ValueError('Previous experiment has not completed')
    damping, stiffness, force_gain, rate_gain = previous['best_parameters']
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    tangent = {'damping_ns_m': damping, 'stiffness_n_m': stiffness}
    population = {'constant_a': BASE, 'constant_b': BASE,
        'calibrated_a': [damping, stiffness, 0., 0.],
        'calibrated_b': [damping, stiffness, 0., 0.],
        'inverse_quarter': [damping, stiffness, force_gain*.25, rate_gain*.25],
        'direct_quarter': [damping, stiffness, -force_gain*.25, rate_gain*.25],
        'rate_only': [damping, stiffness, 0., .05]}
    state = {'phase': 'training', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'algorithm': 'cem_tangential_calibration_then_force_stability_policy_search',
        'rotation_contribution': False, 'spindle_rpm': None, 'normal_force_target_n': 20.,
        'normal_axis_controller_unchanged': True, 'gates': GATES,
        'reward': {'base': REWARD, 'extra_worst_force_gain_vs_original': 1.5,
                   'worst_force_gain_vs_calibrated': .5, 'worst_rate_gain_vs_calibrated': .25},
        'training_eligibility': 'Pass original frozen gates; adaptive feedback cannot worsen force RMS or raw rate CV by over 2% versus either calibrated constant replicate.',
        'zero_feedback_is_explicit_candidate': True,
        'parameter_names': previous['parameter_names'],
        'training_seeds': [109, 110, 319], 'new_training_seeds': [319],
        'development_seeds_from_previous_validation': [309, 310, 311],
        'validation_seeds': [409, 410, 411], 'validation_selection_used': False,
        'previous_result': str(prior/'result.json'),
        'previous_result_sha256': hashlib.sha256((prior/'result.json').read_bytes()).hexdigest(),
        'reference_path_sha256': previous['reference_path_sha256'],
        'production_controller_reconfigured': False, 'physical_depth_validated': False,
        'fixed_tangential_controller': tangent, 'population': population,
        'generations': [], 'validation': []}
    save(root/'protocol.json', state)
    files = ['scripts/train_tcp_force_stability.py', 'scripts/train_tcp_control.py',
             'scripts/probe_tcp_control.py', 'scripts/skrl/tcp_removal_actor.py',
             'scripts/skrl/ppo_policy_node.py', 'scripts/skrl/velocity_actor.py']
    manifest = {}
    for relative in files:
        destination = root/'source_snapshot'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/relative, destination)
        manifest[relative] = hashlib.sha256(destination.read_bytes()).hexdigest()
    save(root/'source_manifest.json', manifest)

    def update(**changes):
        state.update(changes, updated_at_utc=datetime.now(timezone.utc).isoformat())
        save(root/'status.json', state)
        print('FORCE_STABILITY_STATUS', state['phase'], flush=True)

    update(current_batch='force_screen')
    summary = batch(root, 'force_screen', population, 319, signed_feedback=True)
    metrics = summary['candidates']
    original = [metrics[n] for n in ('constant_a', 'constant_b')]
    calibrated = [metrics[n] for n in ('calibrated_a', 'calibrated_b')]
    ranking = {'constant_feed': rank_family(metrics, ['calibrated_a', 'calibrated_b'],
        original, calibrated, summary['path_length_mm'], constant=True)}
    for name in ('inverse_quarter', 'direct_quarter', 'rate_only'):
        ranking[name] = rank_family(metrics, [name], original, calibrated, summary['path_length_mm'])
    eligible = [name for name, row in ranking.items() if row['eligible']]
    pool = eligible or list(ranking)
    winner = max(pool, key=lambda name: ranking[name]['score'])
    chosen = population['calibrated_a' if winner == 'constant_feed' else winner]
    state['generations'].append({'generation': 0, 'seed': 319, 'ranking': ranking,
                                'winner': winner, 'parameters': chosen})
    update(best_parameters=chosen, selected_family=winner, training_candidate_eligible=bool(eligible),
           adaptive_velocity_enabled=winner != 'constant_feed')
    feedback = [chosen[2], chosen[3], 0., 0., 0., 0.]
    contract = {**SCHEDULER_CONTRACT, 'schema_version': 3, 'observation_size': 14,
        'robot': 'ur10_cb3', 'control_period_s': .008, 'policy_action_repeat': 1, 'policy_period_s': .008,
        'policy_backend': 'torchscript_tcp_removal', 'policy_family': 'signed_force_stability',
        'objective': 'preston_tcp_spatial_removal', 'rotation_contribution': False, 'spindle_rpm': None,
        'tool_diameter_mm': 30., 'physics_tool_diameter_mm': 30.,
        'training_algorithm': state['algorithm'], 'learned_parameters': feedback,
        'tangential_controller': tangent, 'requires_matching_controller_configuration': True,
        'reference_path_sha256': state['reference_path_sha256'],
        'adaptive_velocity_enabled': state['adaptive_velocity_enabled'],
        'validation_status': 'candidate; consult result.json before integration'}
    checkpoint = root/'checkpoints/actor.pt'
    checkpoint.parent.mkdir()
    (root/'params').mkdir()
    save(root/'params/velocity_policy.json', contract)
    save(root/'params/controller.json', tangent)
    actor = SignedTcpRemovalActor(feedback).eval()
    torch.jit.save(torch.jit.script(actor), str(checkpoint),
                   _extra_files={'velocity_policy.json': json.dumps(contract, sort_keys=True)})
    agent, _ = _build_agent(str(checkpoint))
    _validate_scheduler_contract(agent.velocity_contract, tangent)
    observations = torch.randn(1024, 14)
    observations[:, 10] = 1.
    observations[:, 11] = 0.
    error = float((actor(observations)-agent.act(observations)[0]).abs().max())
    assert error < 1e-7
    save(root/'export_validation.json', {'ok': True, 'max_action_error': error,
         'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
    update(phase='independent_validation', checkpoint=str(checkpoint))
    validation_population = {'constant_a': BASE, 'constant_b': BASE, 'old_d14': BASE,
                             'calibrated_only': [damping, stiffness, 0., 0.], 'learned': chosen}

    def validate(seed):
        summary = batch(root, f'held_out_{seed}', validation_population, seed,
                        checkpoint=checkpoint, old=True, signed_feedback=True)
        m = summary['candidates']
        baselines = [m[n] for n in ('constant_a', 'constant_b')]
        return {'seed': seed, **assess(m['learned'], baselines, summary['path_length_mm'], validation=True),
            'calibration_only': assess(m['calibrated_only'], baselines, summary['path_length_mm'], validation=True),
            'learned_vs_calibration': assess(m['learned'], [m['calibrated_only']], summary['path_length_mm'])}

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(validate, seed) for seed in state['validation_seeds']]
        for future in as_completed(futures):
            state['validation'].append(future.result())
            state['validation'].sort(key=lambda row: row['seed'])
            update()
    passed = all(row['passed'] for row in state['validation'])
    update(phase='complete', validation_passed=passed,
           disposition='validated_simulator_candidate_requires_controller_integration' if passed
                       else 'candidate_failed_frozen_gates')
    save(root/'result.json', state)
    print('FORCE_STABILITY_COMPLETE', passed, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, default=ROOT/'logs/experiments/tcp_removal_retrain_20261005/calibrated_control/training')
    parser.add_argument('--output', type=Path, required=True)
    train(parser.parse_args())
