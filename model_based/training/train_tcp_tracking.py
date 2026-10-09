#!/usr/bin/env python3
"""Learn bounded outer tracking gain without altering the simulator/force target.

The existing velocity actor and MDK are fixed. A scalar tangent-position
feedback gain is selected from physics rewards, then the combined controller
is frozen before three new conditions. Its experimental backend deliberately
cannot be loaded into the old ROS worker without controller integration.
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
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_tcp_refinement import ROOT, PRIOR, ZERO, OLD, REFERENCE, WEIGHTS, PROMOTION, GATES
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_tcp_refinement import NAMES, evaluate_candidate, promotion_decision, sha, snapshot
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_tcp_control import save, SCHEDULER_CONTRACT, _build_agent
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.tcp_removal_actor import SignedTcpRemovalActor
from nrs_rl.tasks.manager_based.nrs_rl.utils.tcp_tracking_compensation import TangentTrackingCompensation, TrackingPolicyAgent


def native_checks(root, parameters):
    sys.path.insert(0, str(ROOT/'logs/experiments/tcp_removal_retrain_20261005/calibrated_control/native'))
    from _tcp_calibrated_mode3 import TangentialMode3
    sys.path.insert(0, str(ROOT/'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/y2_control_pybind'))
    from y2_control_py import config
    args = [config.NAF_MDGRADI_CKPT, .008, config.FORCE_CON_COORDINATE,
        config.FORCE_SWITCH_DESIRED_FORCE_THRESHOLD, config.FORCE_SWITCH_ACTUAL_FORCE_THRESHOLD,
        config.FORCE_SWITCH_PRECONTACT_FORCE_HOLD, config.FORCE_SWITCH_RETURN_TAU, *parameters[:2]]
    original = TangentialMode3(*args)
    zero = TangentTrackingCompensation(TangentialMode3(*args), 0.)
    enabled = TangentTrackingCompensation(TangentialMode3(*args), 1.)
    pose = [870., 340., 100., 0., 0., 0.]
    for core in (original, zero, enabled):
        core.reset(pose)
    parity_error, normal_error, largest_correction = 0., 0., 0.
    for step in range(500):
        reference = [870.+.03*step, 340.+np.sin(step/100.), 100., 0., 0., 0.]
        wrench = [3.*np.sin(step/30.), -12., 20.+np.cos(step/20.), 0., 0., 0.]
        inputs = [pose, reference, [0., 0., 20.], wrench, np.eye(3).ravel().tolist()]
        a, b, c = [np.asarray(core.step(*inputs)) for core in (original, zero, enabled)]
        parity_error = max(parity_error, float(np.abs(a-b).max()))
        normal_error = max(normal_error, float(np.abs(a[[2, 8, 14, 20]]-c[[2, 8, 14, 20]]).max()))
        largest_correction = max(largest_correction, float(np.linalg.norm(enabled.correction)))
        pose = a[:6].tolist()
    if parity_error > 1e-10 or normal_error > 1e-10 or largest_correction > 4.+1e-10:
        raise ValueError('Controller wrapper failed parity, normal-axis, or bounded-correction checks')
    save(root/'controller_proof.json', {'passed': True, 'steps': 500, 'zero_gain_error': parity_error,
        'normal_axis_error_for_identical_inputs': normal_error, 'largest_correction_mm': largest_correction,
        'ros_runtime_integrated': False, 'physics_or_servo_parameters_changed': False})


def batch(root, name, gain_map, seed, previous, *, length=698., checkpoint=None, old=False):
    population = {key: previous for key in gain_map}
    population.update(constant_a=ZERO, constant_b=ZERO)
    if 'calibrated_only' in population:
        population['calibrated_only'] = previous[:2]+[0.]*6
    if 'old_d14' in population:
        population['old_d14'] = ZERO
    save(root/f'{name}_policies.json', {key: p[2:] for key, p in population.items()})
    save(root/f'{name}_mdk.json', {key: p[:2] for key, p in population.items()})
    save(root/f'{name}_tracking.json', gain_map)
    save(root/f'{name}_checkpoints.json', {'previous': str(PRIOR/'checkpoints/actor.pt')})
    command = [sys.executable, str(ROOT/'scripts/probe_tcp_tracking.py'), '--headless', '--device', 'cpu',
        '--tcp-removal', '--signed-force-feedback', '--parameters-file', str(root/f'{name}_policies.json'),
        '--tangent-parameters', str(root/f'{name}_mdk.json'), '--tracking-parameters', str(root/f'{name}_tracking.json'),
        '--checkpoint-map', str(root/f'{name}_checkpoints.json'), '--path-length-mm', str(length),
        '--max-steps', '26000', '--seed', str(seed), '--surface-cell-mm', '.5', '--output', str(root/name)]
    if checkpoint:
        command += ['--candidate-checkpoint', str(checkpoint)]
    if old:
        command += ['--include-checkpoint', str(OLD)]
    save(root/f'{name}_command.json', command)
    with (root/f'{name}.log').open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env={**os.environ, 'MPLCONFIGDIR': '/tmp/nrs_tracking_mpl',
            'MPLBACKEND': 'Agg'}, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{name} exited {result.returncode}; see {root/name}.log')
    summary = json.loads((root/name/'summary.json').read_text())
    if (summary['tracking_compensation']['gains'] != gain_map or summary['rotation_depth_model'] is not None
            or summary['comparison_checkpoint_sha256']['previous'] != sha(PRIOR/'checkpoints/actor.pt')
            or summary['source_path_sha256'] != sha(REFERENCE) or summary['target_normal_force_n'] != 20.):
        raise ValueError('Controller or measurement protocol mismatch')
    if any(any(v != values[0] for v in values) for values in summary['randomization'].values()):
        raise ValueError('Paired randomization differs')
    return summary


def freeze(root, state, parameters, gain):
    contract = {**SCHEDULER_CONTRACT, 'schema_version': 3, 'observation_size': 14,
        'robot': 'ur10_cb3', 'control_period_s': .008, 'policy_action_repeat': 1, 'policy_period_s': .008,
        'policy_backend': 'torchscript_tcp_tracking', 'policy_family': 'signed_force_stability_with_outer_tracking',
        'objective': 'preston_tcp_spatial_removal', 'rotation_contribution': False, 'spindle_rpm': None,
        'tool_diameter_mm': 30., 'physics_tool_diameter_mm': 30., 'training_algorithm': state['algorithm'],
        'learned_parameters': parameters[2:], 'velocity_coefficients_frozen_from_previous': True,
        'tangential_controller': dict(zip(('damping_ns_m', 'stiffness_n_m'), parameters[:2])),
        'tracking_compensation': {'gain': gain, 'tau_s': .08, 'limit_mm': 4.},
        'requires_matching_controller_configuration': True, 'requires_new_tracking_runtime': True,
        'reference_path_sha256': state['reference_path_sha256'],
        'validation_status': 'experimental; old ROS worker deliberately rejects this backend'}
    (root/'params').mkdir()
    (root/'checkpoints').mkdir()
    save(root/'params/velocity_policy.json', contract)
    save(root/'params/controller.json', {'tangential_controller': contract['tangential_controller'],
                                       'tracking_compensation': contract['tracking_compensation']})
    path = root/'checkpoints/actor.pt'
    actor = SignedTcpRemovalActor(parameters[2:]).eval()
    torch.jit.save(torch.jit.script(actor), str(path),
        _extra_files={'velocity_policy.json': json.dumps(contract, sort_keys=True)})
    agent = TrackingPolicyAgent(path, contract)
    generator = torch.Generator().manual_seed(7006)
    observations = torch.randn(2048, 14, generator=generator)
    observations[:, 10], observations[:, 11] = 1., 0.
    error = float((actor(observations)-agent.act(observations)[-1]['mean_actions']).abs().max())
    legacy_rejected = False
    try:
        _build_agent(str(path))
    except ValueError as exc:
        legacy_rejected = 'Unknown velocity policy backend' in str(exc)
    if error > 1e-7 or not legacy_rejected:
        raise ValueError('Checkpoint parity or missing-controller protection failed')
    save(root/'export_validation.json', {'passed': True, 'max_action_error': error,
        'legacy_worker_rejected': legacy_rejected, 'checkpoint_sha256': sha(path)})
    return path


def train(args):
    torch.set_num_threads(1)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    old = json.loads((PRIOR/'params/velocity_policy.json').read_text())
    previous = [old['tangential_controller'][key] for key in ('damping_ns_m', 'stiffness_n_m')]+old['learned_parameters']
    state = {'phase': 'controller_checks', 'algorithm': 'bounded_tangential_tracking_gain_search',
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'parameter_names': NAMES,
        'best_parameters': previous, 'fixed_velocity_and_mdk_parameters': previous,
        'tracking_compensation_bounds': {'gain': [0., 1.], 'tau_s': .08, 'limit_mm': 4.},
        'reward': WEIGHTS, 'original_frozen_gates': GATES, 'promotion_gates': PROMOTION,
        'screen_seed': 539, 'screen_path_mm': 240., 'screen_gains': [.25, .5, .75, 1.],
        'screen_continuation_rule': 'At least one eligible positive-reward prefix candidate; otherwise stop without promotion.',
        'prefix_screen_excludes_artificial_endpoint': True,
        'full_search_rule': 'Best eligible prefix gain and neighbors +/- 0.15 clipped to [0,1]; no validation selection.',
        'training_seeds': [549, 550], 'validation_seeds': [709, 710, 711],
        'selection_uses_validation': False, 'development_evidence': ['Oct-06 previous traces', 'MDK/feed training seed 529'],
        'prior_refinement_validation_not_used_for_selection': [609, 610, 611],
        'previous_checkpoint': str(PRIOR/'checkpoints/actor.pt'), 'previous_checkpoint_sha256': sha(PRIOR/'checkpoints/actor.pt'),
        'reference_path_sha256': sha(REFERENCE), 'normal_force_target_n': 20.,
        'normal_force_controller_algorithm_unchanged': True, 'physics_and_servo_unchanged': True,
        'rotation_contribution': False, 'spindle_rpm': None, 'production_checkpoint_replaced': False,
        'physical_depth_validated': False, 'runtime_integration_validated': False,
        'screening': [], 'generations': [], 'validation': []}
    save(root/'protocol.json', state)
    source_hashes = snapshot(root, PRIOR)
    for name in ('train_tcp_tracking.py', 'probe_tcp_tracking.py', 'tcp_tracking_compensation.py'):
        source = ROOT/'scripts'/name
        shutil.copy2(source, root/'source_snapshot/scripts'/name)
        source_hashes[str(source)] = sha(source)
    save(root/'source_manifest.json', source_hashes)

    def update(**changes):
        state.update(changes, updated_at_utc=datetime.now(timezone.utc).isoformat())
        save(root/'status.json', state)
        print('TRACKING_TRAIN_STATUS', state['phase'], flush=True)

    def rank(summary, names, full):
        metrics = summary['candidates']
        return {name: evaluate_candidate(metrics[name], metrics, summary['path_length_mm'], full_path=full) for name in names}

    try:
        native_checks(root, previous)
        update(phase='screening')
        gain_map = {'constant_a': 0., 'constant_b': 0., 'previous': 0.,
                    **{f'gain_{int(g*100):03d}': g for g in state['screen_gains']}}
        summary = batch(root, 'screen', gain_map, 539, previous, length=240.)
        names = list(gain_map)[3:]
        ranking = rank(summary, names, False)
        state['screening'].append({'seed': 539, 'gain_map': gain_map, 'ranking': ranking})
        eligible = [name for name in names if ranking[name]['eligible'] and ranking[name]['score'] > 0]
        if not eligible:
            update(phase='complete', validation_passed=False, disposition='screen_rejected_no_new_checkpoint',
                   promotion={'passed': False, 'rejections': ['No eligible positive-reward prefix candidate']})
            save(root/'result.json', state)
            return
        best = max(eligible, key=lambda name: ranking[name]['score'])
        center = gain_map[best]
        candidates = sorted(set(float(np.clip(center+d, 0., 1.)) for d in (-.15, 0., .15)))
        full_map = {'constant_a': 0., 'constant_b': 0., 'previous': 0.,
                    **{f'gain_{int(round(g*1000)):04d}': g for g in candidates}}
        names = list(full_map)[3:]
        update(phase='full_path_training', training_gain_map=full_map)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(batch, root, f'train_{seed}', full_map, seed, previous): seed for seed in state['training_seeds']}
            for future in as_completed(futures):
                seed = futures[future]
                state['generations'].append({'seed': seed, 'ranking': rank(future.result(), names, True)})
                state['generations'].sort(key=lambda row: row['seed'])
                update()
        combined = {name: {'eligible': all(r['ranking'][name]['eligible'] for r in state['generations']),
            'worst_score': min(r['ranking'][name]['score'] for r in state['generations'])} for name in names}
        eligible = [name for name in names if combined[name]['eligible']]
        chosen = max(eligible or names, key=lambda name: combined[name]['worst_score'])
        gain = full_map[chosen]
        update(phase='checkpoint_freeze', selected_family=chosen, training_candidate_eligible=bool(eligible),
            tracking_compensation={'gain': gain, 'tau_s': .08, 'limit_mm': 4.}, aggregate_ranking=combined)
        checkpoint = freeze(root, state, previous, gain)
        update(phase='independent_validation', checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint))
        validation_map = {'constant_a': 0., 'constant_b': 0., 'old_d14': 0., 'previous': 0.,
                          'calibrated_only': gain, 'learned': gain}
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(batch, root, f'held_out_{seed}', validation_map, seed, previous,
                checkpoint=checkpoint, old=True): seed for seed in state['validation_seeds']}
            for future in as_completed(futures):
                summary = future.result()
                record = rank(summary, ['learned', 'calibrated_only'], True)
                state['validation'].append({'seed': futures[future], 'assessment': record['learned'],
                                            'calibration_only': record['calibrated_only']})
                state['validation'].sort(key=lambda row: row['seed'])
                update()
        decision = promotion_decision(state['validation'])
        if not state['training_candidate_eligible']:
            decision['passed'] = False
            decision['rejections'].append('No candidate eligible on both training conditions')
        for path, expected in source_hashes.items():
            if sha(path) != expected:
                raise ValueError('Frozen source changed: '+path)
        if sha(checkpoint) != state['checkpoint_sha256']:
            raise ValueError('Candidate changed during validation')
        update(phase='complete', validation_passed=decision['passed'], promotion=decision,
            disposition='improved_simulator_candidate_requires_new_tracking_runtime' if decision['passed'] else 'retain_previous_validated_candidate')
        save(root/'result.json', state)
        print('TRACKING_TRAIN_COMPLETE', decision['passed'], flush=True)
    except Exception as error:
        update(phase='failed', error=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    train(parser.parse_args())
