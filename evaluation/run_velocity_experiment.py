#!/usr/bin/env python3
"""Snapshot, train and independently compare a 125 Hz velocity policy.

This workflow writes simulation artifacts only. --smoke checks the gSDE runtime
and actor export on 2048 control steps per environment, without a quality claim.
"""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--algorithm', choices=('gsde', 'skrl', 'feedback_screen', 'depth_smoke', 'depth_search'), default='gsde')
parser.add_argument('--smoke', action='store_true')
parser.add_argument('--profile', type=Path)
parser.add_argument('--reuse-initial-evaluation', type=Path)
args = parser.parse_args()
if args.smoke and args.algorithm != 'gsde':
    parser.error('--smoke is available for the new gSDE runtime only')
if args.algorithm in ('depth_smoke', 'depth_search') and args.profile is None:
    parser.error('Depth search requires an explicit --profile from the dwell model')
if args.reuse_initial_evaluation and args.algorithm != 'depth_search':
    parser.error('Saved initial dynamics can only be reused for depth_search')

root = Path('/home/eunseop/nrs_rl')
python = Path('/home/eunseop/anaconda3/envs/env_isaaclab/bin/python')
stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
ticks = 2048 if args.smoke else 32768
experiment = 'pad30_preview16_' + args.algorithm + ('_smoke_' if args.smoke else '_') + stamp
workflow = root / 'logs/experiments' / ('preston_' + experiment)
workflow.mkdir(parents=True)
snapshot = workflow / 'source_snapshot'
files = [
    'scripts/probe_velocity_feedback.py', 'scripts/skrl/bounded_velocity_actor.py',
    'scripts/train_preston_depth.py', 'scripts/skrl/preston_dwell_actor.py',
    'scripts/analyze_rotation_depth.py', 'scripts/diagnose_dwell_authority.py',
    'scripts/reintegrate_depth_candidates.py', 'scripts/audit_preston_depth_run.py',
    'scripts/test_preston_dwell_policy.py', 'scripts/test_rotation_depth.py',
    'scripts/run_velocity_experiment.py', 'scripts/train_velocity_gsde.py',
    'scripts/skrl/velocity_actor.py', 'scripts/test_velocity_actor.py',
    'scripts/skrl/train.py', 'scripts/skrl/play.py', 'scripts/skrl/ppo_policy_node.py',
    'deployment/y2_speed_limiter/include/Y2RobMotion/ppo_trajectory_scheduler.hpp',
    'deployment/y2_speed_limiter/src/ppo_trajectory_scheduler.cpp',
    'deployment/y2_velocity_preston_v3.patch',
    'scripts/evaluate_adaptive_velocity.py', 'scripts/analyze_preston_surface.py',
    'scripts/validate_velocity_worker.py',
    'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/mdp/action.py',
    'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/mdp/rewards.py',
    'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/mdp/terminations.py',
    'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/nrs_rl_env_cfg.py',
    'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/agents/skrl_ppo_cfg.yaml',
    'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/utils/velocity_policy.py',
    'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/utils/control_action_repeat.py',
    'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/utils/preston_surface.py',
    'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/utils/velocity_evaluation.py',
    'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/assets/assets/robots/ur10_w_spindle.usda',
    'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/assets/assets/robots/ur10e_w_spindle.py',
]
manifest = {}
for relative in files:
    source, target = root / relative, snapshot / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    manifest[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
dataset = 'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/datasets/cmd_continue9D_flat.h5'
manifest[dataset] = hashlib.sha256((root / dataset).read_bytes()).hexdigest()
(snapshot / 'manifest.json').write_text(json.dumps(manifest, indent=2))
if args.algorithm in ('feedback_screen', 'depth_smoke', 'depth_search'):
    result_directory = workflow / 'screen'
    if args.algorithm == 'feedback_screen':
        command = [str(python), 'scripts/probe_velocity_feedback.py', '--headless', '--device', 'cpu',
                   '--path-length-mm', '160', '--output', str(result_directory)]
    else:
        shutil.copyfile(args.profile, workflow / 'initial_dwell_profile.json')
        command = [str(python), 'scripts/train_preston_depth.py',
                   '--profile', str(workflow / 'initial_dwell_profile.json'), '--output', str(result_directory)]
        if args.algorithm == 'depth_smoke':
            command.append('--smoke')
        if args.reuse_initial_evaluation:
            command.extend(['--reuse-initial-evaluation', str(args.reuse_initial_evaluation.resolve())])
    state = {'phase': 'screening' if args.algorithm != 'depth_search' else 'training',
             'kind': args.algorithm, 'learned_policy': False,
             'physical_depth_validated': False, 'promoted_to_robot': False,
             'started_at_utc': datetime.now(timezone.utc).isoformat(),
             'reuse_initial_evaluation': str(args.reuse_initial_evaluation.resolve()) if args.reuse_initial_evaluation else None,
             'independent_validation_passed': False,
             'validation_seeds': [43, 44] if args.algorithm == 'depth_search' else [],
             'workflow': str(workflow), 'output': str(result_directory), 'command': command}

    def save_screen():
        state['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        temporary = workflow / 'status.next.json'
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(workflow / 'status.json')
        print(json.dumps(state), flush=True)

    save_screen()
    try:
        with (workflow / 'screen.log').open('w') as output:
            result = subprocess.run(command, cwd=root, stdout=output, stderr=subprocess.STDOUT)
        state['training_exit_code'] = result.returncode
        if result.returncode:
            raise RuntimeError(f'Candidate simulation failed; see {workflow / "screen.log"}')
        if args.algorithm != 'depth_search':
            state.update(phase='finished', exit_code=0)
            save_screen()
            sys.exit(0)
        training = json.loads((result_directory / 'status.json').read_text())
        state.update(learned_policy=training['learned_policy'], training=training,
                     reference_roi=training['reference_roi'])
        if not training['learned_policy']:
            state.update(phase='finished', quality_passed=False, exit_code=1,
                         stop_reason=training.get('stop_reason', 'No improving candidate exported'))
            save_screen()
            sys.exit(1)
        checkpoint = Path(training['checkpoint'])
        state.update(checkpoint=str(checkpoint), checkpoint_sha256=training['checkpoint_sha256'])
        shutil.copytree(snapshot, result_directory / 'source_snapshot')
        with (result_directory / 'final_inference_validation.json').open('w') as output:
            subprocess.run([str(python), 'scripts/skrl/ppo_policy_node.py', '--self-test',
                            '--checkpoint', str(checkpoint)], cwd=root, stdout=output, check=True)
        # This is an experimental model-quality gate, not a real-robot depth claim.
        # Fix the threshold before examining either held-out seed.
        minimum_gain = training['generations'][training['best_generation']]['minimum_reliable_gain']
        state.update(minimum_depth_cv_improvement_fraction=minimum_gain, evaluations=[])
        for seed in state['validation_seeds']:
            evaluation = root / 'logs/velocity_evaluation' / f'preston_{experiment}_seed{seed}_full'
            state.update(phase='evaluation', evaluation=str(evaluation), evaluation_seed=seed)
            save_screen()
            command = [str(python), 'scripts/evaluate_adaptive_velocity.py', '--headless', '--device', 'cpu',
                       '--seed', str(seed), '--checkpoint', str(checkpoint), '--require-improvement',
                       '--min-cv-improvement-percent', str(100*minimum_gain), '--output', str(evaluation)]
            (workflow / f'evaluation_seed{seed}_command.json').write_text(json.dumps(command, indent=2))
            with (workflow / f'evaluation_seed{seed}.log').open('w') as output:
                result = subprocess.run(command, cwd=root, stdout=output, stderr=subprocess.STDOUT)
            summary = json.loads((evaluation / 'summary.json').read_text())
            passed = bool(result.returncode == 0 and summary['comparison']['passed'])
            record = {'seed': seed, 'summary': str(evaluation / 'summary.json'),
                      'exit_code': result.returncode, 'passed': passed,
                      'comparison': summary['comparison'],
                      'rotation_depth_comparison': summary['rotation_depth_comparison']}
            state['evaluations'].append(record)
            save_screen()
            with (workflow / f'worker_seed{seed}.log').open('w') as output:
                subprocess.run([str(python), 'scripts/validate_velocity_worker.py',
                    '--checkpoint', str(checkpoint), '--trace', str(evaluation / 'adaptive_trace.npz'),
                    '--output', str(evaluation / 'worker_validation.json')],
                    cwd=root, stdout=output, stderr=subprocess.STDOUT, check=True)
            record['worker_validation'] = str(evaluation / 'worker_validation.json')
            if not passed:
                state['stop_reason'] = 'Independent depth-quality gate failed; remaining seeds skipped'
                break
        passed = (len(state['evaluations']) == len(state['validation_seeds'])
                  and all(record['passed'] for record in state['evaluations']))
        state.update(phase='finished', quality_passed=passed, independent_validation_passed=passed,
                     exit_code=0 if passed else 1, finished_at_utc=datetime.now(timezone.utc).isoformat())
        save_screen()
        sys.exit(state['exit_code'])
    except Exception as error:
        state.update(phase='error', error=repr(error), exit_code=2)
        save_screen()
        traceback.print_exc()
        sys.exit(2)
state = {
    'phase': 'training', 'algorithm': args.algorithm, 'runtime_smoke_only': args.smoke,
    'physics_tool_diameter_mm': 30., 'target_normal_force_n': 20.,
    'policy_hz': 125, 'turn_preview_mm': 16., 'control_ticks_per_env': ticks, 'num_envs': 2,
    'minimum_rate_cv_improvement_percent': 10., 'physical_depth_validated': False,
    'promoted_to_robot': False, 'workflow': str(workflow),
}
if args.algorithm == 'gsde':
    state.update(noise_resample_steps=32, exploration_weight_log_std=-3., rollout_steps=1024)
    run = root / 'logs/sb3/preston_20n' / experiment
    state['training_run'] = str(run)
else:
    state.update(initial_std=0.36787944117144233, rollout_steps=512)


def save():
    # Atomic status replacement lets a monitor read while phases change.
    temporary = workflow / 'status.next.json'
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(workflow / 'status.json')
    print(json.dumps(state), flush=True)


save()
try:
    if args.algorithm == 'gsde':
        command = [str(python), 'scripts/train_velocity_gsde.py', '--headless', '--device', 'cpu',
                   '--num-envs', '2', '--control-steps', str(ticks), '--output', str(run)]
    else:
        command = [str(python), 'scripts/skrl/train.py', '--headless', '--device', 'cpu',
                   '--task', 'Template-Nrs-Rl-v0', '--num_envs', '2', '--max_iterations', '64',
                   '--policy-action-repeat', '1', 'env.actions.arm_action.integration.turn_preview_mm=16.0',
                   'env.visualization.enable_visualizer=False',
                   'env.actions.arm_action.integration.enable_debug_print=False',
                   'agent.agent.experiment.directory=preston_20n',
                   'agent.agent.experiment.experiment_name=' + experiment]
    (workflow / 'training_command.json').write_text(json.dumps(command, indent=2))
    with (workflow / 'training.log').open('w') as output:
        subprocess.run(command, cwd=root, stdout=output, stderr=subprocess.STDOUT, check=True)
    if args.algorithm == 'skrl':
        candidates = list((root / 'logs/skrl/preston_20n').glob('*' + experiment))
        if len(candidates) != 1:
            raise RuntimeError('Expected one matching training run')
        run = candidates[0]
        state['training_run'] = str(run)
    shutil.copytree(snapshot, run / 'source_snapshot')
    coverage = json.loads((run / 'training_coverage.json').read_text())
    state['training_coverage'] = coverage
    if coverage['control_failures'] or (not args.smoke and coverage['path_completions'] < 2):
        raise RuntimeError('Training did not complete enough reference paths without control failures')
    checkpoint = run / 'checkpoints' / f'{"actor" if args.algorithm == "gsde" else "agent"}_{ticks}.pt'
    state['checkpoint'] = str(checkpoint)
    state['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with (run / 'final_inference_validation.json').open('w') as output:
        subprocess.run([str(python), 'scripts/skrl/ppo_policy_node.py', '--self-test',
                        '--checkpoint', str(checkpoint)], cwd=root, stdout=output, check=True)
    if args.smoke:
        state.update(phase='finished', runtime_smoke_passed=True, quality_passed=None)
        save()
        sys.exit(0)

    # Evaluate against a fresh constant-speed rollout under the same contract.
    evaluation = root / 'logs/velocity_evaluation' / ('preston_' + experiment + f'_{ticks}_full')
    state.update(phase='evaluation', evaluation=str(evaluation))
    save()
    command = [str(python), 'scripts/evaluate_adaptive_velocity.py', '--headless', '--device', 'cpu',
               '--checkpoint', str(checkpoint), '--min-cv-improvement-percent', '10',
               '--require-improvement', '--output', str(evaluation)]
    (workflow / 'evaluation_command.json').write_text(json.dumps(command, indent=2))
    with (workflow / 'evaluation.log').open('w') as output:
        result = subprocess.run(command, cwd=root, stdout=output, stderr=subprocess.STDOUT)
    summary = json.loads((evaluation / 'summary.json').read_text())
    state['comparison'] = summary.get('comparison')
    state['surface_comparison'] = summary.get('surface_comparison')
    passed = bool(state['comparison'] and state['comparison'].get('passed') and result.returncode == 0)
    state.update(phase='finished', quality_passed=passed, evaluation_exit_code=result.returncode)
    save()
    sys.exit(0 if passed else 1)
except Exception as error:
    state.update(phase='error', error=repr(error))
    save()
    traceback.print_exc()
    sys.exit(2)
