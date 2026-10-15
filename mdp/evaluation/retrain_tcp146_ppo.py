#!/usr/bin/env python3
"""Train and validate PPO against the unchanged UR10e USD at 146 mm TCP."""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(next(p / 'source/nrs_rl' for p in Path(__file__).resolve().parents
                              if (p / 'source/nrs_rl/nrs_rl').is_dir())))

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT, TASK_ROOT, REFERENCE
from nrs_rl.tasks.manager_based.nrs_rl.mdp.robot_geometry import TRAINING_CONFIG, configure_training_robot


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def invoke(study, label, command, timeout=10800):
    save(study / 'status.json', {'phase': label, 'started_at_utc': datetime.now(timezone.utc).isoformat()})
    save(study / (label + '_invocation.json'), {'command': command})
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1')
    with (study / (label + '.log')).open('x') as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=timeout)


def run(args):
    study = args.study.resolve()
    study.mkdir(parents=True, exist_ok=True)
    parent = TASK_ROOT / 'mdp/saved_models/ppo_final_cv11p15/candidate'
    controller = ROOT / 'logs/experiments/uniformity_saves/ppo_latest_force_cv7p80_20261012_215113/controller_source'
    scene = ROOT / 'logs/experiments/ppo_latest_force_retrain_20261012/gui_scene'
    native = ROOT / 'logs/experiments/ppo_latest_force_retrain_20261012/native'
    robot_config = study / 'isaac_ur10e_robot.yaml'
    if robot_config.exists():
        if sha(robot_config) != sha(TRAINING_CONFIG):
            raise ValueError('Study and current canonical robot configurations disagree')
    else:
        shutil.copyfile(TRAINING_CONFIG, robot_config)
    geometry = configure_training_robot(controller, robot_config)
    protocol = {'robot_geometry': geometry, 'scene_profile': str(scene),
        'source_usd_modified': False, 'reference_trajectory': str(REFERENCE),
        'reference_sha256': sha(REFERENCE), 'reference_trajectory_modified': False,
        'parent_actor': str(parent / 'checkpoints/actor.pt'),
        'parent_training_state': str(parent / 'checkpoints/ppo_training_state.pt'),
        'controller_snapshot': str(controller), 'native_dir': str(native), 'steady_gain': .2,
        'training_seed': 4701, 'selection_seed': 4710, 'heldout_seeds': [4711, 4712],
        'num_envs': 4, 'rollout_steps': 2048, 'updates': args.updates,
        'critic_warmup_updates': 2, 'learning_rate': 5.e-5,
        'candidate_updates': list(range(4, args.updates + 1, 2)),
        'initialization': 'Resume latest PPO actor, exploration and Adam; reset critic output for corrected geometry and warm up two batches',
        'validation': 'Same fixed controller, 146 mm and UR10e for all Mode 5/6 runs; candidate always in environment slot 3',
        'selection_metric': 'Processing spatial CV including zero-depth cells; first 2 s and initial 18 mm footprint excluded',
        'minimum_mode6_volume_ratio': .5, 'minimum_retrained_parent_volume_ratio': .9,
        'nominal_sensors': True, 'nominal_repeats_are_not_statistical_significance': True,
        'ros_deployment_changed': False}
    if (study / 'protocol.json').exists() and read(study / 'protocol.json') != protocol:
        raise ValueError('Existing study protocol differs; use a new study directory')
    save(study / 'protocol.json', protocol)
    frozen = {str(path): sha(path) for path in [
        TRAINING_CONFIG, robot_config, REFERENCE, parent / 'checkpoints/actor.pt',
        parent / 'checkpoints/ppo_training_state.pt', native / '_latest_force_mode3.so',
        scene / 'manifest.json', scene / 'robot.usdc', scene / 'workpiece.usdc',
        Path(read(scene / 'manifest.json')['source']),
        TASK_ROOT / 'mdp/robot_geometry.py', TASK_ROOT / 'mdp/steady_postcore.py',
        TASK_ROOT / 'mdp/isaac_latest_force_env.py', TASK_ROOT / 'mdp/action.py',
        TASK_ROOT / 'mdp/agents/train_latest_force_ppo.py',
        TASK_ROOT / 'mdp/evaluation/probe_tcp146_ppo.py',
        TASK_ROOT / 'mdp/evaluation/calibrate_latest_force_reward.py',
        TASK_ROOT / 'utils/assess_steady_force_ppo.py']}
    save(study / 'frozen_inputs.json', frozen)
    source = study / 'source_snapshot'
    source.mkdir(exist_ok=True)
    for name, digest in frozen.items():
        path = Path(name)
        if path.suffix == '.py':
            target = source / path.name
            if not target.exists():
                shutil.copyfile(path, target)
    save(study / 'process.json', {'pid': os.getpid(), 'phase': args.phase,
        'started_at_utc': datetime.now(timezone.utc).isoformat()})
    inputs = study / 'baseline_inputs/checkpoints'
    inputs.mkdir(parents=True, exist_ok=True)
    constant = ROOT / 'logs/experiments/ppo_latest_force_continue_20261013/ar_lowstd_validation/physics_candidates/checkpoints/constant.pt'
    for src, name in [(constant, 'constant.pt'), (parent / 'checkpoints/actor.pt', 'saved.pt'),
                      (parent / 'checkpoints/actor.pt', 'selected.pt')]:
        if not (inputs / name).exists():
            shutil.copyfile(src, inputs / name)
        elif sha(inputs / name) != sha(src):
            raise ValueError('Candidate input changed: ' + name)
    common = ['--headless', '--device', 'cpu', '--native-dir', str(native),
        '--control-snapshot', str(controller), '--robot-config', str(robot_config),
        '--gui-scene', str(scene), '--nominal-sensors', '--steady-gain', '.2']

    def probe(label, candidates, seed):
        result = study / label
        invoke(study, label, [sys.executable, str(Path(__file__).with_name('probe_tcp146_ppo.py')),
            *common, '--training-dir', str(candidates), '--output', str(result),
            '--names', 'constant,constant_b,saved,selected', '--seed', str(seed),
            '--hard-force-stop-n', '100', '--uniformity-only'], timeout=3600)
        report = read(result / 'processing_summary.json')
        if not report['duplicate_mode6_verified']:
            raise ValueError('Paired Mode 6 completion/repeatability failed: ' + label)
        return report

    baseline_dir = study / 'baseline_ur10e'
    if (baseline_dir / 'processing_summary.json').exists():
        baseline = read(baseline_dir / 'processing_summary.json')
        if not baseline['duplicate_mode6_verified']:
            raise ValueError('The existing baseline did not pass')
    else:
        baseline = probe('baseline_ur10e', inputs.parent, 4700)
    if not baseline['candidates']['selected']['eligible']:
        raise ValueError('Existing actor failed at corrected geometry; inspect before PPO training')
    if args.phase == 'baseline':
        save(study / 'status.json', {'phase': 'baseline_complete', 'baseline': str(baseline_dir)})
        return

    forecast = study / 'reward_forecast'
    if not (forecast / 'calibration.json').exists():
        invoke(study, 'calibration', [sys.executable,
            str(Path(__file__).with_name('calibrate_latest_force_reward.py')),
            '--source', str(baseline_dir), '--output', str(forecast), '--cell-mm', '.5'])
    training = study / 'training'
    if not (training / 'training_complete.json').exists():
        bias = -100. * baseline['candidates']['selected']['spatial_cv'] ** 2
        invoke(study, 'training', [sys.executable, str(TASK_ROOT / 'mdp/agents/train_latest_force_ppo.py'),
            *common, '--output', str(training), '--resume', str(parent / 'checkpoints/ppo_training_state.pt'),
            '--seed', '4701', '--num-envs', '4', '--rollout-steps', '2048',
            '--updates', str(args.updates), '--save-interval', '1', '--learning-rate', '0.00005',
            '--max-marginal-kl', '.025', '--gae-lambda', '1', '--reward-cell-mm', '.5',
            '--exploration-rho', '.95', '--reward-forecast', str(forecast / 'forecast.npz'),
            '--reset-critic-output', '--critic-potential-residual', '--critic-residual-bias', str(bias),
            '--critic-warmup-updates', '2', '--restart-episodes-each-update', '--actor-complete-returns-only'])
    completion = read(training / 'training_complete.json')
    if not completion['completed'] or completion['actor_updates'] != args.updates - 2:
        raise ValueError('Requested PPO actor updates did not complete')
    candidates = {}
    for update in protocol['candidate_updates']:
        name = f'ppo_{update:04d}'
        directory = study / (name + '_inputs/checkpoints')
        directory.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(inputs / 'constant.pt', directory / 'constant.pt')
        shutil.copyfile(inputs / 'saved.pt', directory / 'saved.pt')
        shutil.copyfile(training / 'checkpoints' / (name + '.pt'), directory / 'selected.pt')
        report = probe('selection_' + name, directory.parent, 4710)
        row = report['candidates']['selected']
        reference = baseline['candidates']['selected']
        volume_ratios = {key: row[key] / reference[key] for key in
                         ('roi_volume_over_k', 'full_roi_volume_over_k', 'whole_grid_volume_over_k')}
        candidates[name] = {'update': update, 'metrics': row,
            'parent_volume_ratios': volume_ratios,
            'eligible': row['eligible'] and min(volume_ratios.values()) >= .9}
        save(study / 'selection.json', candidates)
    valid = {key: value for key, value in candidates.items() if value['eligible']}
    if not valid:
        raise ValueError('No retrained candidate passed completion and volume constraints')
    chosen = min(valid, key=lambda key: valid[key]['metrics']['spatial_cv'])
    update = valid[chosen]['update']
    bundle = study / 'candidate'
    (bundle / 'checkpoints').mkdir(parents=True, exist_ok=False)
    shutil.copyfile(training / 'checkpoints' / (chosen + '.pt'), bundle / 'checkpoints/actor.pt')
    shutil.copyfile(training / 'checkpoints' / f'training_{update:04d}.pt', bundle / 'checkpoints/ppo_training_state.pt')
    save(study / 'frozen_selection.json', {'candidate': chosen,
        'actor_sha256': sha(bundle / 'checkpoints/actor.pt'), 'heldout_seeds': [4711, 4712]})
    heldout = {str(seed): probe(f'heldout_{seed}', study / (chosen + '_inputs'), seed)
               for seed in [4711, 4712]}
    if not all(report['candidates']['selected']['eligible'] for report in heldout.values()):
        raise ValueError('Selected actor failed independent validation')
    changed = [name for name, digest in frozen.items() if sha(name) != digest]
    if changed:
        raise ValueError('Frozen input changed during training: ' + str(changed))
    result = {'status': 'training_and_validation_complete', 'robot_geometry': geometry,
        'selected_candidate': chosen, 'checkpoint': str(bundle / 'checkpoints/actor.pt'),
        'checkpoint_sha256': sha(bundle / 'checkpoints/actor.pt'),
        'training_state': str(bundle / 'checkpoints/ppo_training_state.pt'),
        'training': completion, 'baseline': baseline, 'selection': valid[chosen], 'heldout': heldout,
        'mean_mode6_cv_improvement_percent': sum(100. * r['candidates']['selected']['mode6_cv_gain']
                                                for r in heldout.values()) / len(heldout),
        'source_usd_modified': False, 'reference_trajectory_modified': False,
        'ros_deployed': False, 'frozen_inputs_verified': True,
        'finished_at_utc': datetime.now(timezone.utc).isoformat()}
    for src in [study / 'protocol.json', training / 'protocol.json', robot_config]:
        target = 'training_protocol.json' if src.parent == training else src.name
        shutil.copyfile(src, bundle / target)
    save(study / 'final_report.json', result)
    save(study / 'status.json', {'phase': 'complete', 'checkpoint': result['checkpoint'],
        'mean_mode6_cv_improvement_percent': result['mean_mode6_cv_improvement_percent']})
    print(json.dumps({'status': result['status'], 'checkpoint': result['checkpoint'],
        'mode6_cv_improvement_percent': result['mean_mode6_cv_improvement_percent']}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--phase', choices=['baseline', 'all'], default='all')
    parser.add_argument('--updates', type=int, default=10)
    args = parser.parse_args()
    if args.updates < 4 or args.updates % 2:
        parser.error('Use an even number of batches >= 4, including two critic warmup batches')
    try:
        run(args)
    except Exception:
        args.study.mkdir(parents=True, exist_ok=True)
        save(args.study / 'failure.json', {'error': traceback.format_exc(),
             'time_utc': datetime.now(timezone.utc).isoformat()})
        save(args.study / 'status.json', {'phase': 'failed', 'see': 'failure.json'})
        raise
