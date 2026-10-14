#!/usr/bin/env python3
"""Select latest-force neural PPO on one seed, freeze it, then assess untouched seeds."""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(next(p / 'source/nrs_rl' for p in Path(__file__).resolve().parents
                              if (p / 'source/nrs_rl/nrs_rl').is_dir())))

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from nrs_rl.tasks.manager_based.nrs_rl.utils.assess_latest_force_ppo import evaluate
from nrs_rl.tasks.manager_based.nrs_rl.utils.removal_metrics import save, sha
from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT


def finish(args):
    study = args.study.resolve()
    training = args.training.resolve()
    protocol = json.loads((study / 'protocol.json').read_text())
    old_save = Path(protocol['protected_save'])
    snapshot = old_save / 'controller_source'
    target = protocol['target_processing_cv_improvement_percent']
    output = study / args.name
    output.mkdir(exist_ok=False)
    status = output / 'status.json'
    deadline = time.monotonic() + 10800.
    while not (training / 'training_complete.json').exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Training completion was not produced')
        save(status, {'phase': 'waiting_for_training', 'training': str(training)})
        time.sleep(10.)
    checkpoint_dir = output / 'physics_candidates/checkpoints'
    checkpoint_dir.mkdir(parents=True)
    shutil.copy2(study / 'physics_candidates/checkpoints/constant.pt', checkpoint_dir / 'constant.pt')
    shutil.copy2(old_save / 'checkpoints/actor.pt', checkpoint_dir / 'saved.pt')
    names = ['constant', 'constant_b', 'saved']
    sources = {}
    for update in args.updates:
        name = f'ppo_{update:04d}'
        path = training / 'checkpoints' / f'{name}.pt'
        state = training / 'checkpoints' / f'training_{update:04d}.pt'
        if not path.is_file() or not state.is_file():
            raise ValueError('Missing completed candidate ' + name)
        shutil.copy2(path, checkpoint_dir / path.name)
        sources[name] = {'actor': str(path), 'state': str(state), 'sha256': sha(path)}
        names.append(name)
    save(output / 'candidate_sources.json', sources)
    environment = dict(os.environ, Y2_CONTROL_SOURCE_DIR=str(snapshot),
                       Y2_ROBOT_CONFIG=str(snapshot / 'Y2RobMotion/config/ppo_fixed_force.yaml'))
    probe = Path(__file__).with_name('probe_latest_force_ppo.py')

    def run_probe(label, seed, candidates):
        directory = output / label
        command = [sys.executable, str(probe), '--headless', '--device', 'cpu',
            '--training-dir', str(checkpoint_dir.parent), '--output', str(directory),
            '--names', ','.join(candidates), '--seed', str(seed),
            '--native-dir', str(study / 'native'), '--control-snapshot', str(snapshot),
            '--hard-force-stop-n', '100', '--uniformity-only']
        if args.gui_scene is not None:
            command += ['--gui-scene', str(args.gui_scene.resolve())]
        if label == 'selection' and protocol.get('selection_nominal_sensors', False):
            command.append('--nominal-sensors')
        with (output / (label + '.log')).open('w') as stream:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        result = evaluate(directory, audit=True)
        baseline = result['candidates']['constant']
        duplicate = result['candidates']['constant_b']
        if abs(duplicate['spatial_cv'] / baseline['spatial_cv'] - 1.) > .005:
            raise ValueError('Duplicate Mode 6 differs by more than 0.5%')
        return result

    save(status, {'phase': 'selection_evaluation', 'seed': protocol['selection_seed'], 'names': names})
    selection = run_probe('selection', protocol['selection_seed'], names)
    valid = {n: row for n, row in selection['candidates'].items()
             if n in sources and row['eligible']}
    if not valid:
        raise ValueError('No complete, fault-free PPO candidate passed processing criteria')
    selected = max(valid, key=lambda n: valid[n]['mode6_cv_gain'])
    if 100. * valid[selected]['mode6_cv_gain'] < target - .3:
        report = {'phase': 'selection_below_target', 'best_candidate': selected,
            'improvement_percent': 100. * valid[selected]['mode6_cv_gain'],
            'target_improvement_percent': target, 'heldout_seeds_consumed': [],
            'selection_report': selection, 'independent_validation_complete': False}
        save(output / 'selection_report.json', report)
        save(status, report)
        print('PPO_SELECTION_BELOW_TARGET', json.dumps(report), flush=True)
        return
    candidate = output / 'candidate'
    (candidate / 'checkpoints').mkdir(parents=True)
    shutil.copy2(sources[selected]['actor'], candidate / 'checkpoints/actor.pt')
    shutil.copy2(sources[selected]['state'], candidate / 'checkpoints/ppo_training_state.pt')
    shutil.copy2(candidate / 'checkpoints/actor.pt', checkpoint_dir / 'selected.pt')
    frozen = {'selected': selected, 'actor_sha256': sha(candidate / 'checkpoints/actor.pt'),
        'training_state_sha256': sha(candidate / 'checkpoints/ppo_training_state.pt'),
        'selection_seed': protocol['selection_seed'], 'selection_metrics': valid[selected],
        'heldout_seeds_not_yet_run': protocol['heldout_seeds'],
        'frozen_at_utc': datetime.now(timezone.utc).isoformat()}
    save(candidate / 'frozen_selection.json', frozen)
    print('PPO_FROZEN_SELECTION', json.dumps(frozen), flush=True)
    save(status, {'phase': 'independent_validation', **frozen})
    seeds = protocol['heldout_seeds']
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_probe, f'heldout_{seed}', seed,
                               ['constant', 'constant_b', 'saved', 'selected']) for seed in seeds]
        reports = [f.result() for f in futures]
    assert sha(candidate / 'checkpoints/actor.pt') == frozen['actor_sha256']
    rows = [r['candidates']['selected'] for r in reports]
    old_rows = [r['candidates']['saved'] for r in reports]
    gains = [100. * r['mode6_cv_gain'] for r in rows]
    import torch
    from nrs_rl.tasks.manager_based.nrs_rl.mdp.agents.latest_force_ppo import PPOActorCritic, check_contract
    torch.set_num_threads(1)
    extra = {'fixed_force_policy.json': ''}
    actor = torch.jit.load(str(candidate / 'checkpoints/actor.pt'), _extra_files=extra).eval()
    contract = json.loads(extra['fixed_force_policy.json'])
    check_contract(contract)
    state = torch.load(candidate / 'checkpoints/ppo_training_state.pt', weights_only=False, map_location='cpu')
    model = PPOActorCritic()
    model.load_state_dict(state['model'])
    torch.manual_seed(4199)
    with torch.no_grad():
        observations = torch.randn(4096, 16) * 2.
        torch.testing.assert_close(actor(observations),
            (.125 + .875 * model.actor(observations).tanh())[:, None], atol=0., rtol=0.)
    original = json.loads((old_save / 'SAVE.json').read_text())
    for relative, digest in original['files_sha256'].items():
        if sha(old_save / relative) != digest:
            raise ValueError('Protected 7.80% save changed: ' + relative)
    report = {'algorithm': 'PPO', 'scope': 'independent headless Isaac; ROS trial still required',
        'numerical_audit_passed': True, 'actor_matches_training_state': True,
        'protected_7p80_save_unchanged': True, 'force_controller_unchanged': True,
        'active_deployment_replaced': False, 'target_improvement_percent': target,
        'mean_heldout_cv_improvement_percent': sum(gains) / len(gains),
        'per_seed_improvement_percent': dict(zip(seeds, gains)),
        'mean_saved_actor_improvement_percent': sum(100. * r['mode6_cv_gain'] for r in old_rows) / len(old_rows),
        'eligible_all_seeds': all(r['eligible'] for r in rows),
        'target_met': all(r['eligible'] for r in rows) and sum(gains) / len(gains) >= target,
        'selected': frozen, 'per_seed': {str(seed): r for seed, r in zip(seeds, reports)}}
    save(candidate / 'validation.json', report)
    save(output / 'final_report.json', report)
    manifest = {str(f.relative_to(candidate)): sha(f) for f in candidate.rglob('*') if f.is_file()}
    save(candidate / 'manifest.json', manifest)
    save(status, {'phase': 'finished', 'target_met': report['target_met'],
        'mean_heldout_cv_improvement_percent': report['mean_heldout_cv_improvement_percent']})
    print('PPO_VALIDATION_COMPLETE', json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--updates', nargs='+', type=int, default=[4, 8])
    parser.add_argument('--gui-scene', type=Path)
    a = parser.parse_args()
    try:
        finish(a)
    except Exception:
        error = traceback.format_exc()
        directory = a.study / a.name
        if directory.is_dir():
            save(directory / 'error.json', {'error': error})
        raise
