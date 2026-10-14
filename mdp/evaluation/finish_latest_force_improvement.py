#!/usr/bin/env python3
"""Select a PPO improvement over the saved actor, then validate a frozen candidate.

All comparisons use the preserved latest controller in isolated headless Isaac.
This command never changes the deployed ROS actor or its controller.
"""
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

from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT
from nrs_rl.tasks.manager_based.nrs_rl.utils.assess_latest_force_ppo import evaluate
from nrs_rl.tasks.manager_based.nrs_rl.utils.removal_metrics import save, sha


SCOPE = 'independent headless Isaac, nominal sensors; ROS performance not established'


def read_json(path):
    return json.loads(Path(path).read_text())


def resolve_path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def protected_audit(old_save):
    files = read_json(old_save / 'SAVE.json')['files_sha256']
    for relative, digest in files.items():
        if sha(old_save / relative) != digest:
            raise ValueError('Protected 7.80% save changed: ' + relative)
    return {'protected_7p80_save_unchanged': True, 'protected_files_checked': len(files)}


def reference_audit(protocol, old_save):
    """A saved experimental incumbent can differ from the protected ROS actor."""
    actor = resolve_path(protocol.get('reference_actor', str(old_save / 'checkpoints/actor.pt')))
    digest = sha(actor)
    expected = protocol.get('reference_actor_sha256')
    if expected is not None and digest != expected:
        raise ValueError('Reference actor changed: ' + str(actor))
    checked = 0
    if protocol.get('reference_bundle'):
        bundle = resolve_path(protocol['reference_bundle'])
        for relative, expected_digest in read_json(bundle / 'manifest.json').items():
            if sha(bundle / relative) != expected_digest:
                raise ValueError('Preserved incumbent bundle changed: ' + relative)
            checked += 1
    return {'reference_actor': str(actor), 'reference_actor_sha256': digest,
            'reference_label': protocol.get('reference_label', 'protected ROS actor'),
            'reference_bundle_files_checked': checked}


def controller_audit(snapshot, old_save, native):
    checked = 0
    for relative, digest in read_json(old_save / 'SAVE.json')['files_sha256'].items():
        if relative.startswith('controller_source/'):
            local = relative.removeprefix('controller_source/')
            if sha(snapshot / local) != digest:
                raise ValueError('Evaluation controller differs from protected source: ' + local)
            checked += 1
    if not checked:
        raise ValueError('Protected save contains no controller sources')
    validation = read_json(native / 'validation.json')
    if (not validation['ok'] or validation['tangent_damping_ns_m'] != 6000.
            or validation['tangent_stiffness_n_m'] != 2000.
            or validation['tracking_compensation']):
        raise ValueError('Unexpected native force controller profile')
    for source, digest in validation['source_sha256'].items():
        if sha(resolve_path(source)) != digest:
            raise ValueError('Native controller source hash changed: ' + source)
    return {'force_controller_unchanged': True, 'controller_source_files_checked': checked,
            'native_controller_sha256': sha(native / '_latest_force_mode3.so')}


def matching_training_pids(training):
    """Read process arguments, never signal or modify another process."""
    matches = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            argv = (directory / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            if not any(Path(part).name == 'train_latest_force_ppo.py' for part in argv):
                continue
            if '--output' not in argv:
                continue
            index = argv.index('--output') + 1
            target = Path(argv[index])
            if not target.is_absolute():
                target = (directory / 'cwd').resolve() / target
            if target.resolve() == training:
                matches.append(int(directory.name))
        except (OSError, IndexError, ValueError):
            continue
    return matches


def wait_for_training(training, status, args):
    started = time.monotonic()
    previously_seen = False
    missing_polls = 0
    while True:
        complete = training / 'training_complete.json'
        if complete.is_file():
            try:
                completed = read_json(complete)
            except json.JSONDecodeError:
                completed = None  # The trainer may currently be writing the file.
            if completed is not None:
                if not completed.get('completed', False):
                    raise RuntimeError('Training completion explicitly reports failure')
                return completed
        for filename in ('error.json', 'training_error.json', 'failed.json'):
            error = training / filename
            if error.is_file():
                raise RuntimeError('Training reported failure: ' + str(error))
        elapsed = time.monotonic() - started
        if elapsed > args.wait_timeout_seconds:
            raise TimeoutError('Training completion was not produced before the deadline')
        pids = matching_training_pids(training)
        previously_seen |= bool(pids)
        missing_polls = 0 if pids else missing_polls + 1
        if not pids and missing_polls >= 2 and (previously_seen or elapsed > args.wait_start_grace_seconds):
            raise RuntimeError('No matching trainer process remains and training is incomplete: ' + str(training))
        save(status, {'phase': 'waiting_for_training', 'training': str(training),
                      'trainer_pids': pids, 'elapsed_seconds': round(elapsed, 1)})
        time.sleep(min(10., args.wait_timeout_seconds - elapsed))


def training_provenance(state_path):
    """Follow hashed resume states so warm-start transforms cannot disappear."""
    import torch
    chain = []
    transforms = []
    visited = set()
    path = state_path.resolve()
    expected = None
    for _ in range(32):
        if path in visited:
            raise ValueError('Cycle in PPO resume-state provenance')
        visited.add(path)
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError('PPO parent training-state hash changed: ' + str(path))
        state = torch.load(path, weights_only=False, map_location='cpu')
        protocol = state['protocol']
        parent = protocol.get('parent_training') or {}
        row = {'training_state': str(path), 'training_state_sha256': digest,
               'update': state.get('update'), 'algorithm': protocol.get('algorithm'),
               'seed': protocol.get('seed'), 'parent_training': parent}
        for container_name, container in (('state', state), ('protocol', protocol),
                                           ('parent_training', parent)):
            if 'warm_start_transform' in container:
                transform = {'source_training_state': str(path), 'container': container_name,
                             'warm_start_transform': container['warm_start_transform']}
                transforms.append(transform)
                row[container_name + '_warm_start_transform'] = container['warm_start_transform']
        chain.append(row)
        if not parent.get('training_state'):
            break
        path = resolve_path(parent['training_state'])
        expected = parent.get('training_state_sha256')
    else:
        raise ValueError('PPO resume-state provenance exceeds 32 stages')
    return {'resume_state_chain': chain, 'warm_start_transforms': transforms}


def actor_state_audit(actor_path, state_path):
    import torch
    from nrs_rl.tasks.manager_based.nrs_rl.mdp.agents.latest_force_ppo import PPOActorCritic, check_contract
    torch.set_num_threads(1)
    extra = {'fixed_force_policy.json': ''}
    actor = torch.jit.load(str(actor_path), _extra_files=extra).eval()
    contract = json.loads(extra['fixed_force_policy.json'])
    check_contract(contract)
    state = torch.load(state_path, weights_only=False, map_location='cpu')
    if (state['protocol']['algorithm'] != 'PPO-Clip' or state.get('update', 0) <= 0
            or contract.get('ppo_updates', 0) <= 0 or contract.get('initialization_only', False)):
        raise ValueError('Candidate must contain completed PPO updates, not initialization only')
    model = PPOActorCritic()
    model.load_state_dict(state['model'])
    model.eval()
    generator = torch.Generator().manual_seed(4199)
    with torch.no_grad():
        observations = torch.randn(4096, 16, generator=generator) * 2.
        torch.testing.assert_close(actor(observations),
            (.125 + .875 * model.actor(observations).tanh())[:, None], atol=0., rtol=0.)
    return {'actor_matches_training_state': True, 'exact_export_probe_count': 4096,
            'actor_contract': contract, 'training_provenance': training_provenance(state_path)}


def selection_decision(result, names, minimum_gain_pp):
    saved = result['candidates']['saved']
    if not saved['eligible']:
        raise ValueError('Saved reference actor failed eligibility')
    valid = {name: row for name, row in result['candidates'].items()
             if name in names and row['eligible']}
    if not valid:
        return {'selected': None, 'selection_passed': False,
                'reason': 'no_eligible_retrained_candidate'}
    selected = max(valid, key=lambda name: valid[name]['mode6_cv_gain'])
    gain = 100. * valid[selected]['mode6_cv_gain']
    reference = 100. * saved['mode6_cv_gain']
    return {'selected': selected, 'selection_passed': gain - reference > minimum_gain_pp + 1.e-10,
            'candidate_improvement_percent': gain, 'saved_actor_improvement_percent': reference,
            'gain_over_saved_percentage_points': gain - reference,
            'minimum_selection_gain_percentage_points': minimum_gain_pp,
            'selection_requirement': 'candidate gain minus paired saved gain strictly exceeds threshold'}


def finish(args):
    study = args.study.resolve()
    training = args.training.resolve()
    protocol = read_json(study / 'protocol.json')
    old_save = resolve_path(protocol['protected_save'])
    snapshot = resolve_path(protocol['controller_source'])
    native = resolve_path(protocol['native_dir'])
    scene = args.gui_scene or protocol.get('scene_profile')
    scene = resolve_path(scene) if scene is not None else None
    minimum_gain_pp = float(protocol.get('minimum_selection_gain_pp', .10))
    if minimum_gain_pp < 0.:
        raise ValueError('minimum_selection_gain_pp cannot be negative')
    if not protocol.get('selection_nominal_sensors', False):
        raise ValueError('This saved-actor comparison requires selection_nominal_sensors=true')
    seeds = [int(seed) for seed in protocol['heldout_seeds']]
    if len(seeds) != 2 or len(set(seeds + [int(protocol['selection_seed'])])) != 3:
        raise ValueError('Use two distinct heldout seeds, separate from selection')
    output = study / args.name
    output.mkdir(exist_ok=False)
    status = output / 'status.json'
    context = {'algorithm': 'PPO', 'scope': SCOPE, 'active_deployment_replaced': False,
               'training_directory': str(training), 'nominal_sensors_all_evaluations': True,
               'minimum_selection_gain_percentage_points': minimum_gain_pp,
               'heldout_seeds_consumed': [], 'independent_validation_complete': False,
               'improvement_validated': False}
    initial_controller_audit = None
    initial_reference_audit = reference_audit(protocol, old_save)
    context.update(initial_reference_audit)

    def write_final(phase, **values):
        control = controller_audit(snapshot, old_save, native)
        if reference_audit(protocol, old_save) != initial_reference_audit:
            raise ValueError('Reference actor or bundle changed during validation')
        if initial_controller_audit is not None and control != initial_controller_audit:
            raise ValueError('Controller source or native binary changed during validation')
        report = {**context, **values, 'phase': phase,
                  'finished_at_utc': datetime.now(timezone.utc).isoformat(),
                  **protected_audit(old_save), **control}
        save(output / 'final_report.json', report)
        save(status, {key: report[key] for key in ('phase', 'improvement_validated',
                     'independent_validation_complete', 'heldout_seeds_consumed')})
        print('PPO_IMPROVEMENT_RESULT', json.dumps({key: report.get(key) for key in
              ('phase', 'improvement_validated', 'mean_heldout_cv_improvement_percent',
               'mean_saved_actor_improvement_percent', 'mean_gain_over_saved_percentage_points',
               'heldout_seeds_consumed')}), flush=True)
        return report

    try:
        context.update(protected_audit(old_save))
        initial_controller_audit = controller_audit(snapshot, old_save, native)
        context.update(initial_controller_audit)
        completion = wait_for_training(training, status, args)
        context['training_completion'] = completion
        checkpoint_dir = output / 'physics_candidates/checkpoints'
        checkpoint_dir.mkdir(parents=True)
        constant = resolve_path(protocol.get('constant_actor',
            str(native.parent / 'physics_candidates/checkpoints/constant.pt')))
        shutil.copy2(constant, checkpoint_dir / 'constant.pt')
        shutil.copy2(initial_reference_audit['reference_actor'], checkpoint_dir / 'saved.pt')
        names = ['constant', 'constant_b', 'saved']
        sources = {}
        for update in sorted(set(args.updates)):
            name = f'ppo_{update:04d}'
            path = training / 'checkpoints' / f'{name}.pt'
            state = training / 'checkpoints' / f'training_{update:04d}.pt'
            if not path.is_file() or not state.is_file():
                raise ValueError('Missing completed candidate ' + name)
            shutil.copy2(path, checkpoint_dir / path.name)
            sources[name] = {'actor': str(path), 'state': str(state), 'actor_sha256': sha(path),
                             'training_state_sha256': sha(state)}
            names.append(name)
        save(output / 'candidate_sources.json', sources)
        shutil.copy2(study / 'protocol.json', output / 'study_protocol.json')
        shutil.copy2(training / 'protocol.json', output / 'training_protocol.json')
        shutil.copy2(Path(__file__), output / Path(__file__).name)
        environment = dict(os.environ, Y2_CONTROL_SOURCE_DIR=str(snapshot),
                           Y2_ROBOT_CONFIG=str(snapshot / 'Y2RobMotion/config/ppo_fixed_force.yaml'))
        probe = Path(__file__).with_name('probe_latest_force_ppo.py')

        def run_probe(label, seed, candidates):
            directory = output / label
            command = [sys.executable, str(probe), '--headless', '--device', 'cpu',
                '--training-dir', str(checkpoint_dir.parent), '--output', str(directory),
                '--names', ','.join(candidates), '--seed', str(seed),
                '--native-dir', str(native), '--control-snapshot', str(snapshot),
                '--hard-force-stop-n', '100', '--uniformity-only', '--nominal-sensors']
            if scene is not None:
                command += ['--gui-scene', str(scene)]
            save(output / (label + '_invocation.json'), {'command': command, 'seed': seed})
            with (output / (label + '.log')).open('w') as stream:
                subprocess.run(command, cwd=ROOT, env=environment, stdout=stream,
                               stderr=subprocess.STDOUT, check=True)
            result = evaluate(directory, audit=True)
            baseline = result['candidates']['constant']
            duplicate = result['candidates']['constant_b']
            if not (baseline['eligible'] and duplicate['eligible']):
                raise ValueError('Mode 6 reference or duplicate failed eligibility: ' + label)
            relative_difference = abs(duplicate['spatial_cv'] / baseline['spatial_cv'] - 1.)
            if relative_difference >= .005:
                raise ValueError('Duplicate Mode 6 CV must differ by less than 0.5%: ' + label)
            result['duplicate_mode6_relative_cv_difference'] = relative_difference
            result['nominal_sensors'] = True
            return result

        save(status, {'phase': 'selection_evaluation', 'seed': protocol['selection_seed'], 'names': names})
        selection = run_probe('selection', int(protocol['selection_seed']), names)
        decision = selection_decision(selection, sources, minimum_gain_pp)
        context.update(selection_decision=decision, selection_report=selection)
        save(output / 'selection_report.json', {'decision': decision, 'metrics': selection})
        if not decision['selection_passed']:
            return write_final('selection_below_improvement_threshold')
        selected = decision['selected']
        candidate = output / 'candidate'
        (candidate / 'checkpoints').mkdir(parents=True)
        actor_path = candidate / 'checkpoints/actor.pt'
        state_path = candidate / 'checkpoints/ppo_training_state.pt'
        shutil.copy2(sources[selected]['actor'], actor_path)
        shutil.copy2(sources[selected]['state'], state_path)
        audit = actor_state_audit(actor_path, state_path)
        save(candidate / 'training_provenance.json', audit['training_provenance'])
        shutil.copy2(actor_path, checkpoint_dir / 'selected.pt')
        frozen = {'selected': selected, 'actor_sha256': sha(actor_path),
            'training_state_sha256': sha(state_path), 'selection_seed': protocol['selection_seed'],
            'selection_decision': decision, 'selection_metrics': selection['candidates'][selected],
            'heldout_seeds_not_yet_run': seeds, 'scope': SCOPE,
            'frozen_at_utc': datetime.now(timezone.utc).isoformat(), **audit}
        save(candidate / 'frozen_selection.json', frozen)
        context['selected'] = frozen
        if args.selection_only:
            return write_final('selection_passed_heldouts_not_requested')
        save(status, {'phase': 'independent_validation', 'selected': selected,
                      'heldout_seeds': seeds, 'frozen_actor_sha256': frozen['actor_sha256']})
        context['heldout_seeds_consumed'] = seeds
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_probe, f'heldout_{seed}', seed,
                                   ['constant', 'constant_b', 'saved', 'selected']) for seed in seeds]
            reports = [future.result() for future in futures]
        if (sha(actor_path) != frozen['actor_sha256']
                or sha(state_path) != frozen['training_state_sha256']
                or sha(checkpoint_dir / 'selected.pt') != frozen['actor_sha256']):
            raise ValueError('Frozen candidate changed during independent validation')
        gains = [100. * row['candidates']['selected']['mode6_cv_gain'] for row in reports]
        references = [100. * row['candidates']['saved']['mode6_cv_gain'] for row in reports]
        differences = [gain - reference for gain, reference in zip(gains, references)]
        eligible = [row['candidates']['selected']['eligible'] and row['candidates']['saved']['eligible']
                    for row in reports]
        passed = all(eligible) and all(difference > 0. for difference in differences)
        report = write_final('finished', independent_validation_complete=True,
            improvement_validated=passed, numerical_audit_passed=True,
            actor_matches_training_state=True, force_controller_unchanged=True,
            eligible_all_seeds=all(eligible), candidate_beats_saved_all_seeds=all(x > 0. for x in differences),
            mean_heldout_cv_improvement_percent=sum(gains) / len(gains),
            mean_saved_actor_improvement_percent=sum(references) / len(references),
            mean_gain_over_saved_percentage_points=sum(differences) / len(differences),
            per_seed_gain_over_saved_percentage_points=dict(zip(map(str, seeds), differences)),
            per_seed={str(seed): row for seed, row in zip(seeds, reports)})
        save(candidate / 'validation.json', report)
        save(candidate / 'manifest.json', {str(path.relative_to(candidate)): sha(path)
             for path in candidate.rglob('*') if path.is_file()})
        return report
    except Exception:
        error = traceback.format_exc()
        save(output / 'error.json', {'error': error})
        failure = {**context, 'phase': 'failed', 'error': error,
                   'improvement_validated': False, 'independent_validation_complete': False}
        try:
            failure.update(protected_audit(old_save))
        except Exception as preservation_error:
            failure.update(protected_7p80_save_unchanged=False,
                           preservation_error=str(preservation_error))
        save(output / 'final_report.json', failure)
        save(status, {'phase': 'failed', 'error': error})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--updates', nargs='+', type=int, default=[4, 8])
    parser.add_argument('--gui-scene', type=Path)
    parser.add_argument('--selection-only', action='store_true')
    parser.add_argument('--wait-timeout-seconds', type=float, default=10800.)
    parser.add_argument('--wait-start-grace-seconds', type=float, default=120.)
    args = parser.parse_args()
    if (Path(args.name).name != args.name or args.name in ('.', '..')
            or any(update <= 0 for update in args.updates)
            or args.wait_timeout_seconds <= 0. or args.wait_start_grace_seconds < 0.):
        parser.error('Use a plain unique name, positive checkpoint updates, and valid wait durations')
    finish(args)
