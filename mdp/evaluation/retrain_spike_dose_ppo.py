"""Wait for the authorized retraining, then select and validate its transient response."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from ...paths import ROOT, TASK_ROOT, REFERENCE
from ..spike_dose_reward import SpikeDoseConfig
from .assess_spike_dose_ppo import assess, compare, save


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def run(study, archive):
    invocation = read(study / 'training_invocation.json')
    plan = invocation['plan']
    args = invocation['command']
    argument = lambda name: args[args.index(name) + 1]
    config = SpikeDoseConfig.load(study / 'spike_dose_reward.json')
    parent = Path(invocation['parent_actor'])
    constant = ROOT / 'logs/experiments/ppo_latest_force_continue_20261013/ar_lowstd_validation/physics_candidates/checkpoints/constant.pt'
    if sha(parent) != invocation['parent_actor_sha256']:
        raise ValueError('Parent actor changed since training started')
    files = [parent, constant, Path(argument('--resume')), REFERENCE,
             study / 'isaac_ur10e_robot.yaml', study / 'spike_dose_reward.json',
             TASK_ROOT / 'mdp/spike_dose_reward.py', TASK_ROOT / 'mdp/isaac_latest_force_env.py',
             TASK_ROOT / 'mdp/agents/train_latest_force_ppo.py', TASK_ROOT / 'mdp/evaluation/probe_tcp146_ppo.py',
             TASK_ROOT / 'mdp/agents/spike_ppo_auxiliary.py',
             TASK_ROOT / 'mdp/evaluation/assess_spike_dose_ppo.py', Path(__file__),
             TASK_ROOT / 'utils/reporting/plot_spike_retraining.py',
             Path(argument('--gui-scene')) / 'robot.usdc', Path(argument('--gui-scene')) / 'workpiece.usdc',
             Path(argument('--native-dir')) / '_latest_force_mode3.so']
    frozen = {str(path): sha(path) for path in files}
    if (study / 'validation_frozen_inputs.json').exists():
        if read(study / 'validation_frozen_inputs.json') != frozen:
            raise ValueError('Validation inputs changed; use a new study')
    else:
        save(study / 'validation_frozen_inputs.json', frozen)
    protocol = {'training': plan, 'reward': config.metadata(),
        'parent_actor_sha256': sha(parent), 'candidate_selection': 'minimum transient absolute-error time integral among quality-eligible candidates',
        'quality_tolerance': {'max_relative_contact_cv_increase': .01, 'max_relative_processing_cv_increase': .01,
                              'minimum_parent_volume_ratio': .9, 'minimum_spike_error_improvement_percent': 5.},
        'matching': 'constant,constant_b,parent,candidate; parent reference rerun in candidate slot 3 at every comparison seed',
        'full_contact_metrics': 'include approach from first measured 1.5 N contact; exclude pre-contact',
        'nominal_repeatability_not_statistical_significance': True, 'ros_deployment_changed': False,
        'archive': str(archive)}
    save(study / 'validation_protocol.json', protocol)
    source = study / 'validation_source_snapshot'
    source.mkdir(exist_ok=True)
    for name in frozen:
        path = Path(name)
        if path.suffix == '.py':
            shutil.copy2(path, source / path.name)
    while not (study / 'training/training_complete.json').exists():
        runner = read(study / 'training_process.json')['runner_pid']
        exit_file = study / 'training_exit.json'
        if exit_file.exists() and not read(exit_file)['ok']:
            raise RuntimeError('Training runner reported failure')
        try:
            os.kill(runner, 0)
        except ProcessLookupError:
            raise RuntimeError('Training exited without a completion artifact')
        save(study / 'validation_status.json', {'phase': 'waiting_for_training', 'checked_utc': datetime.now(timezone.utc).isoformat()})
        time.sleep(20.)
    complete = read(study / 'training/training_complete.json')
    if (not complete['completed'] or complete['actor_updates'] != plan['updates'] - plan['critic_warmup_updates']
            or complete['transitions'] != plan['transitions']):
        raise RuntimeError('Requested training did not finish completely')
    training_protocol = read(study / 'training/protocol.json')
    if training_protocol['reward']['spike_dose'] != config.metadata():
        raise ValueError('Training did not use this exact spike reward')

    def inputs(label, selected):
        directory = study / (label + '_inputs')
        (directory / 'checkpoints').mkdir(parents=True, exist_ok=True)
        for src, name in [(constant, 'constant.pt'), (parent, 'saved.pt'), (selected, 'selected.pt')]:
            target = directory / 'checkpoints' / name
            if target.exists() and sha(target) != sha(src):
                raise ValueError('An existing evaluation actor differs: ' + str(target))
            if not target.exists():
                shutil.copy2(src, target)
        return directory

    common = ['--headless', '--device', 'cpu', '--nominal-sensors', '--steady-gain', '.2',
              '--hard-force-stop-n', '100', '--uniformity-only', '--include-precontrol-contact',
              '--names', 'constant,constant_b,saved,selected']
    for flag in ('--native-dir', '--control-snapshot', '--gui-scene', '--robot-config'):
        common += [flag, argument(flag)]

    def probe(label, directory, seed):
        output = study / label
        if not (output / 'processing_summary.json').exists():
            command = [sys.executable, str(TASK_ROOT / 'mdp/evaluation/probe_tcp146_ppo.py'), *common,
                       '--training-dir', str(directory), '--output', str(output), '--seed', str(seed)]
            save(study / 'validation_status.json', {'phase': label, 'started_utc': datetime.now(timezone.utc).isoformat()})
            save(study / (label + '_invocation.json'), {'command': command})
            with (study / (label + '.log')).open('x') as stream:
                subprocess.run(command, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1'),
                               stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=3600)
        source_summary = read(output / 'summary.json')
        for name in ('constant', 'saved', 'selected'):
            if source_summary['checkpoint_sha256'][name] != sha(directory / 'checkpoints' / (name + '.pt')):
                raise ValueError('Checkpoint mismatch in resumed evaluation')
        result = assess(output, config)
        print('SPIKE_PROBE_COMPLETE', label, json.dumps(result['candidates']['selected']), flush=True)
        return result

    baseline_inputs = inputs('baseline', parent)
    baseline_dir = study / 'baseline_4810'
    baseline = probe('baseline_4810', baseline_inputs, plan['selection_seed'])
    selection = {}
    for update in plan['selection_updates']:
        label = f'ppo_{update:04d}'
        directory = inputs(label, study / 'training/checkpoints' / (label + '.pt'))
        result = probe('selection_' + label, directory, plan['selection_seed'])
        metrics = result['candidates']['selected']
        comparison = compare(metrics, baseline['candidates']['selected'], result['candidates']['constant'])
        selection[label] = {'update': update, 'metrics': metrics, 'comparison': comparison,
                            'output': str(study / ('selection_' + label))}
        save(study / 'selection.json', selection)
    eligible = {key: row for key, row in selection.items() if row['comparison']['quality_eligible']}
    pool = eligible or selection
    chosen = min(pool, key=lambda key: pool[key]['metrics']['spike']['event_abs_error_seconds'])
    selected = selection[chosen]
    bundle = study / 'candidate'
    (bundle / 'checkpoints').mkdir(parents=True, exist_ok=True)
    for src, name in [(study / 'training/checkpoints' / (chosen + '.pt'), 'actor.pt'),
                      (study / 'training/checkpoints' / f"training_{selected['update']:04d}.pt", 'ppo_training_state.pt')]:
        target = bundle / 'checkpoints' / name
        if target.exists() and sha(target) != sha(src):
            raise ValueError('The frozen candidate changed')
        if not target.exists():
            shutil.copy2(src, target)
    save(study / 'frozen_selection.json', {'candidate': chosen, 'actor_sha256': sha(bundle / 'checkpoints/actor.pt'),
        'heldout_seeds': plan['heldout_seeds'], 'selection_comparison': selected['comparison'],
        'research_candidate_only': not selected['comparison']['accepted']})
    heldout = {}
    plot_baseline, plot_candidate = baseline_dir, Path(selected['output'])
    if selected['comparison']['accepted']:
        selected_inputs = study / (chosen + '_inputs')
        for seed in plan['heldout_seeds']:
            reference = probe(f'baseline_{seed}', baseline_inputs, seed)
            result = probe(f'heldout_{seed}', selected_inputs, seed)
            heldout[str(seed)] = {'parent': reference['candidates']['selected'],
                'candidate': result['candidates']['selected'], 'mode6': result['candidates']['constant'],
                'comparison': compare(result['candidates']['selected'], reference['candidates']['selected'], result['candidates']['constant'])}
            save(study / 'heldout.json', heldout)
            if seed == plan['heldout_seeds'][0]:
                plot_baseline, plot_candidate = study / f'baseline_{seed}', study / f'heldout_{seed}'
    changed = [name for name, digest in frozen.items() if sha(name) != digest]
    if changed:
        raise ValueError('Inputs changed during the study: ' + str(changed))
    accepted = bool(selected['comparison']['accepted'] and len(heldout) == len(plan['heldout_seeds'])
                    and all(row['comparison']['accepted'] for row in heldout.values()))
    report = {'status': 'training_and_comparison_complete', 'accepted': accepted,
        'selected_candidate': chosen, 'checkpoint': str(bundle / 'checkpoints/actor.pt'),
        'checkpoint_sha256': sha(bundle / 'checkpoints/actor.pt'), 'training': complete,
        'protocol': protocol, 'selection': selection, 'heldout': heldout,
        'heldout_skipped_reason': None if heldout else 'No selected candidate met both quality and 5% transient improvement criteria',
        'plot_baseline': str(plot_baseline), 'plot_candidate': str(plot_candidate),
        'ros_deployed': False, 'frozen_inputs_verified': True, 'finished_utc': datetime.now(timezone.utc).isoformat()}
    for path, name in [(study / 'training/protocol.json', 'training_protocol.json'),
                       (study / 'validation_protocol.json', 'validation_protocol.json'),
                       (study / 'spike_dose_reward.json', 'spike_dose_reward.json'),
                       (study / 'isaac_ur10e_robot.yaml', 'isaac_ur10e_robot.yaml')]:
        shutil.copy2(path, bundle / name)
    save(study / 'final_report.json', report)
    from ...utils.reporting.plot_spike_retraining import archive as write_archive
    write_archive(archive, plot_baseline, plot_candidate, report)
    save(study / 'validation_status.json', {'phase': 'complete', 'accepted': accepted,
                                          'checkpoint': report['checkpoint'], 'archive': str(archive)})
    print('SPIKE_RETRAINING_COMPLETE', json.dumps({'accepted': accepted, 'candidate': chosen, 'archive': str(archive)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.study.resolve(), args.archive.resolve())
    except BaseException:
        save(args.study / 'validation_failure.json', {'error': traceback.format_exc(), 'time_utc': datetime.now(timezone.utc).isoformat()})
        save(args.study / 'validation_status.json', {'phase': 'failed', 'see': 'validation_failure.json'})
        raise


if __name__ == '__main__':
    main()
