#!/usr/bin/env python3
"""Create a separate launch-compatible bundle only after stronger promotion gates pass."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import argparse
import json
from pathlib import Path
import shutil

from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_tcp_refinement import promotion_decision, sha


def package(args):
    training, output = args.training.resolve(), args.output.resolve()
    state = json.loads((training/'result.json').read_text())
    protocol = json.loads((training/'protocol.json').read_text())
    report = json.loads((training/'report.json').read_text())
    export = json.loads((training/'export_validation.json').read_text())
    runtime = json.loads(args.runtime_validation.read_text())
    if (state['phase'] != 'complete' or not state['validation_passed'] or not state['training_candidate_eligible']
            or not report['frozen_gates_passed'] or len(state['validation']) != 3
            or not promotion_decision(state['validation'])['passed']):
        raise ValueError('A rejected refinement must not become a runnable bundle')
    for key in ('original_frozen_gates', 'promotion_gates', 'reward', 'training_seeds', 'validation_seeds'):
        if state[key] != protocol[key]:
            raise ValueError('Protocol changed: '+key)
    checkpoint = Path(state['checkpoint'])
    digest = sha(checkpoint)
    if (digest != state['checkpoint_sha256'] or digest != report['checkpoint_sha256']
            or digest != export['checkpoint_sha256'] or not export['passed']):
        raise ValueError('Checkpoint identity or export validation mismatch')
    binary = Path(runtime['motion_binary'])
    if (not all(runtime[k] for k in ('ok', 'default_controller_parity', 'normal_axis_unchanged', 'invalid_configuration_rejected'))
            or sha(binary) != runtime['motion_binary_sha256']):
        raise ValueError('Native runtime proof is absent or the binary changed')
    contract = json.loads((training/'params/velocity_policy.json').read_text())
    if (contract.get('policy_backend') != 'torchscript_tcp_removal'
            or contract.get('requires_new_tracking_runtime', False)):
        raise ValueError('This packager supports the existing MDK/feed runtime only; tracking compensation needs its own runtime validation')
    tangent = dict(zip(('damping_ns_m', 'stiffness_n_m'), state['best_parameters'][:2]))
    if (tangent != contract['tangential_controller'] or contract['learned_parameters'] != state['best_parameters'][2:]
            or contract['rotation_contribution'] is not False or contract['spindle_rpm'] is not None):
        raise ValueError('Policy/controller contract mismatch')
    # Add aliases required by the existing launch's schema. Keep all original
    # experiment fields and the stronger promotion thresholds in both records.
    normalized_state = {**state, 'gates': {'original': state['original_frozen_gates'], 'promotion': state['promotion_gates']},
        'validation': [{**v, 'passed': v['assessment']['eligible'], 'rejections': v['assessment']['rejections']}
                       for v in state['validation']],
        'source_result_sha256': sha(training/'result.json'),
        'bundle_schema_note': 'gates and per-condition passed aliases added; source experiment is unchanged.'}
    normalized_protocol = {**protocol, 'gates': normalized_state['gates'],
                           'source_protocol_sha256': sha(training/'protocol.json')}
    output.mkdir(parents=True, exist_ok=False)
    (output/'checkpoints').mkdir()
    (output/'params').mkdir()
    copies = {'checkpoints/actor.pt': checkpoint,
              'params/velocity_policy.json': training/'params/velocity_policy.json',
              'report.json': training/'report.json', 'runtime_validation.json': args.runtime_validation}
    for name, source in copies.items():
        shutil.copy2(source, output/name)
    for name, value in [('validation.json', normalized_state), ('protocol.json', normalized_protocol), ('controller.json', tangent)]:
        (output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    names = [*copies, 'validation.json', 'protocol.json', 'controller.json']
    manifest = {'schema_version': 1, 'name': 'Additional RPM-free MDK/feed refinement',
        'backend': 'isaac', 'normal_force_target_n': 20., 'rotation_contribution': False, 'spindle_rpm': None,
        'physical_depth_validated': False, 'training_directory': str(training), 'checkpoint_sha256': digest,
        'reference_h5_sha256': state['reference_path_sha256'], 'motion_binary_sha256': sha(binary),
        'files': {name: sha(output/name) for name in names}, 'attribution': report['attribution'],
        'replaces_default_bundle': False}
    (output/'deployment.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'bundle': str(output), 'checkpoint_sha256': digest, 'default_bundle_unchanged': True}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime-validation', type=Path, required=True)
    package(parser.parse_args())
