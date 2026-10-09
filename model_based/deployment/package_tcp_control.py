#!/usr/bin/env python3
"""Package a passed experiment with its inseparable controller configuration."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package(training, output, runtime_validation):
    training, output = training.resolve(), output.resolve()
    result = json.loads((training / 'result.json').read_text())
    report = json.loads((training / 'report.json').read_text())
    protocol = json.loads((training / 'protocol.json').read_text())
    runtime = json.loads(runtime_validation.read_text())
    if (result['phase'] != 'complete' or not result['validation_passed'] or
            not report['frozen_gates_passed'] or len(result['validation']) != 3 or
            any(not v['passed'] for v in result['validation']) or
            result['gates'] != protocol['gates'] or result['reward'] != protocol['reward']):
        raise ValueError('A failed or incomplete experiment cannot become a runnable bundle')
    if (not runtime['ok'] or not runtime['default_controller_parity'] or
            not runtime['normal_axis_unchanged'] or not runtime['invalid_configuration_rejected']):
        raise ValueError('Native controller validation is missing')
    binary = Path(runtime['motion_binary'])
    if digest(binary) != runtime['motion_binary_sha256']:
        raise ValueError('Motion executable changed after native validation')
    actor = Path(result['checkpoint'])
    contract = json.loads((training / 'params/velocity_policy.json').read_text())
    if (digest(actor) != report['checkpoint_sha256'] or
            contract['rotation_contribution'] is not False or contract['spindle_rpm'] is not None or
            contract['target_normal_force_n'] != 20 or contract['policy_backend'] != 'torchscript_tcp_removal'):
        raise ValueError('Checkpoint identity or RPM-free contract changed')
    tangent = {'damping_ns_m': result['best_parameters'][0], 'stiffness_n_m': result['best_parameters'][1]}
    if tangent != contract['tangential_controller']:
        raise ValueError('Controller configuration does not match the learned checkpoint')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'checkpoints').mkdir()
    (output / 'params').mkdir()
    files = {
        'checkpoints/actor.pt': actor,
        'params/velocity_policy.json': training / 'params/velocity_policy.json',
        'validation.json': training / 'result.json',
        'report.json': training / 'report.json',
        'protocol.json': training / 'protocol.json',
        'runtime_validation.json': runtime_validation,
    }
    for relative, source in files.items():
        shutil.copy2(source, output / relative)
    (output / 'controller.json').write_text(json.dumps(tangent, indent=2))
    manifest = {
        'schema_version': 1, 'name': 'RPM-free TCP removal with learned tangential MDK',
        'backend': 'isaac', 'normal_force_target_n': 20., 'rotation_contribution': False,
        'spindle_rpm': None, 'physical_depth_validated': False,
        'training_directory': str(training), 'checkpoint_sha256': digest(actor),
        'reference_h5_sha256': result['reference_path_sha256'],
        'motion_binary_sha256': runtime['motion_binary_sha256'],
        'files': {name: digest(output / name) for name in [*files, 'controller.json']},
        'attribution': report['attribution'],
    }
    (output / 'deployment.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'ok': True, 'bundle': str(output), 'checkpoint_sha256': digest(actor),
                      'controller': tangent}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime-validation', type=Path, required=True)
    args = parser.parse_args()
    package(args.training, args.output, args.runtime_validation)
