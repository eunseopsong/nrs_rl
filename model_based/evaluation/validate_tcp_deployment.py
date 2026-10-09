#!/usr/bin/env python3
"""Check a real passed bundle and expand both launches without starting motion."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT as _REPO_ROOT
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--motion-binary', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--launch-file', type=Path, help='Optional workspace launch instead of an installed package launch')
    parser.add_argument('--preflight-file', type=Path, help='Preflight belonging to the workspace launch')
    parser.add_argument('--nrs-root', type=Path, default=_REPO_ROOT)
    parser.add_argument('--worker-python', default='/home/eunseop/anaconda3/envs/env_isaaclab/bin/python')
    args = parser.parse_args()
    preflight = module('tcp_preflight_test', args.preflight_file or args.package / 'scripts/preston_tcp_preflight.py')
    checks = []

    def validate(bundle, binary=None):
        return preflight.validate(args.package, args.nrs_root, args.worker_python, bundle,
                                  binary or args.motion_binary, check_worker=False)

    real = preflight.validate(args.package, args.nrs_root, args.worker_python, args.bundle,
                             args.motion_binary, check_worker=True)
    checks.append('real_checkpoint_inference_and_matching_controller')
    with tempfile.TemporaryDirectory(prefix='tcp_deployment_checks_') as tmp:
        root = Path(tmp)
        for case in ('actor_bytes', 'controller_mismatch', 'failed_validation', 'changed_protocol',
                     'manifest_escape', 'stale_motion_binary'):
            bundle = root / case
            shutil.copytree(args.bundle, bundle)
            manifest = json.loads((bundle / 'deployment.json').read_text())

            def change_json(relative, alter):
                path = bundle / relative
                value = json.loads(path.read_text())
                alter(value)
                path.write_text(json.dumps(value))
                # Exercise semantic checks even when a modified file is rehashed.
                manifest['files'][relative] = hashlib.sha256(path.read_bytes()).hexdigest()

            if case == 'actor_bytes':
                path = bundle / 'checkpoints/actor.pt'
                path.write_bytes(path.read_bytes() + b'changed')
            elif case == 'controller_mismatch':
                change_json('controller.json', lambda c: c.update(stiffness_n_m=c['stiffness_n_m'] + 1))
            elif case == 'failed_validation':
                change_json('validation.json', lambda c: c.update(validation_passed=False))
            elif case == 'changed_protocol':
                change_json('protocol.json', lambda c: c['gates'].update(spatial_cv_gain_min=0.))
            elif case == 'manifest_escape':
                manifest['files']['../outside'] = 'invalid'
            (bundle / 'deployment.json').write_text(json.dumps(manifest))
            try:
                validate(bundle, Path('/usr/bin/true') if case == 'stale_motion_binary' else None)
            except (ValueError, RuntimeError):
                checks.append('rejected_' + case)
            else:
                raise AssertionError('Invalid deployment accepted: ' + case)

        from launch import LaunchContext
        from launch.actions import ExecuteProcess
        from launch.utilities import perform_substitutions
        from launch_ros.actions import Node
        import yaml
        launch = module('tcp_launch_test', args.launch_file or args.package / 'launch/tcp_removal.launch.py')
        prefix = root / 'prefix'
        executable = prefix / 'lib/Y2RobMotion/singleArm_motion'
        executable.parent.mkdir(parents=True)
        executable.symlink_to(args.motion_binary.resolve())
        launch.get_package_share_directory = lambda name: str(args.package)
        launch.get_package_prefix = lambda name: str(prefix)
        for mode in (5, 6):
            context = LaunchContext()
            context.launch_configurations.update({
                'mode': str(mode), 'backend': 'isaac', 'run': f'check_{mode}',
                'bundle': str(args.bundle.resolve()), 'nrs_root': str(args.nrs_root),
                'worker_python': args.worker_python, 'removal_model': 'tcp_fixed_roi',
                'return_lift_mm': '10.0', 'log_root': str(root / 'logs'),
                'removal_root': str(Path.home() / 'nrspath_ws/src/polishing_removal'),
            })
            actions = launch.prepare(context)
            processes = [a for a in actions if isinstance(a, ExecuteProcess)]
            assert len(processes) == (4 if mode == 5 else 3)
            nodes = [a for a in processes if isinstance(a, Node)]
            assert len(nodes) == 3
            removal, motion, command = nodes
            context.extend_locals({'ros_specific_arguments': {'name': '__node:=validation_only'}})

            def parameters(node):
                node._perform_substitutions(context)
                argv = [perform_substitutions(context, p) for p in node.cmd]
                path = Path(argv[argv.index('--params-file') + 1])
                return yaml.safe_load(path.read_text())['/**']['ros__parameters']

            mp, cp, rp = parameters(motion), parameters(command), parameters(removal)
            motion_argv = [perform_substitutions(context, p) for p in motion.cmd]
            assert Path(motion_argv[0]).resolve() == args.motion_binary.resolve()
            assert mp['use_sim_time'] and cp['use_sim_time'] and rp['use_sim_time']
            assert mp['force_control_mode'] == mode
            assert mp['preston_tangent_damping_ns_m'] == real['tangential_controller']['damping_ns_m']
            assert mp['preston_tangent_stiffness_n_m'] == real['tangential_controller']['stiffness_n_m']
            assert cp['return_home_after_txtload'] and cp['require_polishing_measurement']
            assert rp['depth_model'] == 'tcp_fixed_roi'
            remaps = dict(motion.expanded_remapping_rules)
            assert remaps['/ur10skku/joint_states'] == '/isaac_joint_states'
            assert remaps['/ppo/action'] == '/ur10skku/tcp_removal/action'
            if mode == 5:
                actor = [a for a in processes if not isinstance(a, Node)][0]
                argv = [perform_substitutions(context, p) for p in actor.cmd]
                assert argv[argv.index('--controller-config')+1] == real['controller_config']
                assert argv[argv.index('--checkpoint')+1] == real['checkpoint']
            checks.append(f'mode_{mode}_launch_expansion')
        for key, value in [('backend', 'robot'), ('mode', '3'),
                           ('removal_model', 'rotation_dominated_normalized')]:
            previous = context.launch_configurations[key]
            context.launch_configurations[key] = value
            try:
                launch.prepare(context)
            except ValueError:
                checks.append('rejected_launch_' + key)
            else:
                raise AssertionError('Unsupported launch configuration accepted: ' + key)
            context.launch_configurations[key] = previous
    report = {'ok': True, 'checks': checks, 'executed_robot_processes': 0,
              'bundle': str(args.bundle.resolve()), 'checkpoint_sha256': real['checkpoint_sha256']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
