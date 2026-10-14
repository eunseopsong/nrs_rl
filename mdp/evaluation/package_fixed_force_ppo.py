#!/usr/bin/env python3
"""Package the cv11p13 PPO with independently tested, pinned ROS dependencies."""
import argparse
import json
from pathlib import Path
import shutil
import sys

if not __package__:
    sys.path.insert(0, str(next(p / 'source/nrs_rl' for p in Path(__file__).resolve().parents
                              if (p / 'source/nrs_rl/nrs_rl').is_dir())))
from nrs_rl.tasks.manager_based.nrs_rl.paths import TASK_ROOT
from nrs_rl.tasks.manager_based.nrs_rl.mdp.runtime.fixed_force_ppo_deployment import (
    ACTOR_SHA, BASE, BUNDLE, SAVE, digest)


def package(validation_file):
    base = json.loads((BASE / 'deployment.json').read_text())
    for path, expected in base['pinned_files'].items():
        if digest(path) != expected:
            raise ValueError('Shared deployment dependency changed: ' + path)
    source = SAVE / 'candidate/checkpoints/actor.pt'
    record = json.loads((SAVE / 'SAVE.json').read_text())
    if digest(source) != ACTOR_SHA or record['checkpoint_sha256'] != ACTOR_SHA:
        raise ValueError('Saved PPO identity changed')
    validation_file = Path(validation_file).resolve()
    validation = json.loads(validation_file.read_text())
    if (not validation['ok'] or validation['checkpoint_sha256'] != ACTOR_SHA
            or validation['tested_modes'] != [5, 6]
            or not validation['isolated_ros_validation_passed']
            or not validation['unit_tests_passed']
            or validation['motion_binary_sha256'] != digest(base['motion_binary'])
            or validation['command_binary_sha256'] != digest(base['command_binary'])):
        raise ValueError('PPO validation incomplete or stale')
    BUNDLE.mkdir(parents=True, exist_ok=True)
    (BUNDLE / 'checkpoints').mkdir(exist_ok=True)
    actor = BUNDLE / 'checkpoints/actor.pt'
    if actor.exists() and digest(actor) != ACTOR_SHA:
        raise ValueError('Refusing to overwrite a different deployed actor')
    shutil.copy2(source, actor)
    shutil.copy2(SAVE / 'candidate/fixed_force_policy.json', BUNDLE / 'fixed_force_policy.json')
    runtime = TASK_ROOT / 'mdp/runtime'
    launch = BUNDLE / 'ppo_fixed_force.launch.py'
    target = runtime / 'ppo_fixed_force.launch.py'
    if not launch.is_symlink():
        if launch.exists():
            raise ValueError('Refusing to overwrite an unrelated launch file')
        launch.symlink_to(target)
    if launch.resolve() != target:
        raise ValueError('PPO launch link points to a different source')
    data = {key: base[key] for key in (
        'backend', 'supported_modes', 'worker_python', 'motion_binary', 'command_binary',
        'removal_script', 'setup_file', 'trajectory_file', 'reference_h5', 'observation_shape',
        'action_shape', 'telemetry_schema', 'action_packet', 'target_force_n', 'policy_controls_force',
        'control_period_s', 'policy_period_s', 'speed_mapping', 'hard_force_stop_n',
        'virtual_spindle_rpm', 'rpm_is_assumed_not_measured', 'return_lift_mm', 'entry_protocol')}
    data.update(name='ppo_fixed20_virtual1000_cv11p13', algorithm='PPO',
        worker=str(runtime / 'fixed_force_ppo_node.py'), checkpoint=str(actor),
        checkpoint_sha256=ACTOR_SHA, saved_checkpoint=str(source), validation_file=str(validation_file),
        launch_file=str(launch), inherited_controller_manifest=str(BASE / 'deployment.json'),
        mean_heldout_metrics=record['mean_heldout_metrics'],
        physical_removal_validated=False, gui_trials_completed=0)
    dependencies = [BASE / 'deployment.json', SAVE / 'SAVE.json', source, actor,
                    BUNDLE / 'fixed_force_policy.json', BUNDLE / 'README.txt', launch,
                    validation_file, Path(__file__).resolve()]
    dependencies += [runtime / name for name in ('__init__.py', 'fixed_force_ppo_runtime.py',
        'fixed_force_ppo_node.py', 'fixed_force_ppo_deployment.py', 'ppo_fixed_force.launch.py')]
    dependencies += [TASK_ROOT / name for name in ('paths.py', 'mdp/removal_config.py',
        'utils/removal_metrics.py', 'utils/spatial_geometry.py', 'utils/rotary_geometry.py',
        'utils/preston_surface.py', '__init__.py', 'mdp/__init__.py', 'utils/__init__.py')]
    dependencies += [TASK_ROOT.parents[2] / '__init__.py', TASK_ROOT.parents[1] / '__init__.py',
                     TASK_ROOT.parent / '__init__.py']
    dependencies += [Path(p) for p in validation.get('reports', [])]
    dependencies += sorted((TASK_ROOT / 'mdp/tests').glob('*fixed_force_ppo*deployment*.py'))
    dependencies += [TASK_ROOT / 'mdp/tests/validate_fixed_force_ppo_ros.py']
    dependencies += [TASK_ROOT / 'mdp/tests/validate_fixed_force_ppo_launch.py']
    data['pinned_files'] = dict(base['pinned_files'])
    data['pinned_files'].update({str(p): digest(p) for p in dependencies})
    (BUNDLE / 'deployment.json').write_text(json.dumps(data, indent=2) + '\n')
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', type=Path, required=True)
    result = package(parser.parse_args().validation)
    print(json.dumps({'bundle': str(BUNDLE), 'checkpoint_sha256': result['checkpoint_sha256'],
                      'pinned_files': len(result['pinned_files'])}, indent=2))
