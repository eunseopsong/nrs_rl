#!/usr/bin/env python3
"""Expand both PPO ROS trial modes without executing robot or simulator nodes."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(next(p / 'source/nrs_rl' for p in Path(__file__).resolve().parents
                          if (p / 'source/nrs_rl/nrs_rl').is_dir())))
from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT
from nrs_rl.tasks.manager_based.nrs_rl.mdp.runtime.fixed_force_ppo_deployment import (
    ACTOR_SHA, BASE, BUNDLE, validate)


def main():
    from launch import LaunchContext
    from launch.actions import ExecuteProcess, RegisterEventHandler
    from launch.utilities import perform_substitutions
    import yaml
    spec = importlib.util.spec_from_file_location('ppo_trial_launch', BUNDLE / 'ppo_fixed_force.launch.py')
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    configs = []
    with tempfile.TemporaryDirectory(prefix='nrs_ppo_launch_') as temporary:
        for mode in (5, 6):
            context = LaunchContext()
            context.launch_configurations.update(mode=str(mode), pair='1', study='validation',
                backend='isaac', log_root=temporary)
            actions = launch.prepare(context)
            assert len([a for a in actions if isinstance(a, RegisterEventHandler)]) == 4
            processes = [a for a in actions if isinstance(a, ExecuteProcess)]
            assert len(processes) == 4
            policy, removal, motion, command = processes
            context.extend_locals({'ros_specific_arguments': {'name': '__node:=validation_only'}})

            def parameters(node):
                node._perform_substitutions(context)
                argv = [perform_substitutions(context, p) for p in node.cmd]
                params = yaml.safe_load(Path(argv[argv.index('--params-file') + 1]).read_text())['/**']['ros__parameters']
                return argv, params

            ma, mp = parameters(motion)
            ca, cp = parameters(command)
            ra, rp = parameters(removal)
            pa = [perform_substitutions(context, p) for p in policy.cmd]
            assert Path(ma[0]).resolve() == BASE / 'build/singleArm_motion'
            assert Path(ca[0]).resolve() == BASE / 'build/singleArm_cmd'
            assert mp['force_control_mode'] == cp['force_control_mode'] == mode
            assert mp['spatial_policy_enabled']
            assert mp['use_sim_time'] and cp['use_sim_time'] and rp['use_sim_time']
            assert mp['preston_tangent_damping_ns_m'] == 3329.844766906654
            assert mp['preston_tangent_stiffness_n_m'] == 12434.33963608066
            assert cp['return_home_after_txtload'] and cp['require_polishing_measurement']
            assert cp['txtload_return_lift_mm'] == 10.
            assert pa[pa.index('--force-control-mode') + 1] == str(mode)
            assert Path(pa[pa.index('--checkpoint') + 1]).resolve() == BUNDLE / 'checkpoints/actor.pt'
            assert pa[2].endswith('/mdp/runtime/fixed_force_ppo_node.py')
            assert str(BASE / 'paired_removal_node.py') in ra
            remaps = dict(motion.expanded_remapping_rules)
            assert remaps['/joint_states'] == remaps['/ur10skku/joint_states'] == '/isaac_joint_states'
            configs.append({k: v for k, v in mp.items() if k not in ('force_control_mode', 'raw_logging_run')})
        assert configs[0] == configs[1]
        manifests = list(Path(temporary).rglob('deployment.json'))
        assert len(manifests) == 2
        for path in manifests:
            saved = json.loads(path.read_text())
            assert saved['algorithm'] == 'PPO' and saved['checkpoint_sha256'] == ACTOR_SHA
        for mode, pair, backend, study in [('7', '1', 'isaac', 'test'), ('5', '0', 'isaac', 'test'),
                ('5', '100', 'isaac', 'test'), ('5', '1', 'robot', 'test'), ('5', '1', 'isaac', '../bad')]:
            context.launch_configurations.update(mode=mode, pair=pair, backend=backend, study=study)
            try:
                launch.prepare(context)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid launch configuration accepted')
        # A manifest must not permit substituting a different actor or changed runtime.
        changed = json.loads((BUNDLE / 'deployment.json').read_text())
        changed['checkpoint_sha256'] = '0' * 64
        (Path(temporary) / 'deployment.json').write_text(json.dumps(changed))
        try:
            validate(temporary, check_worker=False)
        except ValueError:
            pass
        else:
            raise AssertionError('Checkpoint substitution accepted')
        changed = json.loads((BUNDLE / 'deployment.json').read_text())
        changed['pinned_files'][changed['worker']] = '0' * 64
        (Path(temporary) / 'deployment.json').write_text(json.dumps(changed))
        try:
            validate(temporary, check_worker=False)
        except ValueError:
            pass
        else:
            raise AssertionError('Changed runtime accepted')
    result = {'ok': True, 'tested_modes': [5, 6], 'executed_robot_processes': 0,
        'checkpoint_sha256': ACTOR_SHA, 'checks': ['PPO_actor_and_worker_selected',
        'matched_controller_force_and_speed_parameters', 'required_measurement_and_home_return',
        'isaac_topics_and_clock', 'all_process_exits_shutdown_launch', 'unique_trial_manifests',
        'invalid_modes_pairs_backends_names_rejected', 'checkpoint_and_source_changes_rejected']}
    out = ROOT / 'logs/experiments/ppo_fixed_force_11p13_deployment/launch_validation.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
