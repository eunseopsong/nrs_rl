#!/usr/bin/env python3
"""Check all ten planned launch configurations without starting any node."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT as _REPO_ROOT
import importlib.util
import json
from pathlib import Path
import tempfile

ROOT=_REPO_ROOT


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    loaded=importlib.util.module_from_spec(spec);spec.loader.exec_module(loaded)
    return loaded


def main():
    from launch import LaunchContext
    from launch.actions import ExecuteProcess
    from launch.utilities import perform_substitutions
    from launch_ros.actions import Node
    import yaml
    root=ROOT/'deployment/spatial_14p24_paired'
    launch=module('paired_launch_test',root/'paired.launch.py')
    checks=[];configs=[]
    with tempfile.TemporaryDirectory(prefix='paired_launch_test_') as temporary:
        for pair in range(1,6):
            for mode in (5,6):
                context=LaunchContext()
                context.launch_configurations.update({'mode':str(mode),'pair':str(pair),'study':'validation',
                    'backend':'isaac','log_root':temporary})
                processes=[a for a in launch.prepare(context) if isinstance(a,ExecuteProcess)]
                assert len(processes)==4
                policy,removal,motion,command=processes
                context.extend_locals({'ros_specific_arguments':{'name':'__node:=validation_only'}})
                def parameters(node):
                    node._perform_substitutions(context)
                    argv=[perform_substitutions(context,p) for p in node.cmd]
                    params=yaml.safe_load(Path(argv[argv.index('--params-file')+1]).read_text())['/**']['ros__parameters']
                    return argv,params
                ma,mp=parameters(motion);ca,cp=parameters(command);ra,rp=parameters(removal)
                pa=[perform_substitutions(context,p) for p in policy.cmd]
                assert Path(ma[0]).resolve()==root/'build/singleArm_motion'
                assert mp['force_control_mode']==mode and cp['force_control_mode']==mode
                assert mp['spatial_policy_enabled'] and mp['use_sim_time'] and rp['use_sim_time'] and cp['use_sim_time']
                assert mp['preston_tangent_damping_ns_m']==3329.844766906654
                assert mp['preston_tangent_stiffness_n_m']==12434.33963608066
                assert cp['return_home_after_txtload'] and cp['require_polishing_measurement'] and cp['txtload_return_lift_mm']==10.
                assert pa[pa.index('--force-control-mode')+1]==str(mode)
                assert Path(pa[pa.index('--checkpoint')+1]).resolve()==ROOT/'deployment/spatial_14p24/checkpoints/actor.pt'
                assert str(root/'paired_removal_node.py') in ra and '-s' in ra
                remaps=dict(motion.expanded_remapping_rules)
                assert remaps['/joint_states']==remaps['/ur10skku/joint_states']=='/isaac_joint_states'
                configs.append({'mode':mode,'pair':pair,'motion':ma[0],
                    'common_motion_parameters':{k:v for k,v in mp.items() if k not in ['force_control_mode','raw_logging_run']}})
        assert len(list(Path(temporary).rglob('deployment.json')))==10
        assert all(c['common_motion_parameters']==configs[0]['common_motion_parameters'] for c in configs)
        for mode,pair,backend in [('7','1','isaac'),('5','0','isaac'),('5','6','isaac'),('5','1','robot')]:
            context.launch_configurations.update(mode=mode,pair=pair,backend=backend)
            try:launch.prepare(context)
            except ValueError:pass
            else:raise AssertionError('Invalid configuration accepted')
        checks=['ten_trial_expansions','same_binary_and_common_controller_parameters',
            'same_exact_checkpoint_and_explicit_output_mode','same_controller_snapshot_recorder',
            'isolated_user_site_python_for_numpy_matplotlib','home_return_and_required_measurement',
            'unique_mode_pair_run_manifests','isaac_topics_and_clock','invalid_modes_pairs_backends_rejected']
    result={'ok':True,'checks':checks,'executed_robot_processes':0,'trial_configurations':10}
    path=ROOT/'logs/experiments/spatial_14p24_paired/launch_validation.json'
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
