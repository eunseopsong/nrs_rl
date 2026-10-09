#!/usr/bin/env python3
"""Expand the spatial launch and check its settings without starting processes."""
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
    root=ROOT/'deployment/spatial_control'
    real=module('spatial_preflight_test',root/'preflight.py').validate()
    launch=module('spatial_launch_test',root/'spatial_uniformity.launch.py')
    checks=['pinned_files_and_checkpoint_inference']
    with tempfile.TemporaryDirectory(prefix='spatial_launch_check_') as temporary:
        context=LaunchContext()
        context.launch_configurations.update({'mode':'5','backend':'isaac',
            'run':'dry_expansion','return_lift_mm':'10.0','log_root':temporary})
        actions=launch.prepare(context)
        processes=[a for a in actions if isinstance(a,ExecuteProcess)]
        assert len(processes)==4
        policy,removal,motion,command=processes
        assert all(isinstance(n,Node) for n in (removal,motion,command))
        context.extend_locals({'ros_specific_arguments':{'name':'__node:=validation_only'}})
        def parameters(node):
            node._perform_substitutions(context)
            argv=[perform_substitutions(context,p) for p in node.cmd]
            params=yaml.safe_load(Path(argv[argv.index('--params-file')+1]).read_text())['/**']['ros__parameters']
            return argv,params
        ma,mp=parameters(motion);ca,cp=parameters(command);ra,rp=parameters(removal)
        assert Path(ma[0]).resolve()==Path(real['motion_binary']).resolve()
        assert Path(ca[0]).resolve()==Path(real['command_binary']).resolve()
        assert mp['use_sim_time'] and cp['use_sim_time'] and rp['use_sim_time']
        assert mp['force_control_mode']==5 and mp['spatial_policy_enabled']
        assert mp['preston_tangent_damping_ns_m']==3329.844766906654
        assert mp['preston_tangent_stiffness_n_m']==12434.33963608066
        assert cp['return_home_after_txtload'] and cp['require_polishing_measurement']
        assert cp['txtload_return_lift_mm']==10.
        assert cp['expected_trajectory_file']==real['trajectory_file']
        assert rp['depth_model']=='tcp_fixed_roi' and rp['required_control_mode']=='Force'
        assert rp['pad_radius_mm']==15. and rp['cell_mm']==.5
        remaps=dict(motion.expanded_remapping_rules)
        assert remaps['/ur10skku/joint_states']=='/isaac_joint_states'
        assert remaps['/joint_states']=='/isaac_joint_states'
        argv=[perform_substitutions(context,p) for p in policy.cmd]
        assert argv[argv.index('--checkpoint')+1]==real['checkpoint']
        assert argv[argv.index('--reference')+1]==real['reference_h5']
        assert 'use_sim_time:=true' in argv
        checks+=['four_process_launch_expansion','new_spatial_motion_and_actor_selected',
            'isaac_joint_topics_and_sim_clock','home_return_and_removal_enabled']
        for key,value in [('mode','6'),('backend','robot')]:
            previous=context.launch_configurations[key];context.launch_configurations[key]=value
            try:launch.prepare(context)
            except ValueError:checks.append('rejected_unsupported_'+key)
            else:raise AssertionError('Unsupported configuration accepted: '+key)
            finally:context.launch_configurations[key]=previous
    result={'ok':True,'checks':checks,'executed_robot_processes':0,
        'checkpoint_sha256':real['checkpoint_sha256']}
    path=ROOT/'logs/experiments/spatial_uniformity_20261007/ros_integration/launch_validation.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
