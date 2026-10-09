"""Pin the already tested scalar runtime; finalize only after launch expansion."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'deployment/fixed_force_12p59'
REPORTS = ROOT/'logs/experiments/fixed_force_12p59'
SHA = '6d1dc47bc915b393880d2e6e380f44fbcb82e542de0fa12e4f02eb47c97147df'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def main(finalize=False):
    saved = ROOT/'logs/experiments/uniformity_saves/fixed20_virtual1000_dynamics_20261009_1311'
    for rel, expected in json.loads((saved/'save_manifest.json').read_text()).items():
        assert digest(saved/rel) == expected, rel
    assert digest(BUNDLE/'checkpoints/actor.pt') == SHA
    reports = [REPORTS/f'ros_integration/mode{mode}/validation.json' for mode in (5,6)]
    reports += [REPORTS/'measurement_integration/validation.json', REPORTS/'measurement_training_parity.json']
    for path in reports:
        assert json.loads(path.read_text())['ok'], path
    test_log = Path('/tmp/nrs_fixed_force_12p59_tests.log').read_text()
    assert 'Ran 7 tests' in test_log and test_log.rstrip().endswith('OK')
    for source, name in [('/tmp/nrs_fixed_force_12p59_tests.log','unit_tests.log'),
            ('/tmp/nrs_fixed_force_12p59_final_build.log','build.log'),
            ('/tmp/nrs_fixed_force_12p59_ros5.log','ros_mode5.log'),
            ('/tmp/nrs_fixed_force_12p59_ros6.log','ros_mode6.log'),
            ('/tmp/nrs_fixed_force_12p59_measurement.log','measurement.log')]:
        shutil.copyfile(source, REPORTS/name)
    if finalize:
        report = REPORTS/'launch_validation.json'
        assert json.loads(report.read_text())['ok']
        reports.append(report)
    validation = {'ok': True, 'tested_modes': [5,6], 'checkpoint_sha256': SHA,
        'motion_binary_sha256': digest(BUNDLE/'build/singleArm_motion'),
        'command_binary_sha256': digest(BUNDLE/'build/singleArm_cmd'),
        'unit_tests_passed': 7, 'isolated_ros_validation_passed': True,
        'measurement_training_trace_parity_passed': True, 'launch_expansion_passed': finalize,
        'gui_trials_completed': 0, 'actual_robot_connected_during_validation': False,
        'reports': [str(p) for p in reports]}
    save(BUNDLE/'validation.json', validation)
    package = Path('/home/eunseop/dev_ws/install/Y2RobMotion/share/Y2RobMotion')
    data = {'name':'fixed20_virtual1000_12p59', 'backend':'isaac', 'supported_modes':[5,6],
        'worker_python':'/home/eunseop/anaconda3/envs/env_isaaclab/bin/python3.11',
        'worker':str(BUNDLE/'fixed_force_policy_node.py'),
        'checkpoint':str(BUNDLE/'checkpoints/actor.pt'), 'checkpoint_sha256':SHA,
        'validated_training_checkpoint':str(saved/'candidate/checkpoints/actor.pt'),
        'motion_binary':str(BUNDLE/'build/singleArm_motion'),
        'command_binary':str(BUNDLE/'build/singleArm_cmd'),
        'removal_script':str(BUNDLE/'paired_removal_node.py'),
        'setup_file':str(package/'config/preston_d14.yaml'),
        'trajectory_file':str(package/'txtcmd/cmd_continue9D_preston_d14_20n.txt'),
        'reference_h5':str(ROOT/'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/datasets/cmd_continue9D_flat.h5'),
        'observation_shape':[1,16], 'action_shape':[1,1], 'telemetry_schema':2,
        'action_packet':['epoch','sequence','feed_action'], 'target_force_n':20.,
        'policy_controls_force':False, 'forward_speed_bounds_mm_s':[0.,12.],
        'learned_moving_speed_bounds_mm_s':[1.5,12.], 'speed_mapping':'6+6*a mm/s',
        'control_period_s':.008, 'policy_period_s':.08, 'physical_action_slew_mm_s2':18.,
        'speed_acceleration_mm_s2':16., 'speed_jerk_mm_s3':160., 'hard_force_stop_n':100.,
        'tracking_gain':.6, 'virtual_spindle_rpm':1000., 'rpm_is_assumed_not_measured':True,
        'urdf_rotation_unchanged':True, 'local_returns_enabled':False, 'second_pass':False,
        'return_home_after_txtload':True, 'return_lift_mm':10., 'measurement_enabled':True,
        'mode6_targets':{'normal_force_n':20.,'feed_mm_s':6.,'virtual_spindle_rpm':1000.},
        'entry_protocol':{'seconds':2.,'reference_footprint_mm':18.,'zero_depth_cells_retained':True},
        'measurement_source':'atomic 23D controller snapshot, rotating-disk estimator',
        'measurement_midpoint_arithmetic':'float32 to match the training recorder; raw telemetry retained',
        'physical_removal_validated':False, 'gui_trials_completed':0}
    paths = {Path(data[k]) for k in ['worker','checkpoint','motion_binary','command_binary',
              'removal_script','setup_file','trajectory_file','reference_h5']}
    paths.update(p for p in BUNDLE.glob('*') if p.is_file() and p.name!='deployment.json')
    paths.update(p for p in (BUNDLE/'runtime_source').rglob('*') if p.is_file() and '__pycache__' not in str(p))
    paths.update(reports)
    paths.update(REPORTS/name for name in ['unit_tests.log','build.log','ros_mode5.log','ros_mode6.log','measurement.log'])
    paths.update(ROOT/'scripts'/name for name in ['spatial_uniformity.py','fixed_force_velocity.py',
        'fixed_force_refinement.py','test_fixed_force_12p59_deployment.py',
        'validate_fixed_force_12p59_ros.py','validate_fixed_force_12p59_measurement_ros.py',
        'verify_fixed_force_12p59_measurement.py','package_fixed_force_12p59.py',
        'validate_fixed_force_12p59_launch.py'])
    paths.add(ROOT/'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/utils/preston_surface.py')
    paths.add(Path('/home/eunseop/dev_ws/install/Y2ForceCon/share/Y2ForceCon/checkpoints/NAF_MDGradi/NAF_mdGradi_policy_script.pt'))
    for binary in [data['motion_binary'],data['command_binary']]:
        linked = subprocess.run(['ldd',binary],capture_output=True,text=True,check=True).stdout
        assert 'not found' not in linked, linked
        paths.update(Path(p) for p in re.findall(r'=> (/home/eunseop/dev_ws/\S+)',linked))
    data['pinned_files'] = {str(p):digest(p) for p in sorted(paths)}
    save(BUNDLE/'deployment.json',data)
    if finalize:
        requirements = ROOT/'deployment/POLICY_REQUIREMENTS.json'
        config = json.loads(requirements.read_text())
        assert config['checkpoint_sha256']==SHA and config['policy_may_change_force'] is False
        config.update(active_new_checkpoint=data['checkpoint'], gui_runtime_ready=True,
            active_launch=str(BUNDLE/'fixed_force_12p59.launch.py'),
            active_deployment_manifest=str(BUNDLE/'deployment.json'),
            gui_trial_executed=False, isolated_ros_validation_passed=True)
        save(requirements,config)
    print(json.dumps({'ok':True,'finalized':finalize,'pinned_files':len(paths),'checkpoint_sha256':SHA}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--finalize',action='store_true')
    main(parser.parse_args().finalize)
