"""Saved neural PPO trial with the existing fixed-force Isaac ROS controller."""
import datetime
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(next(p / 'source/nrs_rl' for p in Path(__file__).resolve().parents
                          if (p / 'source/nrs_rl/nrs_rl').is_dir())))
from nrs_rl.tasks.manager_based.nrs_rl.mdp.runtime.fixed_force_ppo_deployment import validate
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def prepare(context):
    mode = int(LaunchConfiguration('mode').perform(context))
    pair = int(LaunchConfiguration('pair').perform(context))
    study = LaunchConfiguration('study').perform(context)
    if mode not in (5, 6) or not 1 <= pair <= 99:
        raise ValueError('Use mode:=5/6 and pair:=1..99')
    if LaunchConfiguration('backend').perform(context) != 'isaac':
        raise ValueError('The shared fixed-force controller supports backend:=isaac only')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', study):
        raise ValueError('Invalid study name')
    checked = validate(mode=mode)
    logs = Path(LaunchConfiguration('log_root').perform(context)).expanduser().resolve()
    run = f'{study}_mode{mode}_pair{pair:02d}_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    destination = logs / ('rl_best' if mode == 5 else 'rl_no_speed_modulation') / run
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'deployment.json').write_text(json.dumps({**checked, 'run': run,
        'study': study, 'pair': pair, 'removal_directory': str(destination / 'removal'),
        'home_pose': 'current pose at TxtLoad request', 'return_lift_mm': 10.,
        'scene_reset_is_operator_responsibility': True}, indent=2) + '\n')
    params = {'setup_file': checked['setup_file'], 'force_control_mode': mode,
        'use_sim_time': True, 'expected_trajectory_file': checked['trajectory_file'],
        'raw_logging_enabled': True, 'raw_logging_root': str(logs), 'raw_logging_run': run}
    motion = Node(executable=checked['motion_binary'], name='singleArm_motion', output='screen',
        parameters=[{**params, 'spatial_policy_enabled': True,
            'preston_tangent_damping_ns_m': 3329.844766906654,
            'preston_tangent_stiffness_n_m': 12434.33963608066}],
        remappings=[('/joint_states', '/isaac_joint_states'),
            ('/ur10skku/joint_states', '/isaac_joint_states'),
            ('/ppo/observation', '/ur10skku/fixed_force/legacy_observation_unused')])
    command = Node(executable=checked['command_binary'], name='singleArm_cmd', output='screen',
        parameters=[{**params, 'return_home_after_txtload': True,
            'txtload_return_lift_mm': 10., 'require_polishing_measurement': True}])
    removal = Node(executable='/usr/bin/python3', name='polishing_removal_node', output='screen',
        arguments=['-s', checked['removal_script']],
        additional_env={'MPLBACKEND': 'Agg', 'MPLCONFIGDIR': str(logs / '.matplotlib')},
        parameters=[{'use_sim_time': True, 'logs_root': str(destination / 'removal'),
            'reference_trajectory_file': checked['trajectory_file']}])
    policy = ExecuteProcess(cmd=['/usr/bin/python3', '-s', checked['worker'],
        '--checkpoint', checked['checkpoint'], '--reference', checked['reference_h5'],
        '--worker-python', checked['worker_python'], '--force-control-mode', str(mode),
        '--log-directory', str(destination), '--ros-args', '-p', 'use_sim_time:=true'], output='screen')
    processes = [policy, removal, motion, command]
    handlers = [RegisterEventHandler(OnProcessExit(target_action=process,
        on_exit=[EmitEvent(event=Shutdown(reason='PPO trial process exited'))])) for process in processes]
    return handlers + processes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='5', description='5: neural PPO, 6: constant 6 mm/s'),
        DeclareLaunchArgument('pair', default_value='1'),
        DeclareLaunchArgument('study', default_value='ppo11p13_v1'),
        DeclareLaunchArgument('backend', default_value='isaac'),
        DeclareLaunchArgument('log_root', default_value=str(Path.home() / 'evaluation')),
        OpaqueFunction(function=prepare)])
