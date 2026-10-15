"""Run visible Isaac + production ROS Mode 5/6 from separate fresh stages."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parent
NRS = Path('/home/eunseop/nrs_rl')
ISAAC = Path('/home/eunseop/isaac/isaac-sim-5.1.0')
PYTHON = '/home/eunseop/anaconda3/envs/env_isaaclab/bin/python'
assert os.environ.get('ROS_DOMAIN_ID') == '181'
assert os.environ.get('ROS_LOCALHOST_ONLY') == '1'

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import WrenchStamped
from y2_rob_motion_interfaces.srv import SingleArmCommand

def dump(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')

def stop(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=15.)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10.)

def trial(mode):
    folder = ROOT/f'mode{mode}'
    folder.mkdir(exist_ok=False)
    scene = bridge = launch = None
    logs = []
    node = Node(f'tcp146_gui_supervisor_mode{mode}')
    state = {'pose': None, 'force': None, 'telemetry': None, 'telemetry_count': 0}
    qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
    def telemetry(msg):
        state['telemetry'] = list(msg.data)
        state['telemetry_count'] += 1
    subscriptions = [
        node.create_subscription(Float64MultiArray, '/ur10skku/currentP',
            lambda msg: state.update(pose=list(msg.data)), 10),
        node.create_subscription(WrenchStamped, '/ur10skku/ftdata',
            lambda msg: state.update(force=[msg.wrench.force.x,msg.wrench.force.y,msg.wrench.force.z]), 10),
        node.create_subscription(Float64MultiArray, '/ur10skku/fixed_force/telemetry', telemetry, qos)]
    client = node.create_client(SingleArmCommand, '/singleArm_cmd/single_arm_command')
    def start(command, name, env=None):
        stream = (folder/(name+'.log')).open('w');logs.append(stream)
        return subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
            env=env, start_new_session=True, cwd='/home/eunseop/dev_ws')
    def alive():
        for name, process in [('scene',scene),('bridge',bridge),('launch',launch)]:
            if process is not None and process.poll() is not None:
                raise RuntimeError(f'{name} exited with {process.returncode}; see {folder}')
    def spin(seconds):
        until = time.monotonic()+seconds
        while time.monotonic()<until:
            alive();rclpy.spin_once(node,timeout_sec=.01)
    try:
        env = os.environ.copy()
        env['PYTHONPATH'] = ''
        env['LD_LIBRARY_PATH'] = str(ISAAC/'exts/isaacsim.ros2.bridge/humble/lib')
        env['RMW_IMPLEMENTATION'] = 'rmw_fastrtps_cpp'
        scene = start([str(ISAAC/'python.sh'), str(ROOT/'scene.py'), '--output', str(folder)], 'scene', env)
        deadline = time.monotonic()+240.
        while not (folder/'ready.json').exists():
            spin(.25)
            if time.monotonic()>deadline:raise TimeoutError('GUI scene did not become ready')
        bridge = start(['ros2','run','y2_isaac_bridge','joint_command_bridge',
                        '--ros-args','-p','use_sim_time:=true'], 'bridge')
        run = f'tcp146_mode{mode}_pair01'
        launch = start(['ros2','launch','Y2RobMotion','ppo_fixed_force.launch.py',
            f'mode:={mode}',f'run:={run}','log_root:='+str(ROOT/'recordings')], 'launch')
        recording = ROOT/'recordings'/('rl_best' if mode==5 else 'rl_no_speed_modulation')/run
        deadline = time.monotonic()+100.
        while True:
            spin(.25)
            calibrated = 'Static wrench offset estimation complete' in (folder/'bridge.log').read_text()
            if client.service_is_ready() and state['pose'] and state['force'] is not None and calibrated and (recording/'policy.csv').exists():break
            if time.monotonic()>deadline:raise TimeoutError('ROS readiness/force calibration timed out')
        spin(2.)
        live = json.loads((folder/'scene_status.json').read_text())
        geometry_error = sum((a-b)**2 for a,b in zip(state['pose'][:3],live['contact_face_dh_base_mm']))**.5
        if geometry_error > .1:raise ValueError(f'Live ROS TCP differs from USD contact face: {geometry_error} mm')
        if max(abs(v) for v in state['force']) > 1.:raise ValueError('Unexpected free-space force before TxtLoad')
        dump(folder/'before_txtload.json', {'ros':state, 'scene':live, 'contact_face_error_mm':geometry_error})
        request = SingleArmCommand.Request()
        request.command_mode = 'TxtLoad';request.target_pose=[];request.target_velocity=0.;request.load_file=''
        future = client.call_async(request)
        # TxtLoad responds only after processing, measurement save and home return.
        deadline = time.monotonic()+900.
        last_report=0.;captured=False
        while True:
            spin(.1)
            t=state['telemetry']
            if future.done() and not future.result().success:
                raise RuntimeError(str(future.result()))
            if t and t[12] != 0.:raise RuntimeError(f'Controller fault {t[12]} at cursor {t[4]}')
            if t and t[4] > .5*t[5] and not captured:
                dump(folder/'capture.json',{'filename':'gui_processing.png'});captured=True
            now=time.monotonic()
            if now-last_report>5.:
                dump(ROOT/'status.json',{'phase':'running','mode':mode,'recording':str(recording),'state':state})
                print(json.dumps({'mode':mode,'samples':state['telemetry_count'],
                    'elapsed_s': t[10] if t else None,'cursor_mm':t[4] if t else None,
                    'force_n':t[21] if t else state['force']}),flush=True)
                last_report=now
            sessions=list(recording.glob('removal/session_*/summary.json'))
            log=(folder/'launch.log').read_text()
            if sessions and 'home returned; Idling' in log:
                summary=json.loads(sessions[0].read_text())
                if not summary['valid_completed_run']:raise ValueError('Incomplete measurement')
                break
            if now>deadline:raise TimeoutError('Processing/home return exceeded 15 minutes')
        while not future.done():
            spin(.05)
            if time.monotonic()>deadline:raise TimeoutError('TxtLoad final response timed out')
        response = future.result()
        dump(folder/'txtload_response.json', {k:getattr(response,k) for k in response.get_fields_and_field_types()})
        if not response.success:raise RuntimeError(str(response))
        dump(folder/'capture.json',{'filename':'gui_completed.png'});spin(2.)
        dump(folder/'completed.json', {'recording':str(recording),'summary':summary,'final_state':state})
        return recording
    except Exception as exc:
        dump(folder/'failure.json',{'error':str(exc),'traceback':traceback.format_exc(),'state':state})
        raise
    finally:
        dump(folder/'stop.json', {'requested':True})
        stop(launch);stop(bridge)
        if scene and scene.poll() is None:
            try:scene.wait(timeout=20.)
            except subprocess.TimeoutExpired:stop(scene)
        node.destroy_node()
        for stream in logs:stream.close()

rclpy.init()
try:
    mode5=trial(5)
    mode6=trial(6)
    subprocess.run([PYTHON,str(NRS/'scripts/compare_fixed_force_gui_runs.py'),
        '--mode5',str(mode5),'--mode6',str(mode6),'--output',str(ROOT/'comparison')],check=True)
    dump(ROOT/'status.json', {'phase':'completed','mode5':str(mode5),'mode6':str(mode6),
        'comparison':str(ROOT/'comparison/comparison.json')})
except Exception as exc:
    dump(ROOT/'status.json', {'phase':'failed','error':str(exc)})
    raise
finally:
    rclpy.shutdown()
