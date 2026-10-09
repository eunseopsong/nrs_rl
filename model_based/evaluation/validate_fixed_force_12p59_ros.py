#!/usr/bin/env python3
"""Private-domain ROS transport test with synthetic states, no robot bridge."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT as _REPO_ROOT
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=_REPO_ROOT


def main(force_mode):
    if os.environ.get('ROS_DOMAIN_ID')!='228' or os.environ.get('ROS_LOCALHOST_ONLY')!='1':
        raise RuntimeError('Test requires private ROS_DOMAIN_ID=228 and localhost-only transport')
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
    from std_msgs.msg import Float64MultiArray,String
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import WrenchStamped
    from rosgraph_msgs.msg import Clock
    import numpy as np
    out=ROOT/'logs/experiments/fixed_force_12p59/ros_integration'/f'mode{force_mode}'
    out.mkdir(parents=True,exist_ok=True)
    stamp=str(time.time_ns());run=out/stamp;run.mkdir()
    fixture=json.loads((ROOT/'deployment/fixed_force_12p59/isolated_test_fixture.json').read_text())
    package=Path('/home/eunseop/dev_ws/install/Y2RobMotion/share/Y2RobMotion')
    trajectory=package/'txtcmd/cmd_continue9D_preston_d14_20n.txt'
    motions=ROOT/'deployment/fixed_force_12p59/build/singleArm_motion'
    actor=ROOT/'deployment/fixed_force_12p59/checkpoints/actor.pt'
    reference=ROOT/'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/datasets/cmd_continue9D_flat.h5'
    rclpy.init();node=Node('spatial_isolated_fixture')
    qos=QoSProfile(depth=100,reliability=ReliabilityPolicy.RELIABLE)
    retained=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
    state={'q':fixture['q'],'follow_commands':True,'clock':1.,'force':68.89791117615832,'poses':[],'telemetry':[],'actions':[]}
    joints=node.create_publisher(JointState,'/fixed_force_validation/joint_states',10)
    wrench=node.create_publisher(WrenchStamped,'/fixed_force_validation/ftdata',10)
    clock=node.create_publisher(Clock,'/clock',10)
    route=node.create_publisher(Float64MultiArray,'/fixed_force_validation/ppo/trajectory9d',retained)
    mode=node.create_publisher(String,'/fixed_force_validation/cmdMode',10)
    action=node.create_publisher(Float64MultiArray,'/fixed_force_validation/fixed_force/action',qos)
    subscriptions=[
        node.create_subscription(Float64MultiArray,'/fixed_force_validation/discarded_commands',
            lambda m:state.update(q=list(m.data)) if state['follow_commands'] else None,10),
        node.create_subscription(Float64MultiArray,'/fixed_force_validation/currentP',lambda m:state['poses'].append(list(m.data)),10),
        node.create_subscription(Float64MultiArray,'/fixed_force_validation/fixed_force/telemetry',lambda m:state['telemetry'].append(list(m.data)),qos),
        node.create_subscription(Float64MultiArray,'/fixed_force_validation/fixed_force/action',lambda m:state['actions'].append(list(m.data)),qos)]
    names=['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint']
    normal=np.asarray(fixture['htm'])[:3,2]
    def tick(count=1):
        for _ in range(count):
            state['clock']+=.008
            ns=int(round(state['clock']*1e9));c=Clock();c.clock.sec=ns//10**9;c.clock.nanosec=ns%10**9
            j=JointState();j.name=names;j.position=state['q'];j.header.stamp=c.clock
            f=WrenchStamped();f.header.stamp=c.clock
            f.wrench.force.x=float(normal[0]*state['force']);f.wrench.force.y=float(normal[1]*state['force']);f.wrench.force.z=float(normal[2]*state['force'])
            joints.publish(j);wrench.publish(f);clock.publish(c)
            deadline=time.monotonic()+.012
            while time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.0005)
    motion_cmd=[str(motions),'--ros-args','-r','__node:=spatial_motion_validation',
        '-p','setup_file:='+str(package/'config/preston_d14.yaml'),'-p',f'force_control_mode:={force_mode}',
        '-p','spatial_policy_enabled:=true','-p','use_sim_time:=true',
        '-p','expected_trajectory_file:='+str(trajectory),'-p','raw_logging_enabled:=true',
        '-p','raw_logging_root:='+str(run),'-p','raw_logging_run:=mock',
        '-p','preston_tangent_damping_ns_m:=3329.844766906654',
        '-p','preston_tangent_stiffness_n_m:=12434.33963608066',
        '-r','/forward_position_controller/commands:=/fixed_force_validation/discarded_commands',
        '-r','/joint_states:=/fixed_force_validation/joint_states']
    for suffix in ('currentJ','currentP','currentF','currentMDK','targetJ','targetP','targetF','ctlMode',
                   'cmdMotion','cmdMode','joy_move','joint_states','ftdata','ppo/trajectory9d','ppo/trajectory_done',
                   'fixed_force/telemetry','fixed_force/action'):
        motion_cmd+=['-r',f'/ur10skku/{suffix}:=/fixed_force_validation/{suffix}']
    motion_cmd+=['-r','/ppo/observation:=/fixed_force_validation/legacy_observation']
    policy_cmd=['/usr/bin/python3',str(ROOT/'deployment/fixed_force_12p59/fixed_force_policy_node.py'),'--checkpoint',str(actor),
        '--reference',str(reference),'--force-control-mode',str(force_mode),'--log-directory',str(run/'policy'),'--ros-args','-p','use_sim_time:=true']
    for suffix in ('telemetry','action','observation'):
        policy_cmd+=['-r',f'/ur10skku/fixed_force/{suffix}:=/fixed_force_validation/fixed_force/{suffix}']
    logs=[(run/'motion.log').open('w'),(run/'policy.log').open('w')]
    children=[];checks=[]
    try:
        motion=subprocess.Popen(motion_cmd,stdout=logs[0],stderr=subprocess.STDOUT);children.append(motion)
        policy=subprocess.Popen(policy_cmd,stdout=logs[1],stderr=subprocess.STDOUT);children.append(policy)
        for _ in range(700):
            tick()
            if state['poses'] and policy.poll() is None and (run/'policy/policy.csv').exists():break
        assert state['poses'],'Controller never initialized'
        assert np.linalg.norm(np.array(state['poses'][-1][:3])-fixture['pose'][:3])<.05
        route.publish(Float64MultiArray(data=np.loadtxt(trajectory).ravel().tolist()));tick(25)
        mode.publish(String(data='Force'))
        for _ in range(600):
            if state['telemetry']:
                # Reproduce the measured GUI entry peak, then a settling
                # force trace. The synthetic robot follows actual commands.
                state['force']=state['telemetry'][-1][8]+48.89791117615832*np.exp(-state['telemetry'][-1][10]/.3)
            tick()
            if motion.poll() is not None or policy.poll() is not None:raise RuntimeError('Runtime process exited; inspect '+str(run))
            if len(state['telemetry'])>=250:break
        t=np.asarray(state['telemetry']);a=np.asarray(state['actions'])
        assert t.shape[0]>=250 and a.shape[0]>=200,(t.shape,a.shape)
        assert np.all(t[:,12]==0),np.unique(t[:,12])
        entry=t[t[:,21]>40.]
        assert len(entry)>10 and np.max(entry[:,21])>68.
        assert np.all(entry[:,4]==0.),'Cursor advanced during overload shielding'
        assert np.max(entry[:,17])-fixture['pose'][2]>.05,'Normal controller did not unload entry contact'
        assert t[-1,4]>1.
        np.testing.assert_array_equal(t[:,8],np.full(len(t),20.))
        assert np.all(t[:,7]==0.) and np.all(t[:,9]==0.)
        assert a.shape[1]==3 and np.max(np.abs(a[:,2]))<=1.000001
        assert np.all(t[:,0]==2.)
        if force_mode==5:
            assert np.max(np.abs(a[:,2]))>.1
        else:
            np.testing.assert_array_equal(a[:,2],np.zeros(len(a)))
            import csv
            with (run/'policy/policy.csv').open() as f: recorded=list(csv.DictReader(f))
            assert any(abs(float(v['actor_feed_action']))>.1 for v in recorded)
        checks+=['actual_checkpoint_16D_scalar_action_transport','fixed_target_20N_all_ticks',
            'forward_only_no_return_actions','motion_advances_with_fresh_policy',
            'observed_68_9N_entry_survives_and_normal_controller_unloads']
        policy.send_signal(signal.SIGINT);policy.wait(timeout=5)
        tick(65)
        held=np.asarray(state['telemetry'][-15:]);assert np.ptp(held[:,4])<1e-9
        cursor=held[-1,4];last=state['telemetry'][-1]
        action.publish(Float64MultiArray(data=[last[1]+1.,last[2],1.]));tick(20)
        assert abs(state['telemetry'][-1][4]-cursor)<1e-9
        action.publish(Float64MultiArray(data=[last[1],1.,1.]));tick(20)
        assert abs(state['telemetry'][-1][4]-cursor)<1e-9
        checks+=['dead_worker_holds_cursor','foreign_epoch_and_replayed_action_rejected']
        last=state['telemetry'][-1]
        action.publish(Float64MultiArray(data=[last[1],last[2],0.,1.,-1.]));tick(20)
        assert abs(state['telemetry'][-1][4]-cursor)<1e-9
        assert state['telemetry'][-1][8]==20.
        checks.append('legacy_force_action_packet_rejected')
        if force_mode==6:
            last=state['telemetry'][-1]
            action.publish(Float64MultiArray(data=[last[1],last[2],1.]));tick(20)
            assert abs(state['telemetry'][-1][4]-cursor)<1e-9
            assert state['telemetry'][-1][8]==20.
            checks.append('mode6_rejects_nonconstant_applied_action')
        last=state['telemetry'][-1]
        action.publish(Float64MultiArray(data=[last[1],last[2],0.]));tick(10)
        assert state['telemetry'][-1][4]>cursor
        checks+=['fresh_action_resumes_bounded_scheduler']
        state['force']=100.1
        for _ in range(125):
            tick()
            if int(state['telemetry'][-1][12])&1:break
        assert int(state['telemetry'][-1][12])&1
        fault_cursor=state['telemetry'][-1][4];state['force']=20.
        for _ in range(125):
            tick()
            if abs(state['telemetry'][-1][21]-20.)<.01:break
        assert state['telemetry'][-1][12]!=0 and state['telemetry'][-1][4]==fault_cursor
        checks+=['force_above_100N_latches_and_holds']
        previous_epoch=state['telemetry'][-1][1]
        # TxtLoad normally leaves Force and approaches the path start before
        # starting a new pass. Restore that state in this synthetic fixture;
        # jumping directly from mid-path must still trip the command-step guard.
        mode.publish(String(data='Idling'));tick(25)
        state['follow_commands']=False;state['q']=fixture['q'];tick(25)
        assert np.linalg.norm(np.array(state['poses'][-1][:3])-fixture['pose'][:3])<.05
        route.publish(Float64MultiArray(data=np.loadtxt(trajectory).ravel().tolist()))
        tick(25);mode.publish(String(data='Force'))
        for _ in range(125):
            tick()
            if state['telemetry'][-1][1]==previous_epoch+1:break
        assert state['telemetry'][-1][1]==previous_epoch+1 and state['telemetry'][-1][12]==0, state['telemetry'][-1]
        assert state['telemetry'][-1][4]==0.
        assert abs(state['telemetry'][-1][8]-20.)<1.e-9
        checks+=['new_trajectory_resets_policy_epoch_force_and_cursor']
        result={'ok':True,'checks':checks,'private_domain':228,'actual_robot_connected':False,'mode':force_mode,
            'samples':len(t),'action_messages':len(a),'run':str(run),'checkpoint_sha256':'6d1dc47bc915b393880d2e6e380f44fbcb82e542de0fa12e4f02eb47c97147df'}
        (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    except Exception:
        (run/'failure.json').write_text(json.dumps({'checks':checks,'last_telemetry':state['telemetry'][-30:],
            'last_pose':state['poses'][-1] if state['poses'] else None},indent=2)+'\n')
        raise
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait()
        for f in logs:f.close()
        node.destroy_node();rclpy.shutdown()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--mode',type=int,choices=(5,6),required=True)
    main(parser.parse_args().mode)
