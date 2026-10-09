#!/usr/bin/env python3
"""Private-domain Trigger/telemetry recorder test, no motion processes."""
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


def main():
    if os.environ.get('ROS_DOMAIN_ID')!='229' or os.environ.get('ROS_LOCALHOST_ONLY')!='1':
        raise RuntimeError('Use private domain 229, localhost only')
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile,ReliabilityPolicy
    from std_msgs.msg import Float64MultiArray
    from std_srvs.srv import Trigger
    import numpy as np
    root=ROOT/'deployment/fixed_force_12p59'
    out=ROOT/'logs/experiments/fixed_force_12p59/measurement_integration'/str(time.time_ns())
    out.mkdir(parents=True)
    reference='/home/eunseop/dev_ws/install/Y2RobMotion/share/Y2RobMotion/txtcmd/cmd_continue9D_preston_d14_20n.txt'
    path=np.loadtxt(reference);length=np.linalg.norm(np.diff(path[:,:3],axis=0),axis=1).sum()
    rclpy.init();node=Node('paired_measurement_fixture')
    pub=node.create_publisher(Float64MultiArray,'/paired_measurement_fixture/telemetry',
        QoSProfile(depth=1000,reliability=ReliabilityPolicy.RELIABLE))
    start=node.create_client(Trigger,'/paired_measurement_fixture/start')
    end=node.create_client(Trigger,'/paired_measurement_fixture/end')
    command=['/usr/bin/python3','-s',str(root/'paired_removal_node.py'),'--ros-args',
        '-p','logs_root:='+str(out/'removal'),'-p','reference_trajectory_file:='+reference,
        '-r','/ur10skku/fixed_force/telemetry:=/paired_measurement_fixture/telemetry',
        '-r','/polishing_removal_node/start:=/paired_measurement_fixture/start',
        '-r','/polishing_removal_node/end:=/paired_measurement_fixture/end']
    def wait(future, timeout=30):
        rclpy.spin_until_future_complete(node,future,timeout_sec=timeout)
        if not future.done():raise RuntimeError('Service response timeout')
        return future.result()
    log=(out/'recorder.log').open('w')
    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
    try:
        assert start.wait_for_service(timeout_sec=15)
        while pub.get_subscription_count()==0:
            rclpy.spin_once(node,timeout_sec=.05)
        assert wait(start.call_async(Trigger.Request())).success
        rows=[]
        for seq in range(1,402):
            cursor=length if seq==401 else min(.048*(seq-1),length-.1)
            x,y,z=path[0,:3];x+=(seq-1)*.048
            v=[2.,1.,float(seq),seq*.008,cursor,length,cursor,0.,20.,0.,seq*.008,
                0.,0.,6.,0.,x,y,z,0.,0.,1.,20.,float(seq==401)]
            # Deliberately send /end before the final telemetry sample, to
            # exercise cross-topic completion ordering in the real services.
            if seq==401:finish=end.call_async(Trigger.Request())
            pub.publish(Float64MultiArray(data=v));rows.append(v)
            rclpy.spin_once(node,timeout_sec=.005)
        response=wait(finish,45);assert response.success,response.message
        session,=list((out/'removal').glob('session_*'))
        recorded=np.loadtxt(session/'telemetry.csv',delimiter=',',skiprows=1)
        np.testing.assert_array_equal(recorded,np.asarray(rows))
        summary=json.loads((session/'summary.json').read_text())
        assert summary['valid_completed_run'] and summary['samples']==401
        assert summary['virtual_spindle_rpm']==1000.
        assert summary['metrics']['grid_volume_over_k'] > 20.*400*.048*50.
        assert (session/'01_removal_heatmap.png').stat().st_size>1000
        assert not wait(start.call_async(Trigger.Request())).success
        result={'ok':True,'private_domain':229,'actual_robot_connected':False,
            'checks':['start_end_services','atomic_controller_snapshots_exactly_preserved',
                'same_tick_force_rotating_disk_integral','completion_event_arriving_before_last_snapshot',
                'summary_and_plots_saved','second_txtload_rejected'], 'run':str(out)}
        (out.parent/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=5)
        log.close();node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
