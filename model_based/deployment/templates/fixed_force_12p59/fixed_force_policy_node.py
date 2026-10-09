#!/usr/bin/env python3
"""ROS Humble frontend + isolated Python 3.11 spatial-policy worker."""
import argparse
import csv
import json
from pathlib import Path
import select
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/model_based/runtime/fixed_force_12p59_source'))


def worker(args):
    from fixed_force_policy_runtime import FixedForcePolicyRuntime, TELEMETRY
    from nrs_fixed_force_12p59.spatial_uniformity import OBSERVATIONS
    runtime = FixedForcePolicyRuntime(args.checkpoint,args.reference,args.force_control_mode)
    probe = runtime.actor
    import torch
    with torch.inference_mode():
        shape = list(probe(torch.zeros(1,16)).shape)
    ready = {'ok':True,'checkpoint_sha256':runtime.sha256,'observation_shape':[1,16],
             'action_shape':shape,'policy_period_s':.08,'control_period_s':.008,'force_control_mode':args.force_control_mode}
    print(json.dumps(ready),flush=True)
    if args.self_test:return
    stream = None
    if args.log_directory:
        folder=Path(args.log_directory);folder.mkdir(parents=True,exist_ok=True)
        stream=(folder/'policy.csv').open('x',newline='',buffering=1)
        writer=csv.writer(stream)
        writer.writerow(TELEMETRY+['obs_'+n for n in OBSERVATIONS]+['feed_action','decision','missing_samples','latency_ms','actor_feed_action'])
    try:
        for line in sys.stdin:
            started=time.perf_counter()
            values=json.loads(line)
            try:
                result=runtime.process(values)
            except Exception as exc:
                if args.log_directory:
                    # Preserve the rejected first sample too: policy.csv may
                    # contain only its header when the controller faults at entry.
                    (Path(args.log_directory)/'policy_failure.json').write_text(json.dumps({
                        'error':str(exc),'checkpoint_sha256':runtime.sha256,
                        'telemetry':dict(zip(TELEMETRY,values))},indent=2)+'\n')
                raise
            result['latency_ms']=(time.perf_counter()-started)*1000.
            if stream:
                writer.writerow(values+result['observation']+result['action'][2:]+
                    [int(result['decision']),result['missing_samples'],result['latency_ms']]+result['actor_action'])
            print(json.dumps(result,allow_nan=False),flush=True)
    finally:
        if stream:stream.close()


def ros(args):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Float64MultiArray
    command=[args.worker_python,str(Path(__file__).resolve()),'--worker','--checkpoint',args.checkpoint,'--reference',args.reference,'--force-control-mode',str(args.force_control_mode)]
    if args.log_directory:command += ['--log-directory',args.log_directory]
    child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
    def response(timeout):
        if not select.select([child.stdout],[],[],timeout)[0]:raise RuntimeError('Spatial worker response timed out')
        line=child.stdout.readline()
        if not line:raise RuntimeError('Spatial worker exited')
        value=json.loads(line)
        if 'error' in value:raise RuntimeError(value['error'])
        return value
    try:
        ready=response(30.)
        if not ready.get('ok') or ready['action_shape'] != [1,1]:raise RuntimeError('Spatial worker handshake failed')
        rclpy.init(args=args.ros_args)
        class PolicyNode(Node):
            def __init__(self):
                super().__init__('fixed_force_policy_node')
                qos=QoSProfile(depth=100,reliability=ReliabilityPolicy.RELIABLE)
                self.action=self.create_publisher(Float64MultiArray,'/ur10skku/fixed_force/action',qos)
                self.observation=self.create_publisher(Float64MultiArray,'/ur10skku/fixed_force/observation',qos)
                self.sub=self.create_subscription(Float64MultiArray,'/ur10skku/fixed_force/telemetry',self.callback,qos)
                self.count=0;self.last_count=0;self.latency=0.;self.feed_action=0.
                self.timer=self.create_timer(1.,self.status)
                self.get_logger().info(f'PAIRED Mode {args.force_control_mode}: '+('learned feed; FIXED 20 N' if args.force_control_mode==5 else 'FIXED 20 N / 6 mm/s; actor output not applied'))
                self.get_logger().info('Fixed-force 12.59% checkpoint SHA-256: '+ready['checkpoint_sha256'])
                self.get_logger().info('16D -> scalar feed; target force fixed 20 N; virtual spindle 1000 RPM; no force action or local returns; policy 12.5 Hz, control 125 Hz. Waiting for TxtLoad/Force telemetry.')
            def callback(self,msg):
                child.stdin.write(json.dumps(list(msg.data),allow_nan=False)+'\n');child.stdin.flush()
                try:result=response(.20)
                except Exception as exc:
                    self.get_logger().error('Fixed-force policy stopped: '+str(exc))
                    raise
                self.action.publish(Float64MultiArray(data=result['action']))
                self.observation.publish(Float64MultiArray(data=result['observation']))
                self.count+=1;self.latency=result['latency_ms'];self.feed_action=result['action'][2]
            def status(self):
                self.get_logger().info(f'telemetry/action={self.count-self.last_count} Hz, fixed force=20.00 N, requested feed={6.+6.*self.feed_action:.2f} mm/s, worker={self.latency:.2f} ms')
                self.last_count=self.count
        node=PolicyNode()
        try:rclpy.spin(node)
        finally:
            node.destroy_node()
            if rclpy.ok():rclpy.shutdown()
    finally:
        if child.stdin:child.stdin.close()
        try:child.wait(timeout=2.)
        except subprocess.TimeoutExpired:
            child.terminate();child.wait(timeout=2.)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--force-control-mode',type=int,choices=(5,6),default=5)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--reference',required=True)
    p.add_argument('--worker-python',default=str(Path.home()/'anaconda3/envs/env_isaaclab/bin/python'))
    p.add_argument('--log-directory')
    p.add_argument('--worker',action='store_true')
    p.add_argument('--self-test',action='store_true')
    args,unknown=p.parse_known_args();args.ros_args=unknown
    try:
        if args.worker or args.self_test:worker(args)
        else:ros(args)
    except KeyboardInterrupt:pass
    except Exception as exc:
        if args.worker:print(json.dumps({'error':str(exc)}),flush=True)
        raise
