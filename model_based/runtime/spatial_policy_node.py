#!/usr/bin/env python3
"""ROS Humble frontend + isolated Python 3.11 spatial-policy worker."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import argparse
import csv
import json
from pathlib import Path
import select
import subprocess
import sys
import time


def worker(args):
    from nrs_rl.tasks.manager_based.nrs_rl.model_based.runtime.spatial_policy_runtime import SpatialPolicyRuntime, TELEMETRY
    from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.spatial_uniformity import OBSERVATIONS
    runtime = SpatialPolicyRuntime(args.checkpoint,args.reference)
    probe = runtime.actor
    import torch
    with torch.inference_mode():
        shape = list(probe(torch.zeros(1,16)).shape)
    ready = {'ok':True,'checkpoint_sha256':runtime.sha256,'observation_shape':[1,16],
             'action_shape':shape,'policy_period_s':.08,'control_period_s':.008}
    print(json.dumps(ready),flush=True)
    if args.self_test:return
    stream = None
    if args.log_directory:
        folder=Path(args.log_directory);folder.mkdir(parents=True,exist_ok=True)
        stream=(folder/'policy.csv').open('x',newline='',buffering=1)
        writer=csv.writer(stream)
        writer.writerow(TELEMETRY+['obs_'+n for n in OBSERVATIONS]+['feed_action','force_action','return_action','decision','missing_samples','latency_ms'])
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
                    [int(result['decision']),result['missing_samples'],result['latency_ms']])
            print(json.dumps(result,allow_nan=False),flush=True)
    finally:
        if stream:stream.close()


def ros(args):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Float64MultiArray
    command=[args.worker_python,str(Path(__file__).resolve()),'--worker','--checkpoint',args.checkpoint,'--reference',args.reference]
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
        if not ready.get('ok') or ready['action_shape'] != [1,3]:raise RuntimeError('Spatial worker handshake failed')
        rclpy.init(args=args.ros_args)
        class PolicyNode(Node):
            def __init__(self):
                super().__init__('spatial_policy_node')
                qos=QoSProfile(depth=100,reliability=ReliabilityPolicy.RELIABLE)
                self.action=self.create_publisher(Float64MultiArray,'/ur10skku/spatial/action',qos)
                self.observation=self.create_publisher(Float64MultiArray,'/ur10skku/spatial/observation',qos)
                self.sub=self.create_subscription(Float64MultiArray,'/ur10skku/spatial/telemetry',self.callback,qos)
                self.count=0;self.last_count=0;self.latency=0.;self.force_action=0.
                self.timer=self.create_timer(1.,self.status)
                self.get_logger().info('Spatial checkpoint SHA-256: '+ready['checkpoint_sha256'])
                self.get_logger().info('16D -> 3 actions; policy 12.5 Hz, control 125 Hz. Waiting for TxtLoad/Force telemetry.')
            def callback(self,msg):
                child.stdin.write(json.dumps(list(msg.data),allow_nan=False)+'\n');child.stdin.flush()
                try:result=response(.20)
                except Exception as exc:
                    self.get_logger().error('Spatial policy stopped: '+str(exc))
                    raise
                self.action.publish(Float64MultiArray(data=result['action']))
                self.observation.publish(Float64MultiArray(data=result['observation']))
                self.count+=1;self.latency=result['latency_ms'];self.force_action=result['action'][3]
            def status(self):
                self.get_logger().info(f'telemetry/action={self.count-self.last_count} Hz, force request={20.+6.*self.force_action:.2f} N, worker={self.latency:.2f} ms')
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
