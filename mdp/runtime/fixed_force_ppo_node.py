#!/usr/bin/env python3
"""ROS Humble frontend and Python 3.11 inference worker for the cv11p13 PPO."""
import argparse
import csv
import json
from pathlib import Path
import select
import subprocess
import sys
import time

if not __package__:
    sys.path.insert(0, str(next(p / 'source/nrs_rl' for p in Path(__file__).resolve().parents
                              if (p / 'source/nrs_rl/nrs_rl').is_dir())))


def worker(args):
    from nrs_rl.tasks.manager_based.nrs_rl.mdp.runtime.fixed_force_ppo_runtime import (
        FixedForcePPORuntime, OBSERVATIONS, TELEMETRY)
    import torch
    runtime = FixedForcePPORuntime(args.checkpoint, args.reference, args.force_control_mode)
    with torch.inference_mode():
        action = runtime.actor(torch.zeros(1, 16))
    ready = {'ok': bool(torch.isfinite(action).all()), 'algorithm': 'PPO',
             'checkpoint_sha256': runtime.sha256, 'observation_shape': [1, 16],
             'action_shape': list(action.shape), 'policy_period_s': .08,
             'control_period_s': .008, 'force_control_mode': args.force_control_mode}
    print(json.dumps(ready), flush=True)
    if args.self_test:
        return
    stream = None
    if args.log_directory:
        folder = Path(args.log_directory)
        folder.mkdir(parents=True, exist_ok=True)
        stream = (folder / 'policy.csv').open('x', newline='', buffering=1)
        writer = csv.writer(stream)
        writer.writerow(TELEMETRY + ['obs_' + n for n in OBSERVATIONS]
                        + ['feed_action', 'decision', 'missing_samples', 'latency_ms', 'actor_feed_action'])
    try:
        for line in sys.stdin:
            started = time.perf_counter()
            values = json.loads(line)
            try:
                result = runtime.process(values)
            except Exception as exc:
                if args.log_directory:
                    (Path(args.log_directory) / 'policy_failure.json').write_text(json.dumps({
                        'error': str(exc), 'checkpoint_sha256': runtime.sha256,
                        'telemetry': dict(zip(TELEMETRY, values))}, indent=2) + '\n')
                raise
            result['latency_ms'] = (time.perf_counter() - started) * 1000.
            if stream:
                writer.writerow(values + result['observation'] + result['action'][2:]
                                + [int(result['decision']), result['missing_samples'], result['latency_ms']]
                                + result['actor_action'])
            print(json.dumps(result, allow_nan=False), flush=True)
    finally:
        if stream:
            stream.close()


def ros(args):
    import rclpy
    from rclpy.clock import Clock, ClockType
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Float64MultiArray
    from nrs_rl.tasks.manager_based.nrs_rl.mdp.runtime.fixed_force_ppo_deployment import ACTOR_SHA
    command = [args.worker_python, str(Path(__file__).resolve()), '--worker',
               '--checkpoint', args.checkpoint, '--reference', args.reference,
               '--force-control-mode', str(args.force_control_mode)]
    if args.log_directory:
        command += ['--log-directory', args.log_directory]
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def response(timeout):
        if not select.select([child.stdout], [], [], timeout)[0]:
            raise RuntimeError('PPO worker response timed out')
        line = child.stdout.readline()
        if not line:
            raise RuntimeError('PPO worker exited')
        value = json.loads(line)
        if 'error' in value:
            raise RuntimeError(value['error'])
        return value

    try:
        ready = response(30.)
        if (not ready.get('ok') or ready.get('algorithm') != 'PPO'
                or ready.get('checkpoint_sha256') != ACTOR_SHA
                or ready.get('observation_shape') != [1, 16] or ready.get('action_shape') != [1, 1]
                or ready.get('force_control_mode') != args.force_control_mode):
            raise RuntimeError('PPO worker handshake failed')
        rclpy.init(args=args.ros_args)

        class PolicyNode(Node):
            def __init__(self):
                super().__init__('fixed_force_ppo_policy_node')
                qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
                self.action = self.create_publisher(Float64MultiArray, '/ur10skku/fixed_force/action', qos)
                self.observation = self.create_publisher(Float64MultiArray, '/ur10skku/fixed_force/observation', qos)
                self.subscription = self.create_subscription(Float64MultiArray,
                    '/ur10skku/fixed_force/telemetry', self.callback, qos)
                # Display data only; singleArm_motion prints it with robot state.
                self.status_publisher = self.create_publisher(Float64MultiArray,
                    '/ur10skku/fixed_force/status',
                    QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
                self.count = self.last_count = 0
                self.latency = self.feed_action = 0.
                # Match the robot state monitor's 100 ms wall cadence.
                self.last_status_time = time.monotonic()
                self.status_clock = Clock(clock_type=ClockType.STEADY_TIME)
                self.timer = self.create_timer(.1, self.status, clock=self.status_clock)
                self.get_logger().info('PPO cv11p13 ready; SHA-256: ' + ready['checkpoint_sha256'])
                self.get_logger().info(f'Mode {args.force_control_mode}; fixed 20 N; '
                    '16D observation -> scalar feed; policy 12.5 Hz, control 125 Hz. Waiting for TxtLoad.')

            def callback(self, message):
                child.stdin.write(json.dumps(list(message.data), allow_nan=False) + '\n')
                child.stdin.flush()
                result = response(.20)
                self.action.publish(Float64MultiArray(data=result['action']))
                self.observation.publish(Float64MultiArray(data=result['observation']))
                self.count += 1
                self.latency = result['latency_ms']
                self.feed_action = result['action'][2]

            def status(self):
                now = time.monotonic()
                elapsed = now - self.last_status_time
                rate = (self.count - self.last_count) / elapsed if elapsed > 0. else 0.
                self.status_publisher.publish(Float64MultiArray(data=[
                    rate, 20., 6. + 6. * self.feed_action, self.latency]))
                self.last_count = self.count
                self.last_status_time = now

        node = PolicyNode()
        try:
            rclpy.spin(node)
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
    finally:
        child.stdin.close()
        try:
            child.wait(timeout=2.)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                child.wait(timeout=2.)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force-control-mode', type=int, choices=(5, 6), default=5)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--worker-python', default=str(Path.home() / 'anaconda3/envs/env_isaaclab/bin/python'))
    parser.add_argument('--log-directory')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args, args_ros = parser.parse_known_args()
    args.ros_args = args_ros
    try:
        if args.worker or args.self_test:
            worker(args)
        else:
            ros(args)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if args.worker:
            print(json.dumps({'error': str(exc)}), flush=True)
        raise


if __name__ == '__main__':
    main()
