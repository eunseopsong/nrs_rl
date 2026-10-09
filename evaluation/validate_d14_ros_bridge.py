#!/usr/bin/env python3
"""Replay D14 observations through the ROS bridge on unique test-only topics.

Run with ROS Humble's /usr/bin/python3. This never starts a robot controller,
publishes joint/force commands, or calls the motion service.
"""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT as _REPO_ROOT
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--worker-python", default="/home/eunseop/anaconda3/envs/env_isaaclab/bin/python")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controller-config", type=Path)
    parser.add_argument("--domain-id", type=int, default=187)
    args = parser.parse_args()
    if not 0 <= args.domain_id <= 232:
        parser.error("domain-id must be in 0..232")
    worker = _REPO_ROOT / 'scripts/skrl/ppo_policy_node.py'
    prepare = """
import importlib.util,json,sys,numpy as np,torch
spec=importlib.util.spec_from_file_location('worker',sys.argv[1])
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
agent,_=worker._build_agent(sys.argv[2])
worker._validate_scheduler_contract(agent.velocity_contract,worker.load_controller_config(sys.argv[4] or None))
with np.load(sys.argv[3]) as trace:
    raw=trace['policy_observation']; obs=raw[np.linspace(0,len(raw)-1,128,dtype=int)].astype(np.float32)
with torch.inference_mode():
    expected=agent.act(torch.from_numpy(obs))[2]['mean_actions'].reshape(-1).tolist()
print(json.dumps({'observations':obs.tolist(),'expected':expected}))
"""
    data = json.loads(subprocess.run([args.worker_python, "-c", prepare, str(worker),
        str(args.checkpoint.resolve()), str(args.trace.resolve()),
        str(args.controller_config.resolve()) if args.controller_config else ""], check=True,
        capture_output=True, text=True, timeout=30).stdout)
    os.environ["ROS_DOMAIN_ID"] = str(args.domain_id)
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    os.environ.setdefault("ROS_LOG_DIR", "/tmp/nrs_d14_ros_logs")
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Float64, Float64MultiArray

    prefix = "/nrs_d14_validation_" + uuid.uuid4().hex
    rclpy.init(args=[])
    node = rclpy.create_node("d14_validation")
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    publisher = node.create_publisher(Float64MultiArray, prefix + "/observation", qos)
    received = []
    subscription = node.create_subscription(Float64, prefix + "/action",
        lambda msg: received.append((time.perf_counter(), msg.data)), qos)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.with_suffix(".policy.log")
    process = None
    try:
        with log_path.open("w") as log:
            command = ["/usr/bin/python3", str(worker), "--checkpoint",
                str(args.checkpoint.resolve()), "--worker-python", args.worker_python,
                "--observation-topic", prefix + "/observation", "--action-topic", prefix + "/action"]
            if args.controller_config is not None:
                command += ["--controller-config", str(args.controller_config.resolve())]
            process = subprocess.Popen(command,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 20
            while publisher.get_subscription_count() == 0 or node.count_publishers(prefix + "/action") == 0:
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("ROS policy startup/discovery failed; see " + str(log_path))
                rclpy.spin_once(node, timeout_sec=.02)
            latency, errors = [], []
            for observation, expected in zip(data["observations"], data["expected"]):
                received.clear()
                started = time.perf_counter()
                publisher.publish(Float64MultiArray(data=observation))
                while not received and time.perf_counter() - started < .5:
                    rclpy.spin_once(node, timeout_sec=.001)
                if len(received) != 1:
                    raise RuntimeError("Missing/duplicate ROS action")
                stamp, action = received[0]
                if not math.isfinite(action) or abs(action - expected) > 2.e-6:
                    raise RuntimeError(f"ROS action mismatch: {action} vs {expected}")
                latency.append((stamp - started) * 1000)
                errors.append(abs(action - expected))
                while time.perf_counter() - started < .008:
                    rclpy.spin_once(node, timeout_sec=.0005)
            # The old C++ observation shape must not produce an action.
            received.clear()
            publisher.publish(Float64MultiArray(data=[0.] * 12))
            deadline = time.monotonic() + .12
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=.005)
            if received:
                raise RuntimeError("ROS bridge accepted the obsolete 12D observation")
            publisher.publish(Float64MultiArray(data=data["observations"][0]))
            deadline = time.monotonic() + .5
            while not received and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=.001)
            if not received or abs(received[0][1] - data["expected"][0]) > 2.e-6:
                raise RuntimeError("ROS bridge did not recover after malformed observation")
            report = {"ok": True, "observations": len(errors), "maximum_action_error": max(errors),
                "ros_roundtrip_latency_ms": {"mean": sum(latency)/len(latency), "max": max(latency),
                    "p99": sorted(latency)[int(.99 * (len(latency)-1))]},
                "wrong_shape_rejected": True, "valid_input_recovery": True,
                "domain_id": args.domain_id, "topic_prefix": prefix,
                "robot_controller_started": False, "includes_robot_latency": False,
                "checkpoint": str(args.checkpoint.resolve()),
                "controller_config": str(args.controller_config.resolve()) if args.controller_config else None}
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report))
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
