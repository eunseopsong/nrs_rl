#!/usr/bin/env python3
"""Exercise the real command/measurement nodes against an isolated fake controller.

Requires ROS_DOMAIN_ID=197 and ROS_LOCALHOST_ONLY=1. Never launch a robot driver
or use this domain for the user's Isaac session. No motion node is started.
"""
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
import math
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    if os.environ.get("ROS_DOMAIN_ID") != "197" or os.environ.get("ROS_LOCALHOST_ONLY") != "1":
        raise RuntimeError("Use the dedicated test domain: ROS_DOMAIN_ID=197 ROS_LOCALHOST_ONLY=1")
    import rclpy
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
    from std_msgs.msg import Float64MultiArray, String, Bool
    from std_srvs.srv import Trigger
    from y2_rob_motion_interfaces.srv import SingleArmCommand
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command-binary", type=Path, required=True)
    parser.add_argument("--removal-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sim-clock", action="store_true", help="Replay /clock at half wall-clock speed")
    parser.add_argument("--depth-model", choices=("rotation_dominated_normalized", "tcp_fixed_roi"),
                        default="rotation_dominated_normalized")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    setup = yaml.safe_load(Path("/home/eunseop/dev_ws/src/y2_ur10skku_control/Y2RobMotion/config/preston_d14.yaml").read_text())
    # The legacy profiler requires a whole number of 8 ms samples per window.
    setup.update(DEFAULT_TRAVEL_TIME=0.2, INITIAL_TRANSFER_SPEED=80.0, ANGULAR_VELOCITY_LIMIT=180.0,
                 STARTING_TIME=0.08, LAST_RESTING_TIME=0.08, ACCELERATION_TIME=0.048,
                 PTP_TARGET_VELOCITY=100.0)
    setup_file = args.output / "setup.yaml"
    setup_file.write_text(yaml.safe_dump(setup))
    trajectory = args.output / "cmd_continue9D_test.txt"
    trajectory.write_text("870 340 100 0.1 -0.15 1.5 0 0 20\n875 345 100 0.1 -0.15 1.5 0 0 20\n")
    measure_config = args.output / "measurement.yaml"
    measure_config.write_text(yaml.safe_dump({"/**": {"ros__parameters": {
        "logs_root": str(args.output / "removal"), "show_heatmap": False,
        "required_control_mode": "Force", "recording_rate_hz": 125.0,
        "depth_model": args.depth_model, "use_sim_time": args.sim_clock,
        "reference_trajectory_file": str(trajectory),
        "nrs_root": "/home/eunseop/nrs_rl", "cell_mm": 0.5,
        "pad_radius_mm": 15.0, "contact_threshold_N": 1.5}}}))
    rclpy.init()
    node = rclpy.create_node("d14_fake_controller_test")
    current = [868., 337., 115., 0.1, -0.15, 1.5]
    mode = "Idling"
    force_started = None
    behavior = "normal"
    done_sent = False
    completed_force = False
    commands = []
    events = []
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pose_pub = node.create_publisher(Float64MultiArray, "/ur10skku/currentP", 10)
    force_pub = node.create_publisher(Float64MultiArray, "/ur10skku/currentF", 10)
    mode_pub = node.create_publisher(String, "/ur10skku/ctlMode", 10)
    done_pub = node.create_publisher(Bool, "/ur10skku/ppo/trajectory_done", latched)
    clock_pub = None
    clock_ticks = 0
    if args.sim_clock:
        from rosgraph_msgs.msg import Clock
        clock_pub = node.create_publisher(Clock, "/clock", 10)

    def cmd_mode(message):
        nonlocal mode, force_started, completed_force
        mode = message.data
        events.append((mode, len(commands)))
        if mode == "Force":
            force_started = time.monotonic()
            completed_force = False

    def cmd_motion(message):
        nonlocal current
        commands.append(list(message.data))
        if mode == "Position" and not (behavior == "stall_return" and completed_force):
            current = list(message.data[:6])

    def full_trajectory(message):
        nonlocal done_sent, completed_force
        done_sent = False
        completed_force = False
        done_pub.publish(Bool(data=False))

    def tick():
        nonlocal current, mode, done_sent, completed_force, clock_ticks
        if clock_pub is not None:
            clock_ticks += 1
            stamp = 1000000000 + (clock_ticks // 2) * 8000000
            message = Clock()
            message.clock.sec, message.clock.nanosec = divmod(stamp, 1000000000)
            clock_pub.publish(message)
        if mode == "Force" and force_started is not None:
            elapsed = time.monotonic() - force_started
            fraction = min(1., elapsed / 0.6)
            current = [870 + 5 * fraction, 340 + 5 * fraction, 100., 0.1, -0.15, 1.5]
            if behavior == "cancel" and elapsed > 0.25:
                mode = "Idling"
                events.append(("operator_cancel", len(commands)))
            elif elapsed > 0.6 and not done_sent:
                done_sent = True
                completed_force = True
                done_pub.publish(Bool(data=True))
        pose_pub.publish(Float64MultiArray(data=current))
        # High forces outside Force expose accidental recording of transit/return.
        force_pub.publish(Float64MultiArray(data=[0., 0., 20. if mode == "Force" else 80., 0., 0., 0.]))
        mode_pub.publish(String(data=mode))

    subscriptions = [
        node.create_subscription(String, "/ur10skku/cmdMode", cmd_mode, 10),
        node.create_subscription(Float64MultiArray, "/ur10skku/cmdMotion", cmd_motion, 10),
        node.create_subscription(Float64MultiArray, "/ur10skku/ppo/trajectory9d", full_trajectory, latched),
    ]
    timer = node.create_timer(.008, tick)
    service = node.create_client(SingleArmCommand, "/singleArm_cmd/single_arm_command")
    start_service = node.create_client(Trigger, "/polishing_removal_node/start")

    def spin_for(seconds):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            rclpy.spin_once(node, timeout_sec=.005)

    def run_request(label, expected_success):
        saved_home = list(current)
        first_command = len(commands)
        first_event = len(events)
        request = SingleArmCommand.Request()
        request.command_mode = "TxtLoad"
        request.load_file = str(trajectory)
        future = service.call_async(request)
        until = time.monotonic() + 45
        while not future.done() and time.monotonic() < until:
            rclpy.spin_once(node, timeout_sec=.005)
        assert future.done(), f"{label}: service response timeout"
        response = future.result()
        (args.output / (label + "_trace.json")).write_text(json.dumps({
            "saved_home": saved_home, "current": current, "mode": mode,
            "events": events[first_event:], "commands": commands[first_command:],
            "response": response.message}, indent=2))
        assert response.success == expected_success, f"{label}: {response.message}"
        spin_for(.08)
        emitted = commands[first_command:]
        trace = events[first_event:]
        assert mode == "Idling", (label, mode)
        if expected_success:
            assert math.dist(current[:3], saved_home[:3]) < 1e-6, (label, current, saved_home)
            assert math.dist(current[3:], saved_home[3:]) < 1e-9, "Home angles were not preserved in radians"
            assert "home returned" in response.message
            assert max(row[2] for row in emitted) >= saved_home[2] + 9.99, "Missing +Z clearance"
        elif label == "cancel_during_force":
            cancellation = next(i for state, i in trace if state == "operator_cancel")
            assert len(commands) == cancellation, "Automatic return after operator cancellation"
        elif label == "recording_start_rejected":
            assert not emitted, "Motion started despite failed required measurement"
        elif label == "home_arrival_timeout":
            assert "did not settle" in response.message, response.message
            assert "home returned" not in response.message
        result = {"case": label, "passed": True, "response_success": response.success,
                  "home": saved_home, "final_pose": list(current), "events": trace,
                  "motion_samples": len(emitted), "message": response.message}
        print(json.dumps({"case": label, "passed": True}), flush=True)
        return result

    processes = []
    logs = []
    results = []
    env = {**os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": "/tmp/nrs_d14_measurement_mpl"}
    try:
        for label, argv in (
            ("measurement", ["/usr/bin/python3", "-s", str(args.removal_script), "--ros-args", "--params-file", str(measure_config)]),
            ("command", [str(args.command_binary), "--ros-args", "-r", "__node:=singleArm_cmd",
                "-p", "setup_file:=" + str(setup_file), "-p", "force_control_mode:=5",
                "-p", "use_sim_time:=" + str(args.sim_clock).lower(),
                "-p", "return_home_after_txtload:=true", "-p", "require_polishing_measurement:=true",
                "-p", "txtload_return_lift_mm:=10.0"]),
        ):
            log = (args.output / f"{label}.log").open("w")
            logs.append(log)
            processes.append(subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env))
        until = time.monotonic() + 20
        while time.monotonic() < until and not (service.service_is_ready() and start_service.service_is_ready()):
            spin_for(.05)
            assert all(p.poll() is None for p in processes), "Node exited; inspect test logs"
        assert service.service_is_ready() and start_service.service_is_ready()
        spin_for(.5)
        results.append(run_request("normal_return", True))
        current = [867., 336., 116., 0.2, -0.1, 1.4]
        spin_for(.2)
        results.append(run_request("second_request_saves_new_home", True))
        behavior = "cancel"
        spin_for(.1)
        results.append(run_request("cancel_during_force", False))
        current = [868., 337., 115., 0.1, -0.15, 1.5]
        behavior = "stall_return"
        spin_for(.2)
        results.append(run_request("home_arrival_timeout", False))
        sessions = sorted((args.output / "removal").glob("session_*"))
        assert len(sessions) == 4, len(sessions)
        for session in sessions:
            report = json.loads((session / "summary.json").read_text())
            assert report["metrics"]["spatial_depth_cv"] is not None
            assert report["model"]["contact_diameter_mm"] == 30.
            assert report["model"]["cell_size_mm"] == .5
            assert report["required_control_mode"] == "Force"
            assert report["physical_depth_validated"] is False
            assert len(list(session.glob("*.png"))) == 4
            with (session / "recording.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            assert len(rows) >= 10
            assert all(float(row["fz_N"]) == 20. for row in rows), "Non-Force samples entered the measurement"
            assert all(abs(float(row["z_mm"]) - 100.) < 1e-9 for row in rows), "Return trip entered the depth map"
        processes[0].send_signal(signal.SIGINT)
        processes[0].wait(timeout=10)
        spin_for(.3)

        def refuse_recording(request, response):
            response.success = False
            response.message = "Test: recording unavailable"
            return response

        rejected_start = node.create_service(Trigger, "/polishing_removal_node/start", refuse_recording)
        spin_for(.5)
        behavior = "normal"
        results.append(run_request("recording_start_rejected", False))
        (args.output / "result.json").write_text(json.dumps({"ok": True, "ros_domain_id": 197,
            "simulation_clock": args.sim_clock, "depth_model": args.depth_model,
            "robot_driver_started": False, "cases": results,
            "recording_excludes_connection_and_return": True, "measurement_sessions": len(sessions)}, indent=2))
        print("PASS: isolated command lifecycle and depth-map recording", flush=True)
    finally:
        for process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for log in logs:
            log.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
