#!/usr/bin/env python3
"""ROS 2 bridge for deterministic SKRL PPO checkpoint inference.

ROS 2 Humble uses Python 3.10 on this workstation, while the existing IsaacLab
SKRL environment uses Python 3.11. The ROS process therefore starts a small
Python 3.11 worker over pipes. The worker loads the original SKRL checkpoint
without converting it and owns the checkpoint RunningStandardScaler.
"""

from __future__ import annotations
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT as _REPO_ROOT

import argparse
import hashlib
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path


DEFAULT_CHECKPOINT = (
    "/home/eunseop/nrs_rl/logs/skrl/adaptive_velocity/"
    "best_agent.pt"
)
DEFAULT_WORKER_PYTHON = "/home/eunseop/anaconda3/envs/env_isaaclab/bin/python"
OBSERVATION_SIZE = 14  # schema 3, with explicit legacy schema-2 evaluation support
SCHEDULER_CONTRACT = {
    "nominal_speed_mm_s": 6.0, "target_mrr_n_mm_s": 120.0,
    "target_normal_force_n": 20.0,
    "residual_speed_fraction": 0.50, "force_rate_compensation": False,
    "policy_signal_tau_s": 0.08, "action_filter_tau_s": 0.08,
    "turn_preview_mm": 16.0,
    "action_slew_per_s": 4.0, "max_speed_acceleration_mm_s2": 16.0,
    "max_speed_jerk_mm_s3": 160.0,
    "speed_limiter_type": "legacy",
}


class PolicyActionHold:
    """Use the checkpoint's decision rate while still replying at 125 Hz."""

    def __init__(self, agent, repeat: int):
        if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
            raise ValueError("policy_action_repeat must be a positive integer")
        self.agent, self.repeat = agent, repeat
        self.reset()

    def reset(self):
        self.action = None
        self.next_sequence = 0
        self.last_sequence = -1

    def act(self, observation, sequence):
        if self.action is None or sequence >= self.next_sequence or sequence < self.last_sequence:
            outputs = self.agent.act(observation, timestep=0, timesteps=0)
            self.action = outputs[-1]["mean_actions"]
            self.next_sequence = sequence + self.repeat
        self.last_sequence = sequence
        return self.action


DEFAULT_TANGENTIAL_CONTROLLER = {"damping_ns_m": 6000., "stiffness_n_m": 2000.}


def _validate_tangential_config(config):
    if (not isinstance(config, dict) or set(config) != {"damping_ns_m", "stiffness_n_m"}
            or not all(type(x) in (int, float) and math.isfinite(x) for x in config.values())
            or not 500. <= config["damping_ns_m"] <= 6000.
            or not 2000. <= config["stiffness_n_m"] <= 16000.):
        raise ValueError("Invalid tangential controller contract")
    return config


def load_controller_config(path):
    """Read the explicit configuration that the launch also gives to C++."""
    if path is None:
        return None
    return _validate_tangential_config(json.loads(Path(path).expanduser().read_text()))


def _validate_scheduler_contract(contract, controller_config=None):
    tangent = _validate_tangential_config(contract.get(
        "tangential_controller", DEFAULT_TANGENTIAL_CONTROLLER))
    if controller_config is not None:
        _validate_tangential_config(controller_config)
    actual = DEFAULT_TANGENTIAL_CONTROLLER if controller_config is None else controller_config
    if actual != tangent:
        raise ValueError("Checkpoint requires explicit matching tangential controller settings; legacy Mode 5 cannot use this checkpoint alone")
    if contract.get("physics_tool_diameter_mm", 30.0) != 30.0:
        raise ValueError("Checkpoint used the old 56 mm contact cylinder; retrain for the 30 mm tool")
    for name, expected in SCHEDULER_CONTRACT.items():
        actual = contract.get(name)
        matches = actual == expected if isinstance(expected, (bool, str)) else (
            isinstance(actual, (int, float)) and math.isclose(actual, expected, rel_tol=1e-9))
        if not matches:
            raise ValueError(
                f"Checkpoint {name}={actual} differs from staged Y2 scheduler {expected}; "
                "synchronize the C++ scheduler and inference contract before deployment."
            )


def _resolve_checkpoint(path: str) -> str:
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    if candidate.name == "best_agent.pt":
        matches = sorted(
            candidate.parent.rglob("best_agent.pt"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if matches:
            return str(matches[0].resolve())
    return str(candidate.resolve())


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_policy_contract(checkpoint: str):
    """Read the shared timing and observation contract without loading Torch."""
    contract_path = Path(checkpoint).resolve().parent.parent / "params/velocity_policy.json"
    if not contract_path.is_file():
        raise ValueError(
            f"Missing velocity policy contract: {contract_path}. "
            "Legacy 12-input policies require retraining; copy params with checkpoints."
        )
    contract = json.loads(contract_path.read_text())
    contract.setdefault("policy_backend", "skrl_shared_ppo")
    if contract["policy_backend"] not in ("skrl_shared_ppo", "torchscript_sb3_ppo", "torchscript_preston_depth", "torchscript_tcp_removal"):
        raise ValueError("Unknown velocity policy backend")
    if contract['policy_backend'] == 'torchscript_tcp_removal' and (
            contract.get('spindle_rpm') is not None or contract.get('rotation_contribution') is not False
            or contract.get('objective') != 'preston_tcp_spatial_removal'):
        raise ValueError('TCP-removal checkpoint must explicitly exclude spindle rotation')
    # Schema-2 runs made before the OTG comparison all used the legacy filter.
    contract.setdefault("speed_limiter_type", "legacy")
    # Older runs used the trajectory's 10 N without an override. Restoring
    # them for evaluation must never silently replace that with today's 20 N.
    contract.setdefault("target_normal_force_n", None)
    contract.setdefault("tool_diameter_mm", 30.0)
    contract.setdefault("physics_tool_diameter_mm", 56.0)
    contract.setdefault("spindle_rpm", None)
    contract.setdefault("objective", "preston_force_tcp_sliding_rate")
    if contract.get("schema_version") == 2:
        contract.setdefault("turn_preview_mm", 0.)
    elif (not isinstance(contract.get("turn_preview_mm"), (int, float))
          or isinstance(contract["turn_preview_mm"], bool)
          or not math.isfinite(contract["turn_preview_mm"]) or contract["turn_preview_mm"] <= 0):
        raise ValueError("Schema 3 requires a positive path-turn preview horizon")
    contract.setdefault("policy_action_repeat", 1)
    repeat = contract["policy_action_repeat"]
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
        raise ValueError("Invalid policy_action_repeat in checkpoint contract")
    contract.setdefault("policy_period_s", .008 * repeat)
    if not math.isclose(contract["policy_period_s"], .008 * repeat):
        raise ValueError("Policy period differs from control period times action repeat")
    if (contract.get("schema_version") not in (2, 3) or contract.get("observation_size") != OBSERVATION_SIZE
            or not math.isclose(contract.get("control_period_s", 0.0), 0.008)):
        raise ValueError("Policy contract must use schema 2/3, 14 observations and 125 Hz control")
    return contract


def _build_agent(checkpoint: str):
    """Build the saved shared PPO model and load all SKRL checkpoint modules."""
    import copy
    contract = load_policy_contract(checkpoint)

    if contract["policy_backend"] in ("torchscript_sb3_ppo", "torchscript_preston_depth", "torchscript_tcp_removal"):
        import importlib.util
        import torch
        spec = importlib.util.spec_from_file_location("velocity_actor", _REPO_ROOT/'scripts/skrl/velocity_actor.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.TorchscriptVelocityAgent(checkpoint, contract), torch

    import torch
    # A 64x64 CPU policy should not fan out onto a large BLAS thread pool in
    # the 8 ms inference budget. Force control runs in the C++ controller.
    torch.set_num_threads(1)
    from gymnasium.spaces import Box
    from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
    from skrl.memories.torch import RandomMemory
    from skrl.resources.preprocessors.torch import RunningStandardScaler
    from skrl.utils.model_instantiators.torch import shared_model

    observation_space = Box(low=-float("inf"), high=float("inf"), shape=(OBSERVATION_SIZE,))
    action_space = Box(low=-1.0, high=1.0, shape=(1,))
    device = "cpu"

    parameters = [
        {
            "clip_actions": False,
            "clip_log_std": True,
            "min_log_std": -4.0,
            "max_log_std": -1.0,
            "initial_log_std": -2.0,
            "network": [
                {
                    "name": "net",
                    "input": "OBSERVATIONS",
                    "layers": [64, 64],
                    "activations": "elu",
                }
            ],
            "output": "ACTIONS",
        },
        {
            "clip_actions": False,
            "network": [
                {
                    "name": "net",
                    "input": "OBSERVATIONS",
                    "layers": [64, 64],
                    "activations": "elu",
                }
            ],
            "output": "ONE",
        },
    ]
    model = shared_model(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        structure=["GaussianMixin", "DeterministicMixin"],
        roles=["policy", "value"],
        parameters=parameters,
        single_forward_pass=True,
    )

    cfg = copy.deepcopy(PPO_DEFAULT_CONFIG)
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": observation_space, "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    cfg["experiment"]["write_interval"] = 0
    cfg["experiment"]["checkpoint_interval"] = 0

    memory = RandomMemory(memory_size=32, num_envs=1, device=device)
    agent = PPO(
        models={"policy": model, "value": model},
        memory=memory,
        cfg=cfg,
        observation_space=observation_space,
        action_space=action_space,
        device=device,
    )
    agent.load(checkpoint)
    agent.set_running_mode("eval")
    agent.velocity_contract = contract
    return agent, torch


def _worker_main(args: argparse.Namespace) -> int:
    checkpoint = _resolve_checkpoint(args.checkpoint)
    if not Path(checkpoint).is_file():
        print(json.dumps({"ready": False, "error": f"checkpoint not found: {checkpoint}"}), flush=True)
        return 2

    try:
        agent, torch = _build_agent(checkpoint)
        controller_config = load_controller_config(getattr(args, "controller_config", None))
        _validate_scheduler_contract(agent.velocity_contract, controller_config)
        scaler_count = float(agent._state_preprocessor.current_count.item())
        checkpoint_sha256 = _sha256(checkpoint)

        # Warm up PyTorch/SKRL before advertising readiness so the first live
        # action does not pay one-time kernel/thread initialization latency.
        with torch.inference_mode():
            warmup_observation = torch.zeros((1, OBSERVATION_SIZE), dtype=torch.float32)
            for _ in range(3):
                agent.act(warmup_observation, timestep=0, timesteps=0)
        action_hold = PolicyActionHold(agent, agent.velocity_contract["policy_action_repeat"])
        print(
            json.dumps(
                {
                    "ready": True,
                    "checkpoint": checkpoint,
                    "checkpoint_sha256": checkpoint_sha256,
                    "observation_shape": [1, OBSERVATION_SIZE],
                    "action_shape": [1, 1],
                    "state_preprocessor": getattr(agent, "normalizer_name", "RunningStandardScaler"),
                    "state_preprocessor_count": scaler_count,
                    "selection": "mean_actions",
                    "policy_action_repeat": agent.velocity_contract["policy_action_repeat"],
                    "policy_period_s": agent.velocity_contract["policy_period_s"],
                    "tangential_controller": controller_config or DEFAULT_TANGENTIAL_CONTROLLER,
                }
            ),
            flush=True,
        )
    except Exception as exc:  # worker must report startup failure to the ROS parent
        print(json.dumps({"ready": False, "error": repr(exc)}), flush=True)
        return 3

    for line in sys.stdin:
        try:
            request = json.loads(line)
            observation = request["observation"]
            if len(observation) != OBSERVATION_SIZE:
                raise ValueError(f"expected {OBSERVATION_SIZE} values, got {len(observation)}")
            if not all(math.isfinite(float(value)) for value in observation):
                raise ValueError("observation contains NaN or Inf")

            tensor = torch.tensor(observation, dtype=torch.float32).reshape(1, OBSERVATION_SIZE)
            started = time.perf_counter()
            with torch.inference_mode():
                action = float(action_hold.act(tensor, int(request["sequence"])).reshape(-1)[0].item())
            latency_ms = (time.perf_counter() - started) * 1000.0
            print(
                json.dumps(
                    {
                        "sequence": int(request["sequence"]),
                        "action": action,
                        "latency_ms": latency_ms,
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            print(json.dumps({"error": repr(exc)}), flush=True)
    return 0


def _ros_main(args: argparse.Namespace) -> int:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Float64, Float64MultiArray

    class PpoPolicyNode(Node):
        def __init__(self):
            super().__init__("ppo_policy_node")
            self._checkpoint = _resolve_checkpoint(args.checkpoint)
            if not Path(self._checkpoint).is_file():
                raise FileNotFoundError(self._checkpoint)
            if not Path(args.worker_python).is_file():
                raise FileNotFoundError(args.worker_python)

            command = [
                args.worker_python,
                str(Path(__file__).resolve()),
                "--worker",
                "--checkpoint",
                self._checkpoint,
            ]
            if args.controller_config is not None:
                command += ["--controller-config", str(Path(args.controller_config).expanduser().resolve())]
            self._worker = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            assert self._worker.stdout is not None
            ready_line = self._worker.stdout.readline()
            if not ready_line:
                raise RuntimeError("SKRL worker exited before startup response")
            ready = json.loads(ready_line)
            if not ready.get("ready", False):
                raise RuntimeError(f"SKRL worker startup failed: {ready.get('error', ready)}")

            qos = QoSProfile(
                depth=1,
                history=HistoryPolicy.KEEP_LAST,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            )
            self._action_pub = self.create_publisher(Float64, args.action_topic, qos)
            self._observation_sub = self.create_subscription(
                Float64MultiArray, args.observation_topic, self._observation_callback, qos
            )
            self._latest_lock = threading.Condition()
            self._latest_observation = None
            self._latest_sequence = 0
            self._processed_sequence = 0
            self._received = 0
            self._published = 0
            self._last_action = 0.0
            self._last_latency_ms = 0.0
            self._stopping = False
            self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
            self._inference_thread.start()
            self._rate_timer = self.create_timer(1.0, self._report_rate)

            self.get_logger().info(f"checkpoint: {ready['checkpoint']}")
            self.get_logger().info(f"checkpoint SHA-256: {ready['checkpoint_sha256']}")
            self.get_logger().info(
                "observation shape: %s, action shape: %s"
                % (ready["observation_shape"], ready["action_shape"])
            )
            self.get_logger().info(
                "SKRL state preprocessor: %s (count=%.0f)"
                % (ready["state_preprocessor"], ready["state_preprocessor_count"])
            )
            self.get_logger().info(
                f"deterministic output: {ready['selection']}; "
                f"{args.observation_topic} -> {args.action_topic}"
            )
            self.get_logger().info(
                "policy decision period: %.3f s; control/action publication: 125 Hz"
                % ready["policy_period_s"]
            )

        def _observation_callback(self, message):
            if len(message.data) != OBSERVATION_SIZE:
                self.get_logger().error(
                    f"rejected observation: expected {OBSERVATION_SIZE}, got {len(message.data)}",
                    throttle_duration_sec=1.0,
                )
                return
            observation = [float(value) for value in message.data]
            if not all(math.isfinite(value) for value in observation):
                self.get_logger().error("rejected non-finite observation", throttle_duration_sec=1.0)
                return
            with self._latest_lock:
                self._latest_sequence += 1
                self._latest_observation = observation
                self._received += 1
                self._latest_lock.notify()

        def _inference_loop(self):
            assert self._worker.stdin is not None
            assert self._worker.stdout is not None
            while True:
                with self._latest_lock:
                    self._latest_lock.wait_for(
                        lambda: self._stopping or self._latest_sequence > self._processed_sequence
                    )
                    if self._stopping:
                        return
                    sequence = self._latest_sequence
                    observation = self._latest_observation
                try:
                    self._worker.stdin.write(
                        json.dumps({"sequence": sequence, "observation": observation}) + "\n"
                    )
                    self._worker.stdin.flush()
                    response_line = self._worker.stdout.readline()
                    if not response_line:
                        raise RuntimeError("SKRL worker pipe closed")
                    response = json.loads(response_line)
                    if "error" in response:
                        raise RuntimeError(response["error"])
                    action = float(response["action"])
                    if not math.isfinite(action):
                        raise ValueError(f"non-finite action: {action}")
                    message = Float64()
                    message.data = action
                    self._action_pub.publish(message)
                    self._last_action = action
                    self._last_latency_ms = float(response["latency_ms"])
                    self._published += 1
                    self._processed_sequence = sequence
                except Exception as exc:
                    self.get_logger().error(f"PPO inference failed: {exc}")
                    return

        def _report_rate(self):
            received, published = self._received, self._published
            self._received = 0
            self._published = 0
            self.get_logger().info(
                "rate: observation=%d Hz action=%d Hz, last_action=%.6f, worker_latency=%.3f ms"
                % (received, published, self._last_action, self._last_latency_ms)
            )

        def close(self):
            with self._latest_lock:
                self._stopping = True
                self._latest_lock.notify_all()
            if self._inference_thread.is_alive():
                self._inference_thread.join(timeout=2.0)
            if self._worker.stdin is not None:
                self._worker.stdin.close()
            self._worker.terminate()
            try:
                self._worker.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._worker.kill()

    rclpy.init(args=None)
    node = None
    try:
        node = PpoPolicyNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


def _self_test(args: argparse.Namespace) -> int:
    checkpoint = _resolve_checkpoint(args.checkpoint)
    agent, torch = _build_agent(checkpoint)
    controller_config = load_controller_config(getattr(args, "controller_config", None))
    _validate_scheduler_contract(agent.velocity_contract, controller_config)
    observation = torch.zeros((1, OBSERVATION_SIZE), dtype=torch.float32)
    with torch.inference_mode():
        outputs = agent.act(observation, timestep=0, timesteps=0)
        mean_action = float(outputs[-1].get("mean_actions", outputs[0]).reshape(-1)[0].item())
    print(
        json.dumps(
            {
                "ok": math.isfinite(mean_action),
                "checkpoint": checkpoint,
                "checkpoint_sha256": _sha256(checkpoint),
                "observation_shape": list(observation.shape),
                "policy_action_repeat": agent.velocity_contract["policy_action_repeat"],
                "policy_period_s": agent.velocity_contract["policy_period_s"],
                "selection": "mean_actions",
                "tangential_controller": controller_config or DEFAULT_TANGENTIAL_CONTROLLER,
                "raw_action": mean_action,
                "clipped_action": max(-1.0, min(1.0, mean_action)),
                "state_preprocessor_count": float(
                    agent._state_preprocessor.current_count.item()
                ),
            }
        )
    )
    return 0 if math.isfinite(mean_action) else 4


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--controller-config", help="Matching tangential MDK JSON, also applied by the experiment launch")
    parser.add_argument("--worker-python", default=DEFAULT_WORKER_PYTHON)
    parser.add_argument("--observation-topic", default="/ppo/observation")
    parser.add_argument("--action-topic", default="/ppo/action")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = _parse_args()
    if parsed.worker:
        raise SystemExit(_worker_main(parsed))
    if parsed.self_test:
        raise SystemExit(_self_test(parsed))
    raise SystemExit(_ros_main(parsed))
