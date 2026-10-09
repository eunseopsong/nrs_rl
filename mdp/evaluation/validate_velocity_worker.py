#!/usr/bin/env python3
"""Replay saved real simulation observations through a local inference worker.

No ROS node is started. Latencies cover CPU inference inside the worker only.
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
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--trace', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--controller-config', type=Path)
args = parser.parse_args()
worker = _REPO_ROOT / 'scripts/skrl/ppo_policy_node.py'
spec = importlib.util.spec_from_file_location('velocity_inference_worker', worker)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with np.load(args.trace) as trace:
    raw = trace['policy_observation']
    observations = raw[np.linspace(0, len(raw)-1, 128, dtype=int)].astype(np.float32)
if observations.shape != (128, 14) or not np.isfinite(observations).all():
    raise ValueError('Replay needs finite 14-dimensional observations')
agent, _ = module._build_agent(str(args.checkpoint.resolve()))
module._validate_scheduler_contract(agent.velocity_contract, module.load_controller_config(args.controller_config))
if agent.velocity_contract['policy_action_repeat'] != 1:
    raise ValueError('This replay validates policies that recompute every 8 ms')
with torch.inference_mode():
    expected = agent.act(torch.from_numpy(observations))[2]['mean_actions'].reshape(-1).numpy().clip(-1, 1)
requests = [{'sequence': i, 'observation': row.tolist()} for i, row in enumerate(observations)]
requests += [
    {'sequence': 128, 'observation': [float('nan')] * 14},
    {'sequence': 129, 'observation': [0.] * 13},
    {'sequence': 130, 'observation': observations[0].tolist()},
]
command = [sys.executable, str(worker), '--worker', '--checkpoint', str(args.checkpoint.resolve())]
if args.controller_config is not None:
    command += ['--controller-config', str(args.controller_config.resolve())]
result = subprocess.run(command,
                        input=''.join(json.dumps(item) + '\n' for item in requests),
                        text=True, capture_output=True, timeout=30, check=True)
responses = [json.loads(line) for line in result.stdout.splitlines()]
ready, answers = responses[0], responses[1:]
assert ready['ready'] and ready['policy_period_s'] == .008
assert len(answers) == len(requests)
assert all(response['sequence'] == i for i, response in enumerate(answers[:128]))
actual = np.array([response['action'] for response in answers[:128]])
error = float(np.max(np.abs(actual-expected)))
assert np.isfinite(actual).all() and error <= 2e-6, error
assert 'NaN or Inf' in answers[128]['error']
assert 'expected 14 values' in answers[129]['error']
assert abs(answers[130]['action']-float(expected[0])) <= 2e-6
latency = np.array([response['latency_ms'] for response in answers[:128]])
report = {
    **ready, 'ok': True, 'replayed_observations': 128, 'trace': str(args.trace.resolve()),
    'maximum_mean_action_error': error, 'nonfinite_input_rejected': True,
    'wrong_shape_rejected': True, 'valid_input_recovery': True,
    'cpu_inference_latency_ms': {'mean': float(latency.mean()), 'max': float(latency.max()),
                                'p99': float(np.quantile(latency, .99))},
    'includes_ros_or_robot_latency': False,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
print(json.dumps(report))
