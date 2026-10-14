"""Verify the PPO actor, shared controller, and measured trial dependencies."""
import hashlib
import json
from pathlib import Path
import subprocess

from ...paths import ROOT

ACTOR_SHA = '363abffb65b4a6078a63baffa9ac864f0609ede73a55605df09e7e1862aa1d0c'
BUNDLE = ROOT / 'deployment/ppo_fixed_force_11p13'
BASE = ROOT / 'deployment/fixed_force_12p59'
SAVE = ROOT / 'logs/experiments/uniformity_saves/ppo_fixed20_virtual1000_cv11p13_20261009_2200'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(root=BUNDLE, mode=5, check_worker=True):
    root = Path(root)
    if mode not in (5, 6):
        raise ValueError('Use Mode 5 or matched Mode 6')
    data = json.loads((root / 'deployment.json').read_text())
    if (data['backend'] != 'isaac' or data['algorithm'] != 'PPO'
            or data['checkpoint_sha256'] != ACTOR_SHA or data['observation_shape'] != [1, 16]
            or data['action_shape'] != [1, 1] or data['target_force_n'] != 20.
            or data['policy_controls_force'] or data['virtual_spindle_rpm'] != 1000.):
        raise ValueError('PPO deployment contract mismatch')
    for path, expected in data['pinned_files'].items():
        if digest(path) != expected:
            raise ValueError('Deployment component changed: ' + path)
    for key in ('checkpoint', 'motion_binary', 'command_binary', 'worker', 'removal_script',
                'reference_h5', 'trajectory_file', 'setup_file', 'validation_file'):
        if data[key] not in data['pinned_files']:
            raise ValueError('Unpinned component: ' + key)
    if digest(data['checkpoint']) != ACTOR_SHA:
        raise ValueError('PPO actor hash mismatch')
    tested = json.loads(Path(data['validation_file']).read_text())
    if (not tested['ok'] or tested['tested_modes'] != [5, 6]
            or tested['checkpoint_sha256'] != ACTOR_SHA
            or tested['motion_binary_sha256'] != digest(data['motion_binary'])
            or tested['command_binary_sha256'] != digest(data['command_binary'])):
        raise ValueError('PPO runtime validation missing or stale')
    data.update(force_control_mode=mode, policy_output_applied=mode == 5,
                applied_action_override=None if mode == 5 else [0.])
    if check_worker:
        result = subprocess.run([data['worker_python'], data['worker'], '--self-test',
            '--force-control-mode', str(mode), '--checkpoint', data['checkpoint'],
            '--reference', data['reference_h5']], capture_output=True, text=True, check=True, timeout=30)
        ready = json.loads(result.stdout.strip().splitlines()[-1])
        if (not ready['ok'] or ready['algorithm'] != 'PPO' or ready['action_shape'] != [1, 1]
                or ready['observation_shape'] != [1, 16] or ready['force_control_mode'] != mode
                or ready['checkpoint_sha256'] != ACTOR_SHA):
            raise ValueError('PPO worker handshake mismatch')
        data['inference'] = ready
    return data
