#!/usr/bin/env python3
"""125 Hz PPO with state-dependent, temporally coherent exploration.

The action mean is recomputed every 8 ms. Only the exploration weights are
reused for 32 ticks; evaluation and deployment use the deterministic mean.
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
from pathlib import Path
import sys
import time
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--num-envs', type=int, default=2)
parser.add_argument('--control-steps', type=int, default=32768, help='125 Hz steps per environment')
parser.add_argument('--rollout-steps', type=int, default=1024)
parser.add_argument('--noise-resample-steps', type=int, default=32)
parser.add_argument('--seed', type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if min(args.num_envs, args.control_steps, args.rollout_steps, args.noise_resample_steps) < 1:
    parser.error('Environment and timestep counts must be positive')
if args.control_steps % args.rollout_steps:
    parser.error('control-steps must be an exact multiple of rollout-steps')
if args.output.exists():
    parser.error('output already exists; use a new run directory')
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
import nrs_rl.tasks
from nrs_rl.tasks.manager_based.nrs_rl.utils.control_action_repeat import ControlActionRepeat
from nrs_rl.tasks.manager_based.nrs_rl.utils.velocity_policy import OBSERVATION_FIELDS, OBSERVATION_VERSION

sys.path.insert(0, str(_REPO_ROOT / 'scripts/skrl'))
from nrs_rl.tasks.manager_based.nrs_rl.runtime.ppo_policy_node import SCHEDULER_CONTRACT
from nrs_rl.tasks.manager_based.nrs_rl.agents.velocity_actor import export_actor


def train():
    torch.set_num_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True)
    (output / 'params').mkdir()
    (output / 'checkpoints').mkdir()
    cfg = parse_env_cfg('Template-Nrs-Rl-v0', device=args.device or 'cpu', num_envs=args.num_envs)
    cfg.seed = args.seed
    cfg.visualization.enable_visualizer = False
    cfg.actions.arm_action.integration.enable_debug_print = False
    cfg.log_dir = str(output)
    env = gym.make('Template-Nrs-Rl-v0', cfg=cfg)
    coverage_env = ControlActionRepeat(env, 1, .999)
    started = time.monotonic()
    try:
        term = env.unwrapped.action_manager.get_term('arm_action')
        if not np.isclose(env.unwrapped.step_dt, .008):
            raise ValueError('Velocity policy training must use 125 Hz control')
        contract = {
            'schema_version': OBSERVATION_VERSION, 'observation_size': len(OBSERVATION_FIELDS),
            'observation_fields': list(OBSERVATION_FIELDS), 'robot': 'ur10_cb3',
            'control_period_s': .008, 'policy_action_repeat': 1, 'policy_period_s': .008,
            'policy_backend': 'torchscript_sb3_ppo', 'objective': 'preston_force_tcp_sliding_rate',
            'tool_diameter_mm': term.int_cfg.tool_diameter_mm,
            'physics_tool_diameter_mm': term.physical_tool_diameter_mm,
            'spindle_rpm': term.int_cfg.spindle_rpm,
            **{key: getattr(term.int_cfg, key) for key in SCHEDULER_CONTRACT},
        }
        (output / 'params/velocity_policy.json').write_text(json.dumps(contract, indent=2))
        dump_yaml(str(output / 'params/env.yaml'), cfg)
        vec = VecNormalize(Sb3VecEnvWrapper(coverage_env), norm_obs=True,
                           norm_reward=False, clip_obs=5., gamma=.999)
        settings = dict(n_steps=args.rollout_steps, batch_size=256, n_epochs=4,
                        learning_rate=1e-4, gamma=.999, gae_lambda=.995,
                        clip_range=.2, target_kl=.015, ent_coef=0., vf_coef=.5,
                        max_grad_norm=.5, use_sde=True, sde_sample_freq=args.noise_resample_steps,
                        seed=args.seed, device='cpu')
        architecture = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), activation_fn=torch.nn.Tanh,
                            log_std_init=-3., use_expln=True)
        (output / 'params/agent.json').write_text(json.dumps({**settings,
            'net_arch': architecture['net_arch'], 'activation': 'tanh', 'log_std_init': -3.,
            'use_expln': True, 'norm_reward': False, 'clip_obs': 5.,
            'control_steps_per_environment': args.control_steps}, indent=2))
        model = PPO('MlpPolicy', vec, policy_kwargs=architecture, **settings)
        model.set_logger(configure(str(output), ['csv', 'tensorboard']))
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.zero_()

        class Progress(BaseCallback):
            def __init__(self):
                super().__init__()
                self.probes = [np.zeros((1, len(OBSERVATION_FIELDS)), dtype=np.float32)]
                self.exports = {}

            def save_actor(self, ticks):
                probe = np.concatenate(self.probes[-256:])
                path = output / f'checkpoints/actor_{ticks}.pt'
                self.exports[str(ticks)] = export_actor(model.policy, vec, path, contract, probe)
                model.save(output / f'checkpoints/training_{ticks}.zip')
                vec.save(str(output / f'checkpoints/normalizer_{ticks}.pkl'))
                (output / 'export_validation.json').write_text(json.dumps(self.exports, indent=2))
                print('EXPORTED_ACTOR', str(path), json.dumps(self.exports[str(ticks)]), flush=True)

            def _on_step(self):
                ticks = coverage_env.decisions
                if not np.isfinite(self.locals['rewards']).all() or not np.isfinite(self.locals['new_obs']).all():
                    raise RuntimeError('Nonfinite training observations or rewards')
                if coverage_env.control_failures:
                    raise RuntimeError('Control failure during gSDE training; inspect before continuing')
                if ticks % 64 == 0:
                    self.probes.append(vec.get_original_obs().copy())
                if ticks % 1000 == 0 or ticks == args.control_steps:
                    elapsed = time.monotonic() - started
                    report = {**coverage_env.coverage(), 'target_control_steps_per_environment': args.control_steps,
                              'elapsed_seconds': elapsed, 'eta_seconds': elapsed * (args.control_steps-ticks)/ticks,
                              'phase': 'training', 'noise_resample_steps': args.noise_resample_steps}
                    (output / 'progress.json').write_text(json.dumps(report, indent=2))
                    print('TRAINING_PROGRESS', json.dumps(report), flush=True)
                if ticks % 8192 == 0 and ticks < args.control_steps:
                    self.save_actor(ticks)
                return True

        callback = Progress()
        model.learn(total_timesteps=args.control_steps*args.num_envs, callback=callback,
                    log_interval=1, progress_bar=False)
        callback.save_actor(coverage_env.decisions)
        coverage = {**coverage_env.coverage(), 'wall_seconds': time.monotonic()-started}
        (output / 'training_coverage.json').write_text(json.dumps(coverage, indent=2))
        (output / 'progress.json').write_text(json.dumps({**coverage, 'phase': 'finished'}, indent=2))
        print('TRAINING_COVERAGE', json.dumps(coverage), flush=True)
    finally:
        env.close()


status = 0
try:
    train()
except Exception:
    traceback.print_exc()
    status = 1
finally:
    sys.stdout.flush()
    sys.stderr.flush()
    app.app.post_quit(status)
    app.close()
sys.exit(status)
