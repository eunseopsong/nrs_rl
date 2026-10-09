#!/usr/bin/env python3
"""Paired short physics screening of a bounded velocity-policy family.

This is a scripted controllability experiment, not a learned-policy result.
Two identical constant baselines measure variability between vector instances.
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
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--path-length-mm', type=float, default=160.)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--max-steps', type=int, default=8000)
parser.add_argument('--dwell-profile', type=Path)
parser.add_argument('--parameters-file', type=Path)
parser.add_argument('--surface-cell-mm', type=float, default=2.)
parser.add_argument('--tcp-removal', action='store_true', help='RPM-free six-parameter policy family')
parser.add_argument('--include-checkpoint', type=Path, help='Frozen old_d14 comparator in the same rollout batch')
parser.add_argument('--candidate-checkpoint', type=Path, help='Load the exported learned candidate for held-out evaluation')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists() or args.path_length_mm <= 0 or args.max_steps < 1:
    parser.error('Use a new output directory and positive path/step limits')
app = AppLauncher(args).app

import gymnasium as gym
import h5py
import numpy as np
import torch
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.io import dump_yaml
import nrs_rl.tasks
from nrs_rl.tasks.manager_based.nrs_rl.utils import visualization as vis
from nrs_rl.tasks.manager_based.nrs_rl.utils.preston_surface import PlanarPrestonSurface
from nrs_rl.tasks.manager_based.nrs_rl.utils.velocity_evaluation import compare_rollouts, compare_surfaces
from nrs_rl.tasks.manager_based.nrs_rl.mdp.action import spatial_to_rotmat

sys.path.insert(0, str(_REPO_ROOT / 'scripts/skrl'))
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.bounded_velocity_actor import BoundedVelocityActor, PARAMETER_NAMES
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.preston_dwell_actor import PrestonDwellActor, PARAMETER_NAMES as DWELL_PARAMETER_NAMES
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.tcp_removal_actor import TcpRemovalActor, PARAMETER_NAMES as TCP_PARAMETER_NAMES
from nrs_rl.tasks.manager_based.nrs_rl.utils.analyze_rotation_depth import RotationDominatedSurface


def run():
    torch.set_num_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True)
    candidates = {
        'constant_a': [0., 0., 0., 0.], 'constant_b': [0., 0., 0., 0.],
        'force_only': [1., 0., 0., 0.], 'rate_03': [0., .3, 0., 0.],
        'rate_07': [0., .7, 0., 0.], 'force_rate': [.7, .3, 0., 0.],
        'preview_positive': [.7, .3, .3, 0.], 'preview_negative': [.7, .3, -.3, 0.],
    }
    if args.parameters_file is not None:
        candidates = json.loads(args.parameters_file.read_text())
        zero = [0.]*(6 if args.tcp_removal else 4)
        if (not isinstance(candidates, dict) or not 3 <= len(candidates) <= 16
                or 'constant_a' not in candidates or 'constant_b' not in candidates
                or any(candidates[key] != zero for key in ('constant_a', 'constant_b'))
                or any(not name.replace('_', '').isalnum() for name in candidates)):
            raise ValueError('Supply 3--16 named policies including two zero-action baselines')
    if args.tcp_removal and (args.parameters_file is None or args.dwell_profile is not None):
        raise ValueError('TCP policy needs explicit parameters and must not use a rotation dwell profile')
    if args.include_checkpoint and (not args.tcp_removal or 'old_d14' not in candidates):
        raise ValueError('Frozen comparator needs an explicit old_d14 candidate in a TCP batch')
    if args.candidate_checkpoint and (not args.tcp_removal or 'learned' not in candidates):
        raise ValueError('Exported TCP candidate requires a learned row')
    if args.dwell_profile is not None and args.parameters_file is None:
        raise ValueError('Dwell policies need explicit named coefficient candidates')
    names = list(candidates)
    cfg = parse_env_cfg('Template-Nrs-Rl-v0', device=args.device or 'cpu', num_envs=len(names))
    cfg.seed = args.seed
    cfg.visualization.enable_visualizer = False
    cfg.actions.arm_action.integration.enable_debug_print = False
    cfg.log_dir = str(output)
    integration = cfg.actions.arm_action.integration
    if args.tcp_removal and integration.spindle_rpm is not None:
        raise ValueError('TCP-only training excludes spindle RPM')
    source_path = Path(integration.hdf5_file_path)
    with h5py.File(source_path) as source:
        positions = source['position'][:]
        arc = np.r_[0., np.linalg.norm(np.diff(positions[:, :3], axis=0), axis=1).cumsum()]
        end = min(len(positions), max(2, int(np.searchsorted(arc, args.path_length_mm)) + 1))
        with h5py.File(output / 'path.h5', 'w') as target:
            target['position'] = positions[:end]
            target['force'] = source['force'][:end]
    integration.hdf5_file_path = str(output / 'path.h5')
    dump_yaml(str(output / 'env.yaml'), cfg)
    env = gym.make('Template-Nrs-Rl-v0', cfg=cfg)
    started = time.monotonic()
    try:
        raw = env.unwrapped
        if not np.isclose(raw.step_dt, .008):
            raise ValueError('Screening must use 125 Hz control')
        term = raw.action_manager.get_term('arm_action')
        completed = np.zeros(len(names), dtype=bool)
        ended = np.zeros(len(names), dtype=bool)
        rows = [[] for _ in names]
        def capture(i, pose, force):
            if not ended[i]:
                rows[i].append((pose.cpu().numpy().copy(), abs(float(force)),
                    float(term.current_sliding_velocity_mm_s[i]), float(term.current_mrr_n_mm_s[i]),
                    float(term.realized_removal_step[i]), vis.control_trace_snapshot(term, i),
                    term.policy_state[i].cpu().numpy().copy(), term._previous_command_pose[i].cpu().numpy().copy()))
        term._record_visualization = capture
        obs, _ = env.reset(seed=args.seed)
        # Match the existing paired evaluator's sampled sensor/delay conditions.
        for attribute in ('_force_scale', '_force_bias', '_action_delay'):
            values = getattr(term, attribute)
            values.fill_(values[0].item())
        randomization = {key: getattr(term, key).cpu().tolist()
                         for key in ('_force_scale', '_force_bias', '_action_delay')}
        profile = json.loads(args.dwell_profile.read_text()) if args.dwell_profile else None
        actor = (TcpRemovalActor(list(candidates.values())) if args.tcp_removal else
                 PrestonDwellActor(profile, list(candidates.values()), term.path_length_mm) if profile
                 else BoundedVelocityActor(list(candidates.values()))).to(raw.device).eval()
        old_actor = None
        if args.include_checkpoint:
            old_actor = torch.jit.load(str(args.include_checkpoint), map_location=raw.device).eval()
        learned_agent = None
        if args.candidate_checkpoint:
            from nrs_rl.tasks.manager_based.nrs_rl.mdp.runtime.ppo_policy_node import _build_agent, _validate_scheduler_contract
            learned_agent, _ = _build_agent(str(args.candidate_checkpoint))
            _validate_scheduler_contract(learned_agent.velocity_contract)
        summary = {'kind': 'tcp_removal_parameter_evaluation' if args.tcp_removal else 'depth_policy_candidate_evaluation' if profile else 'scripted_feedback_screen',
                   'learned_policy': False,
                   'dt_s': raw.step_dt, 'physics_dt_s': cfg.sim.dt, 'seed': args.seed,
                   'path_length_mm': term.path_length_mm,
                   'parameter_names': TCP_PARAMETER_NAMES if args.tcp_removal else DWELL_PARAMETER_NAMES if profile else PARAMETER_NAMES,
                   'dwell_profile_sha256': hashlib.sha256(args.dwell_profile.read_bytes()).hexdigest() if profile else None,
                   'randomization': randomization, 'target_normal_force_n': integration.target_normal_force_n,
                   'target_mrr_n_mm_s': integration.target_mrr_n_mm_s,
                   'physics_tool_diameter_mm': term.physical_tool_diameter_mm,
                   'source_path_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
                   'completion_criterion': 'reference cursor; actual endpoint settling not simulated'}
        with torch.inference_mode():
            for step in range(args.max_steps):
                actions = actor(obs['policy'])
                if old_actor is not None:
                    old_index = names.index('old_d14')
                    actions[old_index:old_index+1] = old_actor(obs['policy'][old_index:old_index+1])
                if learned_agent is not None:
                    learned_index = names.index('learned')
                    actions[learned_index:learned_index+1] = learned_agent.act(obs['policy'][learned_index:learned_index+1])[-1]['mean_actions']
                actions[torch.as_tensor(ended, device=raw.device)] = 0.
                obs, reward, terminated, truncated, _ = env.step(actions)
                if not torch.isfinite(obs['policy']).all() or not torch.isfinite(reward).all():
                    raise RuntimeError('Nonfinite screening observation or reward')
                done = (terminated | truncated).cpu().numpy() & ~ended
                if done.any():
                    completed[done] = raw.termination_manager.get_term('trajectory_finished').cpu().numpy()[done]
                    ended |= done
                if step % 500 == 0 or ended.all():
                    report = {'phase': 'screening', 'control_steps': step+1,
                              'elapsed_seconds': time.monotonic()-started,
                              'completed': completed.tolist(), 'ended': ended.tolist(),
                              'cursor_mm': term.path_cursor_mm.cpu().tolist()}
                    (output / 'progress.json').write_text(json.dumps(report, indent=2))
                    print('FEEDBACK_SCREEN', json.dumps(report), flush=True)
                if ended.all():
                    break
        normal = spatial_to_rotmat(term.traj_positions[-1, 3:6].unsqueeze(0))[0, :, 2].cpu().numpy()
        surface = PlanarPrestonSurface(positions[:end, :3], normal,
                                     tool_diameter_mm=integration.tool_diameter_mm,
                                     cell_size_mm=args.surface_cell_mm)
        rotation_surface = None if args.tcp_removal else RotationDominatedSurface(positions[:end, :3], normal,
                                     tool_diameter_mm=integration.tool_diameter_mm,
                                     cell_size_mm=args.surface_cell_mm)
        summary['preston_surface_model'] = surface.metadata
        summary['rotation_depth_model'] = rotation_surface.metadata if rotation_surface else None
        summary['old_checkpoint_sha256'] = hashlib.sha256(args.include_checkpoint.read_bytes()).hexdigest() if args.include_checkpoint else None
        summary['candidate_checkpoint_sha256'] = hashlib.sha256(args.candidate_checkpoint.read_bytes()).hexdigest() if args.candidate_checkpoint else None
        from scipy.spatial import cKDTree
        path_tree = cKDTree((positions[:end,:3]-surface.origin) @ surface.basis.T)
        summary['candidates'] = {}
        for i, name in enumerate(names):
            records = rows[i]
            trace = {'time_s': np.arange(len(records))*.008,
                     'tcp_pose': np.stack([r[0] for r in records]),
                     'normal_force_n': np.array([r[1] for r in records]),
                     'measured_speed_mm_s': np.array([r[2] for r in records]),
                     'raw_mrr_n_mm_s': np.array([r[3] for r in records]),
                     'removal_step_n_mm': np.array([r[4] for r in records]),
                     'policy_observation': np.stack([r[6] for r in records]),
                     'command_pose': np.stack([r[7] for r in records])}
            columns = np.stack([r[5] for r in records])
            trace.update({key: columns[:, j] for j, key in enumerate(vis.CONTROL_TRACE_FIELDS)})
            np.savez_compressed(output / f'{name}_trace.npz', **trace)
            active = trace['polishing_active'] > 0
            rate = trace['raw_mrr_n_mm_s'][active]
            force = trace['normal_force_n'][active]
            cross_track, _ = path_tree.query((trace['tcp_pose'][active,:3]-surface.origin) @ surface.basis.T)
            endpoint = (trace['tcp_pose'][-1,:3]-positions[end-1,:3]) @ surface.basis.T
            metrics = {'parameters': candidates[name], 'completed': bool(completed[i]),
                       'processing_rate_cv': float(rate.std()/rate.mean()) if rate.mean()>0 else None,
                       'processing_mean_rate': float(rate.mean()), 'processing_samples': int(active.sum()),
                       'processing_rate_target_rmse': float(np.sqrt(np.mean((rate-120.)**2))),
                       'processing_speed_std_mm_s': float(trace['measured_speed_mm_s'][active].std()),
                       'processing_command_std_mm_s': float(trace['commanded_speed_mm_s'][active].std()),
                       'shield_fraction': float(np.mean(trace['safety_shield_active'][active]>0)),
                       'fault_fraction': float(np.mean(trace['safety_fault_reason'][active]!=0)),
                       'force_rmse_n': float(np.sqrt(np.mean((force-integration.target_normal_force_n)**2))),
                       'force_peak_n': float(force.max()),
                       'force_p99_n': float(np.quantile(force,.99)),
                       'command_acceleration_rms_mm_s2': float(np.sqrt(np.mean(trace['command_acceleration_mm_s2'][active]**2))),
                       'command_jerk_rms_mm_s3': float(np.sqrt(np.mean(trace['command_jerk_mm_s3'][active]**2))),
                       'cross_track_rmse_mm': float(np.sqrt(np.mean(cross_track**2))),
                       'cross_track_p95_mm': float(np.quantile(cross_track,.95)),
                       'endpoint_tangential_error_mm': float(np.linalg.norm(endpoint)),
                       'preston_surface': surface.integrate_trace(trace),
                       'rotation_depth': rotation_surface.integrate_trace(trace) if rotation_surface else None}
            surface.save(output / f'{name}_preston_surface.npz')
            if rotation_surface:
                rotation_surface.save(output / f'{name}_rotation_surface.npz')
            summary['candidates'][name] = metrics
        a, b = summary['candidates']['constant_a'], summary['candidates']['constant_b']
        baseline_difference = abs(a['processing_rate_cv'] / b['processing_rate_cv'] - 1)
        summary['duplicate_baseline_relative_cv_difference'] = baseline_difference
        for name, metrics in summary['candidates'].items():
            metrics['comparisons'] = {baseline: compare_rollouts(metrics, summary['candidates'][baseline])
                                      for baseline in ('constant_a', 'constant_b')}
            metrics['surface_comparisons'] = {
                baseline: compare_surfaces(metrics['preston_surface'], summary['candidates'][baseline]['preston_surface'])
                for baseline in ('constant_a', 'constant_b')}
        summary['wall_seconds'] = time.monotonic()-started
        (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
        (output / 'progress.json').write_text(json.dumps({'phase': 'finished', 'wall_seconds': summary['wall_seconds']}, indent=2))
        print('FEEDBACK_SCREEN_COMPLETE', json.dumps(summary), flush=True)
    finally:
        env.close()


status = 0
try:
    run()
except Exception:
    traceback.print_exc()
    status = 1
finally:
    sys.stdout.flush()
    sys.stderr.flush()
    app.app.post_quit(status)
    app.close()
sys.exit(status)
