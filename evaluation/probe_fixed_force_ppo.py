#!/usr/bin/env python3
"""Independent Isaac evaluation of neural PPO and preserved fixed-force baselines."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--training-dir', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--names', default='constant,constant_b,profile')
parser.add_argument('--seed', type=int, default=1501)
parser.add_argument('--path-length-mm', type=float, default=698.)
parser.add_argument('--max-steps', type=int, default=42000)
parser.add_argument('--tracking-parameters', type=Path, help='Optional explicit per-candidate bounded tracking gains')
parser.add_argument('--hard-force-stop-n',type=float,help='Explicit simulation guard; recorded in result')
parser.add_argument('--uniformity-only',action='store_true',help='No volume/time/mean-force selection penalties')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists():
    parser.error('Use a new output directory')
app = AppLauncher(args).app

import gymnasium as gym
import h5py
import numpy as np
import torch
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.io import dump_yaml
import nrs_rl.tasks
from nrs_rl.tasks.manager_based.nrs_rl.mdp import action as action_module
from nrs_rl.tasks.manager_based.nrs_rl.utils import visualization as vis
from nrs_rl.tasks.model_based.policies.spatial_uniformity import DoseGeometry, PlanarPrestonSurface, ROOT, REFERENCE, CONTRACT, assess
from nrs_rl.tasks.manager_based.nrs_rl.mdp.fixed_force_ppo_action import FixedForcePPOAction as FixedForceRefinementAction
from nrs_rl.tasks.manager_based.nrs_rl.utils.rotary_geometry import FixedForceGeometry, rotating_surface, SPINDLE_RPM
from nrs_rl.tasks.manager_based.nrs_rl.agents.fixed_force_ppo import check_ppo_or_baseline as check_contract


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def run():
    torch.set_num_threads(1)
    names = args.names.split(',')
    if len(set(names)) != len(names) or 'constant' not in names:
        raise ValueError('Need unique names including constant')
    output = args.output.resolve()
    output.mkdir(parents=True)
    cfg = parse_env_cfg('Template-Nrs-Rl-v0', device=args.device or 'cpu', num_envs=len(names))
    cfg.seed = args.seed
    cfg.visualization.enable_visualizer = False
    cfg.actions.arm_action.class_type = FixedForceRefinementAction
    cfg.actions.arm_action.integration.enable_debug_print = False
    cfg.log_dir = str(output)
    integration = cfg.actions.arm_action.integration
    if args.hard_force_stop_n is not None:
        if not 40.<=args.hard_force_stop_n<=100.:raise ValueError('Simulation guard must be between 40 and 100 N')
        integration.max_force_abort_n=args.hard_force_stop_n
    if args.uniformity_only:
        # Keep the global episode deadline, but no separate entry-overload timeout.
        integration.shield_timeout_s=1000.
    if integration.spindle_rpm is not None:
        raise ValueError('The dynamics stay nonrotating; virtual RPM belongs to the removal estimator')
    with h5py.File(REFERENCE) as h:
        positions = h['position'][:]
        arc = np.r_[0., np.linalg.norm(np.diff(positions[:, :3], axis=0), axis=1).cumsum()]
        end = min(len(positions), max(2, int(np.searchsorted(arc, args.path_length_mm))+1))
        with h5py.File(output/'path.h5', 'w') as target:
            target['position'] = positions[:end]
            target['force'] = h['force'][:end]
    integration.hdf5_file_path = str(output/'path.h5')
    dump_yaml(str(output/'env.yaml'), cfg)
    native = ROOT/'logs/experiments/tcp_removal_retrain_20261005/calibrated_control/native'
    sys.path.insert(0, str(native))
    from _tcp_calibrated_mode3 import TangentialMode3
    damping, stiffness = 3329.844766906654, 12434.33963608066
    tracking = ({name: .6 for name in names} if args.tracking_parameters is None
                else json.loads(args.tracking_parameters.read_text()))
    if set(tracking) != set(names) or any(not 0. <= value <= 1. for value in tracking.values()):
        raise ValueError('Tracking gains must explicitly match candidate names and stay within [0, 1]')
    if any(value != .6 for value in tracking.values()):
        raise ValueError('Mode 5/6 must use the identical, frozen 0.6 tracking gain')
    from nrs_rl.tasks.manager_based.nrs_rl.utils.tcp_tracking_compensation import TangentTrackingCompensation
    built = []
    def make_controller(*values):
        name = names[len(built)]
        controller = TangentialMode3(*values, damping, stiffness)
        if tracking[name] != 0.:
            controller = TangentTrackingCompensation(controller, tracking[name])
        built.append(name)
        return controller
    constructor = action_module.y2_pb.Mode3ForceController
    action_module.y2_pb.Mode3ForceController = make_controller
    try:
        env = gym.make('Template-Nrs-Rl-v0', cfg=cfg)
    finally:
        action_module.y2_pb.Mode3ForceController = constructor
    started = time.monotonic()
    try:
        raw = env.unwrapped
        term = raw.action_manager.get_term('arm_action')
        if raw.step_dt != .008:
            raise ValueError('Expected 125 Hz control')
        geometry = FixedForceGeometry(output/'path.h5')
        surfaces = [rotating_surface(positions[:end, :3], cell_mm=2.) for _ in names]
        residence = [np.zeros_like(surface.depth) for surface in surfaces]
        rows = [[] for _ in names]
        last_position = [None for _ in names]
        ended = np.zeros(len(names), dtype=bool)
        completed = np.zeros(len(names), dtype=bool)
        final_returns = [{} for _ in names]

        def capture(i, pose, normal_force):
            if ended[i]:
                return
            point = pose.cpu().numpy().copy()
            previous = last_position[i]
            last_position[i] = point[:3].copy()
            if not bool(term.polishing_active[i]):
                return
            force = abs(float(normal_force))
            speed = float(term.current_sliding_velocity_mm_s[i])
            delta = np.zeros(3) if previous is None else point[:3]-previous
            delta[2] = 0.
            velocity = delta/max(np.linalg.norm(delta), 1e-12)*speed
            midpoint = point[:3]-.5*delta
            contact = force >= 1.5 and int(term.safety_fault_reason[i]) == 0
            surface = surfaces[i]
            surface.deposit(midpoint, velocity, force, .008, contact=contact)
            if contact:
                center = (midpoint-surface.origin) @ surface.basis.T
                rr, cc, _, _, weights, _ = surface._footprint(center)
                residence[i][rr, cc] += .008*(weights > 0)
            rs = term.return_states[i]
            final_returns[i] = {'events': rs.events, 'reverse_mm': rs.reverse_mm, 'frontier_mm': rs.frontier}
            rows[i].append((point, force, speed, float(term.target_force[i]),
                vis.control_trace_snapshot(term, i), float(term.path_cursor_mm[i]),
                term.raw_actions[i].cpu().numpy().copy(),
                [rs.reverse_mm, rs.events], term._previous_command_pose[i].cpu().numpy().copy(),
                velocity.copy(), midpoint.copy()))

        term._record_visualization = capture
        obs, _ = env.reset(seed=args.seed)
        for attribute in ('_force_scale', '_force_bias', '_action_delay'):
            values = getattr(term, attribute)
            values.fill_(values[0].item())
        randomization = {key: getattr(term, key).cpu().tolist()
                         for key in ('_force_scale', '_force_bias', '_action_delay')}
        actors, hashes = {}, {}
        policy_contracts = {}
        for i, name in enumerate(names):
            checkpoint = args.training_dir/'checkpoints'/f'{"constant" if name == "constant_b" else name}.pt'
            extra = {'fixed_force_policy.json': ''}
            actors[name] = torch.jit.load(str(checkpoint), _extra_files=extra, map_location=raw.device).eval()
            contract = json.loads(extra['fixed_force_policy.json'])
            check_contract(contract)
            term.configure_velocity(i, contract)
            policy_contracts[name] = contract
            hashes[name] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        actions = torch.full((len(names), 1), 1./3., device=raw.device)
        with torch.inference_mode():
            for step in range(args.max_steps):
                if step % 10 == 0:
                    depth = np.stack([s.depth.ravel()[geometry.roi_flat] for s in surfaces])
                    dwell = np.stack([r.ravel()[geometry.roi_flat] for r in residence])
                    cursor = term.path_cursor_mm.cpu().numpy()
                    features = geometry.features(depth, dwell, cursor,
                        np.array([r.frontier for r in term.return_states]), term.current_abs_fz.cpu().numpy(),
                        term.current_sliding_velocity_mm_s.cpu().numpy(),
                        np.array([len(r)*.008 for r in rows]), np.array([r.reverse_mm for r in term.return_states]),
                        term.target_force, np.zeros(len(names)),
                        tracking=term.current_path_tracking_error_mm.cpu().numpy(),
                        shield=term.safety_shield_active.cpu().numpy())
                    features = torch.as_tensor(features, device=raw.device)
                    for i, name in enumerate(names):
                        actions[i] = actors[name](features[i:i+1])[0]
                actions[torch.as_tensor(ended, device=raw.device)] = 1./3.
                obs, reward, terminated, truncated, _ = env.step(actions)
                if not torch.isfinite(obs['policy']).all() or not torch.isfinite(reward).all():
                    raise RuntimeError('Nonfinite physics state')
                done = (terminated | truncated).cpu().numpy() & ~ended
                if done.any():
                    completed[done] = raw.termination_manager.get_term('trajectory_finished').cpu().numpy()[done]
                    ended |= done
                if step % 1000 == 0 or ended.all():
                    progress = {'phase': 'isaac_validation', 'step': step+1, 'elapsed_s': time.monotonic()-started,
                                'cursor_mm': term.path_cursor_mm.cpu().tolist(), 'completed': completed.tolist()}
                    save(output/'progress.json', progress)
                    print('SPATIAL_ISAAC', json.dumps(progress), flush=True)
                if ended.all():
                    break
        from scipy.spatial import cKDTree
        tree = cKDTree(positions[:end, :2])
        fine = rotating_surface(positions[:end, :3], cell_mm=.5)
        summary = {'kind': 'isaac_neural_ppo_fixed_force_with_virtual_rotation', 'seed': args.seed, 'names': names,
            'checkpoint_sha256': hashes, 'randomization': randomization,
            'path_length_mm': term.path_length_mm, 'normal_force_controller_weights_unchanged': True,
            'tangent_mdk': [damping, stiffness], 'rotation_contribution': True, 'assumed_spindle_rpm': SPINDLE_RPM,
            'rotation_applied_to': 'removal estimator only; URDF dynamics unchanged',
            'tracking_gains': tracking,
            'hard_force_stop_n':integration.max_force_abort_n,
            'shield_timeout_s':integration.shield_timeout_s,
            'force_bounds_n':[[20.,20.] for _ in names],
            'action_dimension':1, 'target_force_fixed_n':20.,
            'policy_contracts':policy_contracts, 'feed_limits_mm_s':[[0.,2.*x] for x in term.half_range],
            'feed_action_slew_mm_s2':18.,
            'uniformity_only':args.uniformity_only,
            'observation_surface_cell_mm': 2., 'evaluation_surface_cell_mm': .5,
            'roi': fine.metadata, 'control_period_s': .008, 'policy_period_s': .08,
            'completion_criterion': 'reference cursor; endpoint settling not simulated',
            'wall_seconds': time.monotonic()-started, 'candidates': {}}
        for i, name in enumerate(names):
            records = rows[i]
            if len(records) < 2:
                raise RuntimeError(f'Insufficient samples for {name}')
            trace = {'time_s': np.arange(len(records))*.008,
                'tcp_pose': np.stack([r[0] for r in records]),
                'normal_force_n': np.array([r[1] for r in records]),
                'measured_speed_mm_s': np.array([r[2] for r in records]),
                'target_force_n': np.array([r[3] for r in records]),
                'cursor_mm': np.array([r[5] for r in records]),
                'spatial_action': np.stack([r[6] for r in records]),
                'return_state': np.array([r[7] for r in records]),
                'command_pose': np.stack([r[8] for r in records]),
                'tangent_velocity_mm_s': np.stack([r[9] for r in records]),
                'deposition_midpoint_mm': np.stack([r[10] for r in records])}
            columns = np.stack([r[4] for r in records])
            trace.update({key: columns[:, j] for j, key in enumerate(vis.CONTROL_TRACE_FIELDS)})
            force, speed = trace['normal_force_n'], trace['measured_speed_mm_s']
            rate = np.where((force >= 1.5) & (trace['safety_fault_reason'] == 0), force*speed, 0.)
            trace['tcp_only_force_speed_n_mm_s'] = rate.copy()
            assert np.all(trace['target_force_n'] == 20.), 'Force target changed'
            assert trace['spatial_action'].shape[1] == 1, 'Unexpected policy authority'
            # The first active sample can move relative to the last approach
            # sample. Its measured vector/midpoint are captured above so no
            # valid entry removal is dropped or inferred from a missing row.
            fine.reset()
            rate = np.zeros(len(records))
            for j in range(len(records)):
                before = fine.integrated_volume
                fine.deposit(trace['deposition_midpoint_mm'][j], trace['tangent_velocity_mm_s'][j],
                    force[j], .008, contact=bool(force[j] >= 1.5 and trace['safety_fault_reason'][j] == 0))
                rate[j] = (fine.integrated_volume-before)/.008
            trace['raw_mrr_n_mm_s'] = rate
            np.savez_compressed(output/f'{name}_trace.npz', **trace)
            metrics = fine.metrics()
            fine.save(output/f'{name}_surface.npz')
            np.savez_compressed(output/f'{name}_residence.npz', residence_s=residence[i],
                                roi=surfaces[i].roi, u_mm=surfaces[i].u, v_mm=surfaces[i].v)
            cross_track, _ = tree.query(trace['tcp_pose'][:, :2])
            summary['candidates'][name] = {
                'spatial_cv': metrics['spatial_depth_cv'], 'roi_volume_over_k': metrics['roi_volume_over_k'],
                'time_s': len(records)*.008, 'force_mean_n': float(force.mean()), 'force_peak_n': float(force.max()),
                'force_tracking_rmse_n': float(np.sqrt(np.mean((force-trace['target_force_n'])**2))),
                'force_target_min_n': float(trace['target_force_n'].min()), 'force_target_max_n': float(trace['target_force_n'].max()),
                'rate_cv': float(rate.std()/max(rate.mean(), 1e-12)), 'distance_mm': float(speed.sum()*.008),
                'completed': bool(completed[i]), 'zero_fraction': metrics['zero_depth_fraction'],
                'fault_fraction': float(np.mean(trace['safety_fault_reason'] != 0)),
                'shield_fraction': float(np.mean(trace['safety_shield_active'] != 0)),
                'cross_track_rmse_mm': float(np.sqrt(np.mean(cross_track**2))),
                'endpoint_error_mm': float(np.linalg.norm(trace['tcp_pose'][-1, :2]-positions[end-1, :2])),
                'return_events': final_returns[i]['events'], 'reverse_mm': final_returns[i]['reverse_mm'],
                'surface': metrics}
        baseline = summary['candidates']['constant']
        assessor=assess
        if args.uniformity_only:
            from nrs_rl.tasks.manager_based.nrs_rl.utils.uniformity_priority_metrics import assess_uniformity
            assessor=assess_uniformity
        for candidate in summary['candidates'].values():
            candidate['comparison'] = assessor(candidate, baseline)
        save(output/'summary.json', summary)
        save(output/'progress.json', {'phase': 'finished', 'wall_seconds': time.monotonic()-started})
        print('SPATIAL_ISAAC_COMPLETE', str(output/'summary.json'), flush=True)
    finally:
        env.close()


try:
    run()
except Exception:
    traceback.print_exc()
    raise
finally:
    app.close()
