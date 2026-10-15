"""Matched four-slot Isaac evaluation of fixed/learned virtual spindle RPM."""
import argparse
import json
from pathlib import Path
import time
from isaaclab.app import AppLauncher
from nrs_rl.tasks.manager_based.nrs_rl.mdp.robot_geometry import configure_training_robot
from nrs_rl.tasks.manager_based.nrs_rl.mdp.rpm_control import RPMConfig

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--parent', type=Path, required=True)
p.add_argument('--candidate', type=Path)
p.add_argument('--seed', type=int, default=5211)
p.add_argument('--native-dir', type=Path, required=True)
p.add_argument('--control-snapshot', type=Path, required=True)
p.add_argument('--gui-scene', type=Path, required=True)
p.add_argument('--max-steps', type=int, default=24000)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.output.exists() or args.device != 'cpu':
    p.error('Use a new output directory and --device cpu')
robot = configure_training_robot(args.control_snapshot)
app = AppLauncher(args).app

import numpy as np
import torch
from nrs_rl.tasks.manager_based.nrs_rl.mdp.isaac_feed_rpm_env import IsaacFeedRPMEnv
from nrs_rl.tasks.manager_based.nrs_rl.utils.removal_metrics import save, sha


def run():
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    output = args.output.resolve()
    output.mkdir(parents=True)
    c = RPMConfig()
    parent = torch.jit.load(str(args.parent), map_location='cpu').eval()
    candidate = None
    if args.candidate:
        extra = {'feed_rpm_policy.json': ''}
        candidate = torch.jit.load(str(args.candidate), map_location='cpu', _extra_files=extra).eval()
        metadata = json.loads(extra['feed_rpm_policy.json'])
        if metadata['backend'] != 'ppo_feed_rpm_v1' or metadata['rpm_policy'] != c.metadata():
            raise ValueError('RPM policy contract mismatch')
    save(output / 'protocol.json', {'seed': args.seed, 'robot_geometry': robot,
        'slot_names': ['mode6', 'mode6_duplicate', 'parent_mode5', 'candidate' if candidate else 'parent_slot3'],
        'parent': str(args.parent), 'parent_sha256': sha(args.parent),
        'candidate': str(args.candidate) if candidate else None,
        'candidate_sha256': sha(args.candidate) if candidate else None,
        'rate_objective': c.metadata(), 'policy_periods_s': [None, None, .08, .016 if candidate else .08],
        'full_contact': 'first measured >=1.5N including stationary entry through completion',
        'fine_cell_mm': .5, 'nominal_sensors': True, 'physical_spindle_actuated': False})
    env = IsaacFeedRPMEnv(4, args.seed, args.native_dir, args.control_snapshot, output,
                         c, args.gui_scene, record=True)
    ended = np.zeros(4, dtype=bool)
    saved = {}
    action = torch.zeros(4, 2)
    obs = env.observation()
    started = last_report = time.monotonic()
    try:
        with torch.inference_mode():
            for step in range(args.max_steps):
                if step % 5 == 0:
                    action[2, 0] = parent(obs[2:3, :16])[0, 0]
                    if candidate is None:
                        action[3, 0] = parent(obs[3:4, :16])[0, 0]
                if candidate is not None:
                    action[3] = candidate(obs[3:4])[0]
                obs, _, done, records = env.step(action)
                for i, row in enumerate(records):
                    if row is None or ended[i]:
                        continue
                    ended[i] = True
                    captured = env.completed_records[i]
                    trace = captured['trace']
                    fields = ['tcp_pose', 'normal_force_n', 'measured_speed_mm_s', 'cursor_mm',
                        'commanded_speed_mm_s', 'rpm', 'commanded_rpm', 'removal_rate_over_k',
                        'midpoint_mm', 'tangent_velocity_mm_s', 'safety_fault_reason', 'shield',
                        'polishing_active', 'filtered_force_slew_n_s']
                    arrays = {key: np.asarray([record[j] for record in trace]) for j, key in enumerate(fields)}
                    arrays['time_s'] = np.arange(len(trace)) * .008
                    arrays['progress_pct'] = 100. * arrays['cursor_mm'] / env.geometry.length
                    np.savez_compressed(output / f'slot{i}_trace.npz', **arrays)
                    s = env.fine_geometry.surface
                    np.savez_compressed(output / f'slot{i}_depth.npz', depth_over_k=captured['depth'],
                        roi=s.roi, u_mm=s.u, v_mm=s.v, cell_area_mm2=s.area, origin_mm=s.origin, basis=s.basis)
                    q = arrays['removal_rate_over_k']
                    if len(trace) != row['contact_samples']:
                        raise RuntimeError('Contact metric/trace sample mismatch')
                    np.testing.assert_allclose(q.mean(), row['mean_rate_over_k'], rtol=1.e-10)
                    np.testing.assert_allclose(q.std() / q.mean(), row['temporal_rate_cv'], rtol=1.e-7)
                    np.testing.assert_allclose(q.sum() * .008,
                                               captured['depth'].sum() * s.area, rtol=1.e-9)
                    row.update(slot=i, final_cursor_mm=float(arrays['cursor_mm'][-1]),
                        fault_samples=int(np.count_nonzero(arrays['safety_fault_reason'])),
                        rpm_min=float(arrays['rpm'].min()), rpm_max=float(arrays['rpm'].max()),
                        rpm_mean=float(arrays['rpm'].mean()), windows={})
                    for name, mask in [
                        ('entry_first3s', arrays['time_s'] <= 3.),
                        ('path85_90pct', (arrays['progress_pct'] >= 85.) & (arrays['progress_pct'] <= 90.)),
                        ('path98p5_100pct', arrays['progress_pct'] >= 98.5)]:
                        if not mask.any():
                            continue
                        error = q[mask] / c.target_rate_over_k - 1.
                        row['windows'][name] = {'samples': int(mask.sum()),
                            'rate_relative_rmse': float(np.sqrt(np.mean(error**2))),
                            'rate_peak_to_peak_over_target': float(np.ptp(q[mask]) / c.target_rate_over_k),
                            'force_peak_to_peak_n': float(np.ptp(arrays['normal_force_n'][mask]))}
                    saved[str(i)] = row
                    save(output / 'result_partial.json', saved)
                    print('RPM_EVAL_EPISODE', json.dumps(row), flush=True)
                if ended.all():
                    break
                if time.monotonic() - last_report > 30.:
                    progress = {'step': step, 'wall_seconds': time.monotonic() - started,
                                'ended': ended.tolist(), 'cursor_mm': env.term.path_cursor_mm.tolist()}
                    save(output / 'status.json', progress)
                    print('RPM_EVAL_PROGRESS', json.dumps(progress), flush=True)
                    last_report = time.monotonic()
        if not ended.all():
            raise RuntimeError('Evaluation did not produce all four episodes')
        save(output / 'results.json', {'slots': saved, 'wall_seconds': time.monotonic() - started,
                                      'complete': all(row['completed'] for row in saved.values())})
    finally:
        env.close()


try:
    run()
finally:
    app.close()
