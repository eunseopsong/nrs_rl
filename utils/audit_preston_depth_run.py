#!/usr/bin/env python3
"""Audit candidate rejection, grid sensitivity and velocity oscillation offline.

Reintegrates saved 125 Hz traces; no new policy is trained or marked passed.
Every resolution uses the same H5-swept physical ROI for all candidates.
"""
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
import time

import h5py
import numpy as np
from scipy.signal import butter, sosfiltfilt

from nrs_rl.tasks.manager_based.nrs_rl.utils.analyze_rotation_depth import RotationDominatedSurface, TASK


def audit(run, output):
    output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((run / 'summary.json').read_text())
    reference = TASK / 'datasets/cmd_continue9D_flat.h5'
    if hashlib.sha256(reference.read_bytes()).hexdigest() != summary['source_path_sha256']:
        raise ValueError('Reference trajectory differs from the recorded experiment')
    with h5py.File(reference) as stream:
        path = stream['position'][:, :3]
    names = ('constant_a', 'constant_b', 'profile_full')
    traces = {}
    for name in names:
        with np.load(run / f'{name}_trace.npz') as data:
            traces[name] = dict(data)
    report = {'source_run': str(run.resolve()), 'kind': 'offline_saved_trace_audit',
              'new_training': False, 'independent_validation': False,
              'physical_depth_validated': False, 'original_decision_changed': False,
              'grid_results': [], 'velocity_diagnostics': {}}
    started = time.monotonic()
    for cell in (2., 1., .5):
        surface = RotationDominatedSurface(path,
            summary['rotation_depth_model']['plane_normal'],
            tool_diameter_mm=30., cell_size_mm=cell)
        metrics, maps = {}, {}
        for name in names:
            metrics[name] = surface.integrate_trace(traces[name])
            maps[name] = surface.depth.copy()
        comparisons = {}
        for name in ('constant_b', 'profile_full'):
            value, baseline = metrics[name], metrics['constant_a']
            lost = surface.roi & (maps['constant_a'] > 0) & (maps[name] == 0)
            gained = surface.roi & (maps['constant_a'] == 0) & (maps[name] > 0)
            comparisons[name] = {
                'cv_improvement_percent': 100*(1-value['spatial_depth_cv']/baseline['spatial_depth_cv']),
                'mean_depth_ratio': value['mean_depth_over_k_omega']/baseline['mean_depth_over_k_omega'],
                'additional_zero_depth_area_mm2':
                    (value['zero_depth_fraction']-baseline['zero_depth_fraction'])*baseline['roi_area_mm2'],
                'lost_covered_area_mm2': int(lost.sum())*cell**2,
                'newly_covered_area_mm2': int(gained.sum())*cell**2,
            }
        record = {'cell_size_mm': cell, 'metrics': metrics, 'vs_constant_a': comparisons}
        report['grid_results'].append(record)
        np.savez_compressed(output / f'depth_maps_cell_{cell:g}mm.npz',
                            **maps, roi=surface.roi, u_mm=surface.u, v_mm=surface.v)
        (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        print(json.dumps({'cell_size_mm': cell, 'comparisons': comparisons,
                          'elapsed_seconds': time.monotonic()-started}), flush=True)
    for name, trace in traces.items():
        active = trace['polishing_active'] > 0
        start = trace['time_s'][active][0]
        settled = active & (trace['time_s'] >= start + 5.)
        velocity = trace['measured_speed_mm_s'][settled]
        command = trace['commanded_speed_mm_s'][settled]
        metrics = {'scope': 'processing after first 5 seconds; diagnostics only',
                   'measured_speed_std_mm_s': float(velocity.std()),
                   'command_speed_std_mm_s': float(command.std()),
                   'speed_tracking_rmse_mm_s': float(np.sqrt(np.mean((velocity-command)**2))),
                   'maximum_measured_speed_mm_s': float(velocity.max()),
                   'highpass_speed_rms_mm_s': {}}
        for cutoff in (1., 2., 5.):
            filtered = sosfiltfilt(butter(4, cutoff, btype='highpass', fs=125., output='sos'), velocity)
            # Trim both filter edges for the oscillation diagnostic only.
            metrics['highpass_speed_rms_mm_s'][str(cutoff)] = float(np.sqrt(np.mean(filtered[125:-125]**2)))
        report['velocity_diagnostics'][name] = metrics
    report['wall_seconds'] = time.monotonic()-started
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report['velocity_diagnostics']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.run, args.output)
