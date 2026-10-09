#!/usr/bin/env python3
"""Estimate feed/dwell authority under a frozen, rotation-dominated trace model.

This counterfactual holds measured contact forces and the spatial footprint
fixed while retiming the pass. It is not a simulator rollout or an RL result.
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

import h5py
import numpy as np
from scipy.optimize import minimize
from nrs_rl.tasks.manager_based.nrs_rl.utils.analyze_rotation_depth import RotationDominatedSurface, TASK


def diagnose(evaluation, output, knot_spacing=20., maximum_dwell_change=.2):
    source = json.loads((evaluation / 'summary.json').read_text())
    with np.load(evaluation / 'constant_trace.npz') as data:
        trace = dict(data)
    with h5py.File(TASK / 'datasets/cmd_continue9D_flat.h5') as stream:
        path = stream['position'][:, :3]
    surface = RotationDominatedSurface(path, source['preston_surface_model']['plane_normal'],
        tool_diameter_mm=source['physics_tool_diameter_mm'],
        cell_size_mm=source['preston_surface_model']['cell_size_mm'])
    metrics = surface.integrate_trace(trace)
    knots = np.linspace(0, source['path_length_mm'], int(np.ceil(source['path_length_mm']/knot_spacing))+1)
    matrix = np.zeros((surface.roi.size, len(knots)))
    dwell = np.zeros(len(knots))
    xyz = np.asarray(trace['tcp_pose'][:, :3], dtype=np.float64)
    midpoint = xyz-.5*np.diff(xyz, axis=0, prepend=xyz[:1])
    for i in np.flatnonzero(trace['polishing_active'] > 0):
        coordinate = trace['path_cursor_mm'][i]/knots[-1]*(len(knots)-1)
        left = min(len(knots)-2, int(np.floor(coordinate)))
        fraction = np.clip(coordinate-left, 0, 1)
        dwell[left:left+2] += .008*np.array([1-fraction, fraction])
        if trace['normal_force_n'][i] < 1.5 or trace['safety_fault_reason'][i] != 0:
            continue
        center = (midpoint[i]-surface.origin) @ surface.basis.T
        rows, cols, dx, dy, weights, norm = surface._footprint(center)
        ids = rows*surface.shape[1]+cols
        value = trace['normal_force_n'][i]*weights/norm*np.hypot(dx, dy)*.008
        matrix[ids, left] += value*(1-fraction)
        matrix[ids, left+1] += value*fraction
    np.testing.assert_allclose(matrix.sum(axis=1).reshape(surface.shape), surface.depth, atol=1e-9)
    roi = matrix[surface.roi.ravel()]
    scale = roi.sum(axis=1).mean()
    a = roi/scale
    mean = a.mean(axis=0)
    centered = a-mean[None, :]
    differences = np.diff(np.eye(len(knots)), axis=0)
    hessian = centered.T@centered/len(a) + .001*differences.T@differences/len(knots)
    timing = dwell/dwell.sum()
    constraints = [
        {'type': 'eq', 'fun': lambda x: timing@x-1, 'jac': lambda x: timing},
        {'type': 'eq', 'fun': lambda x: mean@x-1, 'jac': lambda x: mean},
        {'type': 'ineq', 'fun': lambda x: maximum_dwell_change-differences@x,
         'jac': lambda x: -differences},
        {'type': 'ineq', 'fun': lambda x: maximum_dwell_change+differences@x,
         'jac': lambda x: differences},
    ]
    result = minimize(lambda x: x@hessian@x, np.ones(len(knots)), jac=lambda x: 2*hessian@x,
                      bounds=[(2/3, 2.)]*len(knots), constraints=constraints,
                      method='SLSQP', options={'ftol': 1e-10, 'maxiter': 500})
    if not result.success:
        raise RuntimeError(result.message)
    predicted = (matrix@result.x).reshape(surface.shape)
    values = predicted[surface.roi]
    cv = float(values.std()/values.mean())
    report = {
        'kind': 'frozen_trace_counterfactual', 'physics_validated': False, 'learned_policy': False,
        'source_evaluation': str(evaluation.resolve()),
        'reference_path_sha256': hashlib.sha256((TASK / 'datasets/cmd_continue9D_flat.h5').read_bytes()).hexdigest(),
        'assumptions': ['constant spindle RPM; rotation-dominated Preston law',
                        'measured force and spatial path unchanged by retiming',
                        'no prediction of servo response, contact dynamics, acceleration or jerk'],
        'model': surface.metadata, 'baseline': metrics,
        'predicted_spatial_cv': cv, 'predicted_cv_improvement_fraction': 1-cv/metrics['spatial_depth_cv'],
        'predicted_duration_ratio': float(timing@result.x), 'predicted_mean_depth_ratio': float(mean@result.x),
        'nominal_feed_limits_mm_s': [3., 9.], 'optimizer_iterations': result.nit,
        'knot_spacing_mm': float(knots[1]-knots[0]), 'maximum_adjacent_dwell_change': maximum_dwell_change,
        'cursor_knots_mm': knots.tolist(), 'dwell_multipliers': result.x.tolist(),
        'feed_schedule_mm_s': (6/result.x).tolist(),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    np.savez_compressed(output / 'maps.npz', baseline=surface.depth, predicted=predicted,
                        roi=surface.roi, u_mm=surface.u, v_mm=surface.v)
    print(json.dumps({key: value for key,value in report.items()
                      if key not in ('model', 'baseline', 'cursor_knots_mm', 'dwell_multipliers', 'feed_schedule_mm_s')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluation', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--knot-spacing-mm', type=float, default=20.)
    parser.add_argument('--maximum-dwell-change', type=float, default=.2)
    args = parser.parse_args()
    if args.knot_spacing_mm <= 0 or args.maximum_dwell_change <= 0:
        parser.error('Spacing and dwell-change limits must be positive')
    diagnose(args.evaluation, args.output, args.knot_spacing_mm, args.maximum_dwell_change)
