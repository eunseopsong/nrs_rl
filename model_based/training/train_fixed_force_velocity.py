#!/usr/bin/env python3
"""Fit scalar feed profiles with fixed 20 N and an assumed constant spindle.

Initial training optimizes residence time on a fixed path. The selected
checkpoint is accepted only after new Isaac rollouts, never from this model.
"""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import torch

from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.fixed_force_velocity import CONTRACT, REFERENCE, rotating_surface, export_actor











from nrs_rl.tasks.manager_based.nrs_rl.utils.removal_metrics import save, sha, processing_mask


def build_matrix(path, knots, *, trace=None):
    """Linear residence response; exact rotary kernel at reference/measured feed.

    Retiming keeps this kernel/footprint fixed. Fresh rollouts later include
    the changed servo response and actual feed in the local sliding vectors.
    """
    surface = rotating_surface(path)
    mask = processing_mask(surface, path)
    inverse = np.full(mask.size, -1, dtype=int)
    inverse[mask.ravel()] = np.arange(mask.sum())
    arc = np.r_[0., np.linalg.norm(np.diff(path, axis=0), axis=1).cumsum()]
    length = arc[-1]
    if trace is None:
        s = np.linspace(0., length, int(np.ceil(length/.25))+1)
        xyz = np.stack([np.interp(s, arc, path[:, j]) for j in range(3)], axis=1)
        delta = np.diff(xyz, axis=0)
        ds = np.linalg.norm(delta, axis=1)
        velocity = 6.*delta/np.maximum(ds[:, None], 1.e-12)
        midpoints = (xyz[:-1]+xyz[1:])*.5
        cursor = (s[:-1]+s[1:])*.5
        force = np.full(len(ds), 20.)
        dt = ds/6.
        keep = cursor >= 12.  # The 18 mm spatial exclusion is stricter.
    else:
        with np.load(trace) as t:
            if not np.all(t['target_force_n'] == 20.):
                raise ValueError('Training dynamics must have a fixed 20 N target')
            midpoints = t['deposition_midpoint_mm'].copy()
            velocity = t['tangent_velocity_mm_s'].copy()
            cursor = t['cursor_mm'].copy()
            force = t['normal_force_n'].copy()
            dt = np.full(len(force), .008)
            keep = (t['time_s'] >= 2.) & (force >= 1.5) & (t['safety_fault_reason'] == 0)
    matrix = np.zeros((mask.sum(), knots))
    for i in np.flatnonzero(keep):
        rr, cc, dx, dy, w, norm = surface._footprint((midpoints[i]-surface.origin) @ surface.basis.T)
        ids = inverse[rr*surface.shape[1]+cc]
        valid = (ids >= 0) & (w > 0)
        feed = velocity[i] @ surface.basis.T
        relative = np.hypot(feed[0]-surface.omega*dy, feed[1]+surface.omega*dx)
        values = force[i]*w[valid]/norm*relative[valid]*dt[i]
        phase = np.clip(cursor[i]/length, 0., 1.)*(knots-1)
        left = min(int(phase), knots-2)
        fraction = phase-left
        matrix[ids[valid], left] += values*(1.-fraction)
        matrix[ids[valid], left+1] += values*fraction
    return matrix, length


def fit(matrix, knots, desired_volume_ratio, max_delta):
    baseline = matrix.sum(axis=1)
    a = matrix/baseline.mean()
    mean = a.mean(axis=0)
    q = a.T@a/len(a)
    d = np.diff(np.eye(knots), axis=0)
    smooth = d.T@d/knots
    def objective(x):
        m = mean@x
        square = x@q@x
        return (square/m**2 - 1. + 1.e-5*(x@smooth@x),
                2.*q@x/m**2 - 2.*square*mean/m**3 + 2.e-5*smooth@x)
    constraints = [LinearConstraint(mean[None, :], desired_volume_ratio, desired_volume_ratio),
                   LinearConstraint(d, -max_delta, max_delta)]
    result = minimize(objective, np.full(knots, desired_volume_ratio), jac=True,
                      bounds=[(2./3., 2.)]*knots, constraints=constraints,
                      method='SLSQP', options={'ftol': 1.e-11, 'maxiter': 1000})
    if (not result.success or abs(mean@result.x-desired_volume_ratio) > 1.e-6
            or np.max(np.abs(np.diff(result.x))) > max_delta+1.e-6):
        raise RuntimeError(f'Residence optimization failed: {result.message}')
    depth = a@result.x
    return 6./result.x, {
        'solver_success': True, 'iterations': int(result.nit),
        'predicted_spatial_cv': float(depth.std()/depth.mean()),
        'predicted_mode6_cv_gain': float(1.-(depth.std()/depth.mean())/(baseline.std()/baseline.mean())),
        'predicted_mode6_volume_ratio': float(mean@result.x),
        'nominal_mode6_spatial_cv': float(baseline.std()/baseline.mean()),
        'dwell_multipliers': result.x.tolist(),
        'qualification': 'model fit only; fixed footprint approximation, not Isaac validation',
    }


def main(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root/'checkpoints').mkdir()
    torch.set_num_threads(1)
    protocol = {**CONTRACT, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'selection_seeds': [2601, 2602], 'heldout_seeds': [2611, 2612],
        'holdouts_used_for_selection': False, 'minimum_mode6_cv_gain_each_holdout': .005,
        'rpm_sensitivity_values': [300., 1000., 3000.],
        'acceptance': 'Both fresh heldouts: completed, no faults, exactly 20 N target, >=50% volume in all scopes, CV gain above paired repeat variability and 0.5%',
        'training_data': str(args.trace) if args.trace else 'reference geometry only',
        'training_data_sha256': sha(args.trace) if args.trace else sha(REFERENCE),
        'entry_peak_is_objective': False, 'hard_stop_above_100n': True,
        'previous_variable_force_checkpoints_are_not_eligible': True}
    save(root/'protocol.json', protocol)
    with h5py.File(REFERENCE) as h:
        path = h['position'][:, :3]
    definitions, fits = {}, {}
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    definitions['constant'] = {'speed_profile': [6., 6.]}
    for knots, volume, smooth in [(49, 1., .4), (97, .9, .25), (97, 1.1, .25), (193, 1., .15)]:
        name = f'profile{knots}_{int(round(volume*100))}'
        matrix, length = build_matrix(path, knots, trace=args.trace)
        speed, metrics = fit(matrix, knots, volume, smooth)
        definitions[name] = {'speed_profile': speed.tolist()}
        fits[name] = metrics
        save(root/'fit_results.json', fits)
        print('FIXED_FORCE_PROFILE_FIT', name, json.dumps({k:v for k,v in metrics.items() if k!='dwell_multipliers'}), flush=True)
    definitions['profile97_feedback'] = {**definitions['profile97_90'], 'feedback': [-2., -2., 0., -1., 0.]}
    definitions['profile97_pause'] = {**definitions['profile97_90'], 'pause_period_s': 8., 'pause_duration_s': .64}
    exports = {}
    for name, parameters in definitions.items():
        exports[name] = export_actor(root/'checkpoints'/f'{name}.pt', parameters,
                                     name=name, length=length, protocol=protocol)
    save(root/'definitions.json', definitions)
    save(root/'checkpoint_manifest.json', exports)
    save(root/'status.json', {'phase': 'model_fit_complete_pending_physics', 'candidates': list(definitions),
                              'protocol': protocol, 'deployment_ready': False})
    print('FIXED_FORCE_FIT_COMPLETE', str(root), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--trace', type=Path, help='Optional fixed-force Mode 6 physics trace for refinement')
    main(p.parse_args())
