#!/usr/bin/env python3
"""Fit a reward forecast to actual Mode 6 deposition, without providing actor targets."""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(next(p / 'source/nrs_rl' for p in Path(__file__).resolve().parents
                              if (p / 'source/nrs_rl/nrs_rl').is_dir())))
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.optimize import minimize
from nrs_rl.tasks.manager_based.nrs_rl.utils.rotary_geometry import rotating_surface
from nrs_rl.tasks.manager_based.nrs_rl.utils.removal_metrics import processing_mask, save, sha


def calibrate(source, output, cell_mm):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((source / 'summary.json').read_text())
    row = summary['candidates']['constant']
    assert row['completed'] and row['fault_fraction'] == 0.
    assert summary['tangent_mdk'] == [6000., 2000.]
    if 'controller_experiment' in summary:
        from nrs_rl.tasks.manager_based.nrs_rl.mdp.steady_postcore import experimental_contract
        gain = summary['controller_experiment']['config']['gain']
        assert summary['controller_experiment'] == experimental_contract(gain)
        assert all(value == gain for value in summary['tracking_gains'].values())
    else:
        assert not any(summary['tracking_gains'].values())
    with np.load(source / 'constant_trace.npz') as archive:
        trace = dict(archive)
    with h5py.File(source / 'path.h5') as f:
        path = f['position'][:, :3]
    surface = rotating_surface(path, cell_mm=cell_mm)
    mask = processing_mask(surface, path)
    length = float(summary['path_length_mm'])
    arc = np.arange(0., length, .5)
    arc = np.r_[arc, length]
    cumulative = np.zeros((len(arc)-1, int(mask.sum())), dtype=np.float32)
    bin_count = int(np.ceil(length / 5.))
    dose = np.zeros((bin_count, int(mask.sum())), dtype=np.float64)
    previous = np.zeros(int(mask.sum()))
    next_forecast = 0
    previous_bin = 0
    for i, cursor in enumerate(trace['cursor_mm']):
        current_bin = min(int(cursor / 5.), bin_count - 1)
        if current_bin != previous_bin:
            current = surface.depth[mask]
            dose[previous_bin] = current - previous
            previous = current.copy()
            previous_bin = current_bin
        while next_forecast < len(cumulative) and cursor >= arc[next_forecast]:
            cumulative[next_forecast] = surface.depth[mask]
            next_forecast += 1
        if trace['time_s'][i] >= 2.:
            surface.deposit(trace['deposition_midpoint_mm'][i], trace['tangent_velocity_mm_s'][i],
                            float(trace['normal_force_n'][i]), .008,
                            contact=bool(trace['normal_force_n'][i] >= 1.5))
    total = surface.depth[mask]
    dose[previous_bin] = total - previous
    cumulative[next_forecast:] = total
    remaining = np.maximum(total[None, :] - cumulative, 0.)
    np.testing.assert_allclose(dose.sum(axis=0), total, atol=1.e-8)
    np.savez_compressed(output / 'forecast.npz', remaining=remaining, arc=arc,
        processing_mask=mask, baseline_processing=total.sum(), cell_mm=cell_mm)
    # This is only an optimistic fixed-path diagnostic. It never initializes
    # actor weights, supplies training actions, or replaces physics validation.
    normalized = dose / total.mean()
    q = normalized @ normalized.T / len(total)
    c = normalized.mean(axis=1)
    def objective(w):
        mu = c @ w
        return float(w @ q @ w / (mu * mu) - 1.)
    def derivative(w):
        mu = c @ w
        return 2. * q @ w / mu**2 - 2. * (w @ q @ w) * c / mu**3
    optimized = minimize(objective, np.ones(bin_count), jac=derivative, method='L-BFGS-B',
        bounds=[(.5, 4.)] * bin_count, options={'maxiter': 1500, 'ftol': 1.e-13, 'gtol': 1.e-9})
    baseline_cv = float(total.std() / total.mean())
    optimistic_cv = float(np.sqrt(max(0., optimized.fun)))
    report = {'source': str(source.resolve()), 'source_trace_sha256': sha(source / 'constant_trace.npz'),
        'source_seed': summary['seed'], 'forecast_sha256': sha(output / 'forecast.npz'),
        'cell_mm': cell_mm, 'entry_seconds': 2., 'processing_roi_cells': int(mask.sum()),
        'reward_forecast': 'Actual fixed Mode 6 processing removal remaining at each reference cursor',
        'actor_teacher_used': False, 'terminal_objective_changed': False,
        'robot_geometry': summary.get('probe_provenance', {}).get('robot_geometry'),
        'controller_experiment': summary.get('controller_experiment'),
        'fixed_path_diagnostic': {'baseline_cv': baseline_cv, 'optimistic_cv': optimistic_cv,
            'optimistic_improvement_percent': 100. * (1. - optimistic_cv / baseline_cv),
            'solver_converged': bool(optimized.success), 'duration_scale_bounds': [.5, 4.],
            'is_ppo_performance': False,
            'limitations': 'Holds measured TCP path fixed, approximates feed changes as residence scaling; not a physical feasibility proof.'}}
    save(output / 'calibration.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cell-mm', type=float, choices=[.5, 1., 2.], default=.5)
    a = p.parse_args()
    calibrate(a.source, a.output, a.cell_mm)
