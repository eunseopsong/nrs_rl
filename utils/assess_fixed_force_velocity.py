"""Strict, common-ROI assessment for the fixed-force virtual-rotation study."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import json
from pathlib import Path

import h5py
import numpy as np

from nrs_rl.tasks.manager_based.nrs_rl.utils.rotary_geometry import rotating_surface, SPINDLE_RPM
from nrs_rl.tasks.manager_based.nrs_rl.utils.removal_metrics import save, processing_mask


def evaluate(directory, *, rpm=SPINDLE_RPM, audit=False, write=True):
    directory = Path(directory)
    source = json.loads((directory/'summary.json').read_text())
    if source['target_force_fixed_n'] != 20. or source['action_dimension'] != 1:
        raise ValueError('Wrong action/force contract')
    if any(gain != .6 for gain in source['tracking_gains'].values()):
        raise ValueError('Unequal control settings')
    with h5py.File(directory/'path.h5') as h:
        path = h['position'][:, :3]
    surface = rotating_surface(path, rpm=rpm)
    mask = processing_mask(surface, path)
    report = {'seed': source['seed'], 'assumed_spindle_rpm': rpm,
        'comparison_baseline': 'paired Mode 6, fixed 20 N and 6 mm/s',
        'rotation_applied_to': 'removal estimator only; URDF dynamics unchanged',
        'entry_seconds': 2., 'entry_reference_mm': 18.,
        'full_roi_cells': int(surface.roi.sum()), 'processing_roi_cells': int(mask.sum()),
        'zero_depth_cells_retained': True, 'candidates': {}}
    for name, previous in source['candidates'].items():
        surface.reset()
        processing_depth = np.zeros_like(surface.depth)
        rates = []
        with np.load(directory/f'{name}_trace.npz') as f:
            trace = dict(f)
        t = trace['time_s']; force = trace['normal_force_n']; target = trace['target_force_n']
        if (not np.all(target == 20.) or trace['spatial_action'].shape != (len(t), 1)
                or not np.isfinite(trace['spatial_action']).all()
                or np.max(np.abs(trace['spatial_action'])) > 1.000001
                or not np.allclose(np.diff(t), .008, atol=1.e-10)
                or not np.all(np.diff(trace['cursor_mm']) >= -1.e-5)):
            raise ValueError(f'Force/action/clock/progress invariant violated: {name}')
        entry_depth = None
        for i in range(len(t)):
            if t[i] >= 2. and entry_depth is None:
                entry_depth = surface.depth.copy()
            before = surface.integrated_volume
            surface.deposit(trace['deposition_midpoint_mm'][i], trace['tangent_velocity_mm_s'][i],
                float(force[i]), .008, contact=bool(force[i] >= 1.5 and trace['safety_fault_reason'][i] == 0))
            rates.append((surface.integrated_volume-before)/.008)
        if entry_depth is None:
            raise ValueError('No processing samples after entry; candidate failed before assessment')
        processing_depth = surface.depth-entry_depth
        values = processing_depth[mask]
        keep = t >= 2.
        full = surface.metrics()
        if audit and rpm == SPINDLE_RPM:
            with np.load(directory/f'{name}_surface.npz') as f:
                np.testing.assert_allclose(surface.depth, f['depth_over_k'], atol=1.e-9, rtol=1.e-10)
                np.testing.assert_array_equal(surface.roi, f['roi'])
            np.testing.assert_allclose(rates, trace['raw_mrr_n_mm_s'], atol=1.e-7, rtol=1.e-9)
        m = {'spatial_cv': float(values.std()/values.mean()),
             'roi_volume_over_k': float(values.sum()*surface.area),
             'full_roi_volume_over_k': full['roi_volume_over_k'],
             'whole_grid_volume_over_k': full['grid_volume_over_k'],
             'time_s': len(t)*.008, 'processing_time_s': int(keep.sum())*.008,
             'force_target_min_n': float(target.min()), 'force_target_max_n': float(target.max()),
             'processing_measured_force_mean_n': float(force[keep].mean()),
             'processing_force_rmse_n': float(np.sqrt(np.mean((force[keep]-20.)**2))),
             'force_peak_n': float(force.max()), 'completed': previous['completed'],
             'fault_fraction': float(np.mean(trace['safety_fault_reason'] != 0)),
             'return_events': previous['return_events'], 'reverse_mm': previous['reverse_mm'],
             'zero_fraction': float(np.mean(values == 0)), 'samples': len(t),
             'scalar_action_only': True, 'reintegration_audited': audit and rpm == SPINDLE_RPM}
        report['candidates'][name] = m
        if write and rpm == SPINDLE_RPM:
            np.savez_compressed(directory/f'{name}_processing_surface.npz', depth_over_k=processing_depth,
                roi=mask, u_mm=surface.u, v_mm=surface.v, cell_area_mm2=surface.area)
    baseline = report['candidates']['constant']
    for m in report['candidates'].values():
        ratios = {key: m[key]/baseline[key] for key in
                  ['roi_volume_over_k', 'full_roi_volume_over_k', 'whole_grid_volume_over_k']}
        valid = lambda d: (d['completed'] and d['fault_fraction'] == 0. and d['force_peak_n'] <= 100.
                           and d['force_target_min_n'] == d['force_target_max_n'] == 20.
                           and d['return_events'] == 0 and d['reverse_mm'] == 0.)
        m.update(mode6_cv_gain=1.-m['spatial_cv']/baseline['spatial_cv'], mode6_volume_ratios=ratios,
                 eligible=bool(valid(m) and valid(baseline) and min(ratios.values()) >= .5))
    if write:
        save(directory/('processing_summary.json' if rpm == SPINDLE_RPM else f'rpm_{rpm:g}_summary.json'), report)
    return report
