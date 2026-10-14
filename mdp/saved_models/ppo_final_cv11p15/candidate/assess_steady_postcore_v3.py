"""Study-private common-ROI assessment of the explicitly tagged steady post-core controller."""
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
    from steady_postcore_v3 import experimental_contract
    contract = source['controller_experiment']
    gain = contract['config']['gain']
    if contract != experimental_contract(gain):
        raise ValueError('Unexpected experimental post-core controller contract')
    if not source['paired_same_controller_variant'] or any(value != gain for value in source['tracking_gains'].values()):
        raise ValueError('Every actor and Mode6 reference must use the same controller variant')
    if source['tangent_mdk'] != [6000., 2000.]:
        raise ValueError('Force controller was changed')
    native = source['native_validation']
    if native['tangent_damping_ns_m'] != 6000. or native['tangent_stiffness_n_m'] != 2000.:
        raise ValueError('Native force controller was changed')
    with h5py.File(directory/'path.h5') as h:
        path = h['position'][:, :3]
    surface = rotating_surface(path, rpm=rpm)
    mask = processing_mask(surface, path)
    report = {'seed': source['seed'], 'assumed_spindle_rpm': rpm,
        'comparison_baseline': 'paired Mode6 under the SAME explicitly experimental post-core controller',
        'controller_experiment': contract, 'experimental': True,
        'position_to_force_source_preservation': source['probe_provenance']['position_to_force_transition_sources_preserved'],
        'scope': 'isolated headless study; no ROS deployment or ROS performance established',
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
        active = trace['steady_active']
        releasing = trace['steady_releasing']
        applied = trace['steady_applied']
        correction = trace['steady_correction_mm']
        np.testing.assert_array_equal(active, trace['steady_eligible'])
        np.testing.assert_array_equal(releasing, trace['steady_soft_releasing'])
        np.testing.assert_array_equal(applied, active | releasing)
        if (np.any(trace['steady_epoch_s'][applied] <= 2.)
                or np.any(trace['steady_stable_s'][active] < .5-1.e-12)
                or np.any(np.linalg.norm(correction, axis=1) > 1.5+1.e-10)
                or np.any(trace['steady_normal_component_mm'] > 1.e-10)
                or np.any(correction[~applied] != 0.)
                or np.any(trace['steady_gate_force_n'][active] < 18.)
                or np.any(trace['steady_gate_force_n'][active] > 22.)
                or np.any(trace['steady_gate_tracking_mm'][applied] > 10.)
                or np.any(trace['safety_fault_reason'][applied] != 0.)
                or np.any(trace['steady_reason'][releasing] != 2)
                or np.any(trace['steady_gate_force_n'][releasing] <= 1.5)
                or np.any(trace['steady_gate_force_n'][releasing] > 32.)
                or np.any(trace['safety_shield_active'][releasing] != 0.)):
            raise ValueError('Steady-only correction eligibility/bounds violated: ' + name)
        decay = np.exp(-.008/.08)
        for i in np.flatnonzero(releasing):
            if i == 0 or not applied[i-1] or not np.any(correction[i-1]):
                raise ValueError('Soft release did not originate from a qualified offset: ' + name)
            before_norm = np.linalg.norm(correction[i-1])
            if np.linalg.norm(correction[i]) > before_norm * decay + 1.e-12:
                raise ValueError('Unqualified correction grew during release: ' + name)
            step = np.linalg.norm(correction[i] - correction[i-1])
            bound = before_norm * (1.-decay) + 1.e-6 + trace['steady_soft_projection_adjustment_mm'][i]
            if step > bound + 1.e-12 or abs(step-trace['steady_soft_release_step_mm'][i]) > 1.e-12:
                raise ValueError('Soft release step bound violated: ' + name)
        raw_valid = trace['steady_raw_output_valid'] & (trace['safety_fault_reason'] == 0)
        # capture() receives the same pre-physics force snapshot action.py used
        # before invoking the wrapper; this guards against stale gate samples.
        np.testing.assert_array_equal(force[raw_valid], trace['steady_gate_force_n'][raw_valid])
        expected_command = trace['steady_raw_command_pose'].copy()
        expected_command[:, :3] += correction
        np.testing.assert_array_equal(trace['command_pose'][raw_valid], expected_command[raw_valid].astype(np.float32))
        if gain == 0. and (active.any() or np.any(correction != 0.)):
            raise ValueError('Zero-gain controller is not an exact passthrough')
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
    duplicate = report['candidates']['constant_b']
    relative_difference = abs(duplicate['spatial_cv'] / baseline['spatial_cv'] - 1.)
    report['duplicate_mode6_relative_cv_difference'] = relative_difference
    report['duplicate_mode6_verified'] = bool(relative_difference < .005
        and baseline['eligible'] and duplicate['eligible'])
    if write:
        save(directory/('processing_summary.json' if rpm == SPINDLE_RPM else f'rpm_{rpm:g}_summary.json'), report)
    return report


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directories', nargs='+', type=Path)
    a = p.parse_args()
    for directory in a.directories:
        result = evaluate(directory, audit=True)
        print(json.dumps({'directory': str(directory), **result}, indent=2))
