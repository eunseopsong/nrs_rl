#!/usr/bin/env python3
"""Entry-separated assessment requested during the spatial-policy study.

The first two active seconds are reported separately. The primary spatial
ROI excludes the fixed entry footprint swept over the first 18 reference mm
(2 s * maximum 9 mm/s). No candidate-dependent ROI or moving-sample filter.
All subsequent stationary/reverse/fault samples remain in the evaluation.
"""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from nrs_rl.tasks.model_based.policies.spatial_uniformity import PlanarPrestonSurface, assess

ENTRY_SECONDS = 2.
ENTRY_REFERENCE_MM = 18.


def evaluate(directory):
    original = json.loads((directory/'summary.json').read_text())
    with h5py.File(directory/'path.h5') as h:
        path = h['position'][:, :3]
    surface = PlanarPrestonSurface(path, cell_size_mm=.5)
    arc = np.r_[0., np.linalg.norm(np.diff(path, axis=0), axis=1).cumsum()]
    mask = surface.roi.copy()
    for position in path[arc <= ENTRY_REFERENCE_MM]:
        center = (position-surface.origin) @ surface.basis.T
        rows, cols, _, _, weights, _ = surface._footprint(center)
        mask[rows[weights > 0], cols[weights > 0]] = False
    result = {'seed': original['seed'], 'entry_seconds': ENTRY_SECONDS,
        'entry_reference_mm': ENTRY_REFERENCE_MM,
        'roi_definition': 'Fixed full reference ROI excluding the footprint swept by the first 18 mm; equal cells, zeros retained',
        'full_roi_cells': int(surface.roi.sum()), 'processing_roi_cells': int(mask.sum()),
        'entry_force_is_a_selection_gate': False,
        'steady_zero_velocity_samples_retained': True, 'candidates': {}}
    for name, previous in original['candidates'].items():
        with np.load(directory/f'{name}_surface.npz') as s:
            assert np.array_equal(s['roi'], surface.roi)
            depth = s['depth_over_k'].copy()
        with np.load(directory/f'{name}_trace.npz') as t:
            entry = t['time_s'] < ENTRY_SECONDS
            keep = ~entry
            if not keep.any():
                raise ValueError('No processing samples after fixed entry window')
            surface.reset()
            for i in np.flatnonzero(entry):
                surface.deposit(t['deposition_midpoint_mm'][i], t['tangent_velocity_mm_s'][i],
                    float(t['normal_force_n'][i]), .008,
                    contact=bool(t['normal_force_n'][i] >= 1.5 and t['safety_fault_reason'][i] == 0))
            remaining = depth-surface.depth
            if remaining.min() < -1e-9:
                raise ValueError('Entry subtraction exceeds the saved full depth')
            remaining = np.maximum(remaining, 0.)
            values = remaining[mask]
            force, speed = t['normal_force_n'][keep], t['measured_speed_mm_s'][keep]
            rate = t['raw_mrr_n_mm_s'][keep]
            metrics = {
                'spatial_cv': float(values.std()/values.mean()),
                'roi_volume_over_k': float(values.sum()*surface.area),
                'time_s': int(keep.sum())*.008,
                'force_mean_n': float(force.mean()), 'force_peak_n': float(force.max()),
                'force_tracking_rmse_n': float(np.sqrt(np.mean((force-t['target_force_n'][keep])**2))),
                'rate_cv': float(rate.std()/rate.mean()), 'distance_mm': float(speed.sum()*.008),
                'zero_fraction': float(np.mean(values == 0)), 'completed': previous['completed'],
                'fault_fraction': float(np.mean(t['safety_fault_reason'][keep] != 0)),
                'return_events': previous['return_events'], 'reverse_mm': previous['reverse_mm'],
                'entry_force_peak_n': float(t['normal_force_n'][entry].max()),
                'entry_force_tracking_rmse_n': float(np.sqrt(np.mean((t['normal_force_n'][entry]-t['target_force_n'][entry])**2))),
            }
            result['candidates'][name] = metrics
        np.savez_compressed(directory/f'{name}_processing_surface.npz', depth_over_k=remaining,
                            roi=mask, u_mm=surface.u, v_mm=surface.v, cell_area_mm2=surface.area)
    for name, m in result['candidates'].items():
        m['comparison'] = assess(m, result['candidates']['constant'])
        if 'matched_constant' in result['candidates']:
            m['matched_tracking_comparison'] = assess(m, result['candidates']['matched_constant'])
    (directory/'processing_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    result = evaluate(parser.parse_args().directory)
    print(json.dumps(result, indent=2))
