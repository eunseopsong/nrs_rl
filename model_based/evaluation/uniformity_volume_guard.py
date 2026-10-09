"""Constrain both processing and whole-path removal against paired Mode 6."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import math


def assess_volume_guard(candidate, mode6, saved=None, minimum_gain=0., floor=.5):
    def finite_positive(value):
        return isinstance(value, (int,float)) and math.isfinite(value) and value > 0.
    keys = ['spatial_cv','roi_volume_over_k','full_roi_volume_over_k']
    valid = (candidate.get('eligible',False) and mode6.get('eligible',False)
             and all(finite_positive(m.get(key)) for m in [candidate,mode6] for key in keys))
    gain = 1.-candidate['spatial_cv']/mode6['spatial_cv'] if valid else None
    volume = candidate['roi_volume_over_k']/mode6['roi_volume_over_k'] if valid else None
    full_volume = candidate['full_roi_volume_over_k']/mode6['full_roi_volume_over_k'] if valid else None
    checks = {'valid_completed_runs':bool(valid),
              'processing_removal_at_least_50_percent':bool(valid and volume >= floor),
              'full_removal_at_least_50_percent':bool(valid and full_volume >= floor),
              'mode6_cv_gain_above_minimum':bool(valid and gain >= minimum_gain)}
    saved_gain = None
    if saved is not None:
        saved_valid = (saved.get('eligible',False) and all(finite_positive(saved.get(k)) for k in keys)
                       and saved['roi_volume_over_k'] >= floor*mode6['roi_volume_over_k']
                       and saved['full_roi_volume_over_k'] >= floor*mode6['full_roi_volume_over_k'])
        saved_gain = 1.-saved['spatial_cv']/mode6['spatial_cv'] if saved_valid else None
        checks['saved_comparator_valid_and_feasible'] = bool(saved_valid)
        checks['improves_saved_same_seed'] = bool(valid and saved_valid and gain > saved_gain+1.e-6)
    return {'passed':all(checks.values()), 'checks':checks, 'comparison_baseline':'Mode 6',
            'mode6_spatial_cv_gain':gain, 'mode6_volume_ratio':volume, 'mode6_full_volume_ratio':full_volume,
            'saved_mode6_spatial_cv_gain':saved_gain, 'minimum_mode6_gain':minimum_gain, 'volume_floor':floor}
