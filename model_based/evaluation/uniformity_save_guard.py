"""Accept only Mode 6 gains above the user's saved threshold, without regression."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import math


def assess_save_guard(candidate,saved,mode6,minimum_gain=.1114):
    eligible=all(m.get('eligible',False) for m in [candidate,saved,mode6])
    values=[m['spatial_cv'] for m in [candidate,saved,mode6]]
    finite=all(math.isfinite(v) and v>=0. for v in values) and values[-1]>0.
    gain=1.-values[0]/values[-1] if finite else None
    saved_gain=1.-values[1]/values[-1] if finite else None
    checks={'all_runs_valid':eligible and finite,
        'mode6_gain_at_least_11p14_percent':finite and gain>=minimum_gain,
        'better_than_preserved_on_same_seed':finite and gain>saved_gain+1.e-6}
    return {'passed':all(checks.values()),'checks':checks,
        'mode6_spatial_cv_gain':gain,'preserved_mode6_spatial_cv_gain':saved_gain,
        'comparison_baseline':'Mode 6','minimum_mode6_gain':minimum_gain}
