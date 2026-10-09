"""Selection contract requested by the user: uniformity, not total volume."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import math


def assess_uniformity(candidate,baseline):
    def ratio(key):
        a,b=candidate[key],baseline[key]
        return a/b if math.isfinite(a) and math.isfinite(b) and b>0. else None
    cv_ratio=ratio('spatial_cv')
    gains={'spatial_cv_gain':None if cv_ratio is None else 1.-cv_ratio,
        'volume_ratio':ratio('roi_volume_over_k'),'time_ratio':ratio('time_s'),
        'raw_cv_ratio':ratio('rate_cv')}
    valid_depth=(math.isfinite(candidate['spatial_cv']) and candidate['spatial_cv']>=0.
        and math.isfinite(candidate['roi_volume_over_k']) and candidate['roi_volume_over_k']>0.)
    violations={'incomplete':float(not candidate['completed']),
        'fault':float(candidate.get('fault_fraction',0.)>0.),
        'invalid_depth':float(not valid_depth)}
    return {'score':(-candidate['spatial_cv'] if valid_depth else -1.e9)-10.*sum(violations.values()),
        'passed':not any(violations.values()),'gains':gains,'violations':violations,
        'total_removal_penalty':0.,'time_penalty':0.,'entry_force_penalty':0.}
