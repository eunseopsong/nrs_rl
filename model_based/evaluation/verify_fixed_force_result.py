#!/usr/bin/env python3
"""Independent pointwise pressure/relative-velocity quadrature audit.

Uses a full grid and direct formulas, without calling the estimator's deposit
or footprint method for the expected result. Also checks scalar checkpoint
outputs and exact target-force/volume constraints in every held-out trace.
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
import torch

from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.fixed_force_velocity import CONTRACT, check_contract, rotating_surface
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_fixed_force_velocity import save, sha


def verify(root):
    result=json.loads((root/'result.json').read_text())
    checkpoint=Path(result['checkpoint'])
    assert sha(checkpoint)==result['checkpoint_sha256']
    extra={'fixed_force_policy.json':''}
    actor=torch.jit.load(str(checkpoint),_extra_files=extra).eval()
    check_contract(json.loads(extra['fixed_force_policy.json']))
    rng=np.random.default_rng(20261009)
    obs=torch.tensor(rng.uniform(-5.,5.,(4096,16)),dtype=torch.float32)
    with torch.inference_mode():out=actor(obs)
    assert out.shape==(4096,1) and torch.isfinite(out).all() and torch.max(torch.abs(out))<=1.
    checks=[]
    for seed in [2611,2612]:
        folder=root/'physics'/f'heldout_{seed}'
        with h5py.File(folder/'path.h5') as f:path=f['position'][:,:3]
        surface=rotating_surface(path)
        xx,yy=np.meshgrid(surface.u,surface.v)
        for name in ['constant','constant_b','selected']:
            with np.load(folder/f'{name}_trace.npz') as f:trace=dict(f)
            assert np.all(trace['target_force_n']==20.)
            assert trace['spatial_action'].shape==(len(trace['time_s']),1)
            assert not np.any(trace['safety_fault_reason'])
            errors=[]
            # Interior samples allow an independent full-grid normalization;
            # clipped edge footprints are audited in the full trace integrator.
            for i in np.linspace(250,len(trace['time_s'])-1,61).astype(int):
                point=trace['deposition_midpoint_mm'][i]
                center=(point-surface.origin) @ surface.basis.T
                if (center[0]-surface.radius<surface.u[0] or center[0]+surface.radius>surface.u[-1]
                        or center[1]-surface.radius<surface.v[0] or center[1]+surface.radius>surface.v[-1]):
                    continue
                dx=xx-center[0];dy=yy-center[1]
                footprint=(dx*dx+dy*dy)<=surface.radius**2
                feed=trace['tangent_velocity_mm_s'][i] @ surface.basis.T
                relative=np.sqrt((feed[0]-surface.omega*dy)**2+(feed[1]+surface.omega*dx)**2)
                contact=trace['normal_force_n'][i]>=1.5
                expected=(trace['normal_force_n'][i]*footprint/footprint.sum()/surface.area
                          *relative*.008) if contact else np.zeros_like(xx)
                surface.reset()
                surface.deposit(point,trace['tangent_velocity_mm_s'][i],trace['normal_force_n'][i],.008,contact=contact)
                np.testing.assert_allclose(surface.depth,expected,atol=1.e-10,rtol=1.e-9)
                errors.append(float(np.max(np.abs(surface.depth-expected))))
            assert len(errors)>=30
            checks.append({'seed':seed,'mode':name,'samples':len(errors),'max_cell_error':max(errors),'passed':True})
        held=result['heldouts'][str(seed)]
        candidate=held['candidates']['selected']
        assert all(v>=.5 for v in candidate['mode6_volume_ratios'].values())
        assert candidate['eligible']
        # No revised fitting/ranking after the independent holdout decision.
        selected=json.loads((root/'candidate/selection.json').read_text())
        assert selected['checkpoint_sha256']==result['checkpoint_sha256']
        assert selected['selected_before_holdouts_at']==result['selected_before_holdouts_at']
    audit={'passed':True,'checkpoint_sha256':result['checkpoint_sha256'],
           'scalar_output_checked':4096,'quadrature_checks':checks,
           'target_force_exactly_20n_all_heldout_ticks':True,
           'volume_floor_passed_all_heldouts':True,
           'selected_checkpoint_unchanged_after_holdouts':True,
           'not_a_physical_depth_calibration':True}
    save(root/'independent_numerical_audit.json',audit)
    print(json.dumps(audit,indent=2))
    return audit


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    verify(p.parse_args().root.resolve())
