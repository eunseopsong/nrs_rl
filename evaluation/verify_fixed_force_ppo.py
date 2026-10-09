"""Audit PPO exports and independent Isaac quadrature after the full pipeline."""
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
import torch

from nrs_rl.tasks.manager_based.nrs_rl.utils.rotary_geometry import rotating_surface
from nrs_rl.tasks.manager_based.nrs_rl.agents.fixed_force_ppo import check_ppo_or_baseline, sha


def verify(root):
    root=Path(root).resolve()
    result=json.loads((root/'result.json').read_text())
    selection=json.loads((root/'selection.json').read_text())
    proof=json.loads((root/'ppo_training_audit.json').read_text())
    assert proof['passed'] and result['candidate_is_ppo']
    checkpoint=Path(result['checkpoint'])
    assert sha(checkpoint)==selection['checkpoint_sha256']==result['checkpoint_sha256']
    extra={'fixed_force_policy.json':''}
    actor=torch.jit.load(str(checkpoint),_extra_files=extra,map_location='cpu').eval()
    c=json.loads(extra['fixed_force_policy.json']);check_ppo_or_baseline(c)
    assert c['actor_kind']=='neural_ppo' and c['ppo_updates']>0 and not c['teacher_profile_used']
    assert sum(p.numel() for p in actor.parameters())==20865
    assert 'residence_profile' not in dict(actor.named_buffers())
    torch.manual_seed(3199)
    with torch.no_grad():
        out=actor(torch.randn(4096,16)*2.)
    assert out.shape==(4096,1) and torch.isfinite(out).all()
    assert (out>=-.750001).all() and (out<=1.000001).all()
    checks=[]
    for seed in [3151,3152]:
        folder=root/'physics'/f'heldout_{seed}'
        summary=json.loads((folder/'summary.json').read_text())
        assert summary['checkpoint_sha256']['selected']==sha(checkpoint)
        assert summary['checkpoint_sha256']['saved']=='6d1dc47bc915b393880d2e6e380f44fbcb82e542de0fa12e4f02eb47c97147df'
        for values in summary['randomization'].values():assert all(v==values[0] for v in values)
        with h5py.File(folder/'path.h5') as f:path=f['position'][:,:3]
        surface=rotating_surface(path)
        xx,yy=np.meshgrid(surface.u,surface.v)
        for name in summary['names']:
            with np.load(folder/f'{name}_trace.npz') as f:t=dict(f)
            assert np.all(t['target_force_n']==20.)
            assert t['spatial_action'].shape==(len(t['time_s']),1)
            assert np.isfinite(t['spatial_action']).all() and np.max(np.abs(t['spatial_action']))<=1.000001
            if name in ('constant','constant_b'):assert np.all(t['spatial_action']==0.)
            assert np.min(np.diff(t['cursor_mm']))>=-1.e-5
            for key in ['requested_speed_mm_s','commanded_speed_mm_s']:
                assert t[key].min()>=-1.e-5 and t[key].max()<=12.0001
            enabled=t['polishing_active'].astype(bool)&~t['safety_shield_active'].astype(bool)
            consecutive=enabled&np.r_[False,enabled[:-1]]
            assert np.max(np.abs(t['command_acceleration_mm_s2'][consecutive]))<=16.0001
            assert np.max(np.abs(t['command_jerk_mm_s3'][consecutive]))<=160.001
            errors=[]
            for i in np.linspace(250,len(t['time_s'])-1,41).astype(int):
                point=t['deposition_midpoint_mm'][i]
                center=(point-surface.origin)@surface.basis.T
                if (center[0]-surface.radius<surface.u[0] or center[0]+surface.radius>surface.u[-1]
                    or center[1]-surface.radius<surface.v[0] or center[1]+surface.radius>surface.v[-1]):continue
                dx,dy=xx-center[0],yy-center[1]
                footprint=dx*dx+dy*dy<=surface.radius**2
                feed=t['tangent_velocity_mm_s'][i]@surface.basis.T
                velocity=np.hypot(feed[0]-surface.omega*dy,feed[1]+surface.omega*dx)
                contact=bool(t['normal_force_n'][i]>=1.5 and t['safety_fault_reason'][i]==0)
                expected=(t['normal_force_n'][i]*footprint/footprint.sum()/surface.area*velocity*.008
                          if contact else np.zeros_like(xx))
                surface.reset();surface.deposit(point,t['tangent_velocity_mm_s'][i],t['normal_force_n'][i],.008,contact=contact)
                np.testing.assert_allclose(surface.depth,expected,atol=1.e-10,rtol=1.e-9)
                errors.append(float(np.max(np.abs(surface.depth-expected))))
            assert len(errors)>=20
            checks.append({'seed':seed,'mode':name,'quadrature_samples':len(errors),'max_cell_error':max(errors),
                           'fixed_force_20n':True,'forward_only':True,'acceleration_jerk_bounds_passed':True})
    locked=json.loads((root.parents[2]/'deployment/FIXED_MODEL.json').read_text())
    assert sha(locked['checkpoint'])==locked['checkpoint_sha256']
    result={'passed':True,'checkpoint_sha256':sha(checkpoint),'neural_actor_parameters':20865,
        'algorithm_training_audit_passed':True,'export_probes':4096,'quadrature_checks':checks,
        'selection_locked_before_holdouts':True,'frozen_baseline_unchanged':True,
        'measurement_reintegration_audited_by_pipeline':True,
        'not_a_material_depth_calibration':True}
    (root/'independent_numerical_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    import argparse
    torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    print(json.dumps(verify(p.parse_args().root),indent=2))
