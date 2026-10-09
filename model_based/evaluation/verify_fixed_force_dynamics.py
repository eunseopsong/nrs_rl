"""Independent fixed-force/action audit and direct full-grid removal quadrature."""
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

from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.fixed_force_velocity import rotating_surface
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.fixed_force_refinement import check_contract
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_fixed_force_dynamics import HOLDOUTS, SAVED_SHA, check_preserved
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_fixed_force_velocity import save, sha


def verify(root, result):
    check_preserved()
    checkpoint = Path(result['checkpoint'])
    assert sha(checkpoint) == result['checkpoint_sha256']
    extra = {'fixed_force_policy.json': ''}
    actor = torch.jit.load(str(checkpoint), _extra_files=extra).eval()
    minimum, maximum = check_contract(json.loads(extra['fixed_force_policy.json']))
    rng = np.random.default_rng(280109)
    obs = torch.tensor(rng.uniform(-5., 5., (4096, 16)), dtype=torch.float32)
    with torch.inference_mode():
        out = actor(obs)
    assert out.shape == (4096, 1) and torch.isfinite(out).all() and torch.max(torch.abs(out)) <= 1.
    speed = maximum/2. * (1. + out)
    assert speed.min() >= minimum-1.e-5 and speed.max() <= maximum+1.e-5
    selected = json.loads((root/'candidate/selection.json').read_text())
    assert selected['checkpoint_sha256'] == result['checkpoint_sha256']
    assert selected['selected_before_holdouts_at'] == result['selected_before_holdouts_at']
    checks = []
    for seed in HOLDOUTS:
        folder = root/'physics'/f'heldout_{seed}'
        summary = json.loads((folder/'summary.json').read_text())
        assert summary['checkpoint_sha256']['saved'] == SAVED_SHA
        assert summary['checkpoint_sha256']['selected'] == result['checkpoint_sha256']
        # All environments share the same randomized force and delay parameters.
        for values in summary['randomization'].values():
            assert np.all(np.asarray(values) == values[0])
        with h5py.File(folder/'path.h5') as f:
            path = f['position'][:, :3]
        surface = rotating_surface(path)
        xx, yy = np.meshgrid(surface.u, surface.v)
        for name in ['constant', 'constant_b', 'saved', 'selected']:
            _, cap = check_contract(summary['policy_contracts'][name])
            with np.load(folder/f'{name}_trace.npz') as f:
                trace = dict(f)
            assert np.all(trace['target_force_n'] == 20.)
            assert trace['spatial_action'].shape == (len(trace['time_s']), 1)
            assert np.isfinite(trace['spatial_action']).all()
            assert np.max(np.abs(trace['spatial_action'])) <= 1.000001
            for key in ['requested_speed_mm_s', 'commanded_speed_mm_s']:
                assert np.min(trace[key]) >= -1.e-6 and np.max(trace[key]) <= cap+1.e-5
            # Safety resets may jump acceleration to zero. Continuous enabled
            # steps must respect the common acceleration and jerk limits.
            enabled = trace['polishing_active'].astype(bool) & ~trace['safety_shield_active'].astype(bool)
            continuous = enabled & np.r_[False, enabled[:-1]]
            assert np.max(np.abs(trace['command_acceleration_mm_s2'][continuous])) <= 16.0001
            assert np.max(np.abs(trace['command_jerk_mm_s3'][continuous])) <= 160.001
            errors = []
            for i in np.linspace(250, len(trace['time_s'])-1, 61).astype(int):
                point = trace['deposition_midpoint_mm'][i]
                center = (point-surface.origin) @ surface.basis.T
                if (center[0]-surface.radius < surface.u[0] or center[0]+surface.radius > surface.u[-1]
                        or center[1]-surface.radius < surface.v[0] or center[1]+surface.radius > surface.v[-1]):
                    continue
                dx, dy = xx-center[0], yy-center[1]
                footprint = (dx*dx+dy*dy) <= surface.radius**2
                feed = trace['tangent_velocity_mm_s'][i] @ surface.basis.T
                relative = np.hypot(feed[0]-surface.omega*dy, feed[1]+surface.omega*dx)
                contact = bool(trace['normal_force_n'][i] >= 1.5 and trace['safety_fault_reason'][i] == 0)
                expected = (trace['normal_force_n'][i]*footprint/footprint.sum()/surface.area
                            * relative*.008) if contact else np.zeros_like(xx)
                surface.reset()
                surface.deposit(point, trace['tangent_velocity_mm_s'][i], trace['normal_force_n'][i],
                                .008, contact=contact)
                np.testing.assert_allclose(surface.depth, expected, atol=1.e-10, rtol=1.e-9)
                errors.append(float(np.max(np.abs(surface.depth-expected))))
            assert len(errors) >= 30
            checks.append({'seed': seed, 'mode': name, 'samples': len(errors),
                           'max_cell_error': max(errors), 'passed': True})
    audit = {'passed': True, 'checkpoint_sha256': result['checkpoint_sha256'],
             'scalar_outputs_checked': 4096, 'quadrature_checks': checks,
             'fixed_target_force_20n_all_heldout_ticks': True,
             'feed_acceleration_jerk_contract_verified': True,
             'matched_randomization_verified': True, 'saved_checkpoint_preserved': True,
             'selected_before_holdouts_and_unchanged': True,
             'numerical_audit_is_separate_from_performance_acceptance': True,
             'not_a_physical_depth_calibration': True}
    save(root/'independent_numerical_audit.json', audit)
    return audit
