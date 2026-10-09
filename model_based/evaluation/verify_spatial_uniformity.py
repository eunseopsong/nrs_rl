#!/usr/bin/env python3
"""Independently audit exported physics metrics, raw traces and checkpoint."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def main(root):
    state = json.loads((root/'result.json').read_text())
    checkpoint = root/'candidate/checkpoints/actor.pt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == state['checkpoint_sha256']
    extra = {'spatial_policy.json': ''}
    actor = torch.jit.load(str(checkpoint), _extra_files=extra).eval()
    contract = json.loads(extra['spatial_policy.json'])
    assert contract == json.loads((root/'candidate/spatial_policy.json').read_text())
    assert contract == state['runtime_contract']
    assert contract['rotation_contribution'] is False and contract['spindle_rpm'] == 0
    assert contract['second_complete_pass'] is False
    assert len(contract['observation_names']) == 16 and len(contract['action_names']) == 3
    with torch.no_grad():
        action = actor(torch.zeros(64, 16))
    assert action.shape == (64, 3) and torch.isfinite(action).all() and (action.abs() <= 1).all()
    checks = []
    for seed in state['heldout_seeds']:
        directory = root/f'held_out_{seed}'
        summary = json.loads((directory/'summary.json').read_text())
        assert summary['checkpoint_sha256'][state['selected']] == state['checkpoint_sha256']
        reference_roi = None
        for name, m in summary['candidates'].items():
            with np.load(directory/f'{name}_surface.npz') as surface:
                roi = surface['roi'];values = surface['depth_over_k'][roi]
                if reference_roi is None:
                    reference_roi = roi.copy()
                assert np.array_equal(roi, reference_roi)
                assert np.isclose(values.std()/values.mean(), m['spatial_cv'], rtol=1e-10)
                assert np.isclose(values.sum()*surface['cell_area_mm2'], m['roi_volume_over_k'], rtol=1e-10)
            with np.load(directory/f'{name}_trace.npz') as t:
                force, speed = t['normal_force_n'], t['measured_speed_mm_s']
                rate = np.where((force >= 1.5) & (t['safety_fault_reason'] == 0), force*speed, 0.)
                assert np.allclose(t['raw_mrr_n_mm_s'], rate)
                assert np.isclose(rate.std()/rate.mean(), m['rate_cv'], rtol=1e-10)
                assert np.isclose(len(rate)*.008, m['time_s'], rtol=1e-10)
                assert np.isclose(force.mean(), m['force_mean_n'], rtol=1e-10)
                assert np.isclose(force.max(), m['force_peak_n'], rtol=1e-10)
                assert np.isclose(speed.sum()*.008, m['distance_mm'], rtol=1e-10)
                assert np.max(np.abs(np.diff(t['target_force_n'])))/.008 <= 5.001
                assert t['target_force_n'].min() >= 14.-1e-6 and t['target_force_n'].max() <= 26.+1e-6
                assert t['commanded_speed_mm_s'].min() >= -3.-1e-5
                assert t['commanded_speed_mm_s'].max() <= 9.+1e-5
                assert m['reverse_mm'] <= .05*summary['path_length_mm']+1e-3
                assert np.allclose(np.diff(t['time_s']), .008, rtol=1e-8, atol=1e-8)
                first_vector_error = abs(float(speed[0])-float(np.linalg.norm(t['tangent_velocity_mm_s'][0])))
                assert first_vector_error < 1e-4
                # Tiny differences can arise when motion is purely normal to
                # the reference plane but the measured TCP normal is tilted.
                contact = (force >= 1.5) & (t['safety_fault_reason'] == 0)
                plane_volume = float((force*np.linalg.norm(t['tangent_velocity_mm_s'], axis=1)*contact).sum()*.008)
                assert np.isclose(plane_volume, m['surface']['tcp_full_footprint_volume_over_k'], rtol=1e-6)
            checks.append({'seed': seed, 'policy': name, 'passed': True,
                           'samples': len(rate), 'first_vector_error': first_vector_error})
    # Evaluation is reproducible from the exact captured source versions.
    source_manifest = json.loads((root/'source_manifest.json').read_text())
    for relative, expected in source_manifest.items():
        assert hashlib.sha256((root/'source_snapshot'/relative).read_bytes()).hexdigest() == expected
    result = {'passed': True, 'metric_and_contract_audit': checks,
              'checkpoint_sha256': state['checkpoint_sha256'],
              'does_not_claim_physical_material_validation': True}
    (root/'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'passed': True, 'audited_runs': len(checks), 'checkpoint': str(checkpoint)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
