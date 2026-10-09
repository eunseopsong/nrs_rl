"""The user-requested entry exclusion must not discard later stalls or zeros."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from nrs_rl.tasks.manager_based.nrs_rl.utils.assess_processing_uniformity import evaluate
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.spatial_uniformity import PlanarPrestonSurface


class ProcessingAssessmentTests(unittest.TestCase):
    def test_entry_force_does_not_change_ranking_and_later_stalls_remain(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = np.array([[0., 0., 0.], [100., 0., 0.]])
            with h5py.File(directory/'path.h5', 'w') as h:
                h['position'] = path
            time_s = np.arange(260)*.008
            entry = time_s < 2.
            points = np.zeros((len(time_s), 3))
            points[entry, 0] = np.linspace(0., 12., entry.sum())
            points[~entry, 0] = np.linspace(40., 41., (~entry).sum())
            speed = np.full(len(time_s), 6.)
            speed[-1] = 0.  # A stall AFTER the exclusion window.
            velocity = np.c_[speed, np.zeros((len(speed), 2))]
            state = {'seed': 1, 'candidates': {}}
            for name, initial_force in [('constant', 20.), ('large_entry_peak', 90.)]:
                force = np.where(entry, initial_force, 20.)
                surface = PlanarPrestonSurface(path, cell_size_mm=.5)
                for position, v, f in zip(points, velocity, force):
                    surface.deposit(position, v, float(f), .008)
                np.savez(directory/f'{name}_surface.npz', roi=surface.roi, depth_over_k=surface.depth)
                np.savez(directory/f'{name}_trace.npz', time_s=time_s,
                    normal_force_n=force, target_force_n=np.full(len(time_s), 20.),
                    measured_speed_mm_s=speed, raw_mrr_n_mm_s=force*speed,
                    safety_fault_reason=np.zeros(len(time_s)),
                    tangent_velocity_mm_s=velocity, deposition_midpoint_mm=points)
                state['candidates'][name] = {'completed': True, 'return_events': 0, 'reverse_mm': 0.}
            (directory/'summary.json').write_text(json.dumps(state))
            result = evaluate(directory)
            base = result['candidates']['constant'];high = result['candidates']['large_entry_peak']
            self.assertEqual(high['entry_force_peak_n'], 90.)
            self.assertEqual(high['force_peak_n'], 20.)
            self.assertTrue(high['comparison']['passed'])
            self.assertAlmostEqual(high['spatial_cv'], base['spatial_cv'])
            self.assertEqual(high['time_s'], .08)
            self.assertAlmostEqual(high['rate_cv'], 1./3.)
            self.assertGreater(high['zero_fraction'], 0.)
            with np.load(directory/'constant_processing_surface.npz') as a:
                with np.load(directory/'large_entry_peak_processing_surface.npz') as b:
                    np.testing.assert_array_equal(a['roi'], b['roi'])
                    np.testing.assert_allclose(a['depth_over_k'], b['depth_over_k'], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
