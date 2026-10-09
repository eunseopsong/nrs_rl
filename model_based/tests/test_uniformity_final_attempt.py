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
from unittest.mock import patch

import numpy as np

import nrs_rl.tasks.manager_based.nrs_rl.model_based.evaluation.finish_uniformity_final_attempt as finish
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_uniformity_final_attempt import adjusted_clock, bounded_residual
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_uniformity_priority import sha


class FinalAttemptTests(unittest.TestCase):
    def test_small_gain_or_any_volume_violation_preserves_save(self):
        baseline = dict(eligible=True, spatial_cv=.4, roi_volume_over_k=100., full_roi_volume_over_k=110.)
        saved = dict(baseline, spatial_cv=.4 * (1. - .1345), roi_volume_over_k=53., full_roi_volume_over_k=60.)
        candidate = dict(saved, spatial_cv=.4 * (1. - .146), roi_volume_over_k=52.)
        self.assertTrue(finish.final_guard(candidate, baseline, saved)['passed'])
        for change in [dict(spatial_cv=.4 * (1. - .144)), dict(roi_volume_over_k=49.99),
                       dict(full_roi_volume_over_k=54.99), dict(eligible=False)]:
            self.assertFalse(finish.final_guard(dict(candidate, **change), baseline, saved)['passed'])
        nearly_tied_save = dict(saved, spatial_cv=.4 * (1. - .143))
        self.assertFalse(finish.final_guard(candidate, baseline, nearly_tied_save)['passed'])

    def test_clock_preserves_path_order_and_does_not_create_removal_distance(self):
        t = np.arange(100) * .008
        data = dict(time=t, cursor=np.linspace(0., 10., 100), target=np.full(100, 10.))
        old = np.array([.67, 0., 0., 0.])
        new = np.zeros(4)
        original_cursor = data['cursor'].copy()
        np.testing.assert_allclose(adjusted_clock(data, [10., 10.], 10., old, old), t)
        # At this force the old command is 9 mm/s and the new one is 6 mm/s.
        warped = adjusted_clock(data, [10., 10.], 10., old, new)
        np.testing.assert_allclose(warped, t * 1.5)
        self.assertTrue(np.all(np.diff(warped) > 0.))
        np.testing.assert_array_equal(original_cursor, data['cursor'])

    def test_residual_correction_is_bounded(self):
        data = dict(time=np.arange(1000) * .008, target=np.full(1000, 10.), force=np.full(1000, 20.))
        response = dict(delay_s=.2, tau_s=.5, gain=1., bias_n=0.)
        np.testing.assert_allclose(bounded_residual(data, response), 1.5)

    def test_rejected_attempt_pins_original_bytes_without_overwriting_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            saved = root / 'saves/previous'
            actor = saved / 'candidate/checkpoints/actor.pt'
            actor.parent.mkdir(parents=True)
            actor.write_bytes(b'preserved model bytes')
            digest = sha(actor)
            (saved / 'save_manifest.json').write_text(json.dumps({'candidate/checkpoints/actor.pt': digest}))
            (saved.parent / 'CURRENT.json').write_text(json.dumps({'sha256': digest}))
            attempt = root / 'attempt'
            attempt.mkdir()
            (attempt / 'verification.json').write_text(json.dumps({'passed': True}))
            result = dict(phase='complete', accepted=False)
            with patch.object(finish, 'SAVED', saved), patch.object(finish, 'SAVED_SHA', digest), \
                 patch.object(finish, 'check_preserved'):
                decision = finish.fix_model(attempt, result)
            self.assertEqual(sha(actor), digest)
            self.assertEqual(decision['checkpoint'], str(actor))
            self.assertEqual(json.loads((saved.parent / 'CURRENT.json').read_text())['status'], 'fixed')
            self.assertFalse(decision['new_candidate_accepted'])
            self.assertFalse(decision['deployed'])


if __name__ == '__main__':
    unittest.main()
