"""A lower transient cost alone must not qualify a degraded polishing candidate."""
import copy
import unittest

from nrs_rl.tasks.manager_based.nrs_rl.mdp.evaluation.assess_spike_dose_ppo import compare


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.parent = {'contact_cv': .45, 'processing_cv': .45,
            'contact_volume_over_k': 100., 'processing_volume_over_k': 95.,
            'time_s': 130., 'existing_quality_eligible': True,
            'fault_fraction': 0., 'shield_fraction': 0.,
            'spike': {'reference_established_active_s': 1., 'event_abs_error_seconds': 3.}}
        self.mode6 = {**self.parent, 'contact_cv': .5}
        self.candidate = copy.deepcopy(self.parent)
        self.candidate['spike']['event_abs_error_seconds'] = 2.

    def test_longer_processing_is_allowed_with_preserved_quality_and_reduced_error(self):
        self.candidate['time_s'] = 160.
        result = compare(self.candidate, self.parent, self.mode6)
        self.assertTrue(result['accepted'])
        self.assertGreater(result['time_change_percent'], 0.)

    def test_volume_loss_cv_loss_faults_and_unarmed_detector_cannot_win(self):
        for field, value in [('contact_cv', .46), ('processing_cv', .46),
                             ('contact_volume_over_k', 85.), ('processing_volume_over_k', 70.),
                             ('fault_fraction', .01), ('shield_fraction', .01),
                             ('existing_quality_eligible', False)]:
            candidate = copy.deepcopy(self.candidate)
            candidate[field] = value
            self.assertFalse(compare(candidate, self.parent, self.mode6)['accepted'], field)
        self.candidate['spike']['reference_established_active_s'] = None
        self.assertFalse(compare(self.candidate, self.parent, self.mode6)['accepted'])

    def test_tiny_error_change_or_no_baseline_event_does_not_prove_gain(self):
        self.candidate['spike']['event_abs_error_seconds'] = 2.99
        self.assertFalse(compare(self.candidate, self.parent, self.mode6)['accepted'])
        self.parent['spike']['event_abs_error_seconds'] = 0.
        result = compare(self.candidate, self.parent, self.mode6)
        self.assertIsNone(result['spike_abs_error_improvement_percent'])
        self.assertFalse(result['accepted'])


if __name__ == '__main__':
    unittest.main()
