"""User selection preserves the original scientific result and process constraints."""
import copy
import unittest

from nrs_rl.tasks.manager_based.nrs_rl.mdp.evaluation.save_ppo_checkpoint import validate_selection


class SaveSelectionTests(unittest.TestCase):
    def setUp(self):
        self.report = {'algorithm': 'PPO', 'numerical_audit_passed': True, 'accepted': False,
                       'per_seed': {str(seed): {'selected': {
                           'eligible': True, 'mode6_volume_ratios': {
                               'processing': 1.03, 'full': 1.04, 'whole': 1.04}}}
                           for seed in [3351, 3352]}}

    def test_small_gain_requires_explicit_user_selection(self):
        with self.assertRaisesRegex(ValueError, 'explicit user selection'):
            validate_selection(self.report)
        original = copy.deepcopy(self.report)
        validate_selection(self.report, user_selected=True)
        self.assertEqual(self.report, original)

    def test_selection_cannot_skip_numerical_audit(self):
        self.report['numerical_audit_passed'] = False
        with self.assertRaises(ValueError):
            validate_selection(self.report, user_selected=True)

    def test_selection_cannot_skip_volume_floor(self):
        self.report['per_seed']['3351']['selected']['mode6_volume_ratios']['full'] = .49
        with self.assertRaisesRegex(ValueError, 'removal floor'):
            validate_selection(self.report, user_selected=True)

    def test_selection_cannot_skip_eligibility(self):
        self.report['per_seed']['3352']['selected']['eligible'] = False
        with self.assertRaisesRegex(ValueError, 'eligibility'):
            validate_selection(self.report, user_selected=True)


if __name__ == '__main__':
    unittest.main()
