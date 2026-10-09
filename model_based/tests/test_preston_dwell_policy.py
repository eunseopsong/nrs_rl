"""Control-direction, transfer and selection regressions for depth policy search."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT as _REPO_ROOT
from pathlib import Path
import sys
import tempfile
import unittest
import json
import importlib.util
import numpy as np
import torch

sys.path.insert(0, str(_REPO_ROOT / 'scripts/skrl'))
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.preston_dwell_actor import PrestonDwellActor, export_dwell_actor
from nrs_rl.tasks.manager_based.nrs_rl.mdp.agents.velocity_actor import TorchscriptVelocityAgent
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_preston_depth import assess_candidate, rank_candidate, next_population
from nrs_rl.tasks.manager_based.nrs_rl.utils.reintegrate_depth_candidates import validate_initial_population

spec = importlib.util.spec_from_file_location('depth_quality', _REPO_ROOT /
    'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/utils/velocity_evaluation.py')
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


class DwellPolicyTests(unittest.TestCase):
    profile = {'cursor_knots_mm': [0., 20., 40.], 'dwell_multipliers': [1., 1.2, 1.]}

    def observations(self, count=101):
        state = torch.zeros(count, 14)
        state[:, 1], state[:, 3], state[:, 6], state[:, 10] = 1., .5, .5, 1.
        return state

    def test_depth_force_direction_and_negative_speed_feedback(self):
        actor = PrestonDwellActor(self.profile, [.5, 1., .3, 0.])
        state = self.observations()
        state[:, 1] = torch.linspace(.8, 1.2, len(state))
        self.assertTrue(torch.all(actor(state).squeeze().diff() >= 0))
        state = self.observations()
        state[:, 3] = torch.linspace(.3, .7, len(state))
        self.assertTrue(torch.all(actor(state).squeeze().diff() <= 0))

    def test_prefix_keeps_physical_profile_location_and_zero_strength_is_constant(self):
        full = PrestonDwellActor(self.profile, [1., 0., 0., 0.])
        prefix = PrestonDwellActor(self.profile, [1., 0., 0., 0.], 20.)
        state = self.observations()
        state[:, 7] = torch.linspace(0, 1, len(state))
        half = state.clone(); half[:, 7] *= .5
        torch.testing.assert_close(prefix(state), full(half))
        zero = PrestonDwellActor(self.profile, [0., 0., 0., 0.])
        torch.testing.assert_close(zero(state), torch.zeros(len(state), 1))

    def test_export_reload_preserves_profile_and_contract(self):
        contract = {'policy_backend': 'torchscript_preston_depth', 'force_rate_compensation': False,
                    'nominal_speed_mm_s': 6., 'residual_speed_fraction': .5}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'actor.pt'
            actor = export_dwell_actor(self.profile, [.7, .5, .2, 0.], path, contract)
            worker = TorchscriptVelocityAgent(path, contract)
            states = self.observations()
            states[:, 7] = torch.linspace(0, 1, len(states))
            torch.testing.assert_close(actor(states), worker.act(states)[0])
            self.assertEqual(worker.normalizer_name, 'PhysicalScaling')

    def test_search_rejects_depth_gain_bought_with_missing_coverage_or_dose(self):
        def candidate(cv=.5, mean=1., zeros=.1):
            return dict(completed=True, fault_fraction=0., shield_fraction=0., processing_samples=1000,
                        rotation_depth=dict(spatial_depth_cv=cv, mean_depth_over_k_omega=mean,
                                            zero_depth_fraction=zeros))
        baseline = candidate()
        self.assertAlmostEqual(rank_candidate(candidate(.4), [baseline, baseline]), .2)
        self.assertAlmostEqual(rank_candidate(candidate(0.), [baseline]), 1.)
        self.assertEqual(rank_candidate(candidate(.4, .8), [baseline]), -10.)
        self.assertEqual(rank_candidate(candidate(.4, zeros=.11), [baseline]), -10.)
        self.assertEqual(rank_candidate(baseline, [baseline]), 0.)
        self.assertEqual(rank_candidate(candidate(.4), [{**baseline, 'fault_fraction': .01}]), -10.)
        self.assertEqual(rank_candidate(candidate(.4), [{**baseline, 'shield_fraction': .01}]), -10.)
        rejected = assess_candidate(candidate(.4, zeros=.11), [baseline])
        self.assertAlmostEqual(rejected['unconstrained_gain'], .2)
        self.assertEqual(rejected['score'], -10.)
        self.assertIn('increased zero-depth area', rejected['rejections'][0])

    def test_reuse_rejects_different_dynamics_seed_or_policy(self):
        candidates = {'constant_a': [0., 0., 0., 0.], 'constant_b': [0., 0., 0., 0.],
                      'profile_full': [1., 0., 0., 0.]}
        summary = dict(seed=42, dt_s=.008, physics_dt_s=.002, source_path_sha256='reference',
                       dwell_profile_sha256='profile', physics_tool_diameter_mm=30.,
                       target_normal_force_n=20., path_length_mm=40.,
                       candidates={name: {'parameters': params} for name, params in candidates.items()})
        validate_initial_population(summary, self.profile, 'profile', 'reference', candidates, 42)
        for key, changed in [('seed', 43), ('dt_s', .002), ('target_normal_force_n', 10.),
                             ('path_length_mm', 20.), ('dwell_profile_sha256', 'other')]:
            with self.assertRaises(ValueError):
                validate_initial_population({**summary, key: changed}, self.profile,
                                            'profile', 'reference', candidates, 42)
        with self.assertRaises(ValueError):
            validate_initial_population(summary, self.profile, 'profile', 'reference',
                                        {**candidates, 'profile_full': [.5, 0., 0., 0.]}, 42)

    def test_parameter_search_preserves_incumbent_and_bounds(self):
        incumbent = [.5, .7, .1, .01]
        population, mean, std = next_population([incumbent, incumbent], incumbent, np.random.default_rng(3))
        self.assertEqual(population['incumbent'], incumbent)
        self.assertEqual(population['constant_a'], [0., 0., 0., 0.])
        self.assertTrue(np.all(np.asarray(std) > 0))
        for parameters in population.values():
            PrestonDwellActor(self.profile, parameters)

    def test_depth_gate_uses_its_explicit_normalization_and_threshold(self):
        reference = dict(roi_cells=100, roi_area_mm2=400., spatial_depth_cv=.5,
                         mean_depth_over_k_omega=1., zero_depth_fraction=.1)
        improved = {**reference, 'spatial_depth_cv': .4}
        self.assertTrue(quality.compare_rotation_surfaces(improved, reference, .1)['passed'])
        self.assertFalse(quality.compare_rotation_surfaces(improved, reference, .3)['passed'])
        self.assertFalse(quality.compare_rotation_surfaces({**improved, 'mean_depth_over_k_omega': .8}, reference)['passed'])

    def test_depth_gate_cannot_compare_different_roi_or_wrong_units(self):
        reference = dict(roi_cells=100, roi_area_mm2=400., spatial_depth_cv=.5,
                         mean_depth_over_k_omega=1., zero_depth_fraction=.1)
        with self.assertRaises(ValueError):
            quality.compare_rotation_surfaces({**reference, 'roi_cells': 99}, reference)
        wrong = {key: value for key, value in reference.items() if key != 'mean_depth_over_k_omega'}
        wrong['mean_depth_over_k'] = 1.
        with self.assertRaises(KeyError):
            quality.compare_rotation_surfaces(wrong, reference)


if __name__ == '__main__':
    unittest.main()
