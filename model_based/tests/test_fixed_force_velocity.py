"""Contract/physics invariant checks for the new fixed-force scalar policy."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT as _REPO_ROOT
import ast
import copy
from pathlib import Path
import unittest

import numpy as np
import torch

from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.fixed_force_velocity import CONTRACT, FixedForceVelocityActor, check_contract, rotating_surface
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.spatial_uniformity import PlanarPrestonSurface


class FixedForceTests(unittest.TestCase):
    def test_actor_cannot_output_force_or_retrace(self):
        torch.manual_seed(2600)
        actor = FixedForceVelocityActor([3., 9., 5., 7.], [-2., -3., .2, -1., .1],
                                       pause_duration_s=1., nominal_duration_s=100.)
        x = torch.randn(4096, 16)*5.
        action = actor(x)
        self.assertEqual(tuple(action.shape), (4096, 1))
        self.assertTrue(torch.isfinite(action).all())
        self.assertTrue((action.abs() <= 1.).all())
        scripted = torch.jit.script(actor)
        torch.testing.assert_close(scripted(x), action)
        x = torch.zeros(1, 16); x[:, 8] = .085
        self.assertEqual(float(actor(x)[0, 0]), -1.)  # bounded pause at 8.5 s
        x[:, 8] = .095
        self.assertGreater(float(actor(x)[0, 0]), -1.)

    def test_legacy_force_contract_is_rejected(self):
        check_contract(CONTRACT)
        for key, value in [('action_names', ['feed','force','local_return']),
                           ('target_force_n', 10.), ('spindle_rpm', 0.),
                           ('force_bounds_n', [4.,30.])]:
            broken = copy.deepcopy(CONTRACT); broken[key] = value
            with self.assertRaises(ValueError): check_contract(broken)

    def test_actual_force_target_is_constant_independent_of_action(self):
        # Execute the real class methods with a tiny base-controller test double;
        # importing Isaac itself requires an AppLauncher and is covered by rollouts.
        source = ast.parse((_REPO_ROOT/'scripts/fixed_force_velocity_action.py').read_text())
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef))
        class Base:
            def _control_reference(self, env_id):
                return torch.zeros(6), torch.tensor([4.,-2.,99.]), 3
            def process_actions(self, actions): self.action = actions
        namespace = {'AdmittanceControlAction': Base, 'torch': torch, 'np': np}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), '<actual-action-class>', 'exec'), namespace)
        action = object.__new__(namespace['FixedForceVelocityAction'])
        action._num_envs_local = 1; action.target_force = np.array([7.])
        for a in np.linspace(-1., 1., 21):
            action.process_actions(torch.tensor([[a]]))
            _, f, _ = action._control_reference(0)
            torch.testing.assert_close(f, torch.tensor([0., 0., 20.]))
            self.assertEqual(action.target_force[0], 20.)
        for invalid in [torch.zeros((1, 3)), torch.tensor([[float('nan')]]), torch.tensor([[2.]])]:
            with self.assertRaises(ValueError): action.process_actions(invalid)

    def test_stationary_rotation_and_no_rotation_limits(self):
        path = np.array([[0.,0.,0.], [20.,0.,0.]])
        spin = rotating_surface(path, cell_mm=.5)
        stop = PlanarPrestonSurface(path, cell_size_mm=.5)
        spin.deposit([10.,0.,0.], [0.,0.,0.], 20., 1.)
        stop.deposit([10.,0.,0.], [0.,0.,0.], 20., 1.)
        self.assertEqual(stop.integrated_volume, 0.)
        expected = 20.*spin.omega*(2.*spin.radius/3.)
        self.assertAlmostEqual(spin.integrated_volume/expected, 1., delta=.003)
        spin2 = rotating_surface(path, cell_mm=.5, rpm=2000.)
        spin2.deposit([10.,0.,0.], [0.,0.,0.], 20., 1.)
        np.testing.assert_allclose(spin2.depth, 2.*spin.depth, rtol=1.e-12)
        zero = rotating_surface(path, cell_mm=.5, rpm=0.)
        zero.deposit([10.,0.,0.], [6.,0.,0.], 20., 1.)
        stop.deposit([10.,0.,0.], [6.,0.,0.], 20., 1.)
        np.testing.assert_allclose(zero.depth, stop.depth, rtol=1.e-12)


if __name__ == '__main__': unittest.main()
