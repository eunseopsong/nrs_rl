"""Velocity authority, old-policy parity and fixed-force rejection checks."""
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
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.fixed_force_refinement import ResidenceActor, ForwardSpeedLimiter, check_contract, contract_for
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.fixed_force_velocity import CONTRACT as PREVIOUS_CONTRACT
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.spatial_uniformity import SignedSpeedLimiter
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_fixed_force_refinement import check_preserved, SAVED


class RefinementTests(unittest.TestCase):
    def test_scalar_inverse_speed_profile_and_bounds(self):
        actor=ResidenceActor([.5,4.,1.],minimum_speed=1.5,maximum_speed=12.)
        obs=torch.zeros(1,16);obs[:,0]=.25
        feed=(actor(obs)[0,0]+1.)*6.
        self.assertAlmostEqual(float(feed),6./2.25,places=6)
        obs=torch.randn(4096,16)*5.
        eager=actor(obs);scripted=torch.jit.script(actor)(obs)
        torch.testing.assert_close(eager,scripted)
        self.assertEqual(tuple(eager.shape),(4096,1))
        self.assertTrue(torch.isfinite(eager).all())
        self.assertGreaterEqual(float(eager.min()),-1.)
        self.assertLessEqual(float(eager.max()),1.)

    def test_saved_model_is_unchanged_and_mapping_explicit(self):
        check_preserved()
        extra={'fixed_force_policy.json':''}
        torch.jit.load(str(SAVED/'candidate/checkpoints/actor.pt'),_extra_files=extra)
        self.assertEqual(check_contract(json.loads(extra['fixed_force_policy.json'])),(3.,9.))
        contract=contract_for(1.5,12.)
        self.assertEqual(check_contract(contract),(1.5,12.))
        for key,value in [('target_force_n',10.),('spindle_rpm',2000.),
                          ('action_names',['feed','force']),('force_bounds_n',[10.,30.])]:
            bad=copy.deepcopy(contract);bad[key]=value
            with self.assertRaises(ValueError):check_contract(bad)

    def test_legacy_limiter_exact_parity_and_expanded_derivative_limits(self):
        rng=np.random.default_rng(2700)
        old=SignedSpeedLimiter();new=ForwardSpeedLimiter(9.)
        for request in rng.uniform(0.,9.,3000):
            self.assertEqual(old.step(request),new.step(request))
        new=ForwardSpeedLimiter(12.)
        for request in rng.uniform(0.,12.,3000):
            self.assertGreaterEqual(new.step(request),0.)
            self.assertLessEqual(new.speed,12.+1.e-12)
            self.assertLessEqual(abs(new.acceleration),16.+1.e-9)
            self.assertLessEqual(abs(new.jerk),160.+1.e-8)

    def test_old_filter_parity_and_physical_slew_rescaling(self):
        root=_REPO_ROOT
        original=ast.parse((root/'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/mdp/action.py').read_text())
        original_cls=next(n for n in original.body if isinstance(n,ast.ClassDef) and n.name=='AdmittanceControlAction')
        original_method=next(n for n in original_cls.body if isinstance(n,ast.FunctionDef) and n.name=='process_actions')
        old_ns={'torch':torch,'math':math}
        exec(compile(ast.Module(body=[original_method],type_ignores=[]),'<original-filter>','exec'),old_ns)
        new_tree=ast.parse((root/'scripts/fixed_force_refinement_action.py').read_text())
        new_cls=next(n for n in new_tree.body if isinstance(n,ast.ClassDef))
        new_method=next(n for n in new_cls.body if isinstance(n,ast.FunctionDef) and n.name=='process_actions')
        new_ns={'torch':torch,'math':math,'np':np}
        exec(compile(ast.Module(body=[new_method],type_ignores=[]),'<new-filter>','exec'),new_ns)
        def state(half):
            return SimpleNamespace(_num_envs_local=1,device='cpu',
                _raw_actions=torch.zeros(1,1),_processed_actions=torch.zeros(1,1),
                action_delta=torch.zeros(1),_previous_clipped_action=torch.zeros(1),
                _action_history=torch.zeros(1,5),_action_delay=torch.tensor([2]),
                _filtered_action=torch.zeros(1),half_range=np.array([half]),
                int_cfg=SimpleNamespace(action_filter_tau_s=.08,action_slew_per_s=4.),
                _step_dt_local=.008)
        old,new,wide=state(4.5),state(4.5),state(6.)
        torch.manual_seed(2700)
        for value in (torch.rand(2000)*2.-1.):
            action=value.reshape(1,1)
            old_ns['process_actions'](old,action)
            new_ns['process_actions'](new,action)
            torch.testing.assert_close(old._filtered_action,new._filtered_action,atol=0.,rtol=0.)
            before=float(wide._filtered_action[0])
            new_ns['process_actions'](wide,action)
            self.assertLessEqual(abs(float(wide._filtered_action[0])-before)*6.,18.*.008+1.e-6)
        with self.assertRaises(ValueError):new_ns['process_actions'](wide,torch.zeros(1,3))


if __name__=='__main__':unittest.main()
