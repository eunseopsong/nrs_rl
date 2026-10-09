"""Math, feature parity, and actuator-authority checks before PPO training."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from nrs_rl.tasks.manager_based.nrs_rl.agents.fixed_force_ppo import RemovalEnv, PPOActorCritic, export_actor, CONTRACT, check_ppo_or_baseline


class PPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.env=RemovalEnv(3,seed=1,randomize=False)

    def test_observation_matches_existing_geometry(self):
        env=self.env
        for _ in range(40):env.step(torch.tensor([-.5,0.,.5]))
        actual=env.observation().numpy()
        expected=env.geometry.features(env.depth.numpy(),env.residence.numpy(),env.cursor.numpy(),env.cursor.numpy(),
             env.force.numpy(),env.speed.numpy(),env.elapsed.numpy(),np.zeros(3),np.full(3,20.),np.zeros(3))
        np.testing.assert_allclose(actual,expected,atol=2.e-5,rtol=1.e-5)

    def test_ppo_ratio_and_gradients(self):
        torch.manual_seed(3101);model=PPOActorCritic();obs=torch.randn(32,16)
        a,z,old,_,v=model.action_value(obs);old=old.detach();z=z.detach()
        _,_,new,_,_=model.action_value(obs,z)
        torch.testing.assert_close((new-old).exp(),torch.ones(32))
        before=model.actor.net[-1].weight.detach().clone()
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        advantage=z.detach()-z.detach().mean()
        loss=-((new-old).exp()*advantage).mean()+.1*v.square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
        self.assertGreater(float((model.actor.net[-1].weight-before).abs().sum()),0.)

    def test_scalar_export_contains_neural_weights(self):
        model=PPOActorCritic()
        self.assertGreater(sum(p.numel() for p in model.actor.parameters()),10000)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'actor.pt';export_actor(model,path,{'ppo_updates':0})
            actor=torch.jit.load(str(path))
            with torch.inference_mode():action=actor(torch.randn(200,16)*5.)
            self.assertEqual(action.shape,(200,1))
            self.assertTrue(bool(torch.isfinite(action).all()))
            self.assertTrue(bool(((6.+6.*action)>=1.5-1.e-6).all()))
            self.assertTrue(bool(((6.+6.*action)<=12.+1.e-6).all()))
        broken={**CONTRACT,'target_force_n':19.}
        with self.assertRaises(ValueError):check_ppo_or_baseline(broken)

    def test_fixed_force_forward_residence_deposition(self):
        env=self.env;env.reset(torch.arange(3));env.speed.zero_();env.slew.zero_();env.filtered.fill_(-1.)
        previous=env.depth.sum(1).clone()
        env.step(torch.full((3,),-.75))
        self.assertTrue(bool((env.depth.sum(1)>previous).all()))
        self.assertTrue(bool((env.cursor>=0.).all()))
        self.assertTrue(bool((env.force==20.).all()))


if __name__=='__main__':unittest.main()
