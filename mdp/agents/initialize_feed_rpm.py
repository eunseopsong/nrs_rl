"""Auditable physics-prior initialization; no evaluation trajectories are fitted."""
import math
import numpy as np
import torch

from .feed_rpm_ppo import FeedRPMActorCritic


def initialize(parent, replay, seed=5210, steps=3500):
    torch.manual_seed(seed)
    model = FeedRPMActorCritic()
    saved = torch.load(parent, map_location='cpu', weights_only=False)['model']
    model.actor.feed.load_state_dict({k[6:]: v for k, v in saved.items() if k.startswith('actor.')})
    source = np.load(replay)
    key = 'observations' if 'observations' in source else source.files[0]
    observations = torch.as_tensor(source[key], dtype=torch.float32)
    if observations.ndim != 2 or observations.shape[1] != 16:
        raise ValueError('Expected the existing 16D parent observation replay')
    optimizer = torch.optim.Adam(model.actor.rpm.parameters(), lr=1.e-3)

    def examples(n, nominal=False):
        obs = torch.zeros(n, 22)
        obs[:, :16] = observations[torch.randint(len(observations), (n,))]
        # Cover the whole permitted contact interval, emphasizing 20 N operation.
        force = torch.where(torch.rand(n) < .7, 15. + 10. * torch.rand(n),
                            torch.exp(math.log(1.5) + math.log(100. / 1.5) * torch.rand(n)))
        force = torch.where(torch.rand(n) < .15, 20., force)
        if nominal:
            force.fill_(20.)
        obs[:, 1] = force / 20.
        obs[:, 10:12] = torch.tensor([1., 0.])
        obs[:, 13] = obs[:, 1] * obs[:, 2] - 1.
        obs[:, 16:18] = (torch.rand(n, 2) - .5) * 1.5
        obs[:, 18] = torch.randn(n) * .5
        obs[:, 19] = force / 20. * torch.exp(3. * obs[:, 16]) - 1.
        obs[:, 20] = torch.rand(n)
        obs[:, 21] = (force / 20. + torch.randn(n) * .15).clamp(.075, 5.)
        action = torch.log(20. / force) / 3.
        target = torch.atanh(action.clamp(-.995, .995))
        return obs, target, force

    for step in range(steps):
        if step == steps * 3 // 4:
            optimizer.param_groups[0]['lr'] = 2.e-4
        obs, target, force = examples(512)
        predicted = model.actor.rpm(obs)
        # Relative MRR error, not raw RPM error, gives low/high forces equal scale.
        rate_ratio = force / 20. * torch.exp(3. * predicted.tanh())
        loss = (rate_ratio - 1.).square().mean() + .1 * (predicted - target).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.actor.rpm.parameters(), 2.)
        optimizer.step()
    with torch.no_grad():
        obs, _, force = examples(10000)
        error = force / 20. * torch.exp(3. * model.actor.rpm(obs).tanh()) - 1.
        typical = (force >= 16.) & (force <= 24.)
        nominal_obs, _, _ = examples(2000, nominal=True)
        nominal_rpm = 1000. * torch.exp(3. * model.actor.rpm(nominal_obs).tanh())
    return model, {'kind': 'supervised inverse-force physics prior, followed by genuine Isaac PPO',
        'uses_future_measured_force': False, 'runtime_analytic_controller': False,
        'supervised_steps': steps, 'synthetic_force_range_n': [1.5, 100.],
        'synthetic_holdout_rate_rmse': float(error.square().mean().sqrt()),
        'synthetic_typical_rate_rmse': float(error[typical].square().mean().sqrt()),
        'nominal_rpm_mean': float(nominal_rpm.mean()),
        'nominal_rpm_std': float(nominal_rpm.std()),
        'synthetic_validation_is_not_isaac_validation': True}
