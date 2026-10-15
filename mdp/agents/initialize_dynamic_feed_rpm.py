"""Learn a causal force/spindle response prior from synthetic states only."""
import math
import torch

from .feed_rpm_ppo import FeedRPMActorCritic


def initialize(parent, replay=None, seed=5310, steps=8000):
    torch.manual_seed(seed)
    model = FeedRPMActorCritic()
    saved = torch.load(parent, map_location='cpu', weights_only=False)['model']
    model.actor.feed.load_state_dict({k[6:]: v for k, v in saved.items() if k.startswith('actor.')})
    optimizer = torch.optim.Adam(model.actor.rpm.parameters(), lr=.001)
    decay = torch.exp(-torch.tensor([.008, .016]) / .016)

    def examples(n):
        obs = torch.zeros(n, 22)
        force = torch.where(torch.rand(n) < .7, 16. + 8. * torch.rand(n),
                            torch.exp(math.log(1.5) + math.log(100. / 1.5) * torch.rand(n)))
        slew = torch.randn(n) * 40.
        slew = torch.where(force < 15., torch.rand(n) * 140., slew)
        slew = torch.where(torch.rand(n) < .3, torch.zeros(n), slew)
        ideal = 20000. / force
        ratio = (1. + torch.randn(n) * .25).clamp(.35, 2.)
        actual = (ideal * ratio).clamp(50., 20000.)
        commanded = (actual * torch.exp(torch.randn(n) * .15)).clamp(50., 20000.)
        obs[:, 1] = force / 20.
        obs[:, 16] = torch.log(actual / 1000.) / 3.
        obs[:, 17] = torch.log(commanded / 1000.) / 3.
        obs[:, 18] = slew / 100.
        obs[:, 19] = force / 20. * actual / 1000. - 1.
        obs[:, 21] = (force - slew * .024).clamp(0., 100.) / 20.
        future_force = (force[:, None] + slew[:, None] * torch.tensor([.008, .016])).clamp(1.5, 100.)
        a = future_force / 20000. * actual[:, None] * decay
        b = future_force / 20000. * (1. - decay)
        command = ((b * (1. - a)).sum(-1) / b.square().sum(-1)).clamp(50., 20000.)
        target = torch.atanh((torch.log(command / 1000.) / 3.).clamp(-.995, .995))
        return obs, target, command, ideal

    for step in range(steps):
        if step == steps * 3 // 4:
            optimizer.param_groups[0]['lr'] = .00015
        obs, target, command, ideal = examples(512)
        predicted = model.actor.rpm(obs)
        predicted_command = 1000. * torch.exp(3. * predicted.tanh())
        loss = ((predicted_command - command) / ideal).square().mean() + .04 * (predicted - target).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.actor.rpm.parameters(), 2.)
        optimizer.step()
    with torch.no_grad():
        obs, _, command, ideal = examples(10000)
        predicted = 1000. * torch.exp(3. * model.actor.rpm(obs).tanh())
        error = (predicted - command) / ideal
        typical = (obs[:, 1] >= .8) & (obs[:, 1] <= 1.2)
        nominal = torch.zeros(1, 22)
        nominal[:, 1] = nominal[:, 21] = 1.
        nominal_command = float(1000. * torch.exp(3. * model.actor.rpm(nominal).tanh())[0])
    return model, {'kind': 'synthetic causal force forecast and virtual spindle inverse-response neural prior',
        'uses_future_measured_force': False, 'uses_evaluation_trace_for_fitting': False,
        'runtime_analytic_controller': False, 'supervised_steps': steps,
        'forecast': 'current force plus filtered force slew * 8/16 ms',
        'spindle_prior_tau_s': .016, 'teacher': 'least-squares command for the next two physics ticks',
        'synthetic_command_rmse_over_ideal_rpm': float(error.square().mean().sqrt()),
        'synthetic_typical_command_rmse_over_ideal_rpm': float(error[typical].square().mean().sqrt()),
        'nominal_command_rpm': nominal_command, 'synthetic_validation_is_not_isaac_validation': True}
