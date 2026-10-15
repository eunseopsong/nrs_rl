"""Two-output neural policy: spatial feed plus virtual spindle RPM."""
import copy
import json
import math

import torch
from torch import nn

from .latest_force_ppo import CONTRACT as FEED_CONTRACT
from .ppo_network import PolicyNetwork
from ...utils.spatial_geometry import OBSERVATIONS


RPM_OBSERVATIONS = OBSERVATIONS + ['log_rpm_over_nominal_div3', 'log_command_rpm_over_nominal_div3',
    'filtered_force_slew_over_100', 'rotary_rate_relative_error', 'contact_age_over_3s', 'filtered_force_over_20']


class ExtendedNetwork(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.register_buffer('frequencies', torch.arange(1., 9.) * 2. * math.pi)
        self.net = nn.Sequential(nn.Linear(38, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, math.sqrt(2.))
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, obs):
        phase = obs[:, :1] * self.frequencies[None, :]
        return self.net(torch.cat((obs.clamp(-5., 5.), phase.sin(), phase.cos()), -1)).squeeze(-1)


class JointActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.feed = PolicyNetwork()
        self.rpm = RPMNetwork()

    def forward(self, obs):
        spatial = obs[:, :16].clone()
        # RPM handles force. Feed responds to residual removal flux so successful
        # RPM compensation does not trigger a second, unnecessary feed response.
        spatial[:, 1] = (obs[:, 19] + 1.).clamp(0., 5.)
        spatial[:, 13] = spatial[:, 1] * spatial[:, 2] - 1.
        return torch.stack((self.feed(spatial), self.rpm(obs)), -1)


class RPMNetwork(nn.Module):
    """Reactive force/spindle state only; no path-position memorization."""
    def __init__(self):
        super().__init__()
        self.register_buffer('indices', torch.tensor([1, 16, 17, 18, 19, 21]))
        self.net = nn.Sequential(nn.Linear(6, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, math.sqrt(2.))
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, obs):
        return self.net(obs[:, self.indices].clamp(-5., 5.)).squeeze(-1)


class FeedRPMActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = JointActor()
        self.critic = ExtendedNetwork(128)
        self.log_std = nn.Parameter(torch.tensor([-3., -4.]))

    def action_value(self, obs, latent=None):
        distribution = torch.distributions.Normal(self.actor(obs), self.log_std.clamp(-5., -1.).exp())
        if latent is None:
            latent = distribution.sample()
        squashed = latent.tanh()
        action = torch.stack((.125 + .875 * squashed[:, 0], squashed[:, 1]), -1)
        return action, latent, distribution.log_prob(latent).sum(-1), distribution.entropy().sum(-1), self.critic(obs)


class DeterministicFeedRPM(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = copy.deepcopy(actor).cpu().eval()

    def forward(self, obs):
        value = self.actor(obs).tanh()
        return torch.stack((.125 + .875 * value[:, 0], value[:, 1]), -1)


def contract(config):
    result = copy.deepcopy(FEED_CONTRACT)
    result.update(backend='ppo_feed_rpm_v1', action_dimension=2, action_names=['feed', 'log_rpm'],
        action_mapping=['6 + 6*a0 mm/s', '1000 * exp(3*a1) RPM'],
        observation_names=RPM_OBSERVATIONS, observation_dimension=22,
        network='feed 128x128 and critic 128x128 with Fourier path features; RPM 64x64 on six force/spindle features',
        spindle_rpm=None, rpm_policy=config.metadata(), policy_period_s=.016,
        policy_controls_rpm=True, rpm_is_assumed_not_measured=True,
        physical_spindle_actuated=False, scalar_feed_runtime_compatible=False,
        teacher_profile_used=False, rpm_physics_prior_used=True,
        training_method='inverse-force supervised RPM initialization followed by Isaac PPO-Clip',
        entry_seconds=None, entry_reference_footprint_mm=None,
        reward_and_evaluation_contact_scope='first measured contact >=1.5 N through completion; entry included',
        inherited_observation_history='original 16D active-motion history; additional RPM features include entry',
        feed_force_feature='20 * measured rotary removal rate / fixed target rate',
        selection='independent headless Isaac comparison; fixed target-rate error and full-contact depth CV')
    return result


def export_actor(model, path, config, metadata):
    actor = DeterministicFeedRPM(model.actor)
    scripted = torch.jit.script(actor)
    probe = torch.randn(20, 22)
    with torch.no_grad():
        torch.testing.assert_close(actor(probe), scripted(probe))
    torch.jit.save(scripted, str(path), _extra_files={'feed_rpm_policy.json': json.dumps({**contract(config), **metadata})})
