"""Trainable neural actor/critic and deterministic export for PPO-Clip."""
import copy
import json
import math
import torch
from torch import nn
from ..mdp.removal_config import CONTRACT, check_ppo_or_baseline
from ..utils.removal_metrics import sha

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('frequencies', torch.arange(1., 9.)*2.*math.pi)
        self.net = nn.Sequential(nn.Linear(32, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, math.sqrt(2.))
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, math.atanh(-1./7.))

    def forward(self, obs):
        phase = obs[:, :1]*self.frequencies[None, :]
        features = torch.cat((obs.clamp(-5., 5.), torch.sin(phase), torch.cos(phase)), dim=-1)
        return self.net(features).squeeze(-1)


class PPOActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = PolicyNetwork()
        self.critic = copy.deepcopy(self.actor)
        nn.init.zeros_(self.critic.net[-1].weight)
        nn.init.zeros_(self.critic.net[-1].bias)
        self.log_std = nn.Parameter(torch.tensor(-.55))

    def distribution(self, obs):
        return torch.distributions.Normal(self.actor(obs), self.log_std.clamp(-2.5, -.3).exp())

    def action_value(self, obs, latent=None):
        distribution = self.distribution(obs)
        if latent is None:
            latent = distribution.sample()
        # This bijection maps latent Gaussian samples into feed [1.5,12].
        # Log-density Jacobians cancel in PPO's old/new probability ratio.
        action = .125 + .875*torch.tanh(latent)
        return action, latent, distribution.log_prob(latent), distribution.entropy(), self.critic(obs)


class DeterministicPPOActor(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = copy.deepcopy(actor).cpu().eval()

    def forward(self, obs):
        return (.125 + .875*torch.tanh(self.actor(obs))).unsqueeze(-1)


def export_actor(model, path, metadata):
    contract = {**CONTRACT, **metadata}
    check_ppo_or_baseline(contract)
    actor = DeterministicPPOActor(model.actor)
    scripted = torch.jit.script(actor)
    torch.jit.save(scripted, str(path), _extra_files={'fixed_force_policy.json': json.dumps(contract)})
    probe = torch.randn(32, 16)
    with torch.inference_mode():
        torch.testing.assert_close(actor(probe), scripted(probe))
    return sha(path)

