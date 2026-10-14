"""Conditional Gaussian exploration for temporally coherent PPO feed rollouts.

Previous observations and latent actions are retained with each transition.
The previous residual MUST be recomputed with the candidate policy when PPO
evaluates its conditional density, rather than freezing the old-policy noise.
Evaluation uses the deterministic actor with zero exploration noise.
"""
import math
import torch


def action_value(model, observation, previous_observation, previous_latent,
                 has_previous, rho=0., latent=None):
    if not 0. <= rho < 1.:
        raise ValueError('Exploration correlation must be in [0, 1)')
    if rho == 0.:
        return model.action_value(observation, latent)
    residual = (previous_latent - model.actor(previous_observation)) * has_previous
    distribution = torch.distributions.Normal(
        model.actor(observation) + rho * residual,
        model.log_std.clamp(-2.5, -.3).exp() * math.sqrt(1. - rho * rho))
    if latent is None:
        latent = distribution.sample()
    return (.125 + .875 * latent.tanh(), latent, distribution.log_prob(latent),
            distribution.entropy(), model.critic(observation))
