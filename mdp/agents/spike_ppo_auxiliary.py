"""Training-only force compensation prior and independent network clipping.

At fixed 1000 RPM, rotary removal rate is approximately proportional to force.
Preserving nominal dose per travelled distance therefore calls for v ~ F. This
is an auxiliary command target, not a substitute for measured Isaac validation.
"""
import torch


def feed(actor, observation):
    return 6.75 + 5.25 * actor(observation).tanh()


def force_observation(observation, ratio):
    result = observation.clone()
    result[:, 1] = ratio
    # Keep the deployed 16D feature semantics internally consistent: F*v/120-1.
    result[:, 13] = result[:, 1] * result[:, 2] - 1.
    return result


def compensation_loss(actor, parent_actor, observation, eligible):
    """Anchor the nominal spatial policy and refine its force-response slope.

    Synthetic forces are 16 and 24 N, with a paired 20 N anchor. All other
    features come from on-policy rollouts. No auxiliary target is constructed
    for approach, lost contact, shields, or incomplete actor transitions.
    """
    keep = (eligible & (observation[:, 0] > .02) & (observation[:, 10] > .5)
            & (observation[:, 11] < .5) & (observation[:, 1] > .075)
            & (observation[:, 7] < 1.))
    if not bool(keep.any()):
        zero = observation.new_zeros(())
        return zero, zero, 0
    current = observation[keep]
    nominal = force_observation(current, 1.)
    with torch.no_grad():
        reference = feed(parent_actor, nominal)
    anchor = ((feed(actor, nominal) - reference) / 6.).square().mean()
    errors = []
    for ratio in (.8, 1.2):
        altered = force_observation(current, ratio)
        target = (reference * ratio).clamp(1.5, 12.)
        errors.append(((feed(actor, altered) - target) / 6.).square())
    return torch.cat(errors).mean(), anchor, int(keep.sum())


def clip_actor_critic(model, max_norm=.5):
    """A large critic error must not scale the separate actor's gradients."""
    actor = torch.nn.utils.clip_grad_norm_([*model.actor.parameters(), model.log_std], max_norm)
    critic = torch.nn.utils.clip_grad_norm_(model.critic.parameters(), max_norm)
    return actor, critic
