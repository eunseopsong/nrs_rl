# SPDX-License-Identifier: BSD-3-Clause
"""Rewards based exclusively on realized force, motion and spatial removal."""

from __future__ import annotations

import torch


def _term(env, name: str = "arm_action"):
    manager = env.action_manager
    if hasattr(manager, "get_term"):
        try:
            return manager.get_term(name)
        except Exception:
            pass
    if hasattr(manager, "_terms") and name in manager._terms:
        return manager._terms[name]
    raise RuntimeError(f"action term '{name}' not found")


def realized_removal_reward(
    env, target_mrr_n_mm_s: float = 60.0, action_term_name: str = "arm_action"
) -> torch.Tensor:
    """Useful material-removal throughput from measured TCP displacement."""
    term = _term(env, action_term_name)
    mrr = torch.clamp(term.current_mrr_n_mm_s, min=0.0)
    return 1.0 - torch.exp(-mrr / max(target_mrr_n_mm_s, 1.0e-6))


def removal_rate_tracking_penalty(
    env, target_mrr_n_mm_s: float | None = None, action_term_name: str = "arm_action"
) -> torch.Tensor:
    """Track a fixed process rate; a moving mean cannot hide slow speed drift.

    E[(rate/target - 1)^2] equals variance/target^2 plus squared mean bias.
    Penalizing the raw samples therefore targets both CV and useful throughput;
    filtering before squaring would hide high-frequency variation. This target
    is the Preston volume-rate proxy F*v (default 20 N * 6 mm/s). Without a
    calibrated K it is in N mm/s; the definition of v is measured TCP sliding.
    """
    term = _term(env, action_term_name)
    target = max(term.int_cfg.target_mrr_n_mm_s if target_mrr_n_mm_s is None else target_mrr_n_mm_s, 1.0e-6)
    relative_error = (term.current_mrr_n_mm_s - target) / target
    valid = term.polishing_active
    return -torch.square(relative_error) * valid.float()


def force_tracking_reward(
    env, error_scale_n: float = 2.0, action_term_name: str = "arm_action"
) -> torch.Tensor:
    term = _term(env, action_term_name)
    contact = ((term.current_abs_fz >= term.int_cfg.contact_force_n) & term.polishing_active).float()
    return torch.exp(-torch.square(term.force_error_n / max(error_scale_n, 1.0e-6))) * contact


def force_overshoot_penalty(
    env, error_scale_n: float = 2.0, action_term_name: str = "arm_action"
) -> torch.Tensor:
    """Penalize only force above the scheduled target during contact.

    The symmetric tracking reward already handles ordinary tracking error.  A
    separate one-sided term prevents a short force spike from being traded for
    extra removal in one spatial bin, while leaving under-force recovery to the
    production Mode-3 controller.
    """
    term = _term(env, action_term_name)
    overshoot = torch.relu(term.force_error_n)
    contact = ((term.current_abs_fz >= term.int_cfg.contact_force_n) & term.polishing_active).float()
    return -torch.tanh(overshoot / max(error_scale_n, 1.0e-6)) * contact


def spatial_uniformity_reward(
    env, cv_scale: float = 0.25, action_term_name: str = "arm_action"
) -> torch.Tensor:
    """Dense coverage-weighted uniformity; cannot be won by staying in one bin."""
    term = _term(env, action_term_name)
    removal = term.surface_removal_by_index
    visited = removal > 0.0
    count = visited.sum(dim=1)
    safe_count = count.clamp_min(1)
    mean = removal.sum(dim=1) / safe_count
    variance = (
        torch.square(removal - mean[:, None]) * visited
    ).sum(dim=1) / safe_count
    cv = torch.sqrt(variance) / mean.clamp_min(1.0e-6)
    coverage = count.float() / float(removal.shape[1])
    moving_contact = (term.realized_removal_step > 0.0) & ~term.safety_fault_active
    return coverage * torch.exp(-cv / max(cv_scale, 1.0e-6)) * moving_contact.float()


def removal_variation_penalty(
    env, delta_scale: float = 15.0, action_term_name: str = "arm_action"
) -> torch.Tensor:
    term = _term(env, action_term_name)
    # A causal, noise-limited fluctuation around the recent MRR trend. Includes
    # zero-speed samples during polishing; stopping cannot hide fluctuations.
    valid = term.polishing_active & ~term.safety_fault_active
    return -torch.tanh(torch.square(term.mrr_fluctuation_n_mm_s / max(delta_scale, 1.0e-6))) * valid.float()


def action_rate_penalty(env, action_term_name: str = "arm_action") -> torch.Tensor:
    term = _term(env, action_term_name)
    return -torch.square(term.action_delta) * term.command_smoothness_valid.float()


def command_acceleration_penalty(env, action_term_name: str = "arm_action") -> torch.Tensor:
    term = _term(env, action_term_name)
    scale = term.int_cfg.max_speed_acceleration_mm_s2
    return -torch.square(term.command_acceleration_mm_s2 / scale) * term.command_smoothness_valid.float()


def command_jerk_penalty(env, action_term_name: str = "arm_action") -> torch.Tensor:
    term = _term(env, action_term_name)
    scale = term.int_cfg.max_speed_jerk_mm_s3
    return -torch.square(term.command_jerk_mm_s3 / scale) * term.command_smoothness_valid.float()


def safety_shield_penalty(env, action_term_name: str = "arm_action") -> torch.Tensor:
    return -_term(env, action_term_name).safety_shield_active.float()


def completion_quality_reward(
    env, cv_scale: float = 0.25, action_term_name: str = "arm_action"
) -> torch.Tensor:
    term = _term(env, action_term_name)
    removal = term.surface_removal_by_index
    visited = removal > 0.0
    count = visited.sum(dim=1).clamp_min(1)
    mean = removal.sum(dim=1) / count
    variance = (torch.square(removal - mean[:, None]) * visited).sum(dim=1) / count
    cv = torch.sqrt(variance) / mean.clamp_min(1.0e-6)
    coverage = visited.float().mean(dim=1)
    quality = coverage * torch.exp(-cv / max(cv_scale, 1.0e-6))
    return quality * term.path_done.float()


def completion_rate_quality_reward(env, action_term_name: str = "arm_action") -> torch.Tensor:
    """One completion bonus for the same raw-rate objective as the dense loss.

    Isaac multiplies all reward terms by step_dt, including terminal terms.
    Divide here so a configured weight of 1 really is a bonus of at most 1.
    """
    term = _term(env, action_term_name)
    mse = term.rate_squared_error_sum / term.polishing_steps.clamp_min(1)
    return torch.exp(-mse) * term.path_done.float() / env.step_dt
