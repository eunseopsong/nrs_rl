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


def force_tracking_reward(
    env, error_scale_n: float = 2.0, action_term_name: str = "arm_action"
) -> torch.Tensor:
    term = _term(env, action_term_name)
    contact = ((term.current_abs_fz >= term.int_cfg.contact_force_n) & term.polishing_active).float()
    return torch.exp(-torch.square(term.force_error_n / max(error_scale_n, 1.0e-6))) * contact


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
    return coverage * torch.exp(-cv / max(cv_scale, 1.0e-6))


def removal_variation_penalty(
    env, delta_scale: float = 30.0, action_term_name: str = "arm_action"
) -> torch.Tensor:
    term = _term(env, action_term_name)
    return -torch.tanh(torch.abs(term.current_mrr_delta_n_mm_s) / max(delta_scale, 1.0e-6))


def action_rate_penalty(env, action_term_name: str = "arm_action") -> torch.Tensor:
    term = _term(env, action_term_name)
    return -torch.square(term.raw_actions[:, 0] - term.processed_actions[:, 0])


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
