# SPDX-License-Identifier: BSD-3-Clause
"""Reward functions for removal-rate stability and surface uniformity."""

from __future__ import annotations

from typing import TYPE_CHECKING
import importlib
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

local_obs = importlib.import_module("nrs_rl.tasks.manager_based.nrs_rl.mdp.observation")


def _soft_gate(value: torch.Tensor, threshold: float, sharpness: float = 8.0) -> torch.Tensor:
    return torch.sigmoid(sharpness * (value / max(threshold, 1.0e-8) - 1.0))


def _mrr_state(
    env: "ManagerBasedRLEnv",
    action_term_name: str = "arm_action",
    asset_name: str = "robot",
    body_name: str = "spindle_link",
    fixed_joint_name: str = "tool0_to_spindle",
    joint_prim_relpath: str = "joints",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    current_fz, sliding_velocity, current_mrr = local_obs.get_path_sliding_metrics(
        env,
        action_term_name=action_term_name,
        asset_name=asset_name,
        body_name=body_name,
        fixed_joint_name=fixed_joint_name,
        joint_prim_relpath=joint_prim_relpath,
    )

    term = local_obs.get_action_term(env, action_term_name)
    if term is not None and hasattr(term, "current_mrr_delta_n_mm_s"):
        mrr_delta = term.current_mrr_delta_n_mm_s.to(device=env.device, dtype=torch.float32)
        prev_mrr = (
            term.prev_mrr_n_mm_s.to(device=env.device, dtype=torch.float32)
            if hasattr(term, "prev_mrr_n_mm_s")
            else current_mrr - mrr_delta
        )
    else:
        mrr_delta = torch.zeros_like(current_mrr)
        prev_mrr = current_mrr

    return current_fz, sliding_velocity, current_mrr, prev_mrr, mrr_delta


def removal_rate_reward(
    env: "ManagerBasedRLEnv",
    saturation_mrr: float = 500.0,
    min_contact_force: float = 1.0,
    gate_sharpness: float = 8.0,
    action_term_name: str = "arm_action",
) -> torch.Tensor:
    """Dense reward for making total removal high, with exponential saturation."""
    current_fz, _, current_mrr, _, _ = _mrr_state(env, action_term_name=action_term_name)
    force_gate = _soft_gate(current_fz, min_contact_force, gate_sharpness)
    throughput = 1.0 - torch.exp(-torch.clamp(current_mrr, min=0.0) / max(saturation_mrr, 1.0e-6))
    return throughput * force_gate


def removal_constancy_reward(
    env: "ManagerBasedRLEnv",
    delta_tau: float = 35.0,
    min_contact_force: float = 1.0,
    min_active_mrr: float = 80.0,
    gate_sharpness: float = 8.0,
    action_term_name: str = "arm_action",
) -> torch.Tensor:
    """Dense penalty for local removal-rate changes."""
    current_fz, _, current_mrr, prev_mrr, mrr_delta = _mrr_state(env, action_term_name=action_term_name)
    force_gate = _soft_gate(current_fz, min_contact_force, gate_sharpness)
    active_mrr = torch.maximum(torch.abs(prev_mrr), torch.abs(current_mrr))
    active_gate = _soft_gate(active_mrr, min_active_mrr, gate_sharpness)
    change_penalty = 1.0 - torch.exp(-torch.abs(mrr_delta) / max(delta_tau, 1.0e-6))
    return -change_penalty * force_gate * active_gate


def removal_instability_penalty(
    env: "ManagerBasedRLEnv",
    spike_delta: float = 80.0,
    spike_tau: float = 35.0,
    dip_mrr: float = 120.0,
    dip_tau: float = 45.0,
    dip_weight: float = 1.0,
    min_contact_force: float = 1.0,
    min_prev_mrr: float = 120.0,
    gate_sharpness: float = 8.0,
    action_term_name: str = "arm_action",
) -> torch.Tensor:
    """Penalty for the sharp vertical spikes and zero-dips visible in the MRR plot."""
    current_fz, _, current_mrr, prev_mrr, mrr_delta = _mrr_state(env, action_term_name=action_term_name)

    force_gate = _soft_gate(current_fz, min_contact_force, gate_sharpness)
    active_gate = _soft_gate(torch.maximum(torch.abs(prev_mrr), torch.abs(current_mrr)), min_prev_mrr, gate_sharpness)

    spike_excess = torch.clamp(torch.abs(mrr_delta) - spike_delta, min=0.0)
    spike_penalty = 1.0 - torch.exp(-spike_excess / max(spike_tau, 1.0e-6))

    prev_active_gate = _soft_gate(torch.abs(prev_mrr), min_prev_mrr, gate_sharpness)
    dip_depth = torch.clamp(dip_mrr - current_mrr, min=0.0)
    dip_penalty = 1.0 - torch.exp(-dip_depth / max(dip_tau, 1.0e-6))

    return -(spike_penalty * active_gate + dip_weight * dip_penalty * prev_active_gate) * force_gate


def surface_uniformity_reward(
    env: "ManagerBasedRLEnv",
    cv_tau: float = 0.35,
    total_tau: float = 5500.0,
    min_mean_removal: float = 1.0e-6,
    action_term_name: str = "arm_action",
) -> torch.Tensor:
    """Terminal reward for high and uniform accumulated removal over the whole path."""
    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    reset_buf = getattr(env, "reset_buf", None)
    if reset_buf is None or not reset_buf.any():
        return reward

    term = local_obs.get_action_term(env, action_term_name)
    if term is None or not hasattr(term, "surface_removal_by_index"):
        return reward

    removal_by_index = term.surface_removal_by_index.to(device=env.device, dtype=torch.float32)
    path_done = (
        term.path_done.to(device=env.device, dtype=torch.bool)
        if hasattr(term, "path_done")
        else torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    )

    for env_id in torch.nonzero(reset_buf, as_tuple=False).flatten().tolist():
        if not bool(path_done[env_id].item()):
            continue

        values = removal_by_index[env_id]
        mean_removal = torch.mean(values)
        if float(mean_removal.item()) <= min_mean_removal:
            continue

        std_removal = torch.std(values, unbiased=False)
        cv = std_removal / torch.clamp(mean_removal, min=min_mean_removal)
        total_removal = torch.sum(values)
        uniformity = torch.exp(-cv / max(cv_tau, 1.0e-6))
        throughput = 1.0 - torch.exp(-total_removal / max(total_tau, 1.0e-6))
        reward[env_id] = uniformity * throughput

    return reward
