# SPDX-License-Identifier: BSD-3-Clause
"""
Reward functions module for Adaptive Uniform Polishing.
[🔥 에이전트 날먹 꼼수 박살 버전]
"""
 
from __future__ import annotations
 
from typing import TYPE_CHECKING
import importlib
import torch
 
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
 
local_obs = importlib.import_module("nrs_rl.tasks.manager_based.nrs_rl.mdp.observation")
 
# ==========================================
# Helper Functions
# ==========================================
def _soft_gate(
    value: torch.Tensor,
    threshold: float,
    sharpness: float = 8.0,
) -> torch.Tensor:
    return torch.sigmoid(sharpness * (value / max(threshold, 1e-8) - 1.0))
 
def _curriculum_sigma(
    env: "ManagerBasedRLEnv",
    sigma_start: float,
    sigma_end: float,
    ramp: int = 200,
) -> float:
    if not hasattr(env, "_ep_curriculum"):
        env._ep_curriculum = 0
    progress = float(min(env._ep_curriculum, ramp)) / float(max(ramp, 1))
    return sigma_start + (sigma_end - sigma_start) * progress


def _exponential_target_reward(
    value: torch.Tensor,
    target: float | torch.Tensor,
    tau: float,
    progress_tau: float = 0.35,
    max_ratio: float = 4.0,
) -> torch.Tensor:
    target_tensor = torch.as_tensor(target, device=value.device, dtype=value.dtype)
    ratio = torch.clamp(value / torch.clamp(target_tensor, min=1.0e-6), min=0.0, max=max_ratio)
    relative_error = torch.abs(ratio - 1.0)
    tracking_reward = torch.exp(-relative_error / max(tau, 1.0e-6))

    progress_gate = 1.0 - torch.exp(-ratio / max(progress_tau, 1.0e-6))
    target_gate = 1.0 - torch.exp(torch.tensor(-1.0 / max(progress_tau, 1.0e-6), device=value.device, dtype=value.dtype))
    progress_gate = torch.clamp(progress_gate / torch.clamp(target_gate, min=1.0e-6), max=1.0)

    return tracking_reward * progress_gate
 
# ==========================================
# 🚀 1. Core Adaptive MRR Reward
# ==========================================
def adaptive_mrr_reward(
    env: "ManagerBasedRLEnv",
    target_mrr: float = 500.0,
    mrr_sigma_start: float = 1.2,
    mrr_sigma_end: float = 0.3,
    curriculum_ramp: int = 200,
    min_contact_force: float = 1.0,
    min_velocity: float = 1e-3,
    gate_sharpness: float = 8.0,
    asset_name: str = "robot",
    body_name: str = "spindle_link",
    fixed_joint_name: str = "tool0_to_spindle",
    joint_prim_relpath: str = "joints",
    action_term_name: str = "arm_action",
) -> torch.Tensor:
    current_fz, sliding_velocity, current_mrr = local_obs.get_path_sliding_metrics(
        env,
        action_term_name=action_term_name,
        asset_name=asset_name,
        body_name=body_name,
        fixed_joint_name=fixed_joint_name,
        joint_prim_relpath=joint_prim_relpath,
    )
 
    force_gate = _soft_gate(current_fz, min_contact_force, gate_sharpness)
    vel_gate = _soft_gate(sliding_velocity, min_velocity, gate_sharpness)
    activity_gate = force_gate * vel_gate
 
    tau = _curriculum_sigma(env, mrr_sigma_start, mrr_sigma_end, curriculum_ramp)
 
    reward = _exponential_target_reward(current_mrr, target_mrr, tau)
    return reward * activity_gate
 
# ==========================================
# 🚀 2. Inverse Control Bonus
# ==========================================
def inverse_fv_bonus(
    env: "ManagerBasedRLEnv",
    target_mrr: float = 500.0,
    bonus_sigma_start: float = 0.8,
    bonus_sigma_end: float = 0.25,
    curriculum_ramp: int = 200,
    min_contact_force: float = 1.0,
    gate_sharpness: float = 8.0,
    asset_name: str = "robot",
    body_name: str = "spindle_link",
    fixed_joint_name: str = "tool0_to_spindle",
    joint_prim_relpath: str = "joints",
    action_term_name: str = "arm_action",
) -> torch.Tensor:
    current_fz, sliding_velocity, _ = local_obs.get_path_sliding_metrics(
        env,
        action_term_name=action_term_name,
        asset_name=asset_name,
        body_name=body_name,
        fixed_joint_name=fixed_joint_name,
        joint_prim_relpath=joint_prim_relpath,
    )
 
    force_gate = _soft_gate(current_fz, min_contact_force, gate_sharpness)
    ideal_velocity = target_mrr / torch.clamp(current_fz, min=1e-3)
    tau = _curriculum_sigma(env, bonus_sigma_start, bonus_sigma_end, curriculum_ramp)
 
    bonus = _exponential_target_reward(sliding_velocity, ideal_velocity, tau)
    return bonus * force_gate


def mrr_flatness_reward(
    env: "ManagerBasedRLEnv",
    target_mrr: float = 500.0,
    band_tau: float = 0.25,
    delta_tau: float = 60.0,
    min_contact_force: float = 1.0,
    min_velocity: float = 1.0,
    gate_sharpness: float = 8.0,
    asset_name: str = "robot",
    body_name: str = "spindle_link",
    fixed_joint_name: str = "tool0_to_spindle",
    joint_prim_relpath: str = "joints",
    action_term_name: str = "arm_action",
) -> torch.Tensor:
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
        mrr_delta = torch.abs(term.current_mrr_delta_n_mm_s.to(device=env.device, dtype=torch.float32))
    else:
        mrr_delta = torch.zeros_like(current_mrr)

    force_gate = _soft_gate(current_fz, min_contact_force, gate_sharpness)
    vel_gate = _soft_gate(sliding_velocity, min_velocity, gate_sharpness)
    activity_gate = force_gate * vel_gate

    band_reward = _exponential_target_reward(
        current_mrr,
        target_mrr,
        tau=band_tau,
        progress_tau=0.25,
        max_ratio=2.5,
    )
    delta_reward = torch.exp(-mrr_delta / max(delta_tau, 1.0e-6))
    return band_reward * delta_reward * activity_gate
 
# ==========================================
# 🚀 3. Trajectory Tracking Reward
# ==========================================
def trajectory_tracking_reward(
    env: "ManagerBasedRLEnv",
    pos_sigma: float = 0.05,
    vel_bonus_scale: float = 0.3,
    min_tracking_velocity: float = 5e-4,
    gate_sharpness: float = 6.0,
    asset_name: str = "robot",
    body_name: str = "spindle_link",
) -> torch.Tensor:
    current_pose = local_obs.get_ee_pose(env, asset_name=asset_name)
    current_xyz = current_pose[:, :3]
    _, _, current_vel_norm = local_obs.get_current_pose_and_velocity(env, asset_name, body_name)
    target_xyz, _ = local_obs.get_current_target_pose_for_reward(env)
 
    pos_error = torch.norm(current_xyz - target_xyz, dim=-1)
    pos_reward = torch.exp(-torch.square(pos_error / pos_sigma))
 
    vel_gate = _soft_gate(current_vel_norm, min_tracking_velocity, gate_sharpness)
    vel_bonus = vel_bonus_scale * vel_gate
 
    return pos_reward + vel_bonus

# ==========================================
# 🛑 [신설] 3-2. Anti-Lazy Immobility Penalty (가만히 서 있는 새끼 척살)
# ==========================================
def anti_lazy_immobility_penalty(
    env: "ManagerBasedRLEnv",
    min_velocity: float = 5e-3,      # 최소 이 정도 속도로는 움직여야 함
    penalty_scale: float = -5.0,    # 멈춰있으면 매 스텝마다 골로 보냄
    asset_name: str = "robot",
    body_name: str = "spindle_link",
) -> torch.Tensor:
    """움직이지 않고 제자리에 버티며 에러 꼼수 부리는 행동 실시간 가스실 송행"""
    _, _, current_vel_norm = local_obs.get_current_pose_and_velocity(env, asset_name, body_name)
    
    # 속도가 기준치 미만이면 가만히 있는 것으로 간주 (True = 1.0)
    is_lazy = (current_vel_norm < min_velocity).float()
    
    return is_lazy * penalty_scale

# ==========================================
# 📜 4. Physical Completion Reward (헌법 1조 - 꼼수 봉쇄형 개조)
# ==========================================
def physical_completion_reward(
    env: "ManagerBasedRLEnv",
    distance_threshold: float = 0.05,  # 5cm 이내 도달해야 성공
    asset_name: str = "robot",
    body_name: str = "spindle_link",
) -> torch.Tensor:
    reset_buf = env.reset_buf
    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if not reset_buf.any():
        return reward

    term = local_obs.get_action_term(env, "arm_action")
    is_path_done = term.path_done.to(device=env.device, dtype=torch.bool) if term is not None and hasattr(term, "path_done") else reset_buf
    current_xyz = local_obs.get_ee_pose(env, asset_name=asset_name)[:, :3]
    target_xyz, _ = local_obs.get_current_target_pose_for_reward(env)
    distance_error = torch.norm(current_xyz - target_xyz, dim=-1)
    
    success = distance_error < distance_threshold
    completion_score = torch.where(
        success,
        torch.tensor(2000.0, device=env.device, dtype=torch.float32),
        -500.0 * (distance_error + 1.0) 
    )
    early_termination_penalty = torch.tensor(-2500.0, device=env.device, dtype=torch.float32)
    final_score = torch.where(is_path_done, completion_score, early_termination_penalty)
    
    reward[reset_buf] = final_score[reset_buf]
    return reward

# ==========================================
# 📜 5. Surface Coverage Reward (헌법 2조 - 실시간 압박형 개조)
# ==========================================
def surface_coverage_reward(
    env: "ManagerBasedRLEnv",
    target_cells: int = 500,  # 환경에 맞게 조절하세요
) -> torch.Tensor:
    """연마 표면적을 안 채우고 가만히 있으면 보상이 바닥을 기게 만듦"""
    vis_module = importlib.import_module("nrs_rl.tasks.manager_based.nrs_rl.utils.visualization")
    grid = vis_module._surface_grid
    
    touched_cells = len(grid[grid > 0])
    coverage_ratio = min(touched_cells / max(target_cells, 1), 1.0)
    
    # 가만히 있으면 커버리지가 안 올라서 스텝당 이득이 전혀 없도록 제곱근/제곱 구조 유지
    # 완주 보상과 시너지를 내기 위해 패널티성으로 변형 가능 (여기서는 강력한 스케일링 적용)
    reward = (coverage_ratio ** 2) * 5.0
    return torch.full((env.num_envs,), float(reward), device=env.device, dtype=torch.float32)

# ==========================================
# 🎨 6. Surface Uniformity
# ==========================================
def surface_uniformity_reward(env, scale: float = 3.0):
    reward = local_obs.get_surface_uniformity_reward_value(env)
    return reward * scale

# ==========================================
# ⚠️ 7. Penalties
# ==========================================
def action_smoothness_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    if not hasattr(env, "action_manager") or not hasattr(env.action_manager, "action") or not hasattr(env.action_manager, "prev_action"):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    action_diff = env.action_manager.action - env.action_manager.prev_action
    return -torch.norm(action_diff, dim=-1)

def machining_safety_penalty(
    env: "ManagerBasedRLEnv",
    max_force: float = 50.0,
    asset_name: str = "robot",
    fixed_joint_name: str = "tool0_to_spindle",
    joint_prim_relpath: str = "joints",
) -> torch.Tensor:
    current_fz = torch.abs(local_obs.get_current_fz(env, asset_name, fixed_joint_name, joint_prim_relpath))
    violation = current_fz > max_force
    return torch.where(
        violation,
        torch.tensor(-20.0, device=env.device, dtype=torch.float32), # 물리 위험 페널티도 강화
        torch.tensor(0.0, device=env.device, dtype=torch.float32),
    )
