# SPDX-License-Identifier: BSD-3-Clause
"""
Reward functions module for Adaptive Uniform Polishing.

[디벨롭 핵심 원칙]
1. Positive Reinforcement: 페널티(감점)보다는 목표 달성 시 높은 보상(가점) 부여
2. 멈춤 방지: 페널티가 무서워서 멈추는 현상을 막기 위해 Active Mask 도입
3. 단순 명료: 복잡한 커리큘럼/코너링 제거, F * V = C 역비례 관계 학습에 올인
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import importlib
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

local_obs = importlib.import_module("nrs_rl.tasks.manager_based.nrs_rl.mdp.observation")
local_ft_sensor = importlib.import_module("nrs_rl.tasks.manager_based.nrs_rl.assets.assets.sensors.six_axis_ft_sensor")

# ==========================================
# Helper Functions (최소화)
# ==========================================
def _get_ee_idx(env: "ManagerBasedRLEnv", asset_name: str = "robot", body_name: str = "spindle_link") -> int:
    robot = env.scene[asset_name]
    return int(robot.find_bodies(body_name)[0][0])

def _get_current_pose_and_velocity(env: "ManagerBasedRLEnv", asset_name: str = "robot", body_name: str = "spindle_link") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    robot = env.scene[asset_name]
    ee_idx = _get_ee_idx(env, asset_name=asset_name, body_name=body_name)
    ee_pos_w = robot.data.body_pos_w[:, ee_idx, :]
    ee_lin_vel_w = robot.data.body_lin_vel_w[:, ee_idx, :]
    env_origins = env.scene.env_origins
    
    current_xyz = ee_pos_w - env_origins
    vel_norm = torch.norm(ee_lin_vel_w, dim=-1)
    return current_xyz, robot.data.body_quat_w[:, ee_idx, :], vel_norm

def _get_current_fz(env: "ManagerBasedRLEnv", asset_name: str = "robot", fixed_joint_name: str = "tool0_to_spindle", joint_prim_relpath: str = "joints") -> torch.Tensor:
    wrench = local_ft_sensor.get_6axis_ft_fixed_joint(
        env=env, asset_name=asset_name, fixed_joint_name=fixed_joint_name, joint_prim_relpath=joint_prim_relpath, verbose=False
    )
    return wrench[:, 2]

def _get_current_target_pose(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    return local_obs._get_target_pose_for_polishing(env)

# ==========================================
# 🚀 1. Core Adaptive MRR Reward [메인 퀘스트]
# ==========================================
def adaptive_mrr_reward(
    env: "ManagerBasedRLEnv",
    target_mrr: float = 0.1,
    mrr_sigma: float = 0.8,       # 넓은 허용 범위 (이전의 문제 해결)
    min_contact_force: float = 1.0,
    min_velocity: float = 1e-3,
    asset_name: str = "robot",
    body_name: str = "spindle_link",
    fixed_joint_name: str = "tool0_to_spindle",
    joint_prim_relpath: str = "joints",
) -> torch.Tensor:
    """
    [핵심 긍정 보상] F x V = Target MRR 을 달성하면 칭찬합니다.
    페널티 대신, 아예 접촉 안 하거나 안 움직이면 보상을 '0점' 줘서 꼼수를 막습니다.
    """
    current_fz = torch.abs(_get_current_fz(env, asset_name, fixed_joint_name, joint_prim_relpath))
    _, _, current_vel_norm = _get_current_pose_and_velocity(env, asset_name, body_name)

    # 1. 꼼수 방지용 Active Mask: 힘을 주면서 & 움직이고 있어야만 보상 자격 획득
    is_active = (current_fz > min_contact_force) & (current_vel_norm > min_velocity)
    
    # 2. MRR 오차 계산 (Log 비율 사용으로 튀는 값 방지)
    current_mrr = current_fz * current_vel_norm
    ratio = current_mrr / max(target_mrr, 1e-6)
    log_ratio = torch.log(torch.clamp(ratio, min=1e-4, max=1e4))
    
    # 3. 가우시안 보상 산출 (0 ~ 1 사이의 점수)
    reward = torch.exp(-torch.square(log_ratio / mrr_sigma))
    
    # 활성화 상태가 아니면 얄짤없이 0점
    return reward * is_active.float()

# ==========================================
# 🚀 2. Inverse Control Bonus [반비례 깨달음 보너스]
# ==========================================
def inverse_fv_bonus(
    env: "ManagerBasedRLEnv",
    target_mrr: float = 0.1,
    bonus_sigma: float = 0.5,
    min_contact_force: float = 1.0,
    asset_name: str = "robot",
    body_name: str = "spindle_link",
    fixed_joint_name: str = "tool0_to_spindle",
    joint_prim_relpath: str = "joints",
) -> torch.Tensor:
    """
    에이전트에게 "힘이 강하면 속도를 늦춰!"라는 정답을 직접적으로 떠먹여주는 보너스.
    이게 05번 그래프의 예쁜 곡선을 만들어냅니다.
    """
    current_fz = torch.abs(_get_current_fz(env, asset_name, fixed_joint_name, joint_prim_relpath))
    _, _, current_vel_norm = _get_current_pose_and_velocity(env, asset_name, body_name)

    is_contacting = (current_fz > min_contact_force).float()

    # 현재 힘을 기준으로 내가 내야 하는 '이상적인 속도' 계산
    ideal_velocity = target_mrr / torch.clamp(current_fz, min=1e-3)
    
    # 실제 속도와 이상적 속도의 일치율 확인
    vel_ratio = current_vel_norm / torch.clamp(ideal_velocity, min=1e-5)
    log_vel_ratio = torch.log(torch.clamp(vel_ratio, min=1e-4, max=1e4))
    
    bonus = torch.exp(-torch.square(log_vel_ratio / bonus_sigma))
    
    return bonus * is_contacting

# ==========================================
# 🚀 3. Trajectory Tracking Reward [긍정적 궤적 유도]
# ==========================================
def trajectory_tracking_reward(
    env: "ManagerBasedRLEnv",
    pos_sigma: float = 0.05,
    asset_name: str = "robot",
    body_name: str = "spindle_link",
) -> torch.Tensor:
    """
    기존의 무서운 '이탈 페널티'를 없애고, '경로를 잘 따라가면 주는 점수'로 바꿨습니다.
    에이전트가 멈춰있지 않고 목표점을 향해 적극적으로 이동하게 만듭니다.
    """
    current_xyz, _, _ = _get_current_pose_and_velocity(env, asset_name, body_name)
    target_xyz, _ = _get_current_target_pose(env)

    pos_error = torch.norm(current_xyz - target_xyz, dim=-1)
    # 목표점에 가까울수록 1.0에 가까운 보상 획득
    reward = torch.exp(-torch.square(pos_error / pos_sigma))
    
    return reward

# ==========================================
# ⚠️ 4. Action Smoothness Penalty [유일하게 남긴 최소한의 페널티]
# ==========================================
def action_smoothness_penalty(
    env: "ManagerBasedRLEnv",
) -> torch.Tensor:
    """
    L자형 그래프(극단적 제어)를 막기 위해, 명령(Action)을 너무 확확 바꾸는 것만 살짝 감점합니다.
    """
    if not hasattr(env, "action_manager") or not hasattr(env.action_manager, "action"):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    action_diff = env.action_manager.action - env.action_manager.prev_action
    return -torch.norm(action_diff, dim=-1)

# ==========================================
# ⚠️ 5. Safety Limit Penalty [물리엔진 보호용]
# ==========================================
def machining_safety_penalty(
    env: "ManagerBasedRLEnv", 
    max_force: float = 50.0, 
    asset_name: str = "robot", 
    fixed_joint_name: str = "tool0_to_spindle", 
    joint_prim_relpath: str = "joints"
) -> torch.Tensor:
    """물리 엔진 과부하를 막기 위해 비정상적인 융단폭격(과도한 힘)을 차단합니다."""
    current_fz = torch.abs(_get_current_fz(env, asset_name, fixed_joint_name, joint_prim_relpath))
    
    # max_force를 초과하면 아주 큰 페널티(-10.0) 부여
    violation = current_fz > max_force
    return torch.where(
        violation, 
        torch.tensor(-10.0, device=env.device, dtype=torch.float32), 
        torch.tensor(0.0, device=env.device, dtype=torch.float32)
    )