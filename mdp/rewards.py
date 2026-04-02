# SPDX-License-Identifier: BSD-3-Clause
"""
Reward functions module for Uniform Polishing.
[26.04.02 업데이트] 
- 프레스턴 방정식 기반 MRR 균일화 보상 추가
- 6축 FT 센서 데이터 및 Cartesian 선속도 연동
- Look-ahead 웨이포인트를 활용한 코너링 감속/힘 조절 유도
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# =========================================================================
# 1. Uniform Material Removal Rate (MRR) Reward
# =========================================================================
def uniform_mrr_reward(
    env: ManagerBasedRLEnv, 
    target_force: float, 
    target_velocity: float,
    mrr_sigma: float = 10.0
) -> torch.Tensor:
    """
    [26.04.02] 가공량 균일화 보상 (Preston's Equation 반영)
    debug.py에서 확인한 '현재 Z축 힘(Fz)'과 'Cartesian 선속도(vel_magnitude)'의 곱이
    목표 가공률(target_force * target_velocity)과 일치하도록 유도합니다.
    """
    # 1. 6축 FT 센서에서 Z축 힘 추출 (Isaac Lab Scene 구조 참조)
    # env.scene["ft_sensor"].data.net_forces_w 형태를 가정. (N, 3) 텐서 중 Z축(2)
    current_fz = env.scene["ft_sensor"].data.net_forces_w[:, 2] 
    
    # 2. EE(엔드이펙터)의 Cartesian 공간 선속도 (debug.py의 vel_magnitude와 동일한 물리량)
    # env.scene["robot"].data.body_lin_vel_w 에서 EE 링크의 속도를 가져와 크기(Norm) 계산
    ee_lin_vel = env.scene["robot"].data.body_lin_vel_w[:, env.unwrapped.ee_link_idx, :]
    current_vel_norm = torch.norm(ee_lin_vel, dim=-1)
    
    # 3. 현재의 MRR 대리값 (Proxy) = Fz * V
    current_mrr_proxy = torch.abs(current_fz) * current_vel_norm
    target_mrr_proxy = target_force * target_velocity
    
    # 4. 오차에 대한 가우시안 보상
    mrr_error = torch.abs(current_mrr_proxy - target_mrr_proxy)
    return torch.exp(-mrr_sigma * torch.square(mrr_error))


# =========================================================================
# 2. Force Tracking Reward
# =========================================================================
def force_tracking_reward(
    env: ManagerBasedRLEnv, 
    target_force: float, 
    force_sigma: float = 0.5
) -> torch.Tensor:
    """
    [26.04.02] 목표 접촉 힘 추종 보상
    직진/코너링 상관없이 표면에 누르는 힘 자체가 target_force를 유지하도록 유도합니다.
    """
    current_fz = env.scene["ft_sensor"].data.net_forces_w[:, 2]
    
    # 부호 주의: 센서 좌표계에 따라 누르는 힘이 음수일 수 있으므로 절댓값 또는 방향에 맞게 보정
    force_error = torch.abs(torch.abs(current_fz) - target_force)
    
    return torch.exp(-force_sigma * torch.square(force_error))


# =========================================================================
# 3. Look-ahead Cornering Smoothness Penalty
# =========================================================================
def lookahead_cornering_penalty(
    env: ManagerBasedRLEnv,
    cornering_threshold_angle: float = 0.5, # 라디안 기준 (약 30도)
    penalty_scale: float = 5.0
) -> torch.Tensor:
    """
    [26.04.02] 코너링 예방 보상 (Look-ahead 반영)
    현재 진행 방향과 Look-ahead(미래 웨이포인트) 방향의 각도 차이가 클 때(코너링이 다가올 때),
    속도가 너무 빠르거나 힘이 튀면 강한 페널티를 부여합니다.
    """
    # 현재 EE 위치 및 Look-ahead 타겟 위치
    current_xyz = env.scene["robot"].data.body_pos_w[:, env.unwrapped.ee_link_idx, :]
    target_xyz = env.command_manager.get_command("trajectory")[:, :3] # 현재 타겟
    lookahead_xyz = env.command_manager.get_command("lookahead")[:, :3] # 미래 타겟 (경로 가공량/오프셋이 반영된 좌표)

    # 진행 방향 벡터
    current_dir = target_xyz - current_xyz
    future_dir = lookahead_xyz - target_xyz
    
    # 정규화
    current_dir_norm = torch.nn.functional.normalize(current_dir, dim=-1)
    future_dir_norm = torch.nn.functional.normalize(future_dir, dim=-1)
    
    # 코사인 유사도를 이용한 방향 전환 각도 계산
    cos_sim = torch.sum(current_dir_norm * future_dir_norm, dim=-1)
    angle_diff = torch.acos(torch.clamp(cos_sim, -1.0 + 1e-6, 1.0 - 1e-6))
    
    # 코너링 상황(각도가 큰 경우) 판단 마스크
    is_cornering = angle_diff > cornering_threshold_angle
    
    # 코너링 구간에서 속도가 높거나, Action의 변화율이 크면 페널티
    ee_lin_vel = env.scene["robot"].data.body_lin_vel_w[:, env.unwrapped.ee_link_idx, :]
    current_vel_norm = torch.norm(ee_lin_vel, dim=-1)
    
    # 코너에서 속도가 높을수록 페널티 증가
    corner_penalty = is_cornering.float() * current_vel_norm * penalty_scale
    
    return -corner_penalty


# =========================================================================
# 4. Trajectory Tracking Penalty (Position & Rotation Error)
# =========================================================================
def trajectory_tracking_penalty(
    env: ManagerBasedRLEnv, 
    pos_sigma: float = 100.0,
    rot_sigma: float = 50.0
) -> torch.Tensor:
    """
    [26.04.02] 경로 이탈 페널티
    debug.py에서 출력하는 pos_norm, rot_norm 값과 직접적으로 연결되는 보상입니다.
    """
    current_xyz = env.scene["robot"].data.body_pos_w[:, env.unwrapped.ee_link_idx, :]
    current_quat = env.scene["robot"].data.body_quat_w[:, env.unwrapped.ee_link_idx, :]
    
    # Command Manager에서 생성된 현재 궤적(Target Pose)
    target_pose = env.command_manager.get_command("trajectory")
    target_xyz = target_pose[:, :3]
    target_quat = target_pose[:, 3:7] # Quaternion (w, x, y, z)
    
    # 위치 오차 (pos_err_norm)
    pos_error_norm = torch.norm(current_xyz - target_xyz, dim=-1)
    
    # 회전 오차 (내적을 이용한 쿼터니언 거리)
    quat_inner = torch.abs(torch.sum(current_quat * target_quat, dim=-1))
    rot_error_norm = 2.0 * torch.acos(torch.clamp(quat_inner, -1.0 + 1e-6, 1.0 - 1e-6))
    
    # 오차가 커질수록 음수(페널티) 부과
    pos_penalty = -torch.exp(pos_sigma * torch.square(pos_error_norm)) + 1.0
    rot_penalty = -torch.exp(rot_sigma * torch.square(rot_error_norm)) + 1.0
    
    return pos_penalty + rot_penalty


# =========================================================================
# 5. Action Smoothness (Chattering Prevention)
# =========================================================================
def action_smoothness_penalty(
    env: ManagerBasedRLEnv
) -> torch.Tensor:
    """
    [26.04.02] 제어 입력 부드러움 페널티
    로봇이 들쑥날쑥하게 힘을 가하는 것을 방지합니다. 
    이전 Action과 현재 Action의 변화량(L2 Norm)을 페널티로 반환합니다.
    """
    action_diff = env.action_manager.action - env.action_manager.prev_action
    
    return -torch.norm(action_diff, dim=-1)