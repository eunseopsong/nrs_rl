# SPDX-License-Identifier: BSD-3-Clause
"""
Reward functions module for Adaptive Uniform Polishing.
 
[디벨롭 핵심 원칙]
1. Positive Reinforcement: 페널티보다 목표 달성 시 높은 보상 부여
2. Soft Gate: 하드 boolean 대신 sigmoid로 초기 학습 부트스트랩 보장
3. Curriculum Sigma: 에피소드가 쌓일수록 sigma가 좁아져 ep1 vs 마지막 ep 차이 명확
4. 단순 명료: F * V = C 역비례 관계 학습에 올인
 
[수정 이력]
- Hard boolean gate → soft sigmoid gate (Bug A/B Fix)
  이전: (fz > threshold).float() → reward=0 → gradient=0 → 학습 불가
  이후: sigmoid(sharpness*(fz/threshold-1)) → 초기에도 gradient 흘러 부트스트랩 가능
- mrr_sigma 고정 → curriculum sigma (Bug D Fix)
  이전: mrr_sigma=0.8 고정 → ep1=ep100 허용범위 동일
  이후: sigma_start(1.2)→sigma_end(0.3) 선형 감소 → 에피소드 진행에 따라 정밀도 요구 증가
- visualization.py on_episode_reset()에서 env._ep_curriculum += 1 필수 (Bug C Fix)
"""
 
from __future__ import annotations
 
from typing import TYPE_CHECKING
import importlib
import torch
import numpy as np
 
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
 
local_obs = importlib.import_module("nrs_rl.tasks.manager_based.nrs_rl.mdp.observation")
local_ft_sensor = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.assets.assets.sensors.six_axis_ft_sensor"
)
 
 
# ==========================================
# Helper Functions
# ==========================================
def _get_ee_idx(
    env: "ManagerBasedRLEnv",
    asset_name: str = "robot",
    body_name: str = "spindle_link",
) -> int:
    robot = env.scene[asset_name]
    return int(robot.find_bodies(body_name)[0][0])
 
 
def _get_current_pose_and_velocity(
    env: "ManagerBasedRLEnv",
    asset_name: str = "robot",
    body_name: str = "spindle_link",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    robot = env.scene[asset_name]
    ee_idx = _get_ee_idx(env, asset_name=asset_name, body_name=body_name)
    ee_pos_w = robot.data.body_pos_w[:, ee_idx, :]
    ee_lin_vel_w = robot.data.body_lin_vel_w[:, ee_idx, :]
    current_xyz = ee_pos_w - env.scene.env_origins
    vel_norm = torch.norm(ee_lin_vel_w, dim=-1)
    return current_xyz, robot.data.body_quat_w[:, ee_idx, :], vel_norm
 
 
def _get_current_fz(
    env: "ManagerBasedRLEnv",
    asset_name: str = "robot",
    fixed_joint_name: str = "tool0_to_spindle",
    joint_prim_relpath: str = "joints",
) -> torch.Tensor:
    wrench = local_ft_sensor.get_6axis_ft_fixed_joint(
        env=env,
        asset_name=asset_name,
        fixed_joint_name=fixed_joint_name,
        joint_prim_relpath=joint_prim_relpath,
        verbose=False,
    )
    return wrench[:, 2]
 
 
def _get_current_target_pose(
    env: "ManagerBasedRLEnv",
) -> tuple[torch.Tensor, torch.Tensor]:
    return local_obs._get_target_pose_for_polishing(env)
 
 
# ==========================================
# [Fix] Soft Gate  (Bug A/B 수정 핵심)
# ==========================================
def _soft_gate(
    value: torch.Tensor,
    threshold: float,
    sharpness: float = 8.0,
) -> torch.Tensor:
    """
    하드 boolean 게이트를 sigmoid 소프트 게이트로 대체합니다.
 
    [이전 코드의 문제]
        is_active = (fz > threshold).float()
        → threshold 미만이면 reward = 0, gradient = 0
        → 에이전트가 접촉/이동 방향을 전혀 탐색 못함 → ep1 = ep100
 
    [수정 코드의 효과]
        gate = sigmoid(sharpness * (value/threshold - 1))
        → threshold 미만이어도 0이 아닌 작은 값 → gradient 항상 흐름
        → 초기 학습에서 방향을 찾을 수 있음 (부트스트랩)
        → 학습 진행 → threshold 충족 → gate ≈ 1.0 → full reward
        → 이 과정이 ep1(흩어진 산점도) → 후기(쌍곡선 정렬)의 시각적 변화
    """
    return torch.sigmoid(sharpness * (value / max(threshold, 1e-8) - 1.0))
 
 
# ==========================================
# [Fix] Curriculum Sigma  (Bug D 수정 핵심)
# ==========================================
def _curriculum_sigma(
    env: "ManagerBasedRLEnv",
    sigma_start: float,
    sigma_end: float,
    ramp: int = 200,
) -> float:
    """
    에피소드가 쌓일수록 sigma가 sigma_start → sigma_end로 선형 감소합니다.
 
    [이전 코드의 문제]
        mrr_sigma = 0.8  # 하드코딩 고정
        → ep1이나 ep100이나 허용범위 동일 → 그래프가 같아 보임
 
    [수정 코드의 효과]
        초기 에피소드: sigma 큼 → 대충 해도 보상 → 탐색 장려
        후기 에피소드: sigma 작음 → 정밀해야 보상 → 정밀 제어 강제
        → ep1(흐릿한 MRR) vs 마지막ep(목표선 수렴)의 차이가 명확해짐
 
    ⚠️ 이 함수가 작동하려면 visualization.py의 on_episode_reset()에서
       env._ep_curriculum += 1 이 반드시 실행돼야 합니다.
    """
    if not hasattr(env, "_ep_curriculum"):
        env._ep_curriculum = 0
    progress = float(min(env._ep_curriculum, ramp)) / float(max(ramp, 1))
    return sigma_start + (sigma_end - sigma_start) * progress
 
 
# ==========================================
# 🚀 1. Core Adaptive MRR Reward  [메인 퀘스트]
# ==========================================
def adaptive_mrr_reward(
    env: "ManagerBasedRLEnv",
    target_mrr: float = 0.1,
    mrr_sigma_start: float = 1.5,    # 초기: 넓은 허용 → 탐색 장려
    mrr_sigma_end: float = 0.08,      # 후기: 좁은 허용 → 정밀 제어 강제
    curriculum_ramp: int = 50,
    min_contact_force: float = 1.0,
    min_velocity: float = 1e-3,
    gate_sharpness: float = 8.0,
    asset_name: str = "robot",
    body_name: str = "spindle_link",
    fixed_joint_name: str = "tool0_to_spindle",
    joint_prim_relpath: str = "joints",
) -> torch.Tensor:
    """
    [핵심 긍정 보상] F x V = Target MRR을 달성하면 칭찬합니다.
 
    [힘-속도 역비례 원리]
    힘이 강하면 속도를 낮추고, 힘이 약하면 속도를 높여야 MRR이 일정해집니다.
    에이전트는 이 관계를 보상 최대화를 통해 스스로 학습합니다.
 
    [에피소드 진행에 따른 변화]
    초기: sigma=1.2(넓음) + gate 소프트 → 대충해도 보상 → 방향 탐색
    후기: sigma=0.3(좁음) + gate 강화   → 정밀해야 보상 → 정확도 수렴
    → 06_mrr_tracking_error.png에서 빨간 면적이 에피소드마다 줄어드는 시각적 증거
    """
    current_fz = torch.abs(
        _get_current_fz(env, asset_name, fixed_joint_name, joint_prim_relpath)
    )
    _, _, current_vel_norm = _get_current_pose_and_velocity(env, asset_name, body_name)
 
    # [Bug A Fix] soft gate: 0/1 하드 마스크 → 0.0~1.0 연속값
    force_gate = _soft_gate(current_fz, min_contact_force, gate_sharpness)
    vel_gate = _soft_gate(current_vel_norm, min_velocity, gate_sharpness)
    activity_gate = force_gate * vel_gate
 
    # MRR 오차 (log 스케일: 극단값에 강건)
    current_mrr = current_fz * current_vel_norm
    log_ratio = torch.log(
        torch.clamp(current_mrr / max(target_mrr, 1e-6), min=1e-4, max=1e4)
    )
 
    # [Bug D Fix] 커리큘럼 sigma: 에피소드마다 허용범위 축소
    sigma = _curriculum_sigma(env, mrr_sigma_start, mrr_sigma_end, curriculum_ramp)
 
    reward = torch.exp(-torch.square(log_ratio / sigma))
    return reward * activity_gate
 
 
# ==========================================
# 🚀 2. Inverse Control Bonus  [반비례 깨달음 보너스]
# ==========================================
def inverse_fv_bonus(
    env: "ManagerBasedRLEnv",
    target_mrr: float = 0.1,
    bonus_sigma_start: float = 1.0,  # 초기: 넓게
    bonus_sigma_end: float = 0.05,   # 후기: 좁게
    curriculum_ramp: int = 200,
    min_contact_force: float = 1.0,
    gate_sharpness: float = 8.0,
    asset_name: str = "robot",
    body_name: str = "spindle_link",
    fixed_joint_name: str = "tool0_to_spindle",
    joint_prim_relpath: str = "joints",
) -> torch.Tensor:
    """
    에이전트에게 "힘이 강하면 속도를 늦춰!"라는 정답을 직접적으로 떠먹여주는 보너스.
 
    [시각적 효과 — 05_force_vel_correlation.png]
    초기 에피소드: 점들이 무작위 산포
    후기 에피소드: 점들이 V = target_mrr / F 쌍곡선 위에 정렬
    → 이 패턴 변화가 "학습이 됐다"는 가장 직관적인 시각 증거
 
    [Bug B Fix] 하드 .float() → soft gate
    [Bug D Fix] bonus_sigma 고정 → 커리큘럼 sigma
    """
    current_fz = torch.abs(
        _get_current_fz(env, asset_name, fixed_joint_name, joint_prim_relpath)
    )
    _, _, current_vel_norm = _get_current_pose_and_velocity(env, asset_name, body_name)
 
    # [Bug B Fix] soft gate
    force_gate = _soft_gate(current_fz, min_contact_force, gate_sharpness)
 
    # 이상적인 속도: V_ideal = target_mrr / |Fz|
    ideal_velocity = target_mrr / torch.clamp(current_fz, min=1e-3)
    log_vel_ratio = torch.log(
        torch.clamp(
            current_vel_norm / torch.clamp(ideal_velocity, min=1e-5),
            min=1e-4, max=1e4,
        )
    )
 
    # [Bug D Fix] 커리큘럼 sigma
    sigma = _curriculum_sigma(env, bonus_sigma_start, bonus_sigma_end, curriculum_ramp)
 
    bonus = torch.exp(-torch.square(log_vel_ratio / sigma))
    return bonus * force_gate
 
 
# ==========================================
# 🚀 3. Trajectory Tracking Reward  [긍정적 궤적 유도]
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
    """
    경로를 잘 따라가면 주는 점수.
 
    [Bug C Fix] 속도 유도 항 추가
    이전: 위치 오차만 → 목표점 근처 정지해도 높은 보상 → 멈춤 전략 허용
    이후: pos_reward + vel_bonus 결합
      → 정지 상태에서는 vel_bonus ≈ 0 → 총 보상 낮음 → 멈춤 불리
      → 움직이면서 따라가야 최대 보상 → 이동 동기 부여
    """
    current_xyz, _, current_vel_norm = _get_current_pose_and_velocity(
        env, asset_name, body_name
    )
    target_xyz, _ = _get_current_target_pose(env)
 
    # 위치 보상 (기존 유지)
    pos_error = torch.norm(current_xyz - target_xyz, dim=-1)
    pos_reward = torch.exp(-torch.square(pos_error / pos_sigma))
 
    # 속도 유도 항: 움직이고 있을 때 추가 보상
    vel_gate = _soft_gate(current_vel_norm, min_tracking_velocity, gate_sharpness)
    vel_bonus = vel_bonus_scale * vel_gate
 
    return pos_reward + vel_bonus
 
 
# ==========================================
# ⚠️ 4. Action Smoothness Penalty  [최소한의 페널티]
# ==========================================
def action_smoothness_penalty(
    env: "ManagerBasedRLEnv",
) -> torch.Tensor:
    """L자형 극단 제어 방지. 액션 변화가 클 때만 살짝 감점."""
    if not hasattr(env, "action_manager"):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if not hasattr(env.action_manager, "action") or not hasattr(env.action_manager, "prev_action"):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
 
    action_diff = env.action_manager.action - env.action_manager.prev_action
    return -torch.norm(action_diff, dim=-1)
 
 
# ==========================================
# ⚠️ 5. Safety Limit Penalty  [물리엔진 보호용]
# ==========================================
def machining_safety_penalty(
    env: "ManagerBasedRLEnv",
    max_force: float = 50.0,
    asset_name: str = "robot",
    fixed_joint_name: str = "tool0_to_spindle",
    joint_prim_relpath: str = "joints",
) -> torch.Tensor:
    """물리 엔진 과부하 차단. max_force 초과 시 강한 페널티."""
    current_fz = torch.abs(
        _get_current_fz(env, asset_name, fixed_joint_name, joint_prim_relpath)
    )
    violation = current_fz > max_force
    return torch.where(
        violation,
        torch.tensor(-10.0, device=env.device, dtype=torch.float32),
        torch.tensor(0.0, device=env.device, dtype=torch.float32),
    )

def _get_surface_uniformity_reward(env):

    vis_module = importlib.import_module(
        "nrs_rl.tasks.manager_based.nrs_rl.utils.visualization"
    )
    grid = vis_module._surface_grid
    valid = grid[grid > 0]
    if len(valid) < 10:
        return torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.float32,
        )
    std = float(np.std(valid))
    reward = 1.0 / (1.0 + std)
    return torch.full(
        (env.num_envs,),
        reward,
        device=env.device,
        dtype=torch.float32,
    )

def surface_uniformity_reward(
    env,
    scale: float = 3.0,
):
    reward = _get_surface_uniformity_reward(
        env
    )
    return reward * scale

def force_stability_reward(
    env,
    target_force: float = 10.0,
):
    current_fz = torch.abs(
        _get_current_fz(env)
    )
    error = torch.abs(
        current_fz - target_force
    )
    reward = torch.exp(
        -0.1 * error
    )
    return reward