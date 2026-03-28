# SPDX-License-Identifier: BSD-3-Clause
# shangus
"""
v28: Adaptive Polishing Intelligence (Position + Force + Smoothness)
----------------------------------------------------------------------------------
- Reward Type: Exponential kernel (no tanh)
- Goal: Joint-space + Cartesian-space 병렬 학습 + Orientation wrap-safe error 계산
        + 시각화 시 np.unwrap 적용 (roll/pitch/yaw 연속 표시)
        + [NEW] 적응형 힘 제어 보상 (Adaptive Force Tracking)
        + [NEW] 진동 억제 페널티 (Action Smoothness Penalty)

Notes (port to nrs_rl):
    - imports: local .observations
    - target_vel uses env dt * decimation
    - angle wrap is computed in torch
"""

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import os
import numpy as np
import torch
import datetime  # ✅ [26.03.01. 추가] 시간 모듈 임포트 (폴더명 생성용)
import atexit    # ✅ [26.03.01. 추가] 프로그램 종료 시 요약 출력을 위한 모듈

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ✅ nrs_rl 구조: 같은 mdp 폴더의 observations.py
from .observations import (
    get_hdf5_target_joints,
    get_hdf5_target_positions,
    get_ee_pose,
    get_contact_forces,  # ✅ [26.02.24. 추가] 힘 센서 데이터 가져오기
)

# -----------------------------------------------------------
# Global
# -----------------------------------------------------------
version = "v28"

# -----------------------------------------------------------
# [26.03.01. 추가] 훈련 실행 시각 기반 고유 폴더명 생성 변수
# 변경 사유: 매 훈련(Run)마다 그래프 결과가 덮어씌워지는 것을 방지하고, 실험 이력을 날짜/시간별로 독립적으로 보존하기 위함.
# -----------------------------------------------------------
_run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

_joint_tracking_history = []
_joint_reward_history = []
_position_tracking_history = []
_position_reward_history = []

_episode_counter_joint = 0
_episode_counter_position = 0

# [26.02.24. 추가] 파라미터 탐색용 최고 보상 및 파라미터 추적 변수
_best_joint_reward = -np.inf
_best_joint_episode = -1
_best_position_reward = -np.inf
_best_position_episode = -1
_current_episode_params = {}

# -----------------------------------------------------------
# Utility: angle wrap correction (torch, GPU-safe)
# -----------------------------------------------------------
def angle_diff_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute minimal difference between two angles (radians), wrapped to [-pi, pi].
    a, b: (..., 3) tensor
    """
    two_pi = 2.0 * np.pi
    return torch.remainder(a - b + np.pi, two_pi) - np.pi

# -----------------------------------------------------------
# (1) Joint Tracking Reward
# -----------------------------------------------------------
def joint_tracking_reward(env: "ManagerBasedRLEnv"):
    """Joint-space tracking reward (exponential kernel)"""
    robot = env.scene["robot"]
    q, qd = robot.data.joint_pos[:, :6], robot.data.joint_vel[:, :6]

    dt = getattr(env.sim, "dt", 1.0 / 30.0) * getattr(env, "decimation", 1)
    D = q.shape[1]
    step = int(env.common_step_counter)

    fut = get_hdf5_target_joints(env, horizon=8)  # (N, 8*D)
    q_star_curr, q_star_next = fut[:, :D], fut[:, D:2 * D]
    qd_star = (q_star_next - q_star_curr) / (dt + 1e-8)

    e_q, e_qd = q - q_star_next, qd - qd_star

    wj = torch.tensor([1.0, 2.0, 1.0, 4.0, 1.0, 1.0], device=q.device).unsqueeze(0)
    k_pos = torch.tensor([1.0, 8.0, 2.0, 6.0, 2.0, 2.0], device=q.device).unsqueeze(0)
    k_vel = torch.tensor([0.10, 0.40, 0.10, 0.40, 0.10, 0.10], device=q.device).unsqueeze(0)

    e_q2, e_qd2 = wj * (e_q ** 2), wj * (e_qd ** 2)
    r_pose_jointwise = torch.exp(-k_pos * e_q2)
    r_vel_jointwise = torch.exp(-k_vel * e_qd2)

    r_pose, r_vel = r_pose_jointwise.sum(dim=1), r_vel_jointwise.sum(dim=1)
    total = 0.9 * r_pose + 0.1 * r_vel

    if step % 10 == 0:
        print(f"[Joint Step {step}] mean(|e_q|)={torch.norm(e_q, dim=1).mean():.3f}, total={total.mean():.3f}")
        mean_e_q_abs = torch.mean(torch.abs(e_q), dim=0).detach().cpu().numpy()
        mean_r_pose = torch.mean(r_pose_jointwise, dim=0).detach().cpu().numpy()
        for j in range(D):
            print(f"  joint{j+1}: |mean(e_q)|={mean_e_q_abs[j]:.3f}, r_pose={mean_r_pose[j]:.3f}")

    _joint_tracking_history.append((step, q_star_next[0].detach().cpu().numpy(), q[0].detach().cpu().numpy()))
    _joint_reward_history.append((step, r_pose_jointwise[0].detach().cpu().numpy()))

    # -----------------------------------------------------------
    # [26.03.01. 수정] 그래프 저장 트리거 조건 변경 (안전 장치)
    # 변경 사유: 에피소드 조기 종료 시 저장이 누락되는 문제를 해결하기 위해, 에피소드 길이에 의존하지 않고 300 스텝마다 강제로 저장하도록 변경.
    # -----------------------------------------------------------
    save_interval = 300
    if step > 0 and (step % save_interval == 0):
        print(f"📸 [Joint] {step} 스텝 도달! 그래프 저장을 시도합니다...")
        save_episode_plots_joint(step)

    return total

# -----------------------------------------------------------
# (2) Position Tracking Reward (6D + velocity)
# -----------------------------------------------------------
def position_tracking_reward(env: "ManagerBasedRLEnv"):
    """6D EE pose + velocity tracking reward (wrap-safe orientation)"""
    device = env.device
    step = int(env.common_step_counter)

    dt = getattr(env.sim, "dt", 1.0 / 30.0) * getattr(env, "decimation", 1)

    # (1) FK 기반 EE pose: (N,6) [x,y,z,roll,pitch,yaw]
    ee_pose = get_ee_pose(env)

    robot = env.scene["robot"]
    wrist_id = robot.find_bodies("wrist_3_link")[0]

    ee_vel = robot.data.body_lin_vel_w[:, wrist_id, :]  # (N,3) or (N,1,3)
    ee_ang = robot.data.body_ang_vel_w[:, wrist_id, :]
    if ee_vel.ndim == 3:
        ee_vel = ee_vel.squeeze(1)
    if ee_ang.ndim == 3:
        ee_ang = ee_ang.squeeze(1)

    ee_vel6d = torch.cat([ee_vel, ee_ang], dim=1)  # (N,6)

    # (2) HDF5 target (2-step horizon): fut = (N, 12)
    fut = get_hdf5_target_positions(env, horizon=2)
    if fut.ndim == 3:
        fut = fut.squeeze(1)

    target_curr, target_next = fut[:, :6], fut[:, 6:12]
    target_vel = (target_next - target_curr) / (dt + 1e-8)

    # (3) wrap-safe orientation diff (torch, GPU)
    e_pose = ee_pose.clone()
    e_pose[:, :3] = e_pose[:, :3] - target_next[:, :3]
    e_pose[:, 3:6] = angle_diff_torch(ee_pose[:, 3:6], target_next[:, 3:6])

    # (4) velocity error
    e_vel = ee_vel6d - target_vel

    # (5) reward
    w = torch.tensor([1.0, 2.0, 2.0, 1.0, 1.0, 1.0], device=device).unsqueeze(0)
    k_pose = torch.tensor([8.0, 32.0, 32.0, 4.0, 4.0, 4.0], device=device).unsqueeze(0)
    k_vel  = torch.tensor([0.2, 0.05, 0.05, 0.1, 0.1, 0.1], device=device).unsqueeze(0)

    r_pose_axiswise = torch.exp(-k_pose * (w * e_pose) ** 2)
    r_vel_axiswise  = torch.exp(-k_vel  * (w * e_vel) ** 2)
   
    r_pose = torch.mean(r_pose_axiswise, dim=1)
    r_vel  = torch.mean(r_vel_axiswise, dim=1)
    reward = 0.9 * r_pose + 0.1 * r_vel

    # (6) 기록 및 로그
    global _position_tracking_history, _position_reward_history
    _position_tracking_history.append(
        (step, target_next[0].detach().cpu().numpy(), ee_pose[0].detach().cpu().numpy())
    )
    _position_reward_history.append((step, float(reward[0].detach().cpu().item())))

    if step % 10 == 0:
        mean_e_pose = torch.mean(torch.abs(e_pose), dim=0).detach().cpu().numpy()
        mean_e_vel  = torch.mean(torch.abs(e_vel), dim=0).detach().cpu().numpy()
        mean_r_pose = torch.mean(r_pose_axiswise, dim=0).detach().cpu().numpy()
        mean_r_vel  = torch.mean(r_vel_axiswise, dim=0).detach().cpu().numpy()
        print(
            f"[Position Step {step}] |e_pose|={torch.norm(e_pose, dim=1).mean():.4f}, "
            f"|e_vel|={torch.norm(e_vel, dim=1).mean():.4f}, total={reward.mean():.4f}"
        )
        labels = ["x", "y", "z", "roll", "pitch", "yaw"]
        for i in range(6):
            print(
                f"  {labels[i]:<6} | e_pose={mean_e_pose[i]:+6.4f} | e_vel={mean_e_vel[i]:+6.4f} | "
                f"r_pose={mean_r_pose[i]:.4f} | r_vel={mean_r_vel[i]:.4f}"
            )

    # -----------------------------------------------------------
    # [26.03.01. 수정] 그래프 저장 트리거 조건 변경 (안전 장치)
    # 변경 사유: 에피소드 조기 종료 시 저장이 누락되는 문제를 해결하기 위해, 에피소드 길이에 의존하지 않고 300 스텝마다 강제로 저장하도록 변경.
    # -----------------------------------------------------------
    save_interval = 300
    if step > 0 and (step % save_interval == 0):
        print(f"📸 [Position] {step} 스텝 도달! 그래프 저장을 시도합니다...")
        save_episode_plots_position(step)

    return reward

# -----------------------------------------------------------
# Visualization & Best Episode Tracking
# -----------------------------------------------------------
def save_episode_plots_joint(step: int):
    global _joint_tracking_history, _joint_reward_history, _episode_counter_joint
    global _best_joint_reward, _best_joint_episode, _current_episode_params, _run_timestamp
 
    # [안전 장치] 기록된 데이터가 없으면 함수 종료
    if not _joint_tracking_history or not _joint_reward_history:
        return

    # -----------------------------------------------------------
    # [26.03.01. 수정] 타임스탬프가 포함된 경로로 저장 디렉토리 변경
    # 변경 사유: 이전 훈련 결과 보존을 위해 실행 시점(run_YYYYMMDD_HHMMSS) 기반의 독립된 상위 폴더 내에 png/rewards 폴더를 생성하도록 변경.
    # -----------------------------------------------------------
    save_dir = os.path.expanduser(f"~/nrs_rl/outputs/run_{_run_timestamp}/png/")
    reward_dir = os.path.expanduser(f"~/nrs_rl/outputs/run_{_run_timestamp}/rewards/")
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(reward_dir, exist_ok=True)

    steps, targets, currents = zip(*_joint_tracking_history)
    targets, currents = np.vstack(targets), np.vstack(currents)
    colors = ["r", "g", "b", "orange", "purple", "gray"]

    # 1. Tracking Plot
    plt.figure(figsize=(10, 6))
    for j in range(targets.shape[1]):
        # ✅ X축(steps) 동기화 반영
        plt.plot(steps, targets[:, j], "--", color=colors[j], label=f"Target q{j+1}")
        plt.plot(steps, currents[:, j], "-",  color=colors[j], label=f"Current q{j+1}")
    # ✅ 범례 위치 수정 (좌측 상단)
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.title(f"Joint Tracking ({version})")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"joint_tracking_{version}_ep{_episode_counter_joint+1}.png"))
    plt.close()

    # 2. Reward Plot & Best Logic
    r_steps, r_values = zip(*_joint_reward_history)
    r_values_arr = np.array(r_values)
  
    # [수정됨] 명시적으로 총 보상 계산 및 변수 할당
    episode_total_reward = float(np.sum(r_values_arr))

    # [26.02.24. 추가] 최고 보상 갱신 체크 (Joint)
    if episode_total_reward > _best_joint_reward:
        _best_joint_reward = episode_total_reward
        _best_joint_episode = _episode_counter_joint + 1
        print("\n" + "★"*50)
        print(f"🎉 [NEW BEST JOINT EPISODE] Episode {_best_joint_episode} 🎉")
        print(f"Total Joint Reward: {episode_total_reward:.4f}")
        print(f"Applied Params: {_current_episode_params}")
        print("★"*50 + "\n")

    plt.figure(figsize=(10, 5))
    plt.plot(r_steps, r_values_arr, "k", linewidth=2.0, label="Total Reward")
    plt.legend()
    plt.grid(True)
    plt.title(f"Joint Reward ({version}) - Ep Total: {episode_total_reward:.2f}")
    plt.tight_layout()
    plt.savefig(os.path.join(reward_dir, f"r_total_joint_{version}_ep{_episode_counter_joint+1}.png"))
    plt.close()

    _joint_tracking_history.clear()
    _joint_reward_history.clear()
    _episode_counter_joint += 1


def save_episode_plots_position(step: int):
    global _position_tracking_history, _position_reward_history, _episode_counter_position
    global _best_position_reward, _best_position_episode, _current_episode_params, _run_timestamp
    # [안전 장치] 기록된 데이터가 없으면 함수 종료
    if not _position_tracking_history or not _position_reward_history:
        return

    # -----------------------------------------------------------
    # [26.03.01. 수정] 타임스탬프가 포함된 경로로 저장 디렉토리 변경
    # 변경 사유: 위 joint 함수와 동일하게 Position 훈련 결과도 독립 보존.
    # -----------------------------------------------------------
    save_dir = os.path.expanduser(f"~/nrs_rl/outputs/run_{_run_timestamp}/png/")
    reward_dir = os.path.expanduser(f"~/nrs_rl/outputs/run_{_run_timestamp}/rewards/")
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(reward_dir, exist_ok=True)

    steps, targets, currents = zip(*_position_tracking_history)
    targets, currents = np.vstack(targets), np.vstack(currents)

    targets[:, 3:6]  = np.unwrap(targets[:, 3:6], axis=0)
    currents[:, 3:6] = np.unwrap(currents[:, 3:6], axis=0)

    labels = ["x", "y", "z", "roll", "pitch", "yaw"]
    colors = ["r", "g", "b", "orange", "purple", "gray"]

    # 1. Tracking Plot
    plt.figure(figsize=(12, 8))
    for j in range(6):
        # ✅ X축(steps) 동기화 반영
        plt.plot(steps, targets[:, j], "--", color=colors[j], label=f"Target {labels[j]}")
        plt.plot(steps, currents[:, j], "-",  color=colors[j], label=f"Current {labels[j]}")
    # ✅ 범례 위치 수정 (좌측 상단, 3열 배치)
    plt.legend(ncol=3, loc="upper left")
    plt.grid(True)
    plt.title(f"EE 6D Pose Tracking ({version})")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"pos_tracking_{version}_ep{_episode_counter_position+1}.png"))
    plt.close()

    # 2. Reward Plot & Best Logic
    r_steps, r_values = zip(*_position_reward_history)
    r_values_arr = np.array(r_values).flatten()
    
    # [수정됨] 명시적으로 총 보상 계산 및 변수 할당
    episode_total_reward = float(np.sum(r_values_arr))

    # [26.02.24. 추가] 최고 보상 갱신 체크 (Position)
    if episode_total_reward > _best_position_reward:
        _best_position_reward = episode_total_reward
        _best_position_episode = _episode_counter_position + 1
        print("\n" + "🚀"*25)
        print(f"🎉 [NEW BEST POSITION EPISODE] Episode {_best_position_episode} 🎉")
        print(f"Total Position Reward: {episode_total_reward:.4f}")
        print(f"Applied Params: {_current_episode_params}")
        print("🚀"*25 + "\n")

    plt.figure(figsize=(10, 5))
    plt.plot(r_steps, r_values_arr, "g", linewidth=2.5, label="Total Reward(6D)")
    plt.legend()
    plt.grid(True)
    plt.title(f"6D Pose Reward ({version}) - Ep Total: {episode_total_reward:.2f}")
    plt.tight_layout()
    plt.savefig(os.path.join(reward_dir, f"r_total_pos_{version}_ep{_episode_counter_position+1}.png"))
    plt.close()

    _position_tracking_history.clear()
    _position_reward_history.clear()
    _episode_counter_position += 1

# -----------------------------------------------------------
# [26.03.01. 추가] (3) Adaptive Force Tracking Reward (적응형 힘 제어)
# 변경 사유: 로봇이 표면 굴곡을 스스로 파악하여 일정한 압력(target_force)으로 연마하도록 유도하는 보상 함수입니다.
# 단순히 궤적을 따라가는 것을 넘어서, 실제 연마 작업에서 중요한 힘 제어 능력을 학습할 수 있도록 합니다.
# -----------------------------------------------------------
def force_tracking_reward(env: "ManagerBasedRLEnv", target_force: float = 15.0):
    # 1. 센서에서 힘 데이터 가져오기
    contact_wrench = get_contact_forces(env, sensor_name="contact_forces")
    forces = contact_wrench[:, :3]
    force_magnitude = torch.norm(forces, dim=-1)
    
    # 2. 힘 오차 및 보상 계산
    force_error = torch.abs(force_magnitude - target_force)
    reward = torch.exp(-0.05 * torch.square(force_error))

    # 3. 콘솔 출력 (기존과 동일)
    step = int(env.common_step_counter)
    if step > 0 and step % 100 == 0:
        mean_f = force_magnitude.mean().item()
        mean_err = force_error.mean().item()
        print(f"[Adaptive Force Step {step}] Target: {target_force}N | Mean: {mean_f:.2f}N | Err: {mean_err:.2f}N | Rwd: {reward.mean():.3f}")
        
    return reward

# -----------------------------------------------------------
# [26.03.01. 추가] (4) Action Smoothness Penalty (진동 억제)
# 변경 사유: 학습 과정에서 로봇이 덜덜 떨리거나(Jittering) 과격하게 움직여 표면이 손상되는 것을 방지하기 위해, 이전 액션과의 차이에 페널티를 부여하는 보상 함수입니다.
# -----------------------------------------------------------
def action_smoothness_penalty(env: "ManagerBasedRLEnv"):
    """
    로봇이 덜덜 떨거나(Jittering) 스텝마다 갑자기 확 움직이는 것을 방지합니다.
    이전 스텝의 명령(Action)과 현재 명령의 차이가 클수록 강한 감점을 줍니다.
    """
    # 에이전트가 내린 연속된 두 액션의 차이 계산
    action_diff = env.action_manager.action - env.action_manager.prev_action
    
    # 차이의 제곱합을 페널티로 반환
    penalty = torch.sum(torch.square(action_diff), dim=-1)
    
    return penalty

# -----------------------------------------------------------
# [26.03.08 추가] (5) Off-Surface Penalty (표면 이탈 페널티)
# 목적: 로봇이 연마 대상을 벗어나 허공에서 헛스윙하는 것을 강력히 방지합니다.
# -----------------------------------------------------------
def off_surface_penalty(env: "ManagerBasedRLEnv", contact_threshold: float = 1.0):
    """
    힘 센서의 측정값이 contact_threshold(기본 1.0N)보다 낮으면 
    로봇이 표면을 이탈한 것으로 간주하고 강력한 마이너스 보상을 줍니다.
    단, 에피소드 시작 직후(Warm-up 기간)에는 페널티를 주지 않습니다.
    """
    # 1. 센서에서 힘 데이터 가져오기 (N, 3)
    contact_wrench = get_contact_forces(env, sensor_name="contact_forces")
    forces = contact_wrench[:, :3]
    force_magnitude = torch.norm(forces, dim=-1)
    
    # 2. 힘이 임계치(1.0N) 미만인지 확인 (True/False 텐서 생성)
    is_off_surface = force_magnitude < contact_threshold
    
    # 3. [수정] 에피소드 초기 Warm-up 처리 (예: 5스텝 미만은 무시)
    # env.episode_length_buf는 각 환경(env)별로 리셋 후 몇 스텝이 지났는지 기록된 텐서입니다.
    warmup_steps = 5 
    is_warmup = env.episode_length_buf < warmup_steps
    
    # 4. 페널티 계산: 이탈했더라도 Warm-up 기간 중이면 0.0을 줌
    penalty = torch.where(
        is_off_surface & (~is_warmup), # 이탈했고, 동시에 웜업 기간이 아닐 때만!
        torch.tensor(-1.0, device=env.device), 
        torch.tensor(0.0, device=env.device)
    )
    
    # 5. 콘솔 확인용 (옵션)
    step = int(env.common_step_counter)
    if step > 0 and step % 100 == 0:
        # 웜업을 제외하고 실제로 이탈 중인 개수만 카운트
        actual_off_count = (is_off_surface & (~is_warmup)).sum().item()
        if actual_off_count > 0:
            print(f"⚠️ [경고] {actual_off_count}개의 환경이 표면을 이탈했습니다! (Warm-up 제외)")

    return penalty

# -----------------------------------------------------------
# [26.03.28 수정] (6) Perfect Polishing Quality Reward (완벽 연마 종합 보상 - 수직 정렬 통합)
# 목적: 1) 표면에 닿아있고(힘>3N), 2) 경로(X,Y)를 정확히 따르며, 3) 수직(Roll, Pitch)을 완벽히 유지할 때만 극대화된 보상을 줌.
# 기존 'Perpendicular Alignment Reward' 로직을 완전히 흡수하여 연산 낭비를 없앴습니다.
# -----------------------------------------------------------
def perfect_polishing_quality_reward(env: "ManagerBasedRLEnv"):
    device = env.device
    
    # 1. 데이터 추출
    ee_pose = get_ee_pose(env)
    fut = get_hdf5_target_positions(env, horizon=2)
    if fut.ndim == 3:
        fut = fut.squeeze(1)
    target_next = fut[:, 6:12]
    
    # [26.03.28 수정] 목표 자세 강제화: HDF5에 비스듬한 데이터가 섞여 있을 가능성을 배제하기 위해
    # 목표 Roll/Pitch를 강제로 0.0(완전 수직)으로 고정하거나, 목표값과의 오차를 매우 엄격하게 잡습니다.
    # 만약 절대적인 수직을 원하신다면 target_rp를 직접 0으로 세팅할 수도 있습니다.
    ee_rp = ee_pose[:, 3:5]
    target_rp = target_next[:, 3:5] # 혹은 torch.zeros_like(ee_rp) 로 테스트 가능
    
    # 2. 오차 계산
    pos_error = torch.norm(ee_pose[:, :2] - target_next[:, :2], dim=-1)
    rp_error = angle_diff_torch(ee_rp, target_rp)
    rp_error_magnitude = torch.norm(rp_error, dim=-1)
    
    # 3. 보상 설계 (Exponential Kernel 강화)
    # [26.03.28 수정] 위치 보상보다 수직 보상의 k값(민감도)을 높여 조금만 기울어져도 점수가 폭락하게 만듭니다.
    r_pos = torch.exp(-15.0 * torch.square(pos_error)) 
    r_perp = torch.exp(-40.0 * torch.square(rp_error_magnitude)) # 기존 15.0 -> 40.0 (강력한 수직 강제)
    
    # 4. 접촉 판단
    contact_wrench = get_contact_forces(env, sensor_name="contact_forces")
    force_magnitude = torch.norm(contact_wrench[:, :3], dim=-1)
    is_in_contact = force_magnitude > 3.0
    
    # 5. 최종 보상 (곱연산)
    # 수직이 맞지 않으면(r_perp가 낮으면) 위치 점수가 아무리 좋아도 전체 보상이 0에 수렴합니다.
    quality_score = r_pos * r_perp
    final_reward = torch.where(is_in_contact, quality_score, torch.zeros_like(quality_score))
    
    # 모니터링 로그 추가
    step = int(env.common_step_counter)
    if step % 100 == 0:
        print(f"📐 [Alignment Check] Pitch/Roll Error: {rp_error_magnitude.mean():.4f} rad")
        if rp_error_magnitude.mean() > 0.05:
            print("⚠️ [경고] 로봇이 비스듬합니다! 수직 정렬 보상이 더 필요합니다.")

    return final_reward