# SPDX-License-Identifier: BSD-3-Clause
# shangus
"""
v29: action.py 가 정상 작동함에 따라, 움직임 통제 관련 reward 함수는 불필요하다고 판단되어 삭제했음.

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
    get_hdf5_target_positions,
    get_ee_pose,
    get_contact_forces,  # ✅ [26.02.24. 추가] 힘 센서 데이터 가져오기
)

# -----------------------------------------------------------
# Global
# -----------------------------------------------------------
version = "v29"

# -----------------------------------------------------------
# [26.03.01. 추가] 훈련 실행 시각 기반 고유 폴더명 생성 변수
# 변경 사유: 매 훈련(Run)마다 그래프 결과가 덮어씌워지는 것을 방지하고, 실험 이력을 날짜/시간별로 독립적으로 보존하기 위함.
# -----------------------------------------------------------
_run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

_position_tracking_history = []
_position_reward_history = []

_episode_counter_position = 0

# [26.02.24. 추가] 파라미터 탐색용 최고 보상 및 파라미터 추적 변수
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