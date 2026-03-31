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

# ✅ nrs_rl 구조: 같은 mdp 폴더의 observation.py
from .observation import (
    get_hdf5_target_positions,
    get_ee_pose,
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
