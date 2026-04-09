# SPDX-License-Identifier: BSD-3-Clause
"""
Unified RL + Polishing Removal + Visualization Module
"""

from __future__ import annotations

import os
import numpy as np
import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # GUI 없는 환경에서도 저장 가능하도록 설정
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter  # [26.04.10 추가] 신호 정제(Smoothing)를 위한 라이브러리

# ============================================================
# GLOBAL
# ============================================================
# 260406 랩미팅: 서준님 코드 이식 및 Isaac Sim 환경에 맞춘 경로 설정
version = "v30_meeting_ref_refined"
_run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# [26.04.10 추가] 호환성 체크: logs 경로를 실행 위치 기준으로 고정
BASE_LOG_DIR = Path(os.getcwd()) / "logs"

# [26.04.10 수정] 메모리 효율을 위해 사전 할당(Pre-allocation) 구조 도입 제안 (선택 사항)
_rl_time_buffers = {}   # env별 시간 분리
_rl_state_buffers = {}
_rl_force_buffers = {}
_rl_start_time = None

_episode_counter = 0
_best_reward = -np.inf


# ============================================================
# ------------------ CORE PHYSICS -----------------------------
# ============================================================

def smooth_signal(data, window=11):
    """[26.04.10 추가] 서준님 코드와 유사한 깔끔한 그래프를 위해 노이즈 제거 (Jerk 등)"""
    if len(data) < window:
        return data
    return savgol_filter(data, window, 3)


def compute_velocity_mm_s(t, xyz):
    """260406 랩미팅: 속도, 가속도, 저크 계산을 위한 물리 로직"""
    # [26.04.10 수정] np.gradient를 사용하여 서준님 코드의 미분 방식과 동기화
    dt = np.diff(t, prepend=t[0])
    dt = np.clip(dt, 1e-6, None)

    vxyz = np.zeros_like(xyz)
    if len(xyz) >= 2:
        # [26.04.10 수정] 각 축별 성분 미분으로 정밀도 향상
        for i in range(3):
            vxyz[:, i] = np.gradient(xyz[:, i], t)

    axyz = np.gradient(vxyz, axis=0)
    jxyz = np.gradient(axyz, axis=0)

    return vxyz, axyz, jxyz, dt


def compute_removal(t, state6, force3):
    """260406 랩미팅: 가공량(Removal) 및 6축 상태 분리"""
    xyz = state6[:, :3]
    wxyz = state6[:, 3:]

    vxyz, axyz, jxyz, dt = compute_velocity_mm_s(t, xyz)
    
    # [26.04.10 수정] 속도 계산 시 2D(평면)와 3D 성분 구분 (서준님 스타일 반영)
    speed = np.linalg.norm(vxyz[:, :2], axis=1)

    fn = np.maximum(force3[:, 2], 0.0)

    # [26.04.10 수정] 접촉 임계값 조정 (0.5N 미만은 가공에서 제외)
    contact = (fn > 0.5) & (speed > 0.1)

    removal_rate = np.zeros_like(fn)
    removal_rate[contact] = fn[contact] * speed[contact]

    dremoval = removal_rate * dt

    return dremoval, removal_rate, xyz, wxyz, vxyz, axyz, jxyz


# ============================================================
# ------------------ RECORDING -------------------------------
# ============================================================

def record_step(env_ids, state6, force3, sim_time):
    """
    260406 랩미팅: 128개 로봇 중 전체를 기록하되 메모리 효율을 위해 관리.
    """
    global _rl_time_buffers, _rl_state_buffers, _rl_force_buffers, _rl_start_time

    if _rl_start_time is None:
        _rl_start_time = sim_time

    t_rel = sim_time - _rl_start_time

    for i, idx in enumerate(env_ids):
        # [26.04.10 추가] env별 time buffer 분리
        if idx not in _rl_time_buffers:
            _rl_time_buffers[idx] = []
            _rl_state_buffers[idx] = []
            _rl_force_buffers[idx] = []

        _rl_time_buffers[idx].append(t_rel)
        _rl_state_buffers[idx].append(state6[i].copy())
        _rl_force_buffers[idx].append(force3[i].copy())


# ============================================================
# ------------------ VISUALIZATION ---------------------------
# ============================================================

def save_plots(out_dir, t, state6, force3, dremoval, removal_rate, vxyz, axyz, jxyz):
    """
    260406 랩미팅: 서준님 코드 스타일 이식
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xyz = state6[:, :3]

    # 1️⃣ 가공량 히트맵
    plt.figure(figsize=(6, 5))
    # [26.04.10 수정] hexbin 적용
    plt.hexbin(xyz[:, 0], xyz[:, 1], C=dremoval, gridsize=30, cmap='jet')
    plt.colorbar(label="Removal Amount")
    plt.title("Removal Heatmap (Top View)")
    plt.savefig(out_dir / "01_removal_heatmap.png")
    plt.close()

    # 2️⃣ 가공량 vs 시간
    plt.figure(figsize=(10, 4))
    plt.plot(t, removal_rate, label="Removal Rate (fn*v)", color='blue')
    plt.xlabel("Time (s)")
    plt.ylabel("Rate")
    plt.title("Removal Rate over Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(out_dir / "02_heatmap_value_vs_time.png")
    plt.close()

    # 3️⃣ 속도 / 가속도 / 저크 / 힘
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # [26.04.10 추가] 신호 Smoothing
    v_norm = smooth_signal(np.linalg.norm(vxyz, axis=1))
    a_norm = smooth_signal(np.linalg.norm(axyz, axis=1))
    j_norm = smooth_signal(np.linalg.norm(jxyz, axis=1))

    axs[0].plot(t, v_norm, color='g')
    axs[0].set_title("Velocity Magnitude (Filtered)")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(t, a_norm, color='r')
    axs[1].set_title("Acceleration Magnitude (Filtered)")
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(t, j_norm, color='orange')
    axs[2].set_title("Jerk Magnitude (Filtered)")
    axs[2].grid(True, alpha=0.3)

    axs[3].plot(t, force3[:, 2], color='purple')
    axs[3].set_title("Normal Force (Z)")
    axs[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "03_signals_subplot.png")
    plt.close()

    # 4️⃣ 3D 경로
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=t, cmap='viridis', s=2)
    ax.set_title("3D Path (Tool Trajectory)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig(out_dir / "04_3d_path_w.png")
    plt.close()


# ============================================================
# ------------------ EPISODE PROCESS -------------------------
# ============================================================

def process_episode():
    """
    260406 랩미팅: 임시방편으로 0번 로봇 데이터 추출 (보상 로직 주석 처리)
    """
    global _rl_time_buffers, _rl_state_buffers, _rl_force_buffers
    global _rl_start_time, _episode_counter, _best_reward

    if not _rl_state_buffers:
        return 0.0

    target_env_id = 0 # [26.04.10 임시] 0번 로봇 고정
    best_data = None
    
    # ------------------------------------------------------------
    # [주석 처리] 보상 기반 최적 로봇 선별 로직 (rewards.py 수정 중)
    # max_ep_reward = -np.inf
    # for env_id, states in _rl_state_buffers.items():
    #     t_arr = np.array(_rl_time_buffers[env_id])
    #     s_arr = np.array(states)
    #     f_arr = np.array(_rl_force_buffers[env_id])
    #     if len(t_arr) < 5: continue
    #     drem, _, _, _, _, _, _ = compute_removal(t_arr, s_arr, f_arr)
    #     current_reward = np.sum(drem)
    #     if current_reward > max_ep_reward:
    #         max_ep_reward = current_reward
    #         target_env_id = env_id
    # ------------------------------------------------------------

    # [26.04.10 추가] 0번 로봇 데이터가 존재하는지 확인 후 처리
    if target_env_id in _rl_state_buffers:
        t = np.array(_rl_time_buffers[target_env_id])
        states_np = np.array(_rl_state_buffers[target_env_id])
        forces_np = np.array(_rl_force_buffers[target_env_id])

        if len(t) >= 5:
            drem, r_rate, xyz, wxyz, vxyz, axyz, jxyz = compute_removal(
                t, states_np, forces_np
            )
            best_data = (t, states_np, forces_np, drem, r_rate, vxyz, axyz, jxyz)
            
            # [주석 처리용] 임시 리워드 계산
            temp_reward = np.sum(drem)
        else:
            temp_reward = 0.0
    else:
        temp_reward = 0.0

    if best_data:
        save_dir = BASE_LOG_DIR / _run_timestamp / f"ep{_episode_counter + 1}"
        save_plots(save_dir, *best_data)
        print(f"[Log] Episode {_episode_counter+1} saved | Target Env: {target_env_id} (Fixed)")

    # 버퍼 초기화
    _rl_time_buffers.clear()
    _rl_state_buffers.clear()
    _rl_force_buffers.clear()
    _rl_start_time = None
    _episode_counter += 1

    return temp_reward


# ============================================================
# ------------------ RL INTERFACE ----------------------------
# ============================================================

def rl_step(env_ids, state6, force3, sim_time):
    record_step(env_ids, state6, force3, sim_time)

def rl_episode_done():
    return process_episode()