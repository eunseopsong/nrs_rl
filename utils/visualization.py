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

# [26.04.10 경로 재생성] Isaac Sim 실행 환경에 상관없이 nrs_rl/logs 경로를 정확히 타겟팅
# __file__을 사용하여 현재 스크립트 위치 기준으로 절대 경로 산출
CURRENT_FILE_PATH = Path(__file__).resolve()
BASE_LOG_DIR = CURRENT_FILE_PATH.parent / "logs"

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
    # t가 정지 상태일 때 (dt=0) 발생하는 RuntimeWarning 방지
    dt_arr = np.diff(t, prepend=t[0] - 0.01) 
    
    vxyz = np.zeros_like(xyz)
    if len(xyz) >= 2:
        # [26.04.10 수정] 각 축별 성분 미분으로 정밀도 향상
        for i in range(3):
            vxyz[:, i] = np.gradient(xyz[:, i], t)

    axyz = np.gradient(vxyz, axis=0)
    jxyz = np.gradient(axyz, axis=0)

    return vxyz, axyz, jxyz, dt_arr


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

    # 가공량 누적 (시간 간격 곱)
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

    # [26.04.10 수정] 넘파이 배열로 변환하여 처리 속도 향상
    env_ids_np = env_ids.cpu().numpy() if hasattr(env_ids, 'cpu') else np.array(env_ids)
    state6_np = state6.cpu().numpy() if hasattr(state6, 'cpu') else np.array(state6)
    force3_np = force3.cpu().numpy() if hasattr(force3, 'cpu') else np.array(force3)

    for i, idx in enumerate(env_ids_np):
        if idx not in _rl_time_buffers:
            _rl_time_buffers[idx] = []
            _rl_state_buffers[idx] = []
            _rl_force_buffers[idx] = []

        _rl_time_buffers[idx].append(t_rel)
        _rl_state_buffers[idx].append(state6_np[i])
        _rl_force_buffers[idx].append(force3_np[i])


# ============================================================
# ------------------ VISUALIZATION ---------------------------
# ============================================================

def save_plots(out_dir, t, state6, force3, dremoval, removal_rate, vxyz, axyz, jxyz):
    """
    260406 랩미팅: 서준님 코드 스타일 이식
    """
    # [26.04.10 경로 재생성] 폴더 생성 재확인
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xyz = state6[:, :3]

    # 1️⃣ 가공량 히트맵
    plt.figure(figsize=(6, 5))
    hb = plt.hexbin(xyz[:, 0], xyz[:, 1], C=dremoval, gridsize=30, cmap='jet')
    plt.colorbar(hb, label="Removal Amount")
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

    v_norm = smooth_signal(np.linalg.norm(vxyz, axis=1))
    a_norm = smooth_signal(np.linalg.norm(axyz, axis=1))
    j_norm = smooth_signal(np.linalg.norm(jxyz, axis=1))

    axs[0].plot(t, v_norm, color='g')
    axs[0].set_ylabel("Vel (mm/s)")
    axs[0].set_title("Velocity Magnitude (Filtered)")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(t, a_norm, color='r')
    axs[1].set_ylabel("Acc (mm/s^2)")
    axs[1].set_title("Acceleration Magnitude (Filtered)")
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(t, j_norm, color='orange')
    axs[2].set_ylabel("Jerk (mm/s^3)")
    axs[2].set_title("Jerk Magnitude (Filtered)")
    axs[2].grid(True, alpha=0.3)

    axs[3].plot(t, force3[:, 2], color='purple')
    axs[3].set_ylabel("Force (N)")
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
    260406 랩미팅: 리워드가 가장 높은 로봇의 데이터를 추출하여 시각화
    """
    global _rl_time_buffers, _rl_state_buffers, _rl_force_buffers
    global _rl_start_time, _episode_counter, _best_reward

    if not _rl_state_buffers:
        print("[Warning] No data recorded in this episode.")
        return 0.0

    target_env_id = None
    max_total_removal = -np.inf
    best_data = None

    # [26.04.10 수정] 모든 로봇 중 가공량(리워드)이 가장 큰 로봇 탐색
    for env_id in _rl_state_buffers.keys():
        t_arr = np.array(_rl_time_buffers[env_id])
        s_arr = np.array(_rl_state_buffers[env_id])
        f_arr = np.array(_rl_force_buffers[env_id])

        if len(t_arr) < 5:
            continue

        # 가공량 계산 루틴
        drem, r_rate, xyz, wxyz, vxyz, axyz, jxyz = compute_removal(t_arr, s_arr, f_arr)
        total_removal = np.sum(drem)

        if total_removal > max_total_removal:
            max_total_removal = total_removal
            target_env_id = env_id
            best_data = (t_arr, s_arr, f_arr, drem, r_rate, vxyz, axyz, jxyz)

    # [26.04.10 추가] 최적 로봇 데이터 저장 (없을 경우 0번 시도)
    if best_data is not None:
        save_dir = BASE_LOG_DIR / _run_timestamp / f"ep{_episode_counter + 1}"
        save_plots(save_dir, *best_data)
        print(f"[Log] Episode {_episode_counter+1} saved | Best Env: {target_env_id} | Total Removal: {max_total_removal:.4f}")
    else:
        print(f"[Log] Episode {_episode_counter+1} skipped (Insufficient data)")

    # 버퍼 초기화
    _rl_time_buffers.clear()
    _rl_state_buffers.clear()
    _rl_force_buffers.clear()
    _rl_start_time = None
    _episode_counter += 1

    return max_total_removal if max_total_removal != -np.inf else 0.0


# ============================================================
# ------------------ RL INTERFACE ----------------------------
# ============================================================

def rl_step(env_ids, state6, force3, sim_time):
    record_step(env_ids, state6, force3, sim_time)

def rl_episode_done():
    return process_episode()