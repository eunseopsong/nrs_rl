# SPDX-License-Identifier: BSD-3-Clause
"""
Unified RL + Polishing Removal + Visualization + Debug Module
하나의 파일에서 데이터 기록, 그래프 저장, 화면 출력을 모두 처리함.
"""

from __future__ import annotations

import os
import numpy as np
import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ============================================================
# GLOBAL CONFIG & PATHS
# ============================================================
_run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CURRENT_FILE_PATH = Path(__file__).resolve()
BASE_LOG_DIR = CURRENT_FILE_PATH.parent / "logs"

# [26.04.10 추가] 학습 시작과 동시에 날짜 폴더 생성
RUN_LOG_DIR = BASE_LOG_DIR / _run_timestamp
RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
print(f"[Init] Log directory created: {RUN_LOG_DIR}")

# 데이터 버퍼 (0번 로봇 전용)
_rl_time_buffer = []   
_rl_state_buffer = []
_rl_force_buffer = []
_rl_start_time = None

_episode_counter = 1

# ============================================================
# ------------------ DEBUG PRINT LOGIC -----------------------
# ============================================================

def print_polishing_status(
    env_id, step, path_index, traj_len, done, 
    pos_err, target_force, current_force, reward_info=None
):
    """
    [26.04.10 통합] 기존 utils.py에 있던 디버그 출력을 이 모듈로 흡수
    """
    global _episode_counter
    
    # 0번 로봇일 때만 화면에 출력 (복잡함 방지)
    if env_id != 0:
        return

    print("\n" + "=" * 80)
    print(f"[Episode {_episode_counter}] Step: {step} | H5_Idx: {path_index}/{traj_len} | Done: {done}")
    print(f"[Status] Pos_Err: {pos_err:.6f} m")
    
    # 힘 정보 출력
    fz = current_force[2] if len(current_force) > 2 else 0.0
    print(f"[Force ] Target_Fz: {target_force[2]:.2f}N | Current_Fz: {fz:.2f}N")
    
    # 리워드 정보가 있다면 출력
    if reward_info:
        print(f"[Reward] Total: {reward_info.get('total', 0):.4f}")
    
    print("=" * 80)

# ============================================================
# ------------------ RECORDING & PROCESSING ------------------
# ============================================================

def record_step(env_ids, state6, force3, sim_time):
    """0번 로봇 데이터 기록"""
    global _rl_time_buffer, _rl_state_buffer, _rl_force_buffer, _rl_start_time

    if _rl_start_time is None:
        _rl_start_time = sim_time

    try:
        # env_ids가 Tensor인 경우 대비
        target_mask = (env_ids == 0)
        if target_mask.any():
            # 0번 로봇의 인덱스 추출
            idx = target_mask.nonzero(as_tuple=True)[0][0].item()
            
            _rl_time_buffer.append(sim_time - _rl_start_time)
            
            s_val = state6[idx].cpu().numpy() if hasattr(state6, 'cpu') else state6[idx]
            f_val = force3[idx].cpu().numpy() if hasattr(force3, 'cpu') else force3[idx]
            
            _rl_state_buffer.append(s_val.copy())
            _rl_force_buffer.append(f_val.copy())
    except:
        pass

def process_episode():
    """에피소드 종료 시 호출: 그래프 저장 및 번호 증가"""
    global _rl_time_buffer, _rl_state_buffer, _rl_force_buffer
    global _rl_start_time, _episode_counter

    if len(_rl_time_buffer) < 10:
        return 0.0

    t = np.array(_rl_time_buffer)
    s_arr = np.array(_rl_state_buffer)
    f_arr = np.array(_rl_force_buffer)

    # 물리량 계산 (간소화 호출)
    xyz = s_arr[:, :3]
    dt = np.diff(t, prepend=t[0]-0.001)
    vxyz = np.zeros_like(xyz)
    for i in range(3): vxyz[:, i] = np.gradient(xyz[:, i], t)
    
    speed = np.linalg.norm(vxyz[:, :2], axis=1)
    fn = np.maximum(f_arr[:, 2], 0.0)
    removal_rate = np.where((fn > 0.5) & (speed > 0.1), fn * speed, 0.0)
    dremoval = removal_rate * dt

    # 폴더 생성 및 저장
    ep_dir = RUN_LOG_DIR / f"ep{_episode_counter}"
    save_plots(ep_dir, t, s_arr, f_arr, dremoval, removal_rate, vxyz)
    
    print(f"\n[STAMP] Episode {_episode_counter} Results Saved to: {ep_dir}\n")

    # 카운트 증가 및 초기화
    _episode_counter += 1
    _rl_time_buffer.clear()
    _rl_state_buffer.clear()
    _rl_force_buffer.clear()
    _rl_start_time = None

    return np.sum(dremoval)

def save_plots(out_dir, t, state6, force3, dremoval, removal_rate, vxyz):
    """그래프 저장 로직 (기존과 동일)"""
    out_dir.mkdir(parents=True, exist_ok=True)
    xyz = state6[:, :3]

    # 1. Heatmap
    plt.figure(figsize=(6, 5))
    plt.hexbin(xyz[:, 0], xyz[:, 1], C=dremoval, gridsize=30, cmap='jet')
    plt.colorbar(label="Removal"); plt.savefig(out_dir / "01_removal_heatmap.png"); plt.close()

    # 2. Rate vs Time
    plt.figure(figsize=(10, 4))
    plt.plot(t, removal_rate); plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "02_heatmap_value_vs_time.png"); plt.close()

    # 3. Signals (Vel, Force)
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    v_norm = savgol_filter(np.linalg.norm(vxyz, axis=1), 11, 3) if len(t)>11 else np.linalg.norm(vxyz, axis=1)
    axs[0].plot(t, v_norm, color='g'); axs[0].set_title("Velocity Magnitude")
    axs[1].plot(t, force3[:, 2], color='purple'); axs[1].set_title("Normal Force (Z)")
    plt.tight_layout(); plt.savefig(out_dir / "03_signals_subplot.png"); plt.close()

    # 4. 3D Path
    fig = plt.figure(figsize=(8, 7)); ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=t, cmap='viridis', s=2)
    plt.savefig(out_dir / "04_3d_path_w.png"); plt.close()

# ============================================================
# ------------------ RL INTERFACE ----------------------------
# ============================================================

def rl_step(env_ids, state6, force3, sim_time, debug_info=None):
    """
    매 스텝 호출됨: 기록 + 화면 출력
    debug_info: {step, path_idx, traj_len, done, pos_err, target_force, reward_dict}
    """
    record_step(env_ids, state6, force3, sim_time)
    
    if debug_info:
        print_polishing_status(
            env_id=0, # 0번 로봇 기준
            step=debug_info['step'],
            path_index=debug_info['path_idx'],
            traj_len=debug_info['traj_len'],
            done=debug_info['done'],
            pos_err=debug_info['pos_err'],
            target_force=debug_info['target_force'],
            current_force=force3[0], # 0번 로봇의 현재 힘
            reward_info=debug_info.get('reward_dict')
        )

def rl_episode_done():
    """에피소드 종료 시 호출"""
    return process_episode()

# # ============================================================
# # ------------------ EPISODE PROCESS -------------------------
# # ============================================================

# def process_episode():
#     """
#     260406 랩미팅: 리워드가 가장 높은 로봇의 데이터를 추출하여 시각화
#     """
#     global _rl_time_buffers, _rl_state_buffers, _rl_force_buffers
#     global _rl_start_time, _episode_counter, _best_reward

#     if not _rl_state_buffers:
#         print("[Warning] No data recorded in this episode.")
#         return 0.0

#     target_env_id = None
#     max_total_removal = -np.inf
#     best_data = None

#     # [26.04.10 수정] 모든 로봇 중 가공량(리워드)이 가장 큰 로봇 탐색
#     for env_id in _rl_state_buffers.keys():
#         t_arr = np.array(_rl_time_buffers[env_id])
#         s_arr = np.array(_rl_state_buffers[env_id])
#         f_arr = np.array(_rl_force_buffers[env_id])

#         if len(t_arr) < 5:
#             continue

#         # 가공량 계산 루틴
#         drem, r_rate, xyz, wxyz, vxyz, axyz, jxyz = compute_removal(t_arr, s_arr, f_arr)
#         total_removal = np.sum(drem)

#         if total_removal > max_total_removal:
#             max_total_removal = total_removal
#             target_env_id = env_id
#             best_data = (t_arr, s_arr, f_arr, drem, r_rate, vxyz, axyz, jxyz)

#     # [26.04.10 추가] 최적 로봇 데이터 저장 (없을 경우 0번 시도)
#     if best_data is not None:
#         save_dir = BASE_LOG_DIR / _run_timestamp / f"ep{_episode_counter + 1}"
#         save_plots(save_dir, *best_data)
#         print(f"[Log] Episode {_episode_counter+1} saved | Best Env: {target_env_id} | Total Removal: {max_total_removal:.4f}")
#     else:
#         print(f"[Log] Episode {_episode_counter+1} skipped (Insufficient data)")

#     # 버퍼 초기화
#     _rl_time_buffers.clear()
#     _rl_state_buffers.clear()
#     _rl_force_buffers.clear()
#     _rl_start_time = None
#     _episode_counter += 1

#     return max_total_removal if max_total_removal != -np.inf else 0.0
