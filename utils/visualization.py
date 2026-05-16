# SPDX-License-Identifier: BSD-3-Clause
"""
Unified RL + Polishing Removal + Visualization Module (With Diagnostics)
"""

from __future__ import annotations

import numpy as np
import datetime
from pathlib import Path
import importlib
from collections import defaultdict # 🚀 추가: 보상 요소를 딕셔너리 형태로 담기 위함

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import torch


# ============================================================
# Local Imports
# ============================================================

local_obs = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.mdp.observation"
)

local_ft_sensor = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.assets.assets.sensors.six_axis_ft_sensor"
)

local_debug = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.utils.debug"
)


# ============================================================
# Path Setup
# ============================================================

_run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

CURRENT_FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT_DIR = CURRENT_FILE_PATH.parent.parent
BASE_LOG_DIR = PROJECT_ROOT_DIR / "logs" / "polishing_results"
RUN_LOG_DIR = BASE_LOG_DIR / _run_timestamp

RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)

local_debug.print_info(f"[Init] Polishing log directory created: {RUN_LOG_DIR}")


# ============================================================
# Global Buffers
# ============================================================

_rl_time_buffer = []
_rl_state_buffer = []
_rl_force_buffer = []
_rl_index_buffer = [] # 🚀 추가: Trajectory Index 추적 버퍼
_rl_reward_components_buffer = defaultdict(list) # 🚀 추가: 개별 보상 추적 버퍼

_current_ep_reward = 0.0

_rl_start_time = None
_episode_counter = 1
_has_seen_any_step = False


# ============================================================
# Surface Memory Map
# ============================================================

GRID_SIZE = 128

_surface_grid = np.zeros(
    (GRID_SIZE, GRID_SIZE),
    dtype=np.float32,
)

_surface_extent = {
    "xmin": -120.0,
    "xmax": 120.0,
    "ymin": -120.0,
    "ymax": 120.0,
}


# ============================================================
# Summary Metrics
# ============================================================

_summary_metrics = {
    "episode": [],
    "total_removal": [],
    "mean_mrr_error": [],
    "std_removal": [],
    "contact_ratio": [],
    "episode_reward": [],
}


# ============================================================
# Utils
# ============================================================

def moving_average(x, w=5):
    x = np.asarray(x)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def update_surface_grid(x, y, removal):
    global _surface_grid

    xmin = _surface_extent["xmin"]
    xmax = _surface_extent["xmax"]
    ymin = _surface_extent["ymin"]
    ymax = _surface_extent["ymax"]

    ix = int((x - xmin) / (xmax - xmin) * (GRID_SIZE - 1))
    iy = int((y - ymin) / (ymax - ymin) * (GRID_SIZE - 1))

    ix = np.clip(ix, 0, GRID_SIZE - 1)
    iy = np.clip(iy, 0, GRID_SIZE - 1)

    _surface_grid[iy, ix] += float(removal)


# ============================================================
# Step Recording
# ============================================================

def record_step(env_ids, state6, force3, sim_time):
    global _rl_time_buffer
    global _rl_state_buffer
    global _rl_force_buffer
    global _rl_start_time
    global _has_seen_any_step

    _has_seen_any_step = True

    if _rl_start_time is None:
        _rl_start_time = float(sim_time)

    try:
        if isinstance(env_ids, torch.Tensor):
            target_mask = (env_ids == 0)
            idx = (
                target_mask.nonzero(as_tuple=True)[0][0].item()
                if bool(target_mask.any())
                else 0
            )
        else:
            idx = 0

        _rl_time_buffer.append(float(sim_time) - float(_rl_start_time))

        s_val = (
            state6[idx].detach().cpu().numpy()
            if hasattr(state6, "detach")
            else np.asarray(state6[idx])
        )

        f_val = (
            force3[idx].detach().cpu().numpy()
            if hasattr(force3, "detach")
            else np.asarray(force3[idx])
        )

        _rl_state_buffer.append(s_val.copy())
        _rl_force_buffer.append(f_val.copy())

    except Exception as e:
        local_debug.print_exception("Visualization record_step failed", e)


# ============================================================
# Global Summary Plot
# ============================================================

def save_global_summary():
    if not _summary_metrics["episode"]:
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    eps = _summary_metrics["episode"]

    # --- 1. Total Removal ---
    axes[0, 0].plot(eps, _summary_metrics["total_removal"], "b-o", markersize=4, alpha=0.7)
    axes[0, 0].set_title("Total Removal per Episode")
    axes[0, 0].set_ylabel("Total Removal [a.u.]")
    axes[0, 0].grid(True, alpha=0.3)

    # --- 2. MRR Error ---
    axes[0, 1].plot(eps, _summary_metrics["mean_mrr_error"], "r-x", markersize=4, alpha=0.7)
    axes[0, 1].set_title("MRR Tracking Error (Lower is better)")
    axes[0, 1].set_ylabel("Mean Abs Error")
    axes[0, 1].grid(True, alpha=0.3)

    # --- 3. Episode Reward ---
    reward_arr = np.array(_summary_metrics["episode_reward"])
    axes[1, 0].plot(eps, reward_arr, color="green", marker="s", markersize=4, alpha=0.3, label="Raw Reward")
    axes[1, 0].plot(eps, moving_average(reward_arr), color="green", linewidth=2.5, label="Trend (MA)")
    axes[1, 0].set_title("Total Reward per Episode (Higher is better)")
    axes[1, 0].set_ylabel("Cumulative Reward")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # --- 4. Contact Ratio ---
    axes[1, 1].plot(eps, _summary_metrics["contact_ratio"], "m-d", markersize=4, alpha=0.7)
    axes[1, 1].set_title("Contact Ratio")
    axes[1, 1].set_ylabel("Ratio (Contact/Total)")
    axes[1, 1].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Episode")

    fig.suptitle(
        f"Overall Polishing Performance Trend (Up to Ep {_episode_counter - 1})",
        fontsize=16,
        fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    summary_path = RUN_LOG_DIR / "overall_learning_trend.png"
    fig.savefig(summary_path, dpi=200)
    plt.close(fig)


# ============================================================
# Episode Processing
# ============================================================

def process_episode():
    global _rl_time_buffer, _rl_state_buffer, _rl_force_buffer
    global _rl_index_buffer, _rl_reward_components_buffer # 🚀 추가
    global _rl_start_time, _episode_counter, _summary_metrics
    global _current_ep_reward

    if len(_rl_time_buffer) < 10:
        _rl_time_buffer.clear()
        _rl_state_buffer.clear()
        _rl_force_buffer.clear()
        _rl_index_buffer.clear() # 🚀 추가
        _rl_reward_components_buffer.clear() # 🚀 추가
        _rl_start_time = None
        _current_ep_reward = 0.0
        return 0.0

    t = np.array(_rl_time_buffer, dtype=float)
    s_arr = np.array(_rl_state_buffer, dtype=float)
    f_arr = np.array(_rl_force_buffer, dtype=float)
    idx_arr = np.array(_rl_index_buffer, dtype=float) # 🚀 추가
    rw_dict = dict(_rl_reward_components_buffer) # 🚀 추가 (복사)

    xyz = s_arr[:, :3]

    # --- Velocity calculation ---
    vxyz = np.zeros_like(xyz)
    for i in range(3):
        vxyz[:, i] = np.gradient(xyz[:, i], t)

    speed = np.linalg.norm(vxyz[:, :2], axis=1)
    fn = np.maximum(f_arr[:, 2], 0.0) # Z-axis force
    dt = np.diff(t, prepend=t[0] - 1e-3)

    # --- Removal logic ---
    removal_rate = np.where((fn > 0.5) & (speed > 0.1), fn * speed, 0.0)
    dremoval = removal_rate * dt

    # --- Surface Memory Update ---
    for i in range(len(dremoval)):
        update_surface_grid(xyz[i, 0], xyz[i, 1], dremoval[i])

    ep_dir = RUN_LOG_DIR / f"ep{_episode_counter}"
    
    # 🚀 수정: index 배열과 reward 딕셔너리도 플롯 함수로 넘김
    save_plots(ep_dir, t, s_arr, f_arr, dremoval, removal_rate, vxyz, idx_arr, rw_dict)

    # --- Metrics calculation ---
    samples = len(t)
    contact_samples = int(np.sum(fn > 0.5))
    rate_mean = float(np.mean(removal_rate))
    mrr_error = np.mean(np.abs(removal_rate - rate_mean))

    # --- Store Metrics ---
    _summary_metrics["episode"].append(_episode_counter)
    _summary_metrics["total_removal"].append(float(np.sum(dremoval)))
    _summary_metrics["mean_mrr_error"].append(float(mrr_error))
    _summary_metrics["std_removal"].append(float(np.std(dremoval)))
    _summary_metrics["contact_ratio"].append(contact_samples / samples if samples > 0 else 0)
    _summary_metrics["episode_reward"].append(_current_ep_reward)

    if _episode_counter % 5 == 0:
        save_global_summary()

    local_debug.print_info(f"\n[STAMP] Episode {_episode_counter} saved. Reward: {_current_ep_reward:.2f}. Results at: {ep_dir}\n")

    _episode_counter += 1
    
    # 🚀 수정: 새 버퍼들까지 모두 초기화
    _rl_time_buffer.clear()
    _rl_state_buffer.clear()
    _rl_force_buffer.clear()
    _rl_index_buffer.clear()
    _rl_reward_components_buffer.clear()
    _rl_start_time = None
    _current_ep_reward = 0.0 

    return float(np.sum(dremoval))


# ============================================================
# Plot Saving
# ============================================================

def save_plots(out_dir, t, state6, force3, dremoval, removal_rate, vxyz, index_arr=None, reward_dict=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    xyz = state6[:, 0:3]

    # --- 1. Local Removal Heatmap ---
    x, y = xyz[:, 0], xyz[:, 1]
    margin = 10.0
    extent = [np.min(x) - margin, np.max(x) + margin, np.min(y) - margin, np.max(y) + margin]

    grid_removal, _, _ = np.histogram2d(
        x, y, bins=150, range=[extent[:2], extent[2:]], weights=dremoval
    )
    grid_removal_smoothed = gaussian_filter(grid_removal.T, sigma=2.5)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(grid_removal_smoothed, origin="lower", extent=extent, cmap="viridis")
    fig.colorbar(im, label="Removal [a.u.]")
    ax.set_title(f"Episode {_episode_counter} Removal Heatmap")
    fig.savefig(out_dir / "01_removal_heatmap.png", dpi=200)
    plt.close(fig)

    # --- 2. Global Surface Map ---
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    global_map = gaussian_filter(_surface_grid, sigma=2.0)
    im2 = ax2.imshow(global_map, origin="lower", cmap="viridis")
    fig2.colorbar(im2)
    ax2.set_title(f"Cumulative Surface Removal (Ep {_episode_counter})")
    fig2.savefig(out_dir / "02_global_surface_map.png", dpi=200)
    plt.close(fig2)

    # 🚀 --- 3. [NEW] Diagnostic Panel ---
    fig3, axes3 = plt.subplots(2, 4, figsize=(24, 10))
    
    # 3-1. Toolpath Trajectory
    axes3[0, 0].plot(x, y, 'b-', linewidth=1.5)
    axes3[0, 0].set_title("Toolpath Trajectory (X-Y)")
    axes3[0, 0].set_xlabel("X")
    axes3[0, 0].set_ylabel("Y")
    axes3[0, 0].axis("equal")
    axes3[0, 0].grid(True)

    # 3-2. Force Magnitude
    force_mag = np.linalg.norm(force3, axis=1)
    axes3[0, 1].plot(t, force_mag, 'r-')
    axes3[0, 1].set_title("Force Magnitude over Time")
    axes3[0, 1].set_xlabel("Time [s]")
    axes3[0, 1].grid(True)

    # 3-3. Index Progression
    if index_arr is not None and len(index_arr) == len(t):
        axes3[0, 2].plot(t, index_arr, 'g-')
    axes3[0, 2].set_title("Trajectory Index Progression")
    axes3[0, 2].set_xlabel("Time [s]")
    axes3[0, 2].grid(True)

    # 3-4. Reward Components Breakdown
    if reward_dict:
        for term_name, values in reward_dict.items():
            # values 리스트 길이가 t와 안 맞을 수 있으므로 슬라이싱 처리
            plot_len = min(len(t), len(values))
            axes3[0, 3].plot(t[:plot_len], values[:plot_len], label=term_name)
        axes3[0, 3].legend(loc='upper left', fontsize='small')
    axes3[0, 3].set_title("Cumulative Rewards Breakdown")
    axes3[0, 3].set_xlabel("Time [s]")
    axes3[0, 3].grid(True)

    # 3-5. Coverage Mask (Binary)
    # 히트맵에서 removal이 1e-4 이상 발생한 구역을 1로 (방문 처리)
    visited_mask = (grid_removal.T > 1e-4).astype(np.uint8)
    coverage_ratio = np.mean(visited_mask)
    im_mask = axes3[1, 0].imshow(visited_mask, origin="lower", extent=extent, cmap="gray")
    axes3[1, 0].set_title(f"Coverage Mask (Ratio: {coverage_ratio:.3f})")

    # 3-6. Contact State
    contact_state = (force_mag > 0.5).astype(int)
    axes3[1, 1].plot(t, contact_state, 'm-')
    axes3[1, 1].set_ylim([-0.1, 1.1])
    axes3[1, 1].set_title("Contact State (F > 0.5)")
    axes3[1, 1].set_xlabel("Time [s]")
    axes3[1, 1].grid(True)

    # 3-7. Tool Speed
    speed = np.linalg.norm(vxyz[:, :2], axis=1)
    axes3[1, 2].plot(t, speed, 'c-')
    axes3[1, 2].set_title("Tool Speed (XY Plane)")
    axes3[1, 2].set_xlabel("Time [s]")
    axes3[1, 2].grid(True)

    # 3-8. 빈 공간 정리
    axes3[1, 3].axis("off")

    fig3.tight_layout()
    fig3.savefig(out_dir / "03_diagnostic_panel.png", dpi=200)
    plt.close(fig3)


# ============================================================
# RL Hooks
# ============================================================

def rl_step(env_ids, state6, force3, sim_time):
    record_step(env_ids, state6, force3, sim_time)


def rl_episode_done():
    return process_episode()


def on_episode_reset(env, env_ids=None):
    global _has_seen_any_step, _current_ep_reward
    try:
        if not _has_seen_any_step:
            return
        if len(_rl_time_buffer) > 0:
            rl_episode_done()

        if env is not None:
            env._ep_curriculum = getattr(env, "_ep_curriculum", 0) + 1
            local_debug.print_info(f"[Curriculum] Episode counter: {env._ep_curriculum}")
            save_global_summary()
            
    except Exception as e:
        local_debug.print_exception("Visualization on_episode_reset failed", e)


# ============================================================
# Step Hook
# ============================================================

def rl_step_hook(
    env,
    action_term_name: str = "arm_action",
    asset_name: str = "robot",
    fixed_joint_name: str = "tool0_to_spindle",
    joint_prim_relpath: str = "joints",
):
    global _current_ep_reward, _rl_index_buffer, _rl_reward_components_buffer
    
    try:
        num_envs = env.num_envs
        device = env.device
        env_ids = torch.arange(num_envs, device=device, dtype=torch.long)

        state6 = local_obs.get_ee_pose(env, asset_name=asset_name)
        wrench6 = local_ft_sensor.get_6axis_ft_fixed_joint(
            env=env,
            asset_name=asset_name,
            fixed_joint_name=fixed_joint_name,
            joint_prim_relpath=joint_prim_relpath,
            verbose=False,
        )
        force3 = wrench6[:, :3]

        sim_time = (
            float(getattr(env, "common_step_counter", 0)) *
            float(getattr(env, "step_dt", 0.02))
        )

        # ----------------------------------------------------
        # 🚀 [NEW] Index 및 개별 보상 추출 로직 추가
        # ----------------------------------------------------
        # 1. Index Progression
        try:
            term = env.action_manager.get_term(action_term_name)
            current_index = float(term._integration_manager.current_index[0].item())
        except Exception:
            current_index = 0.0
        _rl_index_buffer.append(current_index)

        # 2. Reward Components Tracking (0번 환경 기준)
        try:
            if hasattr(env, "reward_manager"):
                for term_name, value_tensor in env.reward_manager.episode_sums.items():
                    _rl_reward_components_buffer[term_name].append(float(value_tensor[0].item()))
        except Exception:
            pass
        # ----------------------------------------------------

        if hasattr(env, "reward_buf"):
            _current_ep_reward += float(env.reward_buf[0].item())

        rl_step(
            env_ids=env_ids,
            state6=state6,
            force3=force3,
            sim_time=sim_time,
        )

    except Exception as e:
        local_debug.print_exception("Visualization rl_step_hook failed", e)

    return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)