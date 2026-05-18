# SPDX-License-Identifier: BSD-3-Clause
"""
Unified RL + Polishing Removal + Visualization Module (Fully Repaired)
"""

from __future__ import annotations

import numpy as np
import datetime
from pathlib import Path
import importlib
from collections import defaultdict

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
_rl_index_buffer = []
_rl_reward_components_buffer = defaultdict(list)

_current_ep_reward = 0.0

_rl_start_time = None
_episode_counter = 1
_has_seen_any_step = False

# 🚀 [FIX] 고정 Grid 대신, 누적 좌표와 제거량을 동적으로 저장하는 버퍼로 변경
_global_removal_history = {"x": [], "y": [], "removal": []}

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

# ============================================================
# Step Recording
# ============================================================

def record_step(env_ids, state6, force3, sim_time):
    global _rl_time_buffer, _rl_state_buffer, _rl_force_buffer
    global _rl_start_time, _has_seen_any_step

    _has_seen_any_step = True

    if _rl_start_time is None:
        _rl_start_time = float(sim_time)

    try:
        if isinstance(env_ids, torch.Tensor):
            target_mask = (env_ids == 0)
            idx = target_mask.nonzero(as_tuple=True)[0][0].item() if bool(target_mask.any()) else 0
        else:
            idx = 0

        _rl_time_buffer.append(float(sim_time) - float(_rl_start_time))

        s_val = state6[idx].detach().cpu().numpy() if hasattr(state6, "detach") else np.asarray(state6[idx])
        f_val = force3[idx].detach().cpu().numpy() if hasattr(force3, "detach") else np.asarray(force3[idx])

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

    axes[0, 0].plot(eps, _summary_metrics["total_removal"], "b-o", markersize=4, alpha=0.7)
    axes[0, 0].set_title("Total Removal per Episode")
    axes[0, 0].set_ylabel("Total Removal [a.u.]")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(eps, _summary_metrics["mean_mrr_error"], "r-x", markersize=4, alpha=0.7)
    axes[0, 1].set_title("MRR Tracking Error (Lower is better)")
    axes[0, 1].set_ylabel("Mean Abs Error")
    axes[0, 1].grid(True, alpha=0.3)

    reward_arr = np.array(_summary_metrics["episode_reward"])
    axes[1, 0].plot(eps, reward_arr, color="green", marker="s", markersize=4, alpha=0.3, label="Raw Reward")
    axes[1, 0].plot(eps, moving_average(reward_arr), color="green", linewidth=2.5, label="Trend (MA)")
    axes[1, 0].set_title("Total Reward per Episode")
    axes[1, 0].set_ylabel("Cumulative Reward")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(eps, _summary_metrics["contact_ratio"], "m-d", markersize=4, alpha=0.7)
    axes[1, 1].set_title("Contact Ratio")
    axes[1, 1].set_ylabel("Ratio (Contact/Total)")
    axes[1, 1].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Episode")

    fig.tight_layout()
    fig.savefig(RUN_LOG_DIR / "overall_learning_trend.png", dpi=200)
    plt.close(fig)

# ============================================================
# Episode Processing
# ============================================================

def process_episode():
    global _rl_time_buffer, _rl_state_buffer, _rl_force_buffer
    global _rl_index_buffer, _rl_reward_components_buffer
    global _rl_start_time, _episode_counter, _summary_metrics
    global _current_ep_reward, _global_removal_history

    if len(_rl_time_buffer) < 5: # 스텝 수가 너무 적어도 진행되도록 허들 낮춤
        _rl_time_buffer.clear()
        _rl_state_buffer.clear()
        _rl_force_buffer.clear()
        _rl_index_buffer.clear()
        _rl_reward_components_buffer.clear()
        _rl_start_time = None
        _current_ep_reward = 0.0
        return 0.0

    t = np.array(_rl_time_buffer, dtype=float)
    s_arr = np.array(_rl_state_buffer, dtype=float)
    f_arr = np.array(_rl_force_buffer, dtype=float)
    idx_arr = np.array(_rl_index_buffer, dtype=float)
    rw_dict = dict(_rl_reward_components_buffer)

    xyz = s_arr[:, :3]

    vxyz = np.zeros_like(xyz)
    for i in range(3):
        vxyz[:, i] = np.gradient(xyz[:, i], t)

    speed = np.linalg.norm(vxyz[:, :2], axis=1)
    fn = np.maximum(f_arr[:, 2], 0.0)
    dt = np.diff(t, prepend=t[0] - 1e-3)

    removal_rate = np.where((fn > 0.5) & (speed > 0.1), fn * speed, 0.0)
    dremoval = removal_rate * dt

    # 🚀 [FIX] 누적 히스토리에 동적 추가
    _global_removal_history["x"].extend(xyz[:, 0].tolist())
    _global_removal_history["y"].extend(xyz[:, 1].tolist())
    _global_removal_history["removal"].extend(dremoval.tolist())

    ep_dir = RUN_LOG_DIR / f"ep{_episode_counter}"
    save_plots(ep_dir, t, s_arr, f_arr, dremoval, removal_rate, vxyz, idx_arr, rw_dict)

    samples = len(t)
    contact_samples = int(np.sum(fn > 0.5))
    rate_mean = float(np.mean(removal_rate))
    mrr_error = np.mean(np.abs(removal_rate - rate_mean))

    _summary_metrics["episode"].append(_episode_counter)
    _summary_metrics["total_removal"].append(float(np.sum(dremoval)))
    _summary_metrics["mean_mrr_error"].append(float(mrr_error))
    _summary_metrics["std_removal"].append(float(np.std(dremoval)))
    _summary_metrics["contact_ratio"].append(contact_samples / samples if samples > 0 else 0)
    _summary_metrics["episode_reward"].append(_current_ep_reward)

    if _episode_counter % 5 == 0:
        save_global_summary()

    local_debug.print_info(f"[STAMP] Ep {_episode_counter} Saved. Reward: {_current_ep_reward:.2f}. Result: {ep_dir}")

    _episode_counter += 1
    
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

    x, y = xyz[:, 0], xyz[:, 1]
    
    # 예외처리: 이동이 전혀 없어서 max==min이 되는 경우 방지
    x_ptp, y_ptp = np.ptp(x), np.ptp(y)
    margin_x = 10.0 if x_ptp < 1.0 else x_ptp * 0.1
    margin_y = 10.0 if y_ptp < 1.0 else y_ptp * 0.1
    
    extent = [np.min(x) - margin_x, np.max(x) + margin_x, np.min(y) - margin_y, np.max(y) + margin_y]

    # --- 1. Local Removal Heatmap ---
    grid_removal, _, _ = np.histogram2d(x, y, bins=150, range=[extent[:2], extent[2:]], weights=dremoval)
    grid_removal_smoothed = gaussian_filter(grid_removal.T, sigma=2.5)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(grid_removal_smoothed, origin="lower", extent=extent, cmap="viridis")
    fig.colorbar(im, label="Removal [a.u.]")
    ax.set_title(f"Episode {_episode_counter} Local Removal Heatmap")
    fig.savefig(out_dir / "01_local_removal_heatmap.png", dpi=200)
    plt.close(fig)

    # 🚀 --- 2. [FIX] Global Surface Map 동적 스케일링 렌더링 ---
    gx = np.array(_global_removal_history["x"])
    gy = np.array(_global_removal_history["y"])
    grem = np.array(_global_removal_history["removal"])
    
    if len(gx) > 0:
        g_extent = [np.min(gx) - 20, np.max(gx) + 20, np.min(gy) - 20, np.max(gy) + 20]
        global_grid, _, _ = np.histogram2d(gx, gy, bins=200, range=[g_extent[:2], g_extent[2:]], weights=grem)
        global_grid_smoothed = gaussian_filter(global_grid.T, sigma=2.0)

        fig2, ax2 = plt.subplots(figsize=(8, 7))
        im2 = ax2.imshow(global_grid_smoothed, origin="lower", extent=g_extent, cmap="viridis")
        fig2.colorbar(im2)
        ax2.set_title(f"Cumulative Global Surface Removal (Up to Ep {_episode_counter})")
        fig2.savefig(out_dir / "02_global_surface_map.png", dpi=200)
        plt.close(fig2)

    # --- 3. Diagnostic Panel ---
    fig3, axes3 = plt.subplots(2, 4, figsize=(24, 10))
    
    axes3[0, 0].plot(x, y, 'b-', linewidth=1.5)
    axes3[0, 0].set_title("Toolpath Trajectory (X-Y)")
    axes3[0, 0].set_xlabel("X")
    axes3[0, 0].set_ylabel("Y")
    axes3[0, 0].axis("equal")
    axes3[0, 0].grid(True)

    force_mag = np.linalg.norm(force3, axis=1)
    axes3[0, 1].plot(t, force_mag, 'r-')
    axes3[0, 1].set_title("Force Magnitude over Time")
    axes3[0, 1].grid(True)

    if index_arr is not None and len(index_arr) == len(t):
        axes3[0, 2].plot(t, index_arr, 'g-')
    axes3[0, 2].set_title("Trajectory Index Progression")
    axes3[0, 2].grid(True)

    if reward_dict:
        for term_name, values in reward_dict.items():
            plot_len = min(len(t), len(values))
            axes3[0, 3].plot(t[:plot_len], values[:plot_len], label=term_name)
        axes3[0, 3].legend(loc='upper left', fontsize='small')
    axes3[0, 3].set_title("Cumulative Rewards Breakdown")
    axes3[0, 3].grid(True)

    visited_mask = (grid_removal.T > 1e-4).astype(np.uint8)
    axes3[1, 0].imshow(visited_mask, origin="lower", extent=extent, cmap="gray")
    axes3[1, 0].set_title(f"Coverage Mask (Ratio: {np.mean(visited_mask):.3f})")

    axes3[1, 1].plot(t, (force_mag > 0.5).astype(int), 'm-')
    axes3[1, 1].set_ylim([-0.1, 1.1])
    axes3[1, 1].set_title("Contact State (F > 0.5)")
    axes3[1, 1].grid(True)

    axes3[1, 2].plot(t, np.linalg.norm(vxyz[:, :2], axis=1), 'c-')
    axes3[1, 2].set_title("Tool Speed (XY Plane)")
    axes3[1, 2].grid(True)

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
            save_global_summary()
            
    except Exception as e:
        local_debug.print_exception("Visualization on_episode_reset failed", e)

# ============================================================
# Step Hook
# ============================================================

def rl_step_hook(env, action_term_name="arm_action", asset_name="robot", fixed_joint_name="tool0_to_spindle", joint_prim_relpath="joints"):
    global _current_ep_reward, _rl_index_buffer, _rl_reward_components_buffer
    
    try:
        num_envs = env.num_envs
        device = env.device
        env_ids = torch.arange(num_envs, device=device, dtype=torch.long)

        state6 = local_obs.get_ee_pose(env, asset_name=asset_name)
        wrench6 = local_ft_sensor.get_6axis_ft_fixed_joint(env, asset_name, fixed_joint_name, joint_prim_relpath, verbose=False)
        force3 = wrench6[:, :3]

        sim_time = float(getattr(env, "common_step_counter", 0)) * float(getattr(env, "step_dt", 0.02))

        try:
            term = env.action_manager.get_term(action_term_name)
            _rl_index_buffer.append(float(term._integration_manager.current_index[0].item()))
        except Exception:
            _rl_index_buffer.append(0.0)

        # 🚀 [FIX] Reward Tracking 보완 (IsaacLab의 protected 속성까지 긁어옴)
        try:
            if hasattr(env, "reward_manager"):
                reward_dict = getattr(env.reward_manager, "episode_sums", getattr(env.reward_manager, "_episode_sums", {}))
                for term_name, value_tensor in reward_dict.items():
                    _rl_reward_components_buffer[term_name].append(float(value_tensor[0].item()))
        except Exception:
            pass

        if hasattr(env, "reward_buf"):
            _current_ep_reward += float(env.reward_buf[0].item())

        rl_step(env_ids, state6, force3, sim_time)

    except Exception as e:
        local_debug.print_exception("Visualization rl_step_hook failed", e)

    return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)