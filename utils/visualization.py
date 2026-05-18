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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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

local_debug.print_info(f"[Init] Polishing log directory configured: {RUN_LOG_DIR}")

# ============================================================
# Global Buffers
# ============================================================

_rl_time_buffer = []
_rl_state_buffer = []
_rl_force_buffer = []
_rl_index_buffer = []
_rl_sliding_velocity_buffer = []
_rl_reward_components_buffer = defaultdict(list)

_current_ep_reward = 0.0

_rl_start_time = None
_episode_counter = 1
_has_seen_any_step = False

_global_removal_history = {"x": [], "y": [], "removal": []}

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
    return

# ============================================================
# Episode Processing
# ============================================================

def process_episode():
    global _rl_time_buffer, _rl_state_buffer, _rl_force_buffer
    global _rl_index_buffer, _rl_sliding_velocity_buffer, _rl_reward_components_buffer
    global _rl_start_time, _episode_counter, _summary_metrics
    global _current_ep_reward, _global_removal_history

    if len(_rl_time_buffer) < 5: # 스텝 수가 너무 적어도 진행되도록 허들 낮춤
        _rl_time_buffer.clear()
        _rl_state_buffer.clear()
        _rl_force_buffer.clear()
        _rl_index_buffer.clear()
        _rl_sliding_velocity_buffer.clear()
        _rl_reward_components_buffer.clear()
        _rl_start_time = None
        _current_ep_reward = 0.0
        return 0.0

    t = np.array(_rl_time_buffer, dtype=float)
    s_arr = np.array(_rl_state_buffer, dtype=float)
    f_arr = np.array(_rl_force_buffer, dtype=float)
    idx_arr = np.array(_rl_index_buffer, dtype=float)
    sliding_velocity_arr = np.array(_rl_sliding_velocity_buffer, dtype=float)
    rw_dict = dict(_rl_reward_components_buffer)

    xyz = s_arr[:, :3]

    vxyz = np.zeros_like(xyz)
    for i in range(3):
        vxyz[:, i] = np.gradient(xyz[:, i], t)

    speed = np.linalg.norm(vxyz[:, :2], axis=1)
    if len(sliding_velocity_arr) == len(t) and np.all(np.isfinite(sliding_velocity_arr)):
        speed = sliding_velocity_arr
    fn = np.maximum(f_arr[:, 2], 0.0)
    dt = np.diff(t, prepend=t[0] - 1e-3)

    removal_rate = np.where((fn > 0.5) & (speed > 0.1), fn * speed, 0.0)
    dremoval = removal_rate * dt

    _global_removal_history["x"].extend(xyz[:, 0].tolist())
    _global_removal_history["y"].extend(xyz[:, 1].tolist())
    _global_removal_history["removal"].extend(dremoval.tolist())
    for i in range(len(dremoval)):
        update_surface_grid(xyz[i, 0], xyz[i, 1], dremoval[i])

    ep_dir = RUN_LOG_DIR / f"ep{_episode_counter}"
    save_plots(ep_dir, t, s_arr, f_arr, dremoval, removal_rate, vxyz, speed)

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

    local_debug.print_info(f"[STAMP] Ep {_episode_counter} Saved. Reward: {_current_ep_reward:.2f}. Result: {ep_dir}")

    _episode_counter += 1
    
    _rl_time_buffer.clear()
    _rl_state_buffer.clear()
    _rl_force_buffer.clear()
    _rl_index_buffer.clear()
    _rl_sliding_velocity_buffer.clear()
    _rl_reward_components_buffer.clear()
    _rl_start_time = None
    _current_ep_reward = 0.0 

    return float(np.sum(dremoval))

# ============================================================
# Plot Saving
# ============================================================

def save_plots(out_dir, t, state6, force3, dremoval, removal_rate, vxyz, sliding_velocity):
    xyz = state6[:, 0:3]
    if len(t) == 0 or xyz.shape[0] == 0:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

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
    fig.savefig(out_dir / "01_removal_heatmap.png", dpi=200)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.plot(t, removal_rate, color="tab:blue", linewidth=1.8)
    ax2.axhline(np.mean(removal_rate), color="tab:orange", linestyle="--", linewidth=1.2, label="episode mean")
    ax2.set_title(f"Episode {_episode_counter} Removal Rate")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Normal Force x Sliding Velocity")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out_dir / "02_heatmap_value_vs_time.png", dpi=200)
    plt.close(fig2)

    normal_force = np.maximum(force3[:, 2], 0.0)
    fig3, axes3 = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes3[0].plot(t, normal_force, color="tab:red", linewidth=1.5)
    axes3[0].set_title("Normal Force")
    axes3[0].set_ylabel("Force [N]")
    axes3[0].grid(True, alpha=0.3)
    axes3[1].plot(t, sliding_velocity, color="tab:green", linewidth=1.5)
    axes3[1].set_title("Sliding Velocity")
    axes3[1].set_xlabel("Time [s]")
    axes3[1].set_ylabel("Velocity [mm/s]")
    axes3[1].grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(out_dir / "03_signals_subplot.png", dpi=200)
    plt.close(fig3)

    fig4 = plt.figure(figsize=(8, 7))
    ax4 = fig4.add_subplot(111, projection="3d")
    ax4.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="tab:blue", linewidth=1.5)
    ax4.scatter(xyz[0, 0], xyz[0, 1], xyz[0, 2], color="green", s=35, label="start")
    ax4.scatter(xyz[-1, 0], xyz[-1, 1], xyz[-1, 2], color="red", s=35, label="end")
    ax4.set_title(f"Episode {_episode_counter} EE Path")
    ax4.set_xlabel("X [mm]")
    ax4.set_ylabel("Y [mm]")
    ax4.set_zlabel("Z [mm]")
    ax4.legend()
    fig4.tight_layout()
    fig4.savefig(out_dir / "04_3d_path_w.png", dpi=200)
    plt.close(fig4)

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
            
    except Exception as e:
        local_debug.print_exception("Visualization on_episode_reset failed", e)

# ============================================================
# Step Hook
# ============================================================

def rl_step_hook(env, action_term_name="arm_action", asset_name="robot", fixed_joint_name="tool0_to_spindle", joint_prim_relpath="joints"):
    global _current_ep_reward, _rl_index_buffer, _rl_sliding_velocity_buffer, _rl_reward_components_buffer
    
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
            if hasattr(term, "path_cursor"):
                _rl_index_buffer.append(float(term.path_cursor[0].item()))
            elif hasattr(term, "current_target_index"):
                _rl_index_buffer.append(float(term.current_target_index[0].item()))
            else:
                _rl_index_buffer.append(0.0)
            if hasattr(term, "current_sliding_velocity_mm_s"):
                _rl_sliding_velocity_buffer.append(float(term.current_sliding_velocity_mm_s[0].item()))
            else:
                _rl_sliding_velocity_buffer.append(float("nan"))
        except Exception:
            _rl_index_buffer.append(0.0)
            _rl_sliding_velocity_buffer.append(float("nan"))

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
