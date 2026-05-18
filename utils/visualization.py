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
from mpl_toolkits.mplot3d.art3d import Line3DCollection

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
_last_processed_reset_step = None

_global_removal_history = {"x": [], "y": [], "removal": []}

GRID_SIZE = 128
CONTACT_FORCE_THRESHOLD_N = 0.5
MIN_CONTACT_FORCE_FRACTION = 0.05
MIN_SLIDING_SPEED_MM_S = 0.1

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


def _env_ids_include_tracked_env(env_ids, tracked_env_id: int = 0) -> bool:
    """Visualization records env0 only, so only env0 resets should finalize plots."""
    if env_ids is None:
        return True
    try:
        if isinstance(env_ids, torch.Tensor):
            return bool((env_ids.detach().to(dtype=torch.long) == tracked_env_id).any().item())
        if isinstance(env_ids, (list, tuple, set)):
            return tracked_env_id in {int(v) for v in env_ids}
        return int(env_ids) == tracked_env_id
    except Exception:
        return True


def _normal_force_from_force3(force3):
    return np.abs(np.asarray(force3, dtype=float)[:, 2])


def _contact_force_threshold(normal_force):
    max_force = float(np.nanmax(normal_force)) if len(normal_force) else 0.0
    if not np.isfinite(max_force) or max_force <= 0.0:
        return CONTACT_FORCE_THRESHOLD_N
    adaptive_threshold = max_force * MIN_CONTACT_FORCE_FRACTION
    return min(CONTACT_FORCE_THRESHOLD_N, adaptive_threshold)


def _spatial_angle_to_rotmat_np(spatial):
    spatial = np.asarray(spatial, dtype=float)
    angle = np.linalg.norm(spatial, axis=-1, keepdims=True)
    axis = np.divide(spatial, angle, out=np.zeros_like(spatial), where=angle > 1.0e-9)
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    c = np.cos(angle[:, 0])
    s = np.sin(angle[:, 0])
    one_c = 1.0 - c

    rot = np.empty((spatial.shape[0], 3, 3), dtype=float)
    rot[:, 0, 0] = c + x * x * one_c
    rot[:, 0, 1] = x * y * one_c - z * s
    rot[:, 0, 2] = x * z * one_c + y * s
    rot[:, 1, 0] = y * x * one_c + z * s
    rot[:, 1, 1] = c + y * y * one_c
    rot[:, 1, 2] = y * z * one_c - x * s
    rot[:, 2, 0] = z * x * one_c - y * s
    rot[:, 2, 1] = z * y * one_c + x * s
    rot[:, 2, 2] = c + z * z * one_c

    zero_mask = angle[:, 0] <= 1.0e-9
    rot[zero_mask] = np.eye(3)
    return rot


def _normal_vectors_from_state6(state6):
    if state6.shape[1] < 6:
        return None
    rot = _spatial_angle_to_rotmat_np(state6[:, 3:6])
    normal = rot[:, :, 2]
    norm = np.linalg.norm(normal, axis=1, keepdims=True)
    return np.divide(normal, norm, out=np.zeros_like(normal), where=norm > 1.0e-9)


def _set_axes_equal_3d(ax, xyz):
    mins = np.nanmin(xyz, axis=0)
    maxs = np.nanmax(xyz, axis=0)
    centers = (mins + maxs) * 0.5
    radius = float(np.nanmax(maxs - mins) * 0.55)
    radius = max(radius, 1.0)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)

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
    fn = _normal_force_from_force3(f_arr)
    dt = np.diff(t, prepend=t[0] - 1e-3)

    contact_threshold = _contact_force_threshold(fn)
    removal_rate = np.where((fn > contact_threshold) & (speed > MIN_SLIDING_SPEED_MM_S), fn * speed, 0.0)
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
    bins = min(140, max(64, int(np.sqrt(len(x)) * 3)))
    grid_removal, _, _ = np.histogram2d(x, y, bins=bins, range=[extent[:2], extent[2:]], weights=dremoval)
    grid_removal_smoothed = gaussian_filter(grid_removal.T, sigma=2.6)
    max_removal = float(np.nanmax(grid_removal_smoothed)) if grid_removal_smoothed.size else 0.0
    if max_removal > 0.0 and np.isfinite(max_removal):
        positive_values = grid_removal_smoothed[grid_removal_smoothed > 0.0]
        robust_max = float(np.nanpercentile(positive_values, 99.5))
        normalizer = max(robust_max, max_removal * 0.25, 1e-12)
        grid_display = np.clip(grid_removal_smoothed / normalizer, 0.0, 1.0)
    else:
        positive_values = np.array([], dtype=float)
        grid_display = grid_removal_smoothed

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(grid_display, origin="lower", extent=extent, cmap="viridis", vmin=0.0, vmax=1.0, interpolation="bilinear")
    fig.colorbar(im, label="Cell removal [a.u.]")
    ax.set_title("Removal Heatmap")
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_aspect("equal", adjustable="box")
    display_positive_values = grid_display[grid_display > 0.0]
    stats_mean = float(np.mean(display_positive_values)) if display_positive_values.size else 0.0
    stats_std = float(np.std(display_positive_values)) if display_positive_values.size else 0.0
    contact_samples = int(np.count_nonzero(dremoval > 0.0))
    stats_text = (
        f"mean = {stats_mean:.6f} [a.u.]\n"
        f"std  = {stats_std:.6f} [a.u.]\n"
        f"samples = {len(t)}\n"
        f"contact = {contact_samples}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "black", "alpha": 0.85},
    )
    fig.tight_layout()
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

    normal_force = _normal_force_from_force3(force3)
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

    fig4 = plt.figure(figsize=(9.5, 8))
    ax4 = fig4.add_subplot(111, projection="3d")

    path_values = removal_rate if len(removal_rate) == len(xyz) else sliding_velocity
    path_values = np.asarray(path_values, dtype=float)
    if len(path_values) == len(xyz) and np.nanmax(path_values) > np.nanmin(path_values):
        segments = np.stack([xyz[:-1], xyz[1:]], axis=1)
        segment_values = 0.5 * (path_values[:-1] + path_values[1:])
        line_collection = Line3DCollection(segments, cmap="viridis", linewidth=2.2, alpha=0.95)
        line_collection.set_array(segment_values)
        ax4.add_collection3d(line_collection)
        cbar4 = fig4.colorbar(line_collection, ax=ax4, pad=0.08, shrink=0.72)
        cbar4.set_label("Removal rate [a.u.]")
    else:
        ax4.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="#1f77b4", linewidth=2.0, label="EE path")

    z_floor = float(np.nanmin(xyz[:, 2]))
    ax4.plot(
        xyz[:, 0],
        xyz[:, 1],
        np.full_like(xyz[:, 2], z_floor),
        color="0.35",
        linestyle="--",
        linewidth=1.0,
        alpha=0.45,
        label="XY projection",
    )

    normals = _normal_vectors_from_state6(state6)
    if normals is not None and len(normals) == len(xyz):
        stride = max(1, len(xyz) // 28)
        normal_scale = max(float(np.ptp(xyz[:, 0])), float(np.ptp(xyz[:, 1])), float(np.ptp(xyz[:, 2])), 1.0) * 0.08
        q_xyz = xyz[::stride]
        q_normals = normals[::stride] * normal_scale
        ax4.quiver(
            q_xyz[:, 0],
            q_xyz[:, 1],
            q_xyz[:, 2],
            q_normals[:, 0],
            q_normals[:, 1],
            q_normals[:, 2],
            color="#d62728",
            linewidth=0.8,
            arrow_length_ratio=0.25,
            alpha=0.85,
            normalize=False,
        )
        ax4.plot([], [], [], color="#d62728", linewidth=1.6, label="TCP normal")

    ax4.scatter(xyz[0, 0], xyz[0, 1], xyz[0, 2], color="#2ca02c", s=55, depthshade=True, label="start")
    ax4.scatter(xyz[-1, 0], xyz[-1, 1], xyz[-1, 2], color="#d62728", s=55, depthshade=True, label="end")
    ax4.set_title(f"Episode {_episode_counter} EE Path and TCP Normals")
    ax4.set_xlabel("X [mm]")
    ax4.set_ylabel("Y [mm]")
    ax4.set_zlabel("Z [mm]")
    _set_axes_equal_3d(ax4, xyz)
    ax4.view_init(elev=28, azim=-58)
    ax4.grid(True, alpha=0.25)
    ax4.legend(loc="upper left")
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

def _clear_episode_buffers():
    global _rl_start_time, _current_ep_reward
    _rl_time_buffer.clear()
    _rl_state_buffer.clear()
    _rl_force_buffer.clear()
    _rl_index_buffer.clear()
    _rl_sliding_velocity_buffer.clear()
    _rl_reward_components_buffer.clear()
    _rl_start_time = None
    _current_ep_reward = 0.0

def _get_visualization_cfg(env):
    return getattr(getattr(env, "cfg", None), "visualization", None)

def _visualization_enabled(env) -> bool:
    cfg = _get_visualization_cfg(env)
    return bool(getattr(cfg, "enable_visualizer", True))

def _should_save_episode(env) -> bool:
    cfg = _get_visualization_cfg(env)
    interval = int(getattr(cfg, "save_interval_episodes", 1))
    if interval <= 0:
        return False
    return (_episode_counter % interval) == 0

def on_episode_reset(env, env_ids=None):
    global _has_seen_any_step, _current_ep_reward, _episode_counter, _last_processed_reset_step
    try:
        if not _visualization_enabled(env):
            return
        if not _env_ids_include_tracked_env(env_ids):
            return
        if not _has_seen_any_step:
            return
        reset_step = int(getattr(env, "common_step_counter", -1)) if env is not None else -1
        if reset_step >= 0 and _last_processed_reset_step == reset_step:
            return
        if len(_rl_time_buffer) > 0:
            if _should_save_episode(env):
                rl_episode_done()
            else:
                _clear_episode_buffers()
                _episode_counter += 1
            _last_processed_reset_step = reset_step

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
        if not _visualization_enabled(env):
            return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)

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
