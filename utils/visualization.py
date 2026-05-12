# SPDX-License-Identifier: BSD-3-Clause
"""
Unified RL + Polishing Removal + Visualization Module
"""

from __future__ import annotations

import numpy as np
import datetime
from pathlib import Path
import importlib

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
    "surface_uniformity": [],
    "uniformity_improvement": [],
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

    # --- Total Removal ---
    axes[0, 0].plot(eps, _summary_metrics["total_removal"], "b-o", markersize=4, alpha=0.7)
    axes[0, 0].set_title("Total Removal per Episode")
    axes[0, 0].set_ylabel("Total Removal [a.u.]")
    axes[0, 0].grid(True, alpha=0.3)

    # --- MRR Error ---
    axes[0, 1].plot(eps, _summary_metrics["mean_mrr_error"], "r-x", markersize=4, alpha=0.7)
    axes[0, 1].set_title("MRR Tracking Error")
    axes[0, 1].set_ylabel("Mean Abs Error")
    axes[0, 1].grid(True, alpha=0.3)

    # --- Uniformity Improvement ---
    improvement = np.array(_summary_metrics["uniformity_improvement"])
    axes[1, 0].plot(eps, improvement, color="green", alpha=0.25, linewidth=1.5)
    axes[1, 0].plot(eps, moving_average(improvement), color="green", linewidth=3)
    axes[1, 0].set_title("Surface Uniformity Improvement (%)")
    axes[1, 0].set_ylabel("Improvement [%]")
    axes[1, 0].grid(True, alpha=0.3)

    # --- Contact Ratio ---
    axes[1, 1].plot(eps, _summary_metrics["contact_ratio"], "m-d", markersize=4, alpha=0.7)
    axes[1, 1].set_title("Contact Ratio")
    axes[1, 1].set_ylabel("Ratio")
    axes[1, 1].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Episode")

    fig.suptitle(
        f"Overall Polishing Performance Trend (Up to Ep {_episode_counter - 1})",
        fontsize=16,
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
    global _rl_start_time, _episode_counter, _summary_metrics

    if len(_rl_time_buffer) < 10:
        _rl_time_buffer.clear()
        _rl_state_buffer.clear()
        _rl_force_buffer.clear()
        _rl_start_time = None
        return 0.0

    t = np.array(_rl_time_buffer, dtype=float)
    s_arr = np.array(_rl_state_buffer, dtype=float)
    f_arr = np.array(_rl_force_buffer, dtype=float)
    xyz = s_arr[:, :3]

    # --- Velocity calculation ---
    vxyz = np.zeros_like(xyz)
    for i in range(3):
        vxyz[:, i] = np.gradient(xyz[:, i], t)

    speed = np.linalg.norm(vxyz[:, :2], axis=1)
    fn = np.maximum(f_arr[:, 2], 0.0)
    dt = np.diff(t, prepend=t[0] - 1e-3)

    # --- Removal logic ---
    removal_rate = np.where((fn > 0.5) & (speed > 0.1), fn * speed, 0.0)
    dremoval = removal_rate * dt

    # --- Surface Memory Update ---
    for i in range(len(dremoval)):
        update_surface_grid(xyz[i, 0], xyz[i, 1], dremoval[i])

    ep_dir = RUN_LOG_DIR / f"ep{_episode_counter}"
    save_plots(ep_dir, t, s_arr, f_arr, dremoval, removal_rate, vxyz)

    # --- Metrics calculation ---
    samples = len(t)
    contact_samples = int(np.sum(fn > 0.5))
    rate_mean = float(np.mean(removal_rate))
    mrr_error = np.mean(np.abs(removal_rate - rate_mean))

    valid_surface = _surface_grid[_surface_grid > 0]
    surface_uniformity = float(np.std(valid_surface)) if len(valid_surface) > 10 else 0.0

    # --- Improvement calculation ---
    if len(_summary_metrics["surface_uniformity"]) == 0:
        improvement = 0.0
    else:
        initial_uniformity = _summary_metrics["surface_uniformity"][0]
        improvement = ((initial_uniformity - surface_uniformity) / max(initial_uniformity, 1e-6)) * 100.0

    # --- Store Metrics ---
    _summary_metrics["episode"].append(_episode_counter)
    _summary_metrics["total_removal"].append(float(np.sum(dremoval)))
    _summary_metrics["mean_mrr_error"].append(float(mrr_error))
    _summary_metrics["std_removal"].append(float(np.std(dremoval)))
    _summary_metrics["contact_ratio"].append(contact_samples / samples if samples > 0 else 0)
    _summary_metrics["surface_uniformity"].append(surface_uniformity)
    _summary_metrics["uniformity_improvement"].append(improvement)

    if _episode_counter % 5 == 0:
        save_global_summary()

    local_debug.print_info(f"\n[STAMP] Episode {_episode_counter} saved. Results at: {ep_dir}\n")

    _episode_counter += 1
    _rl_time_buffer.clear()
    _rl_state_buffer.clear()
    _rl_force_buffer.clear()
    _rl_start_time = None

    return float(np.sum(dremoval))


# ============================================================
# Plot Saving
# ============================================================

def save_plots(out_dir, t, state6, force3, dremoval, removal_rate, vxyz):
    out_dir.mkdir(parents=True, exist_ok=True)
    xyz = state6[:, 0:3]

    # --- Local Removal Heatmap ---
    x, y = xyz[:, 0], xyz[:, 1]
    margin = 10.0
    extent = [np.min(x) - margin, np.max(x) + margin, np.min(y) - margin, np.max(y) + margin]

    grid_removal, _, _ = np.histogram2d(
        x, y, bins=150, range=[extent[:2], extent[2:]], weights=dremoval
    )
    grid_removal = gaussian_filter(grid_removal.T, sigma=2.5)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(grid_removal, origin="lower", extent=extent, cmap="viridis")
    fig.colorbar(im, label="Removal [a.u.]")
    ax.set_title(f"Episode {_episode_counter} Removal Heatmap")
    fig.savefig(out_dir / "01_removal_heatmap.png", dpi=200)
    plt.close(fig)

    # --- Global Surface Map ---
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    global_map = gaussian_filter(_surface_grid, sigma=2.0)
    im2 = ax2.imshow(global_map, origin="lower", cmap="viridis")
    fig2.colorbar(im2)
    ax2.set_title(f"Cumulative Surface Removal (Ep {_episode_counter})")
    fig2.savefig(out_dir / "02_global_surface_map.png", dpi=200)
    plt.close(fig2)


# ============================================================
# RL Hooks
# ============================================================

def rl_step(env_ids, state6, force3, sim_time):
    record_step(env_ids, state6, force3, sim_time)


def rl_episode_done():
    return process_episode()


def on_episode_reset(env, env_ids=None):
    global _has_seen_any_step
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

        rl_step(
            env_ids=env_ids,
            state6=state6,
            force3=force3,
            sim_time=sim_time,
        )

    except Exception as e:
        local_debug.print_exception("Visualization rl_step_hook failed", e)

    return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)