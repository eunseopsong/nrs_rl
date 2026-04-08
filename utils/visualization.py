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
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# GLOBAL
# ============================================================
version = "v30"
_run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# RL buffers
_rl_time_buffer = []
_rl_state_buffer = []
_rl_force_buffer = []
_rl_start_time = None

# RL plot buffers
_position_tracking_history = []
_position_reward_history = []
_episode_counter = 0

_best_reward = -np.inf


# ============================================================
# ------------------ CORE PHYSICS -----------------------------
# ============================================================

def compute_velocity_mm_s(t, xyz):
    dt = np.diff(t, prepend=t[0])
    dt[0] = max(np.median(dt[1:]) if len(dt) > 1 else 0.02, 1e-6)
    dt = np.clip(dt, 1e-6, None)

    vxyz = np.zeros_like(xyz)
    if len(xyz) >= 2:
        dxyz = np.diff(xyz, axis=0)
        dt_seg = np.diff(t)
        dt_seg = np.clip(dt_seg, 1e-6, None)
        vxyz[1:] = dxyz / dt_seg[:, None]
        vxyz[0] = vxyz[1]

    return vxyz, dt


def compute_removal(t, state6, force3):
    xyz = state6[:, :3]
    wxyz = state6[:, 3:]

    vxyz, dt = compute_velocity_mm_s(t, xyz)
    speed = np.linalg.norm(vxyz[:, :2], axis=1)

    fn = np.maximum(force3[:, 2], 0.0)

    contact = (fn > 0.5) & (speed > 0.1)

    removal_rate = np.zeros_like(fn)
    removal_rate[contact] = fn[contact] * speed[contact]

    dremoval = removal_rate * dt

    return dremoval, removal_rate, xyz, wxyz


# ============================================================
# ------------------ RECORDING -------------------------------
# ============================================================

def record_step(state6, force3, sim_time):
    global _rl_time_buffer, _rl_state_buffer, _rl_force_buffer, _rl_start_time

    if _rl_start_time is None:
        _rl_start_time = sim_time

    t_rel = sim_time - _rl_start_time

    _rl_time_buffer.append(t_rel)
    _rl_state_buffer.append(state6.copy())
    _rl_force_buffer.append(force3.copy())


# ============================================================
# ------------------ VISUALIZATION ---------------------------
# ============================================================

def save_plots(out_dir, t, state6, force3, dremoval, removal_rate):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xyz = state6[:, :3]
    wxyz = state6[:, 3:]

    # 1️⃣ reward plot
    plt.figure(figsize=(10, 5))
    plt.plot(t, removal_rate, label="removal_rate")
    plt.grid()
    plt.legend()
    plt.title("Removal Rate")
    plt.savefig(out_dir / "01_reward.png")
    plt.close()

    # 2️⃣ xyz plot
    plt.figure(figsize=(10, 6))
    for i, label in enumerate(["x", "y", "z"]):
        plt.plot(t, xyz[:, i], label=label)
    plt.legend()
    plt.grid()
    plt.title("XYZ Trajectory")
    plt.savefig(out_dir / "02_xyz.png")
    plt.close()

    # 3️⃣ 3D path
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    ax.set_title("3D Path")
    fig.savefig(out_dir / "03_3d.png")
    plt.close()

    # 4️⃣ heatmap
    H, xedges, yedges = np.histogram2d(
        xyz[:, 0], xyz[:, 1],
        bins=50,
        weights=dremoval
    )

    plt.figure(figsize=(6, 5))
    plt.imshow(H.T, origin='lower')
    plt.colorbar(label="removal")
    plt.title("Removal Heatmap")
    plt.savefig(out_dir / "04_heatmap.png")
    plt.close()


# ============================================================
# ------------------ EPISODE PROCESS -------------------------
# ============================================================

def process_episode():
    global _rl_time_buffer, _rl_state_buffer, _rl_force_buffer
    global _rl_start_time, _episode_counter, _best_reward

    if len(_rl_time_buffer) < 2:
        return 0.0

    t = np.array(_rl_time_buffer)
    state6 = np.array(_rl_state_buffer)
    force3 = np.array(_rl_force_buffer)

    dremoval, removal_rate, xyz, wxyz = compute_removal(t, state6, force3)

    total_reward = float(np.sum(dremoval))

    # best tracking
    if total_reward > _best_reward:
        _best_reward = total_reward
        print(f"\n🔥 NEW BEST: {total_reward:.4f}\n")

    # save plots
    save_dir = f"./outputs/run_{_run_timestamp}/ep_{_episode_counter}"
    save_plots(save_dir, t, state6, force3, dremoval, removal_rate)

    # reset buffers
    _rl_time_buffer.clear()
    _rl_state_buffer.clear()
    _rl_force_buffer.clear()
    _rl_start_time = None

    _episode_counter += 1

    return total_reward


# ============================================================
# ------------------ RL INTERFACE ----------------------------
# ============================================================

def rl_step(state6, force3, sim_time):
    record_step(state6, force3, sim_time)


def rl_episode_done():
    return process_episode()


# ============================================================
# ------------------ EXAMPLE USAGE ---------------------------
# ============================================================

if __name__ == "__main__":
    # fake test (replace with real RL env)
    sim_time = 0.0

    for ep in range(3):
        for step in range(200):
            sim_time += 0.02

            state6 = np.random.randn(6) * 10
            force3 = np.random.randn(3)

            rl_step(state6, force3, sim_time)

        reward = rl_episode_done()
        print(f"Episode {ep} reward: {reward:.3f}")