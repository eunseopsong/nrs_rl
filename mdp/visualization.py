# SPDX-License-Identifier: BSD-3-Clause
"""
Visualization utilities for training result plots.
"""

from __future__ import annotations

import os
import numpy as np
import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------
# Global
# -----------------------------------------------------------
version = "v29"

# -----------------------------------------------------------
# Run timestamp for unique output folder
# -----------------------------------------------------------
_run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

_position_tracking_history = []
_position_reward_history = []

_episode_counter_position = 0

_best_position_reward = -np.inf
_best_position_episode = -1
_current_episode_params = {}


def save_episode_plots_position(step: int):
    global _position_tracking_history, _position_reward_history, _episode_counter_position
    global _best_position_reward, _best_position_episode, _current_episode_params, _run_timestamp

    if not _position_tracking_history or not _position_reward_history:
        return

    save_dir = os.path.expanduser(f"~/nrs_rl/outputs/run_{_run_timestamp}/png/")
    reward_dir = os.path.expanduser(f"~/nrs_rl/outputs/run_{_run_timestamp}/rewards/")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(reward_dir, exist_ok=True)

    steps, targets, currents = zip(*_position_tracking_history)
    targets, currents = np.vstack(targets), np.vstack(currents)

    targets[:, 3:6] = np.unwrap(targets[:, 3:6], axis=0)
    currents[:, 3:6] = np.unwrap(currents[:, 3:6], axis=0)

    labels = ["x", "y", "z", "roll", "pitch", "yaw"]
    colors = ["r", "g", "b", "orange", "purple", "gray"]

    # 1. Tracking Plot
    plt.figure(figsize=(12, 8))
    for j in range(6):
        plt.plot(steps, targets[:, j], "--", color=colors[j], label=f"Target {labels[j]}")
        plt.plot(steps, currents[:, j], "-", color=colors[j], label=f"Current {labels[j]}")
    plt.legend(ncol=3, loc="upper left")
    plt.grid(True)
    plt.title(f"EE 6D Pose Tracking ({version})")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"pos_tracking_{version}_ep{_episode_counter_position+1}.png"))
    plt.close()

    # 2. Reward Plot & Best Logic
    r_steps, r_values = zip(*_position_reward_history)
    r_values_arr = np.array(r_values).flatten()

    episode_total_reward = float(np.sum(r_values_arr))

    if episode_total_reward > _best_position_reward:
        _best_position_reward = episode_total_reward
        _best_position_episode = _episode_counter_position + 1
        print("\n" + "🚀" * 25)
        print(f"🎉 [NEW BEST POSITION EPISODE] Episode {_best_position_episode} 🎉")
        print(f"Total Position Reward: {episode_total_reward:.4f}")
        print(f"Applied Params: {_current_episode_params}")
        print("🚀" * 25 + "\n")

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