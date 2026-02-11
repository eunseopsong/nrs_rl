# SPDX-License-Identifier: BSD-3-Clause
"""
v27: Dual Tracking Reward (Joint + Cartesian, Wrap-Safe + Unwrap Visualization)
----------------------------------------------------------------------------------
- Reward Type: Exponential kernel (no tanh)
- Goal: Joint-space + Cartesian-space 병렬 학습 + Orientation wrap-safe error 계산
        + 시각화 시 np.unwrap 적용 (roll/pitch/yaw 연속 표시)

Notes (port to nrs_rl):
    - imports: local .observations
    - target_vel uses env dt * decimation (important if dt != 1/30)
    - angle wrap is computed in torch (no cpu numpy conversion)
    - output dir: ~/nrs_rl/outputs/...
"""

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import os
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ✅ nrs_rl 구조: 같은 mdp 폴더의 observations.py
from .observations import (
    get_hdf5_target_joints,
    get_hdf5_target_positions,
    get_ee_pose,
)

# -----------------------------------------------------------
# Global
# -----------------------------------------------------------
version = "v27"

_joint_tracking_history = []
_joint_reward_history = []
_position_tracking_history = []
_position_reward_history = []

_episode_counter_joint = 0
_episode_counter_position = 0


# -----------------------------------------------------------
# Utility: angle wrap correction (torch, GPU-safe)
# -----------------------------------------------------------
def angle_diff_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute minimal difference between two angles (radians), wrapped to [-pi, pi].
    a, b: (..., 3) tensor
    """
    two_pi = 2.0 * np.pi
    return torch.remainder(a - b + np.pi, two_pi) - np.pi


# -----------------------------------------------------------
# (1) Joint Tracking Reward
# -----------------------------------------------------------
def joint_tracking_reward(env: "ManagerBasedRLEnv"):
    """Joint-space tracking reward (exponential kernel)"""
    robot = env.scene["robot"]
    q, qd = robot.data.joint_pos[:, :6], robot.data.joint_vel[:, :6]

    dt = getattr(env.sim, "dt", 1.0 / 30.0) * getattr(env, "decimation", 1)
    D = q.shape[1]
    step = int(env.common_step_counter)

    fut = get_hdf5_target_joints(env, horizon=8)  # (N, 8*D)
    q_star_curr, q_star_next = fut[:, :D], fut[:, D:2 * D]
    qd_star = (q_star_next - q_star_curr) / (dt + 1e-8)

    e_q, e_qd = q - q_star_next, qd - qd_star

    wj = torch.tensor([1.0, 2.0, 1.0, 4.0, 1.0, 1.0], device=q.device).unsqueeze(0)
    k_pos = torch.tensor([1.0, 8.0, 2.0, 6.0, 2.0, 2.0], device=q.device).unsqueeze(0)
    k_vel = torch.tensor([0.10, 0.40, 0.10, 0.40, 0.10, 0.10], device=q.device).unsqueeze(0)

    e_q2, e_qd2 = wj * (e_q ** 2), wj * (e_qd ** 2)
    r_pose_jointwise = torch.exp(-k_pos * e_q2)
    r_vel_jointwise = torch.exp(-k_vel * e_qd2)

    r_pose, r_vel = r_pose_jointwise.sum(dim=1), r_vel_jointwise.sum(dim=1)
    total = 0.9 * r_pose + 0.1 * r_vel

    if step % 10 == 0:
        print(f"[Joint Step {step}] mean(|e_q|)={torch.norm(e_q, dim=1).mean():.3f}, total={total.mean():.3f}")
        mean_e_q_abs = torch.mean(torch.abs(e_q), dim=0).detach().cpu().numpy()
        mean_r_pose = torch.mean(r_pose_jointwise, dim=0).detach().cpu().numpy()
        for j in range(D):
            print(f"  joint{j+1}: |mean(e_q)|={mean_e_q_abs[j]:.3f}, r_pose={mean_r_pose[j]:.3f}")

    _joint_tracking_history.append((step, q_star_next[0].detach().cpu().numpy(), q[0].detach().cpu().numpy()))
    _joint_reward_history.append((step, r_pose_jointwise[0].detach().cpu().numpy()))

    if hasattr(env, "max_episode_length") and env.max_episode_length > 0:
        episode_steps = int(env.max_episode_length)
        if step > 0 and (step % episode_steps == episode_steps - 1):
            save_episode_plots_joint(step)

    return total


# -----------------------------------------------------------
# (2) Position Tracking Reward (6D + velocity)
# -----------------------------------------------------------
def position_tracking_reward(env: "ManagerBasedRLEnv"):
    """6D EE pose + velocity tracking reward (wrap-safe orientation)"""
    device = env.device
    step = int(env.common_step_counter)

    dt = getattr(env.sim, "dt", 1.0 / 30.0) * getattr(env, "decimation", 1)

    # (1) FK 기반 EE pose: (N,6) [x,y,z,roll,pitch,yaw]
    ee_pose = get_ee_pose(env)

    robot = env.scene["robot"]
    wrist_id = robot.find_bodies("wrist_3_link")[0]

    ee_vel = robot.data.body_lin_vel_w[:, wrist_id, :]  # (N,3) or (N,1,3)
    ee_ang = robot.data.body_ang_vel_w[:, wrist_id, :]
    if ee_vel.ndim == 3:
        ee_vel = ee_vel.squeeze(1)
    if ee_ang.ndim == 3:
        ee_ang = ee_ang.squeeze(1)

    ee_vel6d = torch.cat([ee_vel, ee_ang], dim=1)  # (N,6)

    # (2) HDF5 target (2-step horizon): fut = (N, 12)
    fut = get_hdf5_target_positions(env, horizon=2)
    if fut.ndim == 3:
        fut = fut.squeeze(1)

    target_curr, target_next = fut[:, :6], fut[:, 6:12]
    target_vel = (target_next - target_curr) / (dt + 1e-8)

    # (3) wrap-safe orientation diff (torch, GPU)
    e_pose = ee_pose.clone()
    e_pose[:, :3] = e_pose[:, :3] - target_next[:, :3]
    e_pose[:, 3:6] = angle_diff_torch(ee_pose[:, 3:6], target_next[:, 3:6])

    # (4) velocity error
    e_vel = ee_vel6d - target_vel

    # (5) reward
    w = torch.tensor([1.0, 2.0, 2.0, 1.0, 1.0, 1.0], device=device).unsqueeze(0)
    k_pose = torch.tensor([8.0, 32.0, 32.0, 4.0, 4.0, 4.0], device=device).unsqueeze(0)
    k_vel  = torch.tensor([0.2, 0.05, 0.05, 0.1, 0.1, 0.1], device=device).unsqueeze(0)

    r_pose_axiswise = torch.exp(-k_pose * (w * e_pose) ** 2)
    r_vel_axiswise  = torch.exp(-k_vel  * (w * e_vel) ** 2)

    r_pose = torch.mean(r_pose_axiswise, dim=1)
    r_vel  = torch.mean(r_vel_axiswise, dim=1)
    reward = 0.9 * r_pose + 0.1 * r_vel

    # (6) 기록 및 로그
    global _position_tracking_history, _position_reward_history
    _position_tracking_history.append(
        (step, target_next[0].detach().cpu().numpy(), ee_pose[0].detach().cpu().numpy())
    )
    _position_reward_history.append((step, float(reward[0].detach().cpu().item())))

    if step % 10 == 0:
        mean_e_pose = torch.mean(torch.abs(e_pose), dim=0).detach().cpu().numpy()
        mean_e_vel  = torch.mean(torch.abs(e_vel), dim=0).detach().cpu().numpy()
        mean_r_pose = torch.mean(r_pose_axiswise, dim=0).detach().cpu().numpy()
        mean_r_vel  = torch.mean(r_vel_axiswise, dim=0).detach().cpu().numpy()
        print(
            f"[Position Step {step}] |e_pose|={torch.norm(e_pose, dim=1).mean():.4f}, "
            f"|e_vel|={torch.norm(e_vel, dim=1).mean():.4f}, total={reward.mean():.4f}"
        )
        labels = ["x", "y", "z", "roll", "pitch", "yaw"]
        for i in range(6):
            print(
                f"  {labels[i]:<6} | e_pose={mean_e_pose[i]:+6.4f} | e_vel={mean_e_vel[i]:+6.4f} | "
                f"r_pose={mean_r_pose[i]:.4f} | r_vel={mean_r_vel[i]:.4f}"
            )

    # (7) 시각화
    if hasattr(env, "max_episode_length") and env.max_episode_length > 0:
        episode_steps = int(env.max_episode_length)
        if step > 0 and (step % episode_steps == episode_steps - 1):
            save_episode_plots_position(step)

    return reward


# -----------------------------------------------------------
# Visualization (공통)
# -----------------------------------------------------------
def save_episode_plots_joint(step: int):
    global _joint_tracking_history, _joint_reward_history, _episode_counter_joint

    save_dir = os.path.expanduser("~/nrs_rl/outputs/png/")
    reward_dir = os.path.expanduser("~/nrs_rl/outputs/rewards/")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(reward_dir, exist_ok=True)

    steps, targets, currents = zip(*_joint_tracking_history)
    targets, currents = np.vstack(targets), np.vstack(currents)

    colors = ["r", "g", "b", "orange", "purple", "gray"]

    plt.figure(figsize=(10, 6))
    for j in range(targets.shape[1]):
        plt.plot(targets[:, j], "--", color=colors[j], label=f"Target q{j+1}")
        plt.plot(currents[:, j], "-",  color=colors[j], label=f"Current q{j+1}")
    plt.legend()
    plt.grid(True)
    plt.title(f"Joint Tracking ({version})")
    plt.xlabel("Step")
    plt.ylabel("Joint [rad]")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"joint_tracking_{version}_ep{_episode_counter_joint+1}.png"))
    plt.close()

    r_steps, r_values = zip(*_joint_reward_history)
    r_values = np.vstack(r_values)  # (T,6)
    total_reward = np.sum(r_values, axis=1)

    plt.figure(figsize=(10, 5))
    plt.plot(r_steps, total_reward, "k", linewidth=2.0, label="Total Reward")
    plt.legend()
    plt.grid(True)
    plt.title(f"Joint Reward ({version})")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(os.path.join(reward_dir, f"r_pose_total_joint_{version}_ep{_episode_counter_joint+1}.png"))
    plt.close()

    _joint_tracking_history.clear()
    _joint_reward_history.clear()
    _episode_counter_joint += 1


def save_episode_plots_position(step: int):
    global _position_tracking_history, _position_reward_history, _episode_counter_position

    save_dir = os.path.expanduser("~/nrs_rl/outputs/png/")
    reward_dir = os.path.expanduser("~/nrs_rl/outputs/rewards/")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(reward_dir, exist_ok=True)

    steps, targets, currents = zip(*_position_tracking_history)
    targets, currents = np.vstack(targets), np.vstack(currents)

    # ✅ unwrap only for roll, pitch, yaw
    targets[:, 3:6]  = np.unwrap(targets[:, 3:6], axis=0)
    currents[:, 3:6] = np.unwrap(currents[:, 3:6], axis=0)

    labels = ["x", "y", "z", "roll", "pitch", "yaw"]
    colors = ["r", "g", "b", "orange", "purple", "gray"]

    plt.figure(figsize=(12, 8))
    for j in range(6):
        plt.plot(targets[:, j], "--", color=colors[j], label=f"Target {labels[j]}")
        plt.plot(currents[:, j], "-",  color=colors[j], label=f"Current {labels[j]}")
    plt.legend(ncol=3)
    plt.grid(True)
    plt.title(f"EE 6D Pose Tracking ({version})")
    plt.xlabel("Step")
    plt.ylabel("Pose [m/rad]")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"pos_tracking_{version}_ep{_episode_counter_position+1}.png"))
    plt.close()

    r_steps, r_values = zip(*_position_reward_history)
    r_values = np.array(r_values).flatten()

    plt.figure(figsize=(10, 5))
    plt.plot(r_steps, r_values, "g", linewidth=2.5, label="r_pose(6D pose)")
    plt.legend()
    plt.grid(True)
    plt.title(f"6D Pose Reward ({version})")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(os.path.join(reward_dir, f"r_pose_total_pos_{version}_ep{_episode_counter_position+1}.png"))
    plt.close()

    _position_tracking_history.clear()
    _position_reward_history.clear()
    _episode_counter_position += 1
