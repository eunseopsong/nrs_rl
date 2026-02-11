# SPDX-License-Identifier: BSD-3-Clause
"""
Observation utilities for UR10e spindle environment.
- Integrated with nrs_fk_core (C++ FK module)
- Horizon-based trajectory loaders (joints / positions)
- Includes EE pose (x, y, z, roll, pitch, yaw), contact, and camera sensors
"""

from __future__ import annotations

import sys
import torch
from isaaclab.envs import ManagerBasedRLEnv


# ------------------------------------------------------
# ✅ Conditional import (avoid double registration)
# ------------------------------------------------------
if "nrs_fk_core" not in sys.modules:
    from nrs_fk_core import FKSolver
else:
    FKSolver = sys.modules["nrs_fk_core"].FKSolver


# ------------------------------------------------------
# Global buffers
# ------------------------------------------------------
_hdf5_joints: torch.Tensor | None = None
_hdf5_positions: torch.Tensor | None = None
_step_idx = 0


# ------------------------------------------------------
# ✅ EE pose observation (x, y, z, roll, pitch, yaw)
# ------------------------------------------------------
def get_ee_pose(env: "ManagerBasedRLEnv", asset_name: str = "robot") -> torch.Tensor:
    """
    Returns end-effector pose (x, y, z, roll, pitch, yaw)
    - Reads q1~q6 and runs FK via nrs_fk_core.FKSolver
    - Output: (num_envs, 6) torch tensor on env device
    """
    robot = env.scene[asset_name]
    q = robot.data.joint_pos[:, :6]  # (N,6)
    device = q.device
    num_envs = q.shape[0]

    # NOTE: tool_z, degrees setting must match your FK convention
    fk_solver = FKSolver(tool_z=0.239, use_degrees=False)

    # --------------------------------------------------
    # Try batched FK first (if binding supports it)
    # --------------------------------------------------
    # Expected batched API examples (one of these may exist):
    #   ok, poses = fk_solver.compute_batch(q_tensor_or_numpy)
    #   poses = fk_solver.forward(q)
    # We probe safely; fallback to loop if not available.
    # --------------------------------------------------
    if hasattr(fk_solver, "compute_batch"):
        try:
            # Prefer torch -> cpu numpy only once
            q_np = q.detach().cpu().numpy().astype(float)  # (N,6)
            ok, poses = fk_solver.compute_batch(q_np, as_degrees=False)  # poses: (N,6)
            if not ok:
                ee_pose = torch.full((num_envs, 6), float("nan"), device=device, dtype=torch.float32)
            else:
                ee_pose = torch.tensor(poses, dtype=torch.float32, device=device)
            return ee_pose
        except Exception:
            pass

    if hasattr(fk_solver, "forward"):
        try:
            # Some bindings may accept numpy or torch. Try torch first.
            poses = fk_solver.forward(q)  # expect (N,6)
            ee_pose = poses if isinstance(poses, torch.Tensor) else torch.tensor(poses, dtype=torch.float32, device=device)
            if ee_pose.device != device:
                ee_pose = ee_pose.to(device)
            return ee_pose
        except Exception:
            pass

    # --------------------------------------------------
    # Fallback: per-env loop (slow) — keep for debugging only
    # --------------------------------------------------
    ee_pose_list = []
    q_cpu = q.detach().cpu()
    for i in range(num_envs):
        q_np = q_cpu[i].numpy().astype(float)
        ok, pose = fk_solver.compute(q_np, as_degrees=False)
        if not ok:
            ee_pose_list.append([float("nan")] * 6)
        else:
            ee_pose_list.append([pose.x, pose.y, pose.z, pose.r, pose.p, pose.yaw])

    ee_pose = torch.tensor(ee_pose_list, dtype=torch.float32, device=device)
    assert ee_pose.ndim == 2 and ee_pose.shape[1] == 6, f"[EE_POSE] Invalid shape: {ee_pose.shape}"
    return ee_pose


# ------------------------------------------------------
# HDF5 loader: Joints
# ------------------------------------------------------
def load_hdf5_joints(
    env: ManagerBasedRLEnv,
    env_ids,
    file_path: str,
    dataset_key: str = "target_joints",
):
    """Load HDF5 trajectory (joint targets)."""
    global _hdf5_joints, _step_idx
    import h5py

    with h5py.File(file_path, "r") as f:
        if dataset_key not in f:
            raise KeyError(
                f"[ERROR] HDF5 (joints): '{dataset_key}' not found. Available keys: {list(f.keys())}"
            )
        data = f[dataset_key][:]  # [T, D]

    _hdf5_joints = torch.tensor(data, dtype=torch.float32, device=env.device)
    _step_idx = 0
    print(f"[INFO] Loaded HDF5 joints of shape {_hdf5_joints.shape} from {file_path}")


# ------------------------------------------------------
# HDF5 loader: Positions
# ------------------------------------------------------
def load_hdf5_positions(
    env: ManagerBasedRLEnv,
    env_ids,
    file_path: str,
    dataset_key: str = "target_positions",
):
    """Load HDF5 trajectory (position targets)."""
    global _hdf5_positions, _step_idx
    import h5py

    with h5py.File(file_path, "r") as f:
        if dataset_key not in f:
            raise KeyError(
                f"[ERROR] HDF5 (positions): '{dataset_key}' not found. Available keys: {list(f.keys())}"
            )
        data = f[dataset_key][:]  # [T, D]

    _hdf5_positions = torch.tensor(data, dtype=torch.float32, device=env.device)
    _step_idx = 0
    print(f"[INFO] Loaded HDF5 positions of shape {_hdf5_positions.shape} from {file_path}")


# ------------------------------------------------------
# Observation: target joints (horizon-based)
# ------------------------------------------------------
def get_hdf5_target_joints(env: ManagerBasedRLEnv, horizon: int = 5) -> torch.Tensor:
    """Return future joint targets flattened: (N, horizon*D)."""
    global _hdf5_joints

    if _hdf5_joints is None:
        D = 6  # you're using first 6 joints for control/reward
        return torch.zeros((env.num_envs, horizon * D), device=env.device, dtype=torch.float32)

    T, D = _hdf5_joints.shape

    # episode progress -> map into [0, T)
    step = int(env.episode_length_buf[0].item())
    E = int(env.max_episode_length)
    idx = min(int((step / max(E, 1)) * T), T - 1)

    future_idx = torch.arange(idx, idx + horizon, device=_hdf5_joints.device)
    future_idx = torch.clamp(future_idx, max=T - 1)

    future_targets = _hdf5_joints[future_idx].reshape(1, horizon * D)
    return future_targets.repeat(env.num_envs, 1)


# ------------------------------------------------------
# Observation: target positions (horizon-based)
# ------------------------------------------------------
def get_hdf5_target_positions(env: ManagerBasedRLEnv, horizon: int = 5) -> torch.Tensor:
    """Return future EE pose targets (x,y,z,roll,pitch,yaw) flattened: (N, horizon*6)."""
    global _hdf5_positions

    if _hdf5_positions is None:
        D = 6
        return torch.zeros((env.num_envs, horizon * D), device=env.device, dtype=torch.float32)

    T, D = _hdf5_positions.shape  # D should be 6
    step = int(env.episode_length_buf[0].item())
    E = int(env.max_episode_length)
    idx = min(int((step / max(E, 1)) * T), T - 1)

    future_idx = torch.arange(idx, idx + horizon, device=_hdf5_positions.device)
    future_idx = torch.clamp(future_idx, max=T - 1)

    future_targets = _hdf5_positions[future_idx].reshape(1, horizon * D)
    return future_targets.repeat(env.num_envs, 1)


# ------------------------------------------------------
# ✅ Observation: Contact Sensor Forces
# ------------------------------------------------------
def get_contact_forces(env: ManagerBasedRLEnv, sensor_name: str = "contact_forces") -> torch.Tensor:
    """
    Mean contact wrench [Fx, Fy, Fz, 0, 0, 0]
    - Uses ContactSensorCfg output: sensor.data.net_forces_w
    """
    if sensor_name not in env.scene.sensors:
        raise KeyError(f"[ERROR] Contact sensor '{sensor_name}' not found in scene.sensors.")

    sensor = env.scene.sensors[sensor_name]
    forces_w = sensor.data.net_forces_w  # (N, B, 3) typically
    mean_force = torch.mean(forces_w, dim=1)  # (N,3)

    zeros_torque = torch.zeros_like(mean_force)
    contact_wrench = torch.cat([mean_force, zeros_torque], dim=-1)  # (N,6)

    step = int(env.common_step_counter)
    if step % 100 == 0:
        fx, fy, fz = mean_force[0].detach().cpu().tolist()
        print(f"[ContactSensor DEBUG] Step {step}: Fx={fx:.3f}, Fy={fy:.3f}, Fz={fz:.3f}")

    return contact_wrench


# ------------------------------------------------------
# ✅ Camera distance & normals
# ------------------------------------------------------
def get_camera_distance(env: ManagerBasedRLEnv, sensor_name: str = "camera", debug_interval: int = 100) -> torch.Tensor:
    """Compute mean camera depth (distance-to-image-plane). Output: (N,1)."""
    if sensor_name not in env.scene.sensors:
        raise KeyError(f"[ERROR] Camera sensor '{sensor_name}' not found in scene.sensors.")

    sensor = env.scene.sensors[sensor_name]
    data = sensor.data.output.get("distance_to_image_plane", None)
    if data is None:
        raise RuntimeError("[ERROR] Missing 'distance_to_image_plane' in camera data output.")

    valid_mask = torch.isfinite(data) & (data > 0)
    valid_data = torch.where(valid_mask, data, torch.nan)

    mean_distance = torch.nanmean(valid_data.view(valid_data.shape[0], -1), dim=1).unsqueeze(1)

    if int(env.common_step_counter) % int(debug_interval) == 0:
        md_cpu = mean_distance[0].detach().cpu().item()
        print(f"[Step {int(env.common_step_counter)}] Mean camera distance: {md_cpu:.4f} m")

    return mean_distance


def get_camera_normals(env: ManagerBasedRLEnv, sensor_name: str = "camera") -> torch.Tensor:
    """Compute mean surface normal (x, y, z) from the camera. Output: (N,3)."""
    if sensor_name not in env.scene.sensors:
        raise KeyError(f"[ERROR] Camera sensor '{sensor_name}' not found in scene.sensors.")

    cam_sensor = env.scene.sensors[sensor_name]
    normals = cam_sensor.data.output.get("normals", None)
    if normals is None:
        return torch.zeros((env.num_envs, 3), device=env.device)

    normals_mean = normals.mean(dim=(1, 2))  # (N,3)
    if int(env.common_step_counter) % 100 == 0:
        print(f"[Camera DEBUG] Step {int(env.common_step_counter)}: Mean surface normal = {normals_mean[0].detach().cpu().numpy()}")
    return normals_mean
