# SPDX-License-Identifier: BSD-3-Clause
"""
Observation utilities for UR10e spindle environment.
- Integrated with nrs_fk_core (C++ FK module)
- Horizon-based trajectory loaders (positions)
- Includes EE pose (x, y, z, roll, pitch, yaw), and camera sensors
"""

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import sys
import torch

from ..utils import debug as local_debug

# ------------------------------------------------------
# Conditional import (avoid double registration)
# ------------------------------------------------------
if "nrs_fk_core" not in sys.modules:
    from nrs_fk_core import FKSolver
else:
    FKSolver = sys.modules["nrs_fk_core"].FKSolver


# ------------------------------------------------------
# Global buffers
# ------------------------------------------------------
_hdf5_positions: torch.Tensor | None = None
_step_idx = 0


# ------------------------------------------------------
# EE pose observation (x, y, z, roll, pitch, yaw)
# ------------------------------------------------------
def get_ee_pose(env: "ManagerBasedRLEnv", asset_name: str = "robot") -> torch.Tensor:
    """
    Returns end-effector pose (x, y, z, roll, pitch, yaw)
    - Reads q1~q6 and runs FK via nrs_fk_core.FKSolver
    - Output: (num_envs, 6) torch tensor on env device
    """
    robot = env.scene[asset_name]
    q = robot.data.joint_pos[:, :6]
    device = q.device
    num_envs = q.shape[0]

    fk_solver = FKSolver(tool_z=0.239, use_degrees=False)

    # batched FK first
    if hasattr(fk_solver, "compute_batch"):
        try:
            q_np = q.detach().cpu().numpy().astype(float)
            ok, poses = fk_solver.compute_batch(q_np, as_degrees=False)
            if not ok:
                ee_pose = torch.full((num_envs, 6), float("nan"), device=device, dtype=torch.float32)
            else:
                ee_pose = torch.tensor(poses, dtype=torch.float32, device=device)
            return ee_pose
        except Exception:
            pass

    if hasattr(fk_solver, "forward"):
        try:
            poses = fk_solver.forward(q)
            ee_pose = poses if isinstance(poses, torch.Tensor) else torch.tensor(
                poses, dtype=torch.float32, device=device
            )
            if ee_pose.device != device:
                ee_pose = ee_pose.to(device)
            return ee_pose
        except Exception:
            pass

    # fallback: per-env loop
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
# HDF5 loader: Positions
# ------------------------------------------------------
def load_hdf5_positions(
    env: "ManagerBasedRLEnv",
    env_ids,
    file_path: str,
    dataset_key: str = "target_positions",
):
    """
    Load HDF5 trajectory (position targets).
    """
    global _hdf5_positions, _step_idx
    import h5py

    with h5py.File(file_path, "r") as f:
        if dataset_key not in f:
            raise KeyError(
                f"[ERROR] HDF5 (positions): '{dataset_key}' not found. Available keys: {list(f.keys())}"
            )
        data = f[dataset_key][:]

    _hdf5_positions = torch.tensor(data, dtype=torch.float32, device=env.device)
    _step_idx = 0
    local_debug.print_hdf5_positions_loaded(_hdf5_positions.shape, file_path)


# ------------------------------------------------------
# Observation: target positions (horizon-based)
# ------------------------------------------------------
def get_hdf5_target_positions(env: "ManagerBasedRLEnv", horizon: int = 5) -> torch.Tensor:
    """
    Return future EE pose targets (x,y,z,roll,pitch,yaw) flattened: (N, horizon*6).
    """
    global _hdf5_positions

    if _hdf5_positions is None:
        d = 6
        return torch.zeros((env.num_envs, horizon * d), device=env.device, dtype=torch.float32)

    t_total, d = _hdf5_positions.shape
    step = int(env.episode_length_buf[0].item())
    ep_len = int(env.max_episode_length)
    idx = min(int((step / max(ep_len, 1)) * t_total), t_total - 1)

    future_idx = torch.arange(idx, idx + horizon, device=_hdf5_positions.device)
    future_idx = torch.clamp(future_idx, max=t_total - 1)

    future_targets = _hdf5_positions[future_idx].reshape(1, horizon * d)
    return future_targets.repeat(env.num_envs, 1)


# [26.04.02 추가] 추가 내용: 로봇 현재 속도 Observation 
# ------------------------------------------------------
# 로봇 현재 속도 Observation
# ------------------------------------------------------
def get_joint_velocities(env: "ManagerBasedRLEnv", asset_name: str = "robot") -> torch.Tensor:
    """
    [26.04.02 추가] 로봇의 현재 조인트 속도(qdot)를 반환합니다.
    - Output: (num_envs, 6) torch tensor on env device
    """
    robot = env.scene[asset_name]
    # UR10e의 첫 6개 조인트 속도 추출
    joint_vel = robot.data.joint_vel[:, :6]
    return joint_vel


# [26.04.02 추가] 추가 내용: 속도 기반 동적 타겟 경로 Observation 
# ------------------------------------------------------
# Observation: 속도 기반 동적 타겟 경로 (Velocity-Adjusted Target)
# ------------------------------------------------------
def get_velocity_adjusted_target_positions(
    env: "ManagerBasedRLEnv", 
    horizon: int = 5, 
    lookahead_gain: float = 1.0
) -> torch.Tensor:
    """
    [26.04.02 추가] 현재 속도 크기에 비례하여 HDF5 궤적의 탐색 인덱스를 가공(Look-ahead)하여 반환합니다.
    - 로봇이 빠르게 움직일수록(속도가 높을수록) 궤적의 더 먼 미래를 타겟으로 삼도록 유도합니다.
    - Output: (num_envs, horizon * 6) flattened tensor
    """
    global _hdf5_positions

    if _hdf5_positions is None:
        d = 6
        return torch.zeros((env.num_envs, horizon * d), device=env.device, dtype=torch.float32)

    robot = env.scene["robot"]
    
    # 1. 현재 조인트 속도 가져오기 및 크기(Norm) 계산
    current_vel = robot.data.joint_vel[:, :6] 
    vel_magnitude = torch.norm(current_vel, dim=-1)  # (num_envs,)

    t_total, d = _hdf5_positions.shape
    
    # 2. 각 환경(env)별 진행 스텝 (1D 텐서)
    step = env.episode_length_buf.to(torch.float32)
    ep_len = max(int(env.max_episode_length), 1)
    
    # 3. 속도에 비례하는 인덱스 오프셋 계산 (가공 단계)
    # lookahead_gain을 조절하여 속도에 따른 오프셋 민감도를 튜닝할 수 있습니다.
    step_offset = (vel_magnitude * lookahead_gain).to(torch.int64) 
    
    # 4. 기본 인덱스 + 속도 기반 오프셋
    base_idx = ((step / ep_len) * t_total).to(torch.int64)
    dynamic_idx = base_idx + step_offset
    
    # 최대 인덱스를 초과하지 않도록 클램핑
    dynamic_idx = torch.clamp(dynamic_idx, max=t_total - 1)

    # 5. 환경별로 Horizon만큼 타겟 텐서 할당
    future_targets = torch.zeros((env.num_envs, horizon * d), device=env.device, dtype=torch.float32)
    
    for i in range(horizon):
        # 각 horizon 스텝마다 인덱스 증가 및 클램핑
        h_idx = torch.clamp(dynamic_idx + i, max=t_total - 1)
        
        # _hdf5_positions[h_idx]는 (num_envs, 6) 형태가 됨
        future_targets[:, i*d : (i+1)*d] = _hdf5_positions[h_idx]

    return future_targets


# ------------------------------------------------------
# Camera distance & normals
# ------------------------------------------------------
def get_camera_distance(
    env: "ManagerBasedRLEnv",
    sensor_name: str = "camera",
    debug_interval: int = 100,
) -> torch.Tensor:
    """
    Compute mean camera depth (distance-to-image-plane). Output: (N,1).
    """
    if sensor_name not in env.scene.sensors:
        raise KeyError(f"[ERROR] Camera sensor '{sensor_name}' not found in scene.sensors.")

    sensor = env.scene.sensors[sensor_name]
    data = sensor.data.output.get("distance_to_image_plane", None)
    if data is None:
        raise RuntimeError("[ERROR] Missing 'distance_to_image_plane' in camera data output.")

    valid_mask = torch.isfinite(data) & (data > 0)
    valid_data = torch.where(valid_mask, data, torch.nan)

    mean_distance = torch.nanmean(
        valid_data.view(valid_data.shape[0], -1), dim=1
    ).unsqueeze(1)

    if int(env.common_step_counter) % int(debug_interval) == 0:
        local_debug.print_camera_distance(int(env.common_step_counter), mean_distance[0])

    return mean_distance


def get_camera_normals(env: "ManagerBasedRLEnv", sensor_name: str = "camera") -> torch.Tensor:
    """
    Compute mean surface normal (x, y, z) from the camera. Output: (N,3).
    """
    if sensor_name not in env.scene.sensors:
        raise KeyError(f"[ERROR] Camera sensor '{sensor_name}' not found in scene.sensors.")

    cam_sensor = env.scene.sensors[sensor_name]
    normals = cam_sensor.data.output.get("normals", None)
    if normals is None:
        return torch.zeros((env.num_envs, 3), device=env.device)

    normals_mean = normals.mean(dim=(1, 2))
    if int(env.common_step_counter) % 100 == 0:
        local_debug.print_camera_normals(int(env.common_step_counter), normals_mean[0])

    return normals_mean