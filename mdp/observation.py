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
import importlib

from ..utils import debug as local_debug

local_ft_sensor = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.assets.assets.sensors.six_axis_ft_sensor"
)

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

def _rpy_to_quat_torch(rpy: torch.Tensor) -> torch.Tensor:
    """
    Convert roll-pitch-yaw to quaternion [w, x, y, z].
    Input:  (N, 3)
    Output: (N, 4)
    """
    roll = rpy[:, 0] * 0.5
    pitch = rpy[:, 1] * 0.5
    yaw = rpy[:, 2] * 0.5

    cr = torch.cos(roll)
    sr = torch.sin(roll)
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return torch.stack([qw, qx, qy, qz], dim=-1)


def _get_target_pose_for_polishing(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    """
    Safe target-pose fetcher for polishing observation.

    Priority:
      1) command_manager['trajectory'] if available
      2) _hdf5_positions current-step fallback
      3) zeros + identity quaternion

    Returns:
      target_xyz  : (N, 3)
      target_quat : (N, 4)
    """
    global _hdf5_positions

    device = env.device
    num_envs = env.num_envs

    # --------------------------------------------------
    # 1) Try command_manager trajectory first
    # --------------------------------------------------
    try:
        cmd_mgr = getattr(env, "command_manager", None)
        if cmd_mgr is not None and hasattr(cmd_mgr, "_terms") and ("trajectory" in cmd_mgr._terms):
            target_pose = cmd_mgr.get_command("trajectory")
            if target_pose is not None and target_pose.ndim == 2 and target_pose.shape[1] >= 7:
                target_xyz = target_pose[:, :3].to(device=device, dtype=torch.float32)
                target_quat = target_pose[:, 3:7].to(device=device, dtype=torch.float32)
                return target_xyz, target_quat
    except Exception as e:
        print(f"[TARGET FETCH WARN] command trajectory unavailable | {repr(e)}")

    # --------------------------------------------------
    # 2) Fallback to HDF5 positions
    #    assumes first 3 dims = xyz, next 3 dims = rpy if available
    # --------------------------------------------------
    if _hdf5_positions is not None and _hdf5_positions.ndim == 2 and _hdf5_positions.shape[0] > 0:
        t_total, d = _hdf5_positions.shape

        if hasattr(env, "episode_length_buf"):
            step = env.episode_length_buf.to(torch.float32)
        else:
            step = torch.zeros((num_envs,), device=device, dtype=torch.float32)

        ep_len = max(int(getattr(env, "max_episode_length", 1)), 1)
        idx = ((step / ep_len) * t_total).to(torch.int64)
        idx = torch.clamp(idx, 0, t_total - 1)

        row = _hdf5_positions[idx].to(device=device, dtype=torch.float32)

        # xyz
        target_xyz = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
        if d >= 3:
            target_xyz = row[:, :3]

        # quat
        target_quat = torch.zeros((num_envs, 4), device=device, dtype=torch.float32)
        target_quat[:, 0] = 1.0  # identity quat by default

        if d >= 6:
            target_rpy = row[:, 3:6]
            target_quat = _rpy_to_quat_torch(target_rpy)

        return target_xyz, target_quat

    # --------------------------------------------------
    # 3) Final fallback: zeros + identity quaternion
    # --------------------------------------------------
    target_xyz = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
    target_quat = torch.zeros((num_envs, 4), device=device, dtype=torch.float32)
    target_quat[:, 0] = 1.0
    return target_xyz, target_quat

# ------------------------------------------------------
# Polishing process buffers
# ------------------------------------------------------
_prev_xyz_for_polishing: torch.Tensor | None = None
_cumulative_removal: torch.Tensor | None = None
_cumulative_contact_distance: torch.Tensor | None = None


# ------------------------------------------------------
# Internal helpers
# ------------------------------------------------------
def _update_debug_cache(step: int, wrench_env0: torch.Tensor | None = None, metrics_env0: torch.Tensor | None = None):
    """Update local debug caches directly so action debug can always read them."""
    try:
        if wrench_env0 is not None and hasattr(local_debug, "_last_ft_debug"):
            local_debug._last_ft_debug["step"] = int(step)
            local_debug._last_ft_debug["wrench"] = wrench_env0.detach().cpu().clone()

        if metrics_env0 is not None and hasattr(local_debug, "_last_polishing_debug"):
            local_debug._last_polishing_debug["step"] = int(step)
            local_debug._last_polishing_debug["metrics"] = metrics_env0.detach().cpu().clone()
    except Exception as e:
        print(f"[OBS DEBUG CACHE ERROR] step={step} | {repr(e)}")


def _call_debug_printers(step: int, wrench_env0: torch.Tensor | None = None, metrics_env0: torch.Tensor | None = None):
    """Call debug printer functions without silently swallowing all errors."""
    if wrench_env0 is not None:
        try:
            if hasattr(local_debug, "print_ft_sensor_debug"):
                local_debug.print_ft_sensor_debug(int(step), wrench_env0)
        except Exception as e:
            print(f"[FT DEBUG PRINT ERROR] step={step} | {repr(e)}")

    if metrics_env0 is not None:
        try:
            if hasattr(local_debug, "print_polishing_metrics_debug"):
                local_debug.print_polishing_metrics_debug(int(step), metrics_env0)
        except Exception as e:
            print(f"[POLISHING DEBUG PRINT ERROR] step={step} | {repr(e)}")


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


# ------------------------------------------------------
# 로봇 현재 속도 Observation
# ------------------------------------------------------
def get_joint_velocities(env: "ManagerBasedRLEnv", asset_name: str = "robot") -> torch.Tensor:
    """
    로봇의 현재 조인트 속도(qdot)를 반환합니다.

    - Output: (num_envs, 6) torch tensor on env device
    """
    robot = env.scene[asset_name]
    joint_vel = robot.data.joint_vel[:, :6]
    return joint_vel


# ------------------------------------------------------
# Observation: 속도 기반 동적 타겟 경로 (Velocity-Adjusted Target)
# ------------------------------------------------------
def get_velocity_adjusted_target_positions(
    env: "ManagerBasedRLEnv",
    horizon: int = 5,
    lookahead_gain: float = 1.0,
) -> torch.Tensor:
    """
    현재 속도 크기에 비례하여 HDF5 궤적의 탐색 인덱스를 가공(Look-ahead)하여 반환합니다.

    - 로봇이 빠르게 움직일수록 궤적의 더 먼 미래를 타겟으로 삼도록 유도합니다.
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

    # 2. 각 환경(env)별 진행 스텝
    step = env.episode_length_buf.to(torch.float32)
    ep_len = max(int(env.max_episode_length), 1)

    # 3. 속도에 비례하는 인덱스 오프셋 계산
    step_offset = (vel_magnitude * lookahead_gain).to(torch.int64)

    # 4. 기본 인덱스 + 속도 기반 오프셋
    base_idx = ((step / ep_len) * t_total).to(torch.int64)
    dynamic_idx = base_idx + step_offset
    dynamic_idx = torch.clamp(dynamic_idx, max=t_total - 1)

    # 5. 환경별로 Horizon만큼 타겟 텐서 할당
    future_targets = torch.zeros((env.num_envs, horizon * d), device=env.device, dtype=torch.float32)

    for i in range(horizon):
        h_idx = torch.clamp(dynamic_idx + i, max=t_total - 1)
        future_targets[:, i * d : (i + 1) * d] = _hdf5_positions[h_idx]

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


def get_processed_polishing_target(
    env: "ManagerBasedRLEnv",
    asset_name: str = "robot",
    body_name: str = "spindle_link",
    fixed_joint_name: str = "tool0_to_spindle",
    joint_prim_relpath: str = "joints",
    contact_force_threshold: float = 10.0,
    removal_gain: float = 0.001,
    offset_axis: int = 2,
) -> torch.Tensor:
    """
    현재 타겟 궤적에서 가공량/오프셋이 반영된 최종 타겟 좌표를 반환합니다.

    Returns:
        (num_envs, 14) tensor

        [0]  processed_target_x | [1]  processed_target_y | [2]  processed_target_z | [3]  target_quat_0 | [4]  target_quat_1 | [5]  target_quat_2
        [6]  target_quat_3 | [7]  cartesian_velocity | [8]  Fz | [9]  abs_fz | [10] contact_flag | [11] effective_normal_force
        [12] cumulative_removal | [13] cumulative_contact_distance
    """
    global _prev_xyz_for_polishing
    global _cumulative_removal
    global _cumulative_contact_distance

    device = env.device
    num_envs = env.num_envs
    step_i = int(getattr(env, "common_step_counter", 0))

    # --------------------------------------------------
    # 1) Safe target fetch
    # --------------------------------------------------
    target_xyz, target_quat = _get_target_pose_for_polishing(env)

    # --------------------------------------------------
    # 2) 현재 EE 위치
    # --------------------------------------------------
    robot = env.scene[asset_name]
    body_ids = robot.find_bodies(body_name)[0]
    if len(body_ids) == 0:
        raise ValueError(
            f"[get_processed_polishing_target] body_name='{body_name}' not found. "
            f"Available bodies: {robot.body_names}"
        )

    ee_idx = int(body_ids[0])
    ee_pos_w = robot.data.body_pos_w[:, ee_idx, :]
    env_origins = env.scene.env_origins
    current_xyz = ee_pos_w - env_origins   # env-local position

    # --------------------------------------------------
    # 3) dt
    # --------------------------------------------------
    if hasattr(env, "step_dt"):
        dt = float(env.step_dt)
    elif hasattr(env, "physics_dt"):
        dt = float(env.physics_dt)
    else:
        dt = 0.02

    # --------------------------------------------------
    # 4) Cartesian velocity = ||x_k - x_{k-1}|| / dt
    # --------------------------------------------------
    reset_mask = (env.episode_length_buf == 0)

    if (_prev_xyz_for_polishing is None) or (_prev_xyz_for_polishing.shape[0] != num_envs):
        _prev_xyz_for_polishing = current_xyz.clone()
        cartesian_vel = torch.zeros((num_envs, 1), device=device, dtype=torch.float32)
    else:
        delta = current_xyz - _prev_xyz_for_polishing
        cartesian_vel = torch.norm(delta, dim=-1, keepdim=True) / max(dt, 1e-8)

        if torch.any(reset_mask):
            cartesian_vel[reset_mask] = 0.0
            _prev_xyz_for_polishing[reset_mask] = current_xyz[reset_mask]

        _prev_xyz_for_polishing = current_xyz.clone()

    # --------------------------------------------------
    # 5) FT sensor -> Fz
    # --------------------------------------------------
    wrench = local_ft_sensor.get_6axis_ft_fixed_joint(
        env=env,
        asset_name=asset_name,
        fixed_joint_name=fixed_joint_name,
        joint_prim_relpath=joint_prim_relpath,
        verbose=False,
    )

    fz = wrench[:, 2:3]
    abs_fz = torch.abs(fz)

    # --------------------------------------------------
    # 6) Contact / effective force
    # --------------------------------------------------
    contact_flag = (abs_fz >= contact_force_threshold).to(torch.float32)
    effective_force = torch.clamp(abs_fz - contact_force_threshold, min=0.0)

    # --------------------------------------------------
    # 7) Removal model
    # --------------------------------------------------
    removal_rate = removal_gain * effective_force * cartesian_vel
    removal_amount = removal_rate * dt

    if (_cumulative_removal is None) or (_cumulative_removal.shape[0] != num_envs):
        _cumulative_removal = torch.zeros((num_envs, 1), device=device, dtype=torch.float32)

    if (_cumulative_contact_distance is None) or (_cumulative_contact_distance.shape[0] != num_envs):
        _cumulative_contact_distance = torch.zeros((num_envs, 1), device=device, dtype=torch.float32)

    if torch.any(reset_mask):
        _cumulative_removal[reset_mask] = 0.0
        _cumulative_contact_distance[reset_mask] = 0.0

    _cumulative_removal = _cumulative_removal + removal_amount
    _cumulative_contact_distance = _cumulative_contact_distance + (contact_flag * cartesian_vel * dt)

    # --------------------------------------------------
    # 8) 가공량 반영 target
    # --------------------------------------------------
    processed_target_xyz = target_xyz.clone()
    processed_target_xyz[:, offset_axis:offset_axis + 1] -= _cumulative_removal

    # --------------------------------------------------
    # 9) Debug cache update
    # --------------------------------------------------
    metrics = torch.cat(
        [
            cartesian_vel,                  # [0]
            fz,                             # [1]
            abs_fz,                         # [2]
            contact_flag,                   # [3]
            effective_force,                # [4]
            removal_rate,                   # [5]
            _cumulative_removal,            # [6]
            _cumulative_contact_distance,   # [7]
        ],
        dim=-1,
    )

    _update_debug_cache(
        step=step_i,
        wrench_env0=wrench[0],
        metrics_env0=metrics[0],
    )

    _call_debug_printers(
        step=step_i,
        wrench_env0=wrench[0],
        metrics_env0=metrics[0],
    )

    try:
        if hasattr(local_debug, "polishing_logger"):
            env0_xyz = current_xyz[0].detach().cpu().tolist()
            env0_force = float(fz[0].item())
            env0_reward = 0.0

            x_idx = int(torch.clamp((processed_target_xyz[0, 0] * 100).long(), 0, 49).item())
            y_idx = int(torch.clamp((processed_target_xyz[0, 1] * 100).long(), 0, 49).item())

            local_debug.polishing_logger.step_log(
                current_xyz=env0_xyz,
                x_idx=x_idx,
                y_idx=y_idx,
                force=env0_force,
                reward=env0_reward,
            )
    except Exception as e:
        print(f"[POLISHING LOGGER ERROR] step={step_i} | {repr(e)}")

    # --------------------------------------------------
    # 10) Return
    # --------------------------------------------------
    out = torch.cat(
        [
            processed_target_xyz,          # [0:3]
            target_quat,                   # [3:7]
            cartesian_vel,                 # [7]
            fz,                            # [8]
            abs_fz,                        # [9]
            contact_flag,                  # [10]
            effective_force,               # [11]
            _cumulative_removal,           # [12]
            _cumulative_contact_distance,  # [13]
        ],
        dim=-1,
    )

    return out