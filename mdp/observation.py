# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import torch
import importlib
import numpy as np

from ..utils import debug as local_debug

local_ft_sensor = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.assets.assets.sensors.six_axis_ft_sensor"
)

y2_cfg = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.y2_control_pybind.y2_control_py.config"
)
y2_pb = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.y2_control_pybind.y2_control_py._y2_control_pybind"
)

_hdf5_position: torch.Tensor | None = None
_hdf5_force: torch.Tensor | None = None
_hdf5_traj_len: int = 0

_y2_kin_solver = None

_HOME_Q = [0.5585, -2.0949, -1.5711, -1.0472, 1.5708, 0.5585]
_orientation_offset_rotm: torch.Tensor | None = None


def _get_y2_kin_solver():
    global _y2_kin_solver
    if _y2_kin_solver is None:
        _y2_kin_solver = y2_pb.UR10eKinematics(
            dt=float(y2_cfg.CONTROL_PERIOD),
            ee2tcp=y2_cfg.EE2TCP,
        )
    return _y2_kin_solver


def get_hdf5_trajectory_length() -> int:
    return int(_hdf5_traj_len)


def _get_action_term(env: "ManagerBasedRLEnv", action_term_name: str = "arm_action"):
    am = getattr(env, "action_manager", None)
    if am is None:
        return None

    if hasattr(am, "get_term"):
        try:
            return am.get_term(action_term_name)
        except Exception:
            pass

    if hasattr(am, "_terms"):
        terms = am._terms
        if isinstance(terms, dict) and action_term_name in terms:
            return terms[action_term_name]

    if hasattr(am, action_term_name):
        return getattr(am, action_term_name)

    return None


def get_action_term(env: "ManagerBasedRLEnv", action_term_name: str = "arm_action"):
    return _get_action_term(env, action_term_name=action_term_name)


def get_ee_idx(
    env: "ManagerBasedRLEnv",
    asset_name: str = "robot",
    body_name: str = "spindle_link",
) -> int:
    robot = env.scene[asset_name]
    return int(robot.find_bodies(body_name)[0][0])


def get_current_pose_and_velocity(
    env: "ManagerBasedRLEnv",
    asset_name: str = "robot",
    body_name: str = "spindle_link",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    robot = env.scene[asset_name]
    ee_idx = get_ee_idx(env, asset_name=asset_name, body_name=body_name)
    ee_pos_w = robot.data.body_pos_w[:, ee_idx, :]
    ee_lin_vel_w = robot.data.body_lin_vel_w[:, ee_idx, :]
    current_xyz = ee_pos_w - env.scene.env_origins
    vel_norm = torch.norm(ee_lin_vel_w, dim=-1)
    return current_xyz, robot.data.body_quat_w[:, ee_idx, :], vel_norm


def get_current_fz(
    env: "ManagerBasedRLEnv",
    asset_name: str = "robot",
    fixed_joint_name: str = "tool0_to_spindle",
    joint_prim_relpath: str = "joints",
) -> torch.Tensor:
    wrench = local_ft_sensor.get_6axis_ft_fixed_joint(
        env=env,
        asset_name=asset_name,
        fixed_joint_name=fixed_joint_name,
        joint_prim_relpath=joint_prim_relpath,
        verbose=False,
    )
    return wrench[:, 2]


def get_path_sliding_metrics(
    env: "ManagerBasedRLEnv",
    action_term_name: str = "arm_action",
    asset_name: str = "robot",
    body_name: str = "spindle_link",
    fixed_joint_name: str = "tool0_to_spindle",
    joint_prim_relpath: str = "joints",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    term = _get_action_term(env, action_term_name)
    if term is not None and hasattr(term, "current_sliding_velocity_mm_s"):
        sliding_velocity = term.current_sliding_velocity_mm_s.to(device=env.device, dtype=torch.float32)
        abs_fz = term.current_abs_fz.to(device=env.device, dtype=torch.float32)
        current_mrr = term.current_mrr_n_mm_s.to(device=env.device, dtype=torch.float32)
        return abs_fz, sliding_velocity, current_mrr

    abs_fz = torch.abs(get_current_fz(env, asset_name, fixed_joint_name, joint_prim_relpath))
    _, _, current_vel_norm = get_current_pose_and_velocity(env, asset_name, body_name)
    sliding_velocity = current_vel_norm * 1000.0
    return abs_fz, sliding_velocity, abs_fz * sliding_velocity


def get_current_target_pose_for_reward(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    return _get_target_pose_for_polishing(env)


def get_surface_uniformity_reward_value(env):
    vis_module = importlib.import_module("nrs_rl.tasks.manager_based.nrs_rl.utils.visualization")
    grid = vis_module._surface_grid
    valid = grid[grid > 0]
    if len(valid) < 10:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    std = float(np.std(valid))
    reward = 1.0 / (1.0 + std)
    return torch.full((env.num_envs,), reward, device=env.device, dtype=torch.float32)


def _get_traj_indices_from_action(
    env: "ManagerBasedRLEnv",
    horizon: int = 1,
    action_term_name: str = "arm_action",
) -> torch.Tensor:
    global _hdf5_traj_len

    num_envs = env.num_envs
    device = env.device

    if _hdf5_traj_len <= 0:
        return torch.zeros((num_envs, horizon), device=device, dtype=torch.long)

    term = _get_action_term(env, action_term_name=action_term_name)

    if term is not None:
        if hasattr(term, "current_target_index"):
            base_idx = term.current_target_index.to(device=device, dtype=torch.long)
        elif hasattr(term, "path_index"):
            base_idx = term.path_index.to(device=device, dtype=torch.long)
        else:
            base_idx = None

        if base_idx is not None:
            offsets = torch.arange(horizon, device=device, dtype=torch.long).unsqueeze(0)
            idx = base_idx.unsqueeze(1) + offsets
            idx = torch.clamp(idx, 0, _hdf5_traj_len - 1)
            return idx

    return _get_traj_indices_from_episode(env, horizon=horizon)


def _get_traj_indices_from_episode(env: "ManagerBasedRLEnv", horizon: int = 1) -> torch.Tensor:
    global _hdf5_traj_len

    num_envs = env.num_envs
    device = env.device

    if _hdf5_traj_len <= 0:
        return torch.zeros((num_envs, horizon), device=device, dtype=torch.long)

    if hasattr(env, "episode_length_buf"):
        base_idx = env.episode_length_buf.to(device=device, dtype=torch.long)
    else:
        base_idx = torch.zeros((num_envs,), device=device, dtype=torch.long)

    offsets = torch.arange(horizon, device=device, dtype=torch.long).unsqueeze(0)
    idx = base_idx.unsqueeze(1) + offsets
    idx = torch.clamp(idx, 0, _hdf5_traj_len - 1)
    return idx


def _spatial_angle_to_quat_torch(spatial: torch.Tensor) -> torch.Tensor:
    angle = torch.norm(spatial, dim=-1, keepdim=True)
    eps = 1e-10

    axis = torch.where(
        angle > eps,
        spatial / angle,
        torch.zeros_like(spatial),
    )

    half = 0.5 * angle
    qw = torch.cos(half)
    xyz = axis * torch.sin(half)

    quat = torch.cat([qw, xyz], dim=-1)
    zero_mask = (angle.squeeze(-1) <= eps)
    if torch.any(zero_mask):
        quat[zero_mask, 0] = 1.0
        quat[zero_mask, 1:] = 0.0
    return quat


def _spatial_angle_to_rotmat_torch(spatial: torch.Tensor) -> torch.Tensor:
    device = spatial.device
    dtype = spatial.dtype
    n = spatial.shape[0]

    angle = torch.norm(spatial, dim=-1, keepdim=True)
    eps = 1e-10

    axis = torch.where(angle > eps, spatial / angle, torch.zeros_like(spatial))
    ax = axis[:, 0]
    ay = axis[:, 1]
    az = axis[:, 2]
    th = angle[:, 0]

    c = torch.cos(th)
    s = torch.sin(th)
    one_c = 1.0 - c

    R = torch.zeros((n, 3, 3), device=device, dtype=dtype)

    R[:, 0, 0] = c + ax * ax * one_c
    R[:, 0, 1] = ax * ay * one_c - az * s
    R[:, 0, 2] = ax * az * one_c + ay * s

    R[:, 1, 0] = ay * ax * one_c + az * s
    R[:, 1, 1] = c + ay * ay * one_c
    R[:, 1, 2] = ay * az * one_c - ax * s

    R[:, 2, 0] = az * ax * one_c - ay * s
    R[:, 2, 1] = az * ay * one_c + ax * s
    R[:, 2, 2] = c + az * az * one_c

    zero_mask = angle[:, 0] <= eps
    if torch.any(zero_mask):
        R[zero_mask] = torch.eye(3, device=device, dtype=dtype)

    return R


def _rotmat_to_spatial_angle_torch(R: torch.Tensor) -> torch.Tensor:
    assert R.shape[-2:] == (3, 3), f"Expected (...,3,3), got {R.shape}"

    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    angle = torch.acos(cos_angle)

    eps = 1e-6
    pi = torch.pi

    out = torch.zeros((*R.shape[:-2], 3), dtype=R.dtype, device=R.device)

    small_mask = torch.abs(angle) < eps
    if torch.any(small_mask):
        out[small_mask] = 0.0

    pi_mask = torch.abs(angle - pi) < eps
    if torch.any(pi_mask):
        R_pi = R[pi_mask]
        angle_pi = angle[pi_mask]

        spatial_list = []
        for i in range(R_pi.shape[0]):
            Ri = R_pi[i]
            ai = angle_pi[i]

            if Ri[0, 0] >= Ri[1, 1] and Ri[0, 0] >= Ri[2, 2]:
                axis_x = torch.sqrt(torch.clamp((Ri[0, 0] + 1.0) / 2.0, min=0.0))
                denom = 2.0 * torch.clamp(axis_x, min=1e-8)
                axis_y = Ri[0, 1] / denom
                axis_z = Ri[0, 2] / denom
            elif Ri[1, 1] >= Ri[2, 2]:
                axis_y = torch.sqrt(torch.clamp((Ri[1, 1] + 1.0) / 2.0, min=0.0))
                denom = 2.0 * torch.clamp(axis_y, min=1e-8)
                axis_x = Ri[0, 1] / denom
                axis_z = Ri[1, 2] / denom
            else:
                axis_z = torch.sqrt(torch.clamp((Ri[2, 2] + 1.0) / 2.0, min=0.0))
                denom = 2.0 * torch.clamp(axis_z, min=1e-8)
                axis_x = Ri[0, 2] / denom
                axis_y = Ri[1, 2] / denom

            spatial_list.append(torch.stack([axis_x * ai, axis_y * ai, axis_z * ai]))

        out[pi_mask] = torch.stack(spatial_list, dim=0)

    normal_mask = ~(small_mask | pi_mask)
    if torch.any(normal_mask):
        Rn = R[normal_mask]
        ang = angle[normal_mask]
        sin_ang = torch.sin(ang)

        axis_x = (Rn[:, 2, 1] - Rn[:, 1, 2]) / (2.0 * sin_ang)
        axis_y = (Rn[:, 0, 2] - Rn[:, 2, 0]) / (2.0 * sin_ang)
        axis_z = (Rn[:, 1, 0] - Rn[:, 0, 1]) / (2.0 * sin_ang)

        out[normal_mask, 0] = axis_x * ang
        out[normal_mask, 1] = axis_y * ang
        out[normal_mask, 2] = axis_z * ang

    return out


def _get_orientation_offset_rotm(device: torch.device) -> torch.Tensor:
    global _orientation_offset_rotm

    if _orientation_offset_rotm is not None:
        return _orientation_offset_rotm.to(device=device)

    kin = _get_y2_kin_solver()
    T_home = kin.forward_kinematics(_HOME_Q)
    T_home = torch.tensor(T_home, dtype=torch.float32, device=device)

    R_home_fk = T_home[:3, :3]

    desired_home_spatial = torch.tensor([[0.0, 0.0, 1.5708]], dtype=torch.float32, device=device)
    R_home_desired = _spatial_angle_to_rotmat_torch(desired_home_spatial).squeeze(0)

    _orientation_offset_rotm = R_home_desired @ R_home_fk.T
    return _orientation_offset_rotm


def _get_target_pose_for_polishing(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    global _hdf5_position

    device = env.device
    num_envs = env.num_envs

    if _hdf5_position is not None and _hdf5_position.ndim == 2 and _hdf5_position.shape[0] > 0:
        idx = _get_traj_indices_from_action(env, horizon=1).squeeze(1)
        row = _hdf5_position[idx].to(device=device, dtype=torch.float32)

        target_xyz = row[:, :3]
        target_spatial = row[:, 3:6]
        target_quat = _spatial_angle_to_quat_torch(target_spatial)
        return target_xyz, target_quat

    target_xyz = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
    target_quat = torch.zeros((num_envs, 4), device=device, dtype=torch.float32)
    target_quat[:, 0] = 1.0
    return target_xyz, target_quat


_prev_xyz_for_polishing: torch.Tensor | None = None
_cumulative_removal: torch.Tensor | None = None
_cumulative_contact_distance: torch.Tensor | None = None


def _update_debug_cache(step: int, wrench_env0: torch.Tensor | None = None, metrics_env0: torch.Tensor | None = None):
    try:
        if wrench_env0 is not None and hasattr(local_debug, "_last_ft_debug"):
            local_debug._last_ft_debug["step"] = int(step)
            local_debug._last_ft_debug["wrench"] = wrench_env0.detach().cpu().clone()

        if metrics_env0 is not None and hasattr(local_debug, "_last_polishing_debug"):
            local_debug._last_polishing_debug["step"] = int(step)
            local_debug._last_polishing_debug["metrics"] = metrics_env0.detach().cpu().clone()
    except Exception as e:
        local_debug.print_exception("OBS DEBUG CACHE ERROR", e)


def _call_debug_printers(step: int, wrench_env0: torch.Tensor | None = None, metrics_env0: torch.Tensor | None = None):
    if wrench_env0 is not None:
        try:
            if hasattr(local_debug, "print_ft_sensor_debug"):
                local_debug.print_ft_sensor_debug(int(step), wrench_env0)
        except Exception as e:
            local_debug.print_exception("FT DEBUG PRINT ERROR", e)

    if metrics_env0 is not None:
        try:
            if hasattr(local_debug, "print_polishing_metrics_debug"):
                local_debug.print_polishing_metrics_debug(int(step), metrics_env0)
        except Exception as e:
            local_debug.print_exception("POLISHING DEBUG PRINT ERROR", e)


def get_ee_pose(env: "ManagerBasedRLEnv", asset_name: str = "robot") -> torch.Tensor:
    robot = env.scene[asset_name]
    q = robot.data.joint_pos[:, :6]

    device = q.device
    num_envs = q.shape[0]

    kin = _get_y2_kin_solver()
    R_offset = _get_orientation_offset_rotm(device=device)

    ee_pose_list = []
    q_cpu = q.detach().cpu().to(torch.float64)

    for i in range(num_envs):
        q_i = q_cpu[i].tolist()

        T = kin.forward_kinematics(q_i)
        T_t = torch.tensor(T, dtype=torch.float32, device=device)

        pos = T_t[:3, 3]
        R_fk = T_t[:3, :3]
        R_corr = R_offset @ R_fk
        spatial = _rotmat_to_spatial_angle_torch(R_corr.unsqueeze(0)).squeeze(0)

        ee_pose_list.append(
            [
                float(pos[0]),
                float(pos[1]),
                float(pos[2]),
                float(spatial[0]),
                float(spatial[1]),
                float(spatial[2]),
            ]
        )

    ee_pose = torch.tensor(ee_pose_list, dtype=torch.float32, device=device)
    assert ee_pose.ndim == 2 and ee_pose.shape[1] == 6, f"[EE_POSE] Invalid shape: {ee_pose.shape}"
    return ee_pose


def get_current_pose_and_force(
    env: "ManagerBasedRLEnv",
    asset_name: str = "robot",
    fixed_joint_name: str = "tool0_to_spindle",
    joint_prim_relpath: str = "joints",
) -> torch.Tensor:
    pose = get_ee_pose(env, asset_name=asset_name)
    wrench = local_ft_sensor.get_6axis_ft_fixed_joint(
        env=env,
        asset_name=asset_name,
        fixed_joint_name=fixed_joint_name,
        joint_prim_relpath=joint_prim_relpath,
        verbose=False,
    )
    return torch.cat([pose, wrench], dim=-1)


def load_hdf5_trajectory(
    env: "ManagerBasedRLEnv",
    env_ids,
    file_path: str,
    position_dataset_key: str = "position",
    force_dataset_key: str = "force",
):
    global _hdf5_position, _hdf5_force, _hdf5_traj_len
    import h5py

    with h5py.File(file_path, "r") as f:
        if position_dataset_key not in f:
            raise KeyError(
                f"[ERROR] HDF5 position dataset '{position_dataset_key}' not found. Available keys: {list(f.keys())}"
            )
        if force_dataset_key not in f:
            raise KeyError(
                f"[ERROR] HDF5 force dataset '{force_dataset_key}' not found. Available keys: {list(f.keys())}"
            )

        pos = f[position_dataset_key][:]
        frc = f[force_dataset_key][:]

    if pos.ndim != 2 or pos.shape[1] != 6:
        raise ValueError(f"[ERROR] position dataset must be (T,6), got {pos.shape}")
    if frc.ndim != 2 or frc.shape[1] != 3:
        raise ValueError(f"[ERROR] force dataset must be (T,3), got {frc.shape}")
    if pos.shape[0] != frc.shape[0]:
        raise ValueError(f"[ERROR] position/force length mismatch: {pos.shape[0]} vs {frc.shape[0]}")

    _hdf5_position = torch.tensor(pos, dtype=torch.float32, device=env.device)
    _hdf5_force = torch.tensor(frc, dtype=torch.float32, device=env.device)
    _hdf5_traj_len = int(pos.shape[0])

    local_debug.print_hdf5_positions_loaded(_hdf5_position.shape, file_path)


def load_hdf5_positions(
    env: "ManagerBasedRLEnv",
    env_ids,
    file_path: str,
    position_dataset_key: str = "position",
    force_dataset_key: str = "force",
):
    return load_hdf5_trajectory(
        env=env,
        env_ids=env_ids,
        file_path=file_path,
        position_dataset_key=position_dataset_key,
        force_dataset_key=force_dataset_key,
    )


def get_hdf5_target_positions(env: "ManagerBasedRLEnv", horizon: int = 5) -> torch.Tensor:
    global _hdf5_position

    if _hdf5_position is None:
        return torch.zeros((env.num_envs, horizon * 6), device=env.device, dtype=torch.float32)

    idx = _get_traj_indices_from_action(env, horizon=horizon)
    rows = _hdf5_position[idx]
    return rows.reshape(env.num_envs, horizon * 6)


def get_hdf5_target_forces(env: "ManagerBasedRLEnv", horizon: int = 5) -> torch.Tensor:
    global _hdf5_force

    if _hdf5_force is None:
        return torch.zeros((env.num_envs, horizon * 3), device=env.device, dtype=torch.float32)

    idx = _get_traj_indices_from_action(env, horizon=horizon)
    rows = _hdf5_force[idx]
    return rows.reshape(env.num_envs, horizon * 3)


def get_hdf5_target_pose_force(env: "ManagerBasedRLEnv") -> torch.Tensor:
    global _hdf5_position, _hdf5_force

    if _hdf5_position is None or _hdf5_force is None:
        return torch.zeros((env.num_envs, 9), device=env.device, dtype=torch.float32)

    idx = _get_traj_indices_from_action(env, horizon=1).squeeze(1)
    pos = _hdf5_position[idx]
    frc = _hdf5_force[idx]
    return torch.cat([pos, frc], dim=-1)


def get_joint_velocities(env: "ManagerBasedRLEnv", asset_name: str = "robot") -> torch.Tensor:
    robot = env.scene[asset_name]
    joint_vel = robot.data.joint_vel[:, :6]
    return joint_vel


def get_velocity_adjusted_target_positions(
    env: "ManagerBasedRLEnv",
    horizon: int = 5,
    lookahead_gain: float = 1.0,
) -> torch.Tensor:
    global _hdf5_position

    if _hdf5_position is None:
        return torch.zeros((env.num_envs, horizon * 6), device=env.device, dtype=torch.float32)

    robot = env.scene["robot"]

    current_vel = robot.data.joint_vel[:, :6]
    vel_magnitude = torch.norm(current_vel, dim=-1)

    if _hdf5_traj_len <= 0:
        return torch.zeros((env.num_envs, horizon * 6), device=env.device, dtype=torch.float32)

    idx0 = _get_traj_indices_from_action(env, horizon=1).squeeze(1)
    step_offset = (vel_magnitude * lookahead_gain).to(torch.long)
    dynamic_idx = torch.clamp(idx0 + step_offset, 0, _hdf5_traj_len - 1)

    future_targets = torch.zeros((env.num_envs, horizon * 6), device=env.device, dtype=torch.float32)

    for i in range(horizon):
        h_idx = torch.clamp(dynamic_idx + i, 0, _hdf5_traj_len - 1)
        future_targets[:, i * 6 : (i + 1) * 6] = _hdf5_position[h_idx]

    return future_targets


def get_camera_distance(
    env: "ManagerBasedRLEnv",
    sensor_name: str = "camera",
    debug_interval: int = 100,
) -> torch.Tensor:
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
    global _prev_xyz_for_polishing
    global _cumulative_removal
    global _cumulative_contact_distance

    device = env.device
    num_envs = env.num_envs
    step_i = int(getattr(env, "common_step_counter", 0))

    target_xyz, target_quat = _get_target_pose_for_polishing(env)

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
    current_xyz = ee_pos_w - env_origins

    if hasattr(env, "step_dt"):
        dt = float(env.step_dt)
    elif hasattr(env, "physics_dt"):
        dt = float(env.physics_dt)
    else:
        dt = 0.02

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

    wrench = local_ft_sensor.get_6axis_ft_fixed_joint(
        env=env,
        asset_name=asset_name,
        fixed_joint_name=fixed_joint_name,
        joint_prim_relpath=joint_prim_relpath,
        verbose=False,
    )

    fz = wrench[:, 2:3]
    abs_fz = torch.abs(fz)

    contact_flag = (abs_fz >= contact_force_threshold).to(torch.float32)
    effective_force = torch.clamp(abs_fz - contact_force_threshold, min=0.0)

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

    processed_target_xyz = target_xyz.clone()
    processed_target_xyz[:, offset_axis : offset_axis + 1] -= _cumulative_removal

    metrics = torch.cat(
        [
            cartesian_vel,
            fz,
            abs_fz,
            contact_flag,
            effective_force,
            removal_rate,
            _cumulative_removal,
            _cumulative_contact_distance,
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
        local_debug.print_exception("POLISHING LOGGER ERROR", e)

    out = torch.cat(
        [
            processed_target_xyz,
            target_quat,
            cartesian_vel,
            fz,
            abs_fz,
            contact_flag,
            effective_force,
            _cumulative_removal,
            _cumulative_contact_distance,
        ],
        dim=-1,
    )

    return out
