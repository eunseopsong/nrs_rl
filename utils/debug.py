# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch

_last_ft_debug = {
    "step": None,
    "wrench": None,  # [Fx, Fy, Fz, Tx, Ty, Tz]
}

_last_polishing_debug = {
    "step": None,
    "metrics": None,
    # [0] cartesian_speed
    # [1] Fz
    # [2] abs_fz
    # [3] contact_flag
    # [4] effective_normal_force
    # [5] removal_rate
    # [6] cumulative_removal
    # [7] contact_distance
}


def _as_float_list(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().reshape(-1).tolist()
    if isinstance(x, (list, tuple, np.ndarray)):
        return [float(v) for v in np.array(x).reshape(-1).tolist()]
    return [float(x)]


def print_hdf5_positions_loaded(shape, file_path: str):
    print(f"[INFO] Loaded HDF5 positions of shape {shape} from {file_path}")


def print_fixed_joint_ft_cache(
    robot_prim_path_env0: str,
    joint_prim_path: str,
    child_link_name: str | None = None,
    child_link_index: int | None = None,
):
    print(f"[get_6axis_ft_fixed_joint] robot_prim_path_env0 : {robot_prim_path_env0}")
    print(f"[get_6axis_ft_fixed_joint] joint_prim_path      : {joint_prim_path}")
    if child_link_name is not None:
        print(f"[get_6axis_ft_fixed_joint] child_link_name     : {child_link_name}")
    if child_link_index is not None:
        print(f"[get_6axis_ft_fixed_joint] child_link_index    : {child_link_index}")


def print_fixed_joint_ft_failed(error):
    print(f"[get_6axis_ft_fixed_joint] failed: {error}")


def print_action_init(
    hdf5_file_path: str,
    position_dataset_key: str,
    traj_shape,
    stride: int,
    used_traj_shape,
    body_name: str,
    ee_idx: int,
    num_envs: int,
    tcp_length_offset_m: float,
    tcp_offset_axis: str,
):
    print(f"[Action] HDF5 file: {hdf5_file_path}")
    print(f"[Action] dataset key: {position_dataset_key}")
    print(f"[Action] full traj shape: {traj_shape}")
    print(f"[Action] stride: {stride} -> used traj shape: {used_traj_shape}")
    print(f"[Action] EE body_name: {body_name}, ee_idx: {ee_idx}")
    print(f"[Action] num_envs: {num_envs}")
    print(f"[Action] TCP length offset: {tcp_length_offset_m} m")
    print(f"[Action] TCP offset axis: {tcp_offset_axis}")


def print_camera_distance(step: int, mean_distance_env0):
    md_cpu = float(_as_float_list(mean_distance_env0)[0])
    print(f"[Step {step}] Mean camera distance: {md_cpu:.4f} m")


def print_camera_normals(step: int, normals_mean_env0):
    nx, ny, nz = _as_float_list(normals_mean_env0)[:3]
    print(f"[Camera DEBUG] Step {step}: Mean surface normal = [{nx:.6f} {ny:.6f} {nz:.6f}]")


def print_ft_sensor_debug(step: int, wrench_env0):
    global _last_ft_debug

    vals = _as_float_list(wrench_env0)
    if len(vals) < 6:
        vals = vals + [0.0] * (6 - len(vals))

    _last_ft_debug["step"] = int(step)
    _last_ft_debug["wrench"] = vals[:6]


def print_polishing_metrics_debug(step: int, metrics_env0):
    global _last_polishing_debug

    vals = _as_float_list(metrics_env0)
    if len(vals) < 8:
        vals = vals + [0.0] * (8 - len(vals))

    _last_polishing_debug["step"] = int(step)
    _last_polishing_debug["metrics"] = vals[:8]


def print_action_debug_status(
    env_id: int,
    global_step: int,
    path_index: int,
    traj_length: int,
    waypoint_steps: int,
    path_done: bool,
    raw_target_xyz,
    raw_target_force,
    target_xyz,
    target_wxyz,
    target_force,
    current_xyz,
    current_wxyz,
    pos_err_norm: float,
    rot_err_norm: float,
    reward_total: float | None = None,
    reward_score: float | None = None,
    penalty_score: float | None = None,
    dt: float = 0.02,
):
    raw_target_xyz = _as_float_list(raw_target_xyz)
    raw_target_force = _as_float_list(raw_target_force)
    target_xyz = _as_float_list(target_xyz)
    target_wxyz = _as_float_list(target_wxyz)
    target_force = _as_float_list(target_force)
    current_xyz = _as_float_list(current_xyz)
    current_wxyz = _as_float_list(current_wxyz)

    while len(raw_target_xyz) < 3:
        raw_target_xyz.append(0.0)
    while len(raw_target_force) < 3:
        raw_target_force.append(0.0)
    while len(target_xyz) < 3:
        target_xyz.append(0.0)
    while len(target_wxyz) < 3:
        target_wxyz.append(0.0)
    while len(target_force) < 3:
        target_force.append(0.0)
    while len(current_xyz) < 3:
        current_xyz.append(0.0)
    while len(current_wxyz) < 3:
        current_wxyz.append(0.0)

    print("\n" + "=" * 100)

    print(
        f"[Action Debug ] env={env_id} | "
        f"step={global_step} | "
        f"h5_index={path_index}/{max(traj_length - 1, 0)} | "
        f"waypoint_steps={waypoint_steps} | "
        f"done={path_done} | "
        f"pos_err_norm={pos_err_norm:.6f} | "
        f"rot_err_norm={rot_err_norm:.6f}"
    )

    print(
        "[Current Pose ] "
        f"x={current_xyz[0]: .6f}, y={current_xyz[1]: .6f}, z={current_xyz[2]: .6f}, "
        f"wx={current_wxyz[0]: .6f}, wy={current_wxyz[1]: .6f}, wz={current_wxyz[2]: .6f}"
    )

    print(
        "[Target Pose  ] "
        f"x={target_xyz[0]: .6f}, y={target_xyz[1]: .6f}, z={target_xyz[2]: .6f}, "
        f"wx={target_wxyz[0]: .6f}, wy={target_wxyz[1]: .6f}, wz={target_wxyz[2]: .6f}"
    )

    if _last_ft_debug["wrench"] is not None:
        fx, fy, fz, tx, ty, tz = _last_ft_debug["wrench"]
        ft_step = _last_ft_debug["step"]
        print(
            "[Current Force] "
            f"step={ft_step}, "
            f"Fx={fx: .6f}, Fy={fy: .6f}, Fz={fz: .6f}, "
            f"Tx={tx: .6f}, Ty={ty: .6f}, Tz={tz: .6f}"
        )
    else:
        print("[Current Force] No cached 6-axis FT data")

    print(
        "[Target Force ] "
        f"Fx={target_force[0]: .6f}, Fy={target_force[1]: .6f}, Fz={target_force[2]: .6f}"
    )

    if _last_polishing_debug["metrics"] is not None:
        pm_step = _last_polishing_debug["step"]
        pm = _last_polishing_debug["metrics"]

        cartesian_speed = pm[0]
        fz_polish = pm[1]
        abs_fz = pm[2]
        contact_flag = int(pm[3])
        effective_force = pm[4]
        removal_rate = pm[5]
        cumulative_removal = pm[6]
        contact_distance = pm[7]

        print(
            "[Polishing    ] "
            f"step={pm_step}, "
            f"contact={contact_flag}, "
            f"cartesian_speed={cartesian_speed: .6f} m/s, "
            f"Fz={fz_polish: .6f} N, "
            f"|Fz|={abs_fz: .6f} N, "
            f"effective_F={effective_force: .6f} N, "
            f"removal_rate={removal_rate: .8f}, "
            f"cumulative_removal={cumulative_removal: .8f}, "
            f"contact_distance={contact_distance: .6f} m"
        )
    else:
        print("[Polishing    ] No cached polishing metrics")

    if reward_total is None or reward_score is None or penalty_score is None:
        print("[RL Score     ] N/A (actual reward not connected)")
    else:
        print(
            f"[RL Score     ] TOTAL={reward_total: .6f} | "
            f"REWARD(+)= {reward_score: .6f} | "
            f"PENALTY(-)= {penalty_score: .6f}"
        )

    print("=" * 100)