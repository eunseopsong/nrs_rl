# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch


# ============================================================
# Global caches
# ============================================================
_last_ft_debug = {
    "step": None,
    "wrench": None,
}

_last_polishing_debug = {
    "step": None,
    "metrics": None,
}


# ============================================================
# Utils
# ============================================================
def _as_float_list(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().reshape(-1).tolist()
    if isinstance(x, (list, tuple, np.ndarray)):
        return [float(v) for v in np.array(x).reshape(-1).tolist()]
    return [float(x)]


# ============================================================
# Generic print helpers
# ============================================================
def print_exception(context: str, error):
    print(f"[{context}] {repr(error)}")


def print_info(msg: str):
    print(msg)


# ============================================================
# Simple info prints
# ============================================================
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
    body_name: str,
    ee_idx: int,
    num_envs: int,
):
    print(f"[Action] HDF5 file: {hdf5_file_path}")
    print(f"[Action] dataset key: {position_dataset_key}")
    print(f"[Action] traj shape: {traj_shape}")
    print(f"[Action] EE body_name: {body_name}, ee_idx: {ee_idx}")
    print(f"[Action] num_envs: {num_envs}")


def print_camera_distance(step: int, mean_distance_env0):
    md_cpu = float(_as_float_list(mean_distance_env0)[0])
    print(f"[Step {step}] Mean camera distance: {md_cpu:.4f} m")


def print_camera_normals(step: int, normals_mean_env0):
    nx, ny, nz = _as_float_list(normals_mean_env0)[:3]
    print(f"[Camera DEBUG] Step {step}: Mean surface normal = [{nx:.6f} {ny:.6f} {nz:.6f}]")


# ============================================================
# Cache-only debug recorders
# ============================================================
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


def format_reward_debug(env, env_id: int = 0) -> str:
    reward_parts = []

    try:
        if hasattr(env, "reward_buf"):
            reward_buf = env.reward_buf
            if isinstance(reward_buf, torch.Tensor) and reward_buf.numel() > int(env_id):
                reward_parts.append(f"last_total={float(reward_buf[int(env_id)].item()):.6f}")

        if hasattr(env, "reward_manager"):
            reward_dict = getattr(env.reward_manager, "episode_sums", getattr(env.reward_manager, "_episode_sums", {}))
            for term_name, value_tensor in reward_dict.items():
                if isinstance(value_tensor, torch.Tensor):
                    value = float(value_tensor[int(env_id)].item())
                else:
                    values = _as_float_list(value_tensor)
                    value = values[int(env_id)] if len(values) > int(env_id) else values[0]
                reward_parts.append(f"{term_name}_episode_sum={value:.6f}")
    except Exception as e:
        return f"reward_debug_error={repr(e)}"

    if not reward_parts:
        return "No reward debug data"
    return " | ".join(reward_parts)


# ============================================================
# Unified action runtime print
# ============================================================
def print_action_runtime(
    env_id: int,
    global_step: int,
    current_index: int,
    next_index: int,
    traj_length: int,
    path_done: bool,
    pos_err_norm: float,
    rot_err_norm: float,
    pybind_called: bool,
    pybind_success: bool,
    inner_iters: int,
    dq_norm: float,
    current_xyz,
    current_wxyz,
    target_xyz,
    target_wxyz,
    target_force,
):
    current_xyz = _as_float_list(current_xyz)
    current_wxyz = _as_float_list(current_wxyz)
    target_xyz = _as_float_list(target_xyz)
    target_wxyz = _as_float_list(target_wxyz)
    target_force = _as_float_list(target_force)

    print("\n" + "=" * 100)
    print(
        f"[Pybind IK   ] called={pybind_called} success={pybind_success} "
        f"inner_iters={inner_iters} dq_norm={dq_norm:.6f}"
    )
    print(
        f"[Action Debug ] env={env_id} | step={global_step} "
        f"| h5_index={current_index}/{traj_length} "
        f"| next_index={next_index}/{traj_length} "
        f"| done={path_done} "
        f"| pos_err_norm={pos_err_norm:.6f} "
        f"| rot_err_norm={rot_err_norm:.6f}"
    )
    print(
        f"[Current Pose ] x={current_xyz[0]: .6f}, y={current_xyz[1]: .6f}, z={current_xyz[2]: .6f}, "
        f"wx={current_wxyz[0]: .6f}, wy={current_wxyz[1]: .6f}, wz={current_wxyz[2]: .6f}"
    )
    print(
        f"[Target Pose  ] x={target_xyz[0]: .6f}, y={target_xyz[1]: .6f}, z={target_xyz[2]: .6f}, "
        f"wx={target_wxyz[0]: .6f}, wy={target_wxyz[1]: .6f}, wz={target_wxyz[2]: .6f}"
    )

    if _last_ft_debug["wrench"] is not None:
        fx, fy, fz, tx, ty, tz = _last_ft_debug["wrench"]
        ft_step = _last_ft_debug["step"]
        print(
            f"[Current Force] step={ft_step}, "
            f"Fx={fx: .6f}, Fy={fy: .6f}, Fz={fz: .6f}, "
            f"Tx={tx: .6f}, Ty={ty: .6f}, Tz={tz: .6f}"
        )
    else:
        print("[Current Force] No cached 6-axis FT data")

    print(
        f"[Target Force ] Fx={target_force[0]: .6f}, Fy={target_force[1]: .6f}, Fz={target_force[2]: .6f}"
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
            f"[Polishing    ] step={pm_step}, "
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

    print("=" * 100)
