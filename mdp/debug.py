from __future__ import annotations

import torch


# =========================================================
# Internal cache
# - FT sensor debug는 여기 저장만 하고
# - 실제 출력은 action debug 시점에 함께 묶어서 출력
# =========================================================
_last_ft_debug = {
    "step": None,
    "wrench": None,  # [Fx, Fy, Fz, Tx, Ty, Tz]
}


def _as_float_list(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().reshape(-1).tolist()
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    return [float(x)]


# =========================================================
# HDF5 / init prints
# =========================================================
def print_hdf5_positions_loaded(shape, file_path: str):
    print(f"[INFO] Loaded HDF5 positions of shape {shape} from {file_path}")


def print_fixed_joint_ft_cache(
    robot_prim_path_env0: str,
    joint_prim_path: str,
    child_link_name: str,
    child_link_index: int,
):
    print(f"[get_6axis_ft_fixed_joint] robot_prim_path_env0 : {robot_prim_path_env0}")
    print(f"[get_6axis_ft_fixed_joint] joint_prim_path      : {joint_prim_path}")
    print(f"[get_6axis_ft_fixed_joint] child_link_name     : {child_link_name}")
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


# =========================================================
# Camera debug
# =========================================================
def print_camera_distance(step: int, mean_distance_env0):
    md_cpu = float(_as_float_list(mean_distance_env0)[0])
    print(f"[Step {step}] Mean camera distance: {md_cpu:.4f} m")


def print_camera_normals(step: int, normals_mean_env0):
    nx, ny, nz = _as_float_list(normals_mean_env0)[:3]
    print(f"[Camera DEBUG] Step {step}: Mean surface normal = [{nx:.6f} {ny:.6f} {nz:.6f}]")


# =========================================================
# FT sensor debug cache update
# - 여기서는 print하지 않고 저장만 함
# =========================================================
def print_ft_sensor_debug(step: int, wrench_env0):
    global _last_ft_debug

    vals = _as_float_list(wrench_env0)
    if len(vals) < 6:
        vals = vals + [0.0] * (6 - len(vals))

    _last_ft_debug["step"] = int(step)
    _last_ft_debug["wrench"] = vals[:6]


# =========================================================
# Combined action debug print
# - action + ft sensor 를 한 번에 출력
# =========================================================
def print_action_debug_status(
    env_id: int,
    global_step: int,
    path_index: int,
    traj_length: int,
    waypoint_steps: int,
    path_done: bool,
    raw_target_xyz,
    target_xyz,
    target_rpy,
    current_xyz,
    current_rpy,
    pos_err_norm: float,
    rot_err_norm: float,
):
    raw_target_xyz = _as_float_list(raw_target_xyz)
    target_xyz = _as_float_list(target_xyz)
    target_rpy = _as_float_list(target_rpy)
    current_xyz = _as_float_list(current_xyz)
    current_rpy = _as_float_list(current_rpy)

    print("\n" + "=" * 100)
    print(
        f"[Action Debug] env={env_id} | step={global_step} | "
        f"h5_index={path_index}/{traj_length - 1} | "
        f"waypoint_steps={waypoint_steps} | "
        f"done={path_done}"
    )
    print(
        "[Raw Target   ] "
        f"x={raw_target_xyz[0]: .6f}, y={raw_target_xyz[1]: .6f}, z={raw_target_xyz[2]: .6f}"
    )
    print(
        "[Target Pose  ] "
        f"x={target_xyz[0]: .6f}, y={target_xyz[1]: .6f}, z={target_xyz[2]: .6f}, "
        f"r={target_rpy[0]: .6f}, p={target_rpy[1]: .6f}, yw={target_rpy[2]: .6f}"
    )
    print(
        "[Current Pose ] "
        f"x={current_xyz[0]: .6f}, y={current_xyz[1]: .6f}, z={current_xyz[2]: .6f}, "
        f"r={current_rpy[0]: .6f}, p={current_rpy[1]: .6f}, yw={current_rpy[2]: .6f}"
    )
    print(
        f"[Error        ] pos_norm={pos_err_norm: .6f}, "
        f"rot_norm={rot_err_norm: .6f}"
    )

    if _last_ft_debug["wrench"] is not None:
        fx, fy, fz, tx, ty, tz = _last_ft_debug["wrench"]
        ft_step = _last_ft_debug["step"]
        print(
            "[FT Sensor    ] "
            f"step={ft_step}, "
            f"Fx={fx: .6f}, Fy={fy: .6f}, Fz={fz: .6f}, "
            f"Tx={tx: .6f}, Ty={ty: .6f}, Tz={tz: .6f}"
        )
    else:
        print("[FT Sensor    ] No cached 6-axis FT data")

    print("=" * 100)