from __future__ import annotations

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# =========================================================
# Internal cache
# - FT sensor debug는 여기 저장만 하고
# - 실제 출력은 action debug 시점에 함께 묶어서 출력
# =========================================================
_last_ft_debug = {
    "step": None,
    "wrench": None,  # [Fx, Fy, Fz, Tx, Ty, Tz]
}

# =========================================================
# Polishing metric cache
# =========================================================
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

# 콘솔 출력을 위한 이전 위치 캐시 (fallback 속도 계산용)
_prev_xyz_for_debug = None


def _as_float_list(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().reshape(-1).tolist()
    if isinstance(x, (list, tuple, np.ndarray)):
        return [float(v) for v in np.array(x).reshape(-1).tolist()]
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
# Polishing debug cache update
# - 여기서는 print하지 않고 저장만 함
# =========================================================
def print_polishing_metrics_debug(step: int, metrics_env0):
    global _last_polishing_debug

    vals = _as_float_list(metrics_env0)
    if len(vals) < 8:
        vals = vals + [0.0] * (8 - len(vals))

    _last_polishing_debug["step"] = int(step)
    _last_polishing_debug["metrics"] = vals[:8]


# =========================================================
# Combined action debug print
# - action + ft sensor + polishing metrics 를 한 번에 출력
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
    reward_total: float = 0.0,
    reward_score: float = 0.0,
    penalty_score: float = 0.0,
    dt: float = 0.02
):
    global _prev_xyz_for_debug

    raw_target_xyz = _as_float_list(raw_target_xyz)
    target_xyz = _as_float_list(target_xyz)
    target_rpy = _as_float_list(target_rpy)
    current_xyz = _as_float_list(current_xyz)
    current_rpy = _as_float_list(current_rpy)

    # fallback velocity 계산
    if _prev_xyz_for_debug is None:
        vel_magnitude = 0.0
    else:
        dist = np.linalg.norm(np.array(current_xyz) - np.array(_prev_xyz_for_debug))
        vel_magnitude = dist / max(dt, 1e-8)
    _prev_xyz_for_debug = current_xyz

    # polishing cache에 있으면 그 속도를 우선 사용
    if _last_polishing_debug["metrics"] is not None:
        vel_display = _last_polishing_debug["metrics"][0]
    else:
        vel_display = vel_magnitude

    # FT cache에 있으면 Dynamic Path 줄에도 Fz 같이 표시
    fz_for_dynamic = None
    if _last_ft_debug["wrench"] is not None:
        fz_for_dynamic = float(_last_ft_debug["wrench"][2])

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

    if fz_for_dynamic is None:
        print(
            f"[Dynamic Path ] cartesian_vel={vel_display: .6f} m/s"
        )
    else:
        print(
            f"[Dynamic Path ] cartesian_vel={vel_display: .6f} m/s, "
            f"Fz={fz_for_dynamic: .6f} N"
        )

    print("-" * 100)
    print(
        f"[RL Score    ] TOTAL={reward_total: .6f} | "
        f"REWARD(+)= {reward_score: .6f} | "
        f"PENALTY(-)= {penalty_score: .6f}"
    )
    print("-" * 100)

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

    if _last_polishing_debug["metrics"] is not None:
        pm_step = _last_polishing_debug["step"]
        pm = _last_polishing_debug["metrics"]

        cartesian_speed = pm[0]
        fz = pm[1]
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
            f"Fz={fz: .6f} N, "
            f"|Fz|={abs_fz: .6f} N, "
            f"effective_F={effective_force: .6f} N, "
            f"removal_rate={removal_rate: .8f}, "
            f"cumulative_removal={cumulative_removal: .8f}, "
            f"contact_distance={contact_distance: .6f} m"
        )
    else:
        print("[Polishing    ] No cached polishing metrics")

    print("=" * 100)


# =========================================================
# 폴리싱 로깅 및 시각화 모듈
# =========================================================
class PolishingLoggerAndVisualizer:
    def __init__(self, target_force: float = 10.0, grid_size: int = 50, save_dir: str = "./logs/polishing_results", dt: float = 0.02):
        self.target_force = target_force
        self.grid_size = grid_size
        self.save_dir = save_dir
        self.dt = dt
        os.makedirs(self.save_dir, exist_ok=True)

        self.current_forces = []
        self.current_velocities = []
        self.current_rewards = []
        self.surface_map = np.zeros((self.grid_size, self.grid_size))
        self.current_reward_sum = 0.0
        self.prev_xyz = None

        self.best_reward = -float('inf')
        self.best_forces = []
        self.best_velocities = []
        self.best_rewards = []
        self.best_surface_map = None

    def step_log(self, current_xyz: list, x_idx: int, y_idx: int, force: float, reward: float):
        if self.prev_xyz is None:
            velocity = 0.0
        else:
            dist = np.linalg.norm(np.array(current_xyz) - np.array(self.prev_xyz))
            velocity = dist / self.dt

        self.prev_xyz = current_xyz

        self.current_forces.append(force)
        self.current_velocities.append(velocity)
        self.current_rewards.append(reward)
        self.current_reward_sum += reward

        K = 0.001
        removal_amount = K * max(0, force) * velocity

        x_idx = np.clip(int(x_idx), 0, self.grid_size - 1)
        y_idx = np.clip(int(y_idx), 0, self.grid_size - 1)

        self.surface_map[x_idx, y_idx] -= removal_amount

    def end_episode(self):
        if self.current_reward_sum > self.best_reward:
            self.best_reward = self.current_reward_sum
            self.best_forces = list(self.current_forces)
            self.best_velocities = list(self.current_velocities)
            self.best_rewards = list(self.current_rewards)
            self.best_surface_map = np.copy(self.surface_map)
            print(f"[Polishing Logger] ⭐ New Best Episode Recorded! Reward Sum: {self.best_reward:.4f}")

        self.current_forces = []
        self.current_velocities = []
        self.current_rewards = []
        self.surface_map = np.zeros((self.grid_size, self.grid_size))
        self.current_reward_sum = 0.0
        self.prev_xyz = None

    def visualize_best_case(self):
        if self.best_surface_map is None:
            print("[Polishing Logger] 저장된 베스트 에피소드가 없어 시각화를 건너뜁니다.")
            return

        fig = plt.figure(figsize=(18, 5))

        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        X, Y = np.meshgrid(range(self.grid_size), range(self.grid_size))
        surf = ax1.plot_surface(X, Y, self.best_surface_map, cmap='viridis', edgecolor='none')
        ax1.set_title("Best Uniform Surface Yield (MRR)")
        ax1.set_xlabel("Surface X")
        ax1.set_ylabel("Surface Y")
        ax1.set_zlabel("Depth")
        fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(self.best_forces, color='red', label='Actual Force (Fz)')
        ax2.axhline(self.target_force, color='blue', linestyle='--', label=f'Target Force ({self.target_force}N)')
        ax2.set_title("Force Tracking Profile")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Force (N)")
        ax2.legend()
        ax2.grid(True)

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.plot(self.best_rewards, color='green')
        ax3.set_title("Reward Step History (Best Episode)")
        ax3.set_xlabel("Time Step")
        ax3.set_ylabel("Reward")
        ax3.grid(True)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, "best_polishing_result.png")
        plt.savefig(save_path)
        print(f"\n[Polishing Logger] ✅ Visualization successfully saved to: {save_path}\n")


polishing_logger = PolishingLoggerAndVisualizer(target_force=10.0, dt=0.02)