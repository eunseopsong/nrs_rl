from __future__ import annotations

import torch
# [26.04.02 추가] 추가 내용: 시각화 및 데이터 저장을 위한 라이브러리 임포트
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

# [26.04.02 추가] 추가 내용: 콘솔 출력을 위한 이전 위치 캐시 (속도 계산용)
_prev_xyz_for_debug = None


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
# - [26.04.02 수정] 리워드, 페널티, 총점을 추가하여 출력
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
    # [26.04.02 추가] 추가 내용: 경로 가공량, 제어 주기, 리워드 변수 추가
    reward_total: float = 0.0,
    reward_score: float = 0.0,
    penalty_score: float = 0.0,
    lookahead_offset: int = 0,
    dt: float = 0.02
):
    global _prev_xyz_for_debug

    raw_target_xyz = _as_float_list(raw_target_xyz)
    target_xyz = _as_float_list(target_xyz)
    target_rpy = _as_float_list(target_rpy)
    current_xyz = _as_float_list(current_xyz)
    current_rpy = _as_float_list(current_rpy)

    # [26.04.02 추가] 추가 내용: Cartesian 공간에서의 선속도 자동 계산 (m/s)
    if _prev_xyz_for_debug is None:
        vel_magnitude = 0.0
    else:
        dist = np.linalg.norm(np.array(current_xyz) - np.array(_prev_xyz_for_debug))
        vel_magnitude = dist / dt
    _prev_xyz_for_debug = current_xyz

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
    
    # [26.04.02 추가] 추가 내용: 계산된 카테시안 속도(선속도) 및 동적 경로 가공량 출력
    print(
        f"[Dynamic Path ] cartesian_vel={vel_magnitude: .6f} m/s, "
        f"path_offset={lookahead_offset: d}"
    )

    # [26.04.02 수정] 리워드/페널티/총점 출력 섹션 추가
    print("-" * 100)
    print(
        f"[RL Score    ] TOTAL={reward_total: .6f} | "
        f"REWARD(+)={reward_score: .6f} | "
        f"PENALTY(-)={penalty_score: .6f}"
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

    print("=" * 100)


# =========================================================
# [26.04.02 추가] 추가 내용: 폴리싱 로깅 및 시각화 모듈 (위치 미분 기반 속도 적용)
# =========================================================
class PolishingLoggerAndVisualizer:
    def __init__(self, target_force: float = 10.0, grid_size: int = 50, save_dir: str = "./logs/polishing_results", dt: float = 0.02):
        """
        폴리싱 공정의 힘, 속도(위치 미분), 표면 가공량을 추적하고 시각화하는 클래스
        - dt: 환경의 제어 주기 (예: 50Hz 제어라면 0.02초)
        """
        self.target_force = target_force
        self.grid_size = grid_size
        self.save_dir = save_dir
        self.dt = dt
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 현재 에피소드 버퍼
        self.current_forces = []
        self.current_velocities = []
        self.current_rewards = [] # 스텝별 리워드 저장용
        self.surface_map = np.zeros((self.grid_size, self.grid_size))
        self.current_reward_sum = 0.0
        self.prev_xyz = None  # 속도 미분을 위한 이전 위치 저장
        
        # 베스트 에피소드 버퍼
        self.best_reward = -float('inf')
        self.best_forces = []
        self.best_velocities = []
        self.best_rewards = []
        self.best_surface_map = None

    def step_log(self, current_xyz: list, x_idx: int, y_idx: int, force: float, reward: float):
        """
        현재 위치(current_xyz)를 받아 이전 위치와 비교하여 선속도를 계산하고 가공량을 누적합니다.
        """
        # 1. 위치 미분을 통한 선속도(Cartesian Velocity) 계산
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
        
        # 2. 가공량 계산 (단순화된 프레스턴 모델: 가공량 ∝ 힘 * 속도)
        K = 0.001 
        removal_amount = K * max(0, force) * velocity
        
        # 인덱스 범위 이탈 방지
        x_idx = np.clip(int(x_idx), 0, self.grid_size - 1)
        y_idx = np.clip(int(y_idx), 0, self.grid_size - 1)
        
        # 해당 지점의 표면 깎임 누적
        self.surface_map[x_idx, y_idx] -= removal_amount

    def end_episode(self):
        """에피소드 종료 시 호출. 베스트 케이스 갱신 및 초기화"""
        if self.current_reward_sum > self.best_reward:
            self.best_reward = self.current_reward_sum
            self.best_forces = list(self.current_forces)
            self.best_velocities = list(self.current_velocities)
            self.best_rewards = list(self.current_rewards)
            self.best_surface_map = np.copy(self.surface_map)
            print(f"[Polishing Logger] ⭐ New Best Episode Recorded! Reward Sum: {self.best_reward:.4f}")
            
        # 다음 에피소드를 위해 버퍼 초기화
        self.current_forces = []
        self.current_velocities = []
        self.current_rewards = []
        self.surface_map = np.zeros((self.grid_size, self.grid_size))
        self.current_reward_sum = 0.0
        self.prev_xyz = None  # 위치 초기화

    def visualize_best_case(self):
        """학습 완료 후 베스트 가공 상태와 힘 추종 그래프 시각화"""
        if self.best_surface_map is None:
            print("[Polishing Logger] 저장된 베스트 에피소드가 없어 시각화를 건너뜁니다.")
            return

        fig = plt.figure(figsize=(18, 5))

        # 1. 3D 표면 가공량(Surface Yield) 시각화
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        X, Y = np.meshgrid(range(self.grid_size), range(self.grid_size))
        surf = ax1.plot_surface(X, Y, self.best_surface_map, cmap='viridis', edgecolor='none')
        ax1.set_title("Best Uniform Surface Yield (MRR)")
        ax1.set_xlabel("Surface X")
        ax1.set_ylabel("Surface Y")
        ax1.set_zlabel("Depth")
        fig.colorbar(surf, ax1=ax1, shrink=0.5, aspect=5)

        # 2. 목표 힘 vs 실제 힘 추종 (Force Tracking) 시각화
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(self.best_forces, color='red', label='Actual Force (Fz)')
        ax2.axhline(self.target_force, color='blue', linestyle='--', label=f'Target Force ({self.target_force}N)')
        ax2.set_title("Force Tracking Profile")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Force (N)")
        ax2.legend()
        ax2.grid(True)

        # 3. 리워드 히스토리 추이 그래프 (학습 안정성 확인용)
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


# [26.04.02 추가] 추가 내용: 전역 객체 생성 (dt 값은 환경 제어 주기에 맞춰 수정 필요)
polishing_logger = PolishingLoggerAndVisualizer(target_force=10.0, dt=0.02)