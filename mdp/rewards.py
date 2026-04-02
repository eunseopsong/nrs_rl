# SPDX-License-Identifier: BSD-3-Clause
"""
Reward functions module for Uniform Polishing.
[26.04.02 업데이트]
- 프레스턴 방정식 기반 MRR 균일화 보상
- 6축 FT 센서 데이터 및 Cartesian 선속도 연동
- Look-ahead 웨이포인트를 활용한 코너링 감속/힘 조절 유도
- 현재 nrs_rl 환경 구조에 맞게 안전하게 수정됨
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import importlib
import math
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

local_obs = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.mdp.observation"
)

local_ft_sensor = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.assets.assets.sensors.six_axis_ft_sensor"
)


# =========================================================================
# Internal helper functions
# =========================================================================
def _get_ee_idx(
    env: "ManagerBasedRLEnv",
    asset_name: str = "robot",
    body_name: str = "spindle_link",
) -> int:
    robot = env.scene[asset_name]
    body_ids = robot.find_bodies(body_name)[0]
    if len(body_ids) == 0:
        raise ValueError(
            f"[rewards] body_name='{body_name}' not found. Available bodies: {robot.body_names}"
        )
    return int(body_ids[0])


def _get_current_pose_and_velocity(
    env: "ManagerBasedRLEnv",
    asset_name: str = "robot",
    body_name: str = "spindle_link",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        current_xyz  : (N, 3) env-local
        current_quat : (N, 4) world quat
        vel_norm     : (N,)
    """
    robot = env.scene[asset_name]
    ee_idx = _get_ee_idx(env, asset_name=asset_name, body_name=body_name)

    ee_pos_w = robot.data.body_pos_w[:, ee_idx, :]
    ee_quat_w = robot.data.body_quat_w[:, ee_idx, :]
    ee_lin_vel_w = robot.data.body_lin_vel_w[:, ee_idx, :]

    env_origins = env.scene.env_origins
    current_xyz = ee_pos_w - env_origins
    vel_norm = torch.norm(ee_lin_vel_w, dim=-1)

    return current_xyz, ee_quat_w, vel_norm


def _get_current_fz(
    env: "ManagerBasedRLEnv",
    asset_name: str = "robot",
    fixed_joint_name: str = "tool0_to_spindle",
    joint_prim_relpath: str = "joints",
) -> torch.Tensor:
    """
    Returns:
        current_fz : (N,)
    """
    wrench = local_ft_sensor.get_6axis_ft_fixed_joint(
        env=env,
        asset_name=asset_name,
        fixed_joint_name=fixed_joint_name,
        joint_prim_relpath=joint_prim_relpath,
        verbose=False,
    )
    return wrench[:, 2]


def _safe_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / torch.clamp(torch.norm(x, dim=-1, keepdim=True), min=eps)


def _get_current_target_pose(
    env: "ManagerBasedRLEnv",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Safe current target getter.
    observation.py 쪽 fallback 로직을 그대로 활용
    """
    return local_obs._get_target_pose_for_polishing(env)


def _get_lookahead_xyz(
    env: "ManagerBasedRLEnv",
    lookahead_steps: int = 5,
) -> torch.Tensor:
    """
    Safe future target getter.

    Priority:
      1) command_manager['lookahead'] if available
      2) HDF5 trajectory fallback
      3) current target fallback
    """
    device = env.device
    num_envs = env.num_envs

    # 1) command_manager lookahead
    try:
        cmd_mgr = getattr(env, "command_manager", None)
        if cmd_mgr is not None and hasattr(cmd_mgr, "_terms") and ("lookahead" in cmd_mgr._terms):
            cmd = cmd_mgr.get_command("lookahead")
            if cmd is not None and cmd.ndim == 2 and cmd.shape[1] >= 3:
                return cmd[:, :3].to(device=device, dtype=torch.float32)
    except Exception:
        pass

    # 2) fallback from HDF5
    h5 = getattr(local_obs, "_hdf5_positions", None)
    if h5 is not None and h5.ndim == 2 and h5.shape[0] > 0:
        t_total, d = h5.shape

        if hasattr(env, "episode_length_buf"):
            step = env.episode_length_buf.to(torch.float32)
        else:
            step = torch.zeros((num_envs,), device=device, dtype=torch.float32)

        ep_len = max(int(getattr(env, "max_episode_length", 1)), 1)
        base_idx = ((step / ep_len) * t_total).to(torch.int64)
        future_idx = torch.clamp(base_idx + int(lookahead_steps), 0, t_total - 1)

        row = h5[future_idx].to(device=device, dtype=torch.float32)
        if d >= 3:
            return row[:, :3]

    # 3) final fallback = current target
    target_xyz, _ = _get_current_target_pose(env)
    return target_xyz


def _quat_angle_error(current_quat: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:
    """
    Quaternion angular distance in rad.
    quat format: (w, x, y, z)
    """
    quat_inner = torch.abs(torch.sum(current_quat * target_quat, dim=-1))
    quat_inner = torch.clamp(quat_inner, -1.0 + 1e-6, 1.0 - 1e-6)
    return 2.0 * torch.acos(quat_inner)


# =========================================================================
# 1. Uniform Material Removal Rate (MRR) Reward
# =========================================================================
def uniform_mrr_reward(
    env: "ManagerBasedRLEnv",
    target_force: float = 20.0,
    target_velocity: float = 0.0002,
    mrr_sigma: float = 0.002,
    asset_name: str = "robot",
    body_name: str = "spindle_link",
    fixed_joint_name: str = "tool0_to_spindle",
    joint_prim_relpath: str = "joints",
) -> torch.Tensor:
    """
    Preston proxy = |Fz| * |v|

    현재 디버그 값 기준으로 cartesian_vel 이 매우 작게 나오므로,
    target_velocity 도 같은 스케일(예: 0.0002 m/s 전후)로 두는 게 맞습니다.
    """
    current_fz = _get_current_fz(
        env,
        asset_name=asset_name,
        fixed_joint_name=fixed_joint_name,
        joint_prim_relpath=joint_prim_relpath,
    )

    _, _, current_vel_norm = _get_current_pose_and_velocity(
        env,
        asset_name=asset_name,
        body_name=body_name,
    )

    current_mrr_proxy = torch.abs(current_fz) * current_vel_norm
    target_mrr_proxy = target_force * target_velocity

    mrr_error = current_mrr_proxy - target_mrr_proxy
    reward = torch.exp(-torch.square(mrr_error / max(mrr_sigma, 1e-6)))
    return reward


# =========================================================================
# 2. Force Tracking Reward
# =========================================================================
def force_tracking_reward(
    env: "ManagerBasedRLEnv",
    target_force: float = 20.0,
    force_sigma: float = 5.0,
    asset_name: str = "robot",
    fixed_joint_name: str = "tool0_to_spindle",
    joint_prim_relpath: str = "joints",
) -> torch.Tensor:
    """
    목표 접촉 힘 추종 보상
    """
    current_fz = _get_current_fz(
        env,
        asset_name=asset_name,
        fixed_joint_name=fixed_joint_name,
        joint_prim_relpath=joint_prim_relpath,
    )

    force_error = torch.abs(torch.abs(current_fz) - target_force)
    reward = torch.exp(-torch.square(force_error / max(force_sigma, 1e-6)))
    return reward


# =========================================================================
# 3. Look-ahead Cornering Smoothness Penalty
# =========================================================================
def lookahead_cornering_penalty(
    env: "ManagerBasedRLEnv",
    cornering_threshold_angle: float = 0.5,   # rad
    penalty_scale: float = 0.5,
    lookahead_steps: int = 5,
    speed_ref: float = 0.002,
    action_rate_scale: float = 0.1,
    asset_name: str = "robot",
    body_name: str = "spindle_link",
) -> torch.Tensor:
    """
    코너링이 큰 구간에서 속도가 높거나 action 변화가 크면 페널티.
    """
    current_xyz, _, current_vel_norm = _get_current_pose_and_velocity(
        env,
        asset_name=asset_name,
        body_name=body_name,
    )

    target_xyz, _ = _get_current_target_pose(env)
    lookahead_xyz = _get_lookahead_xyz(env, lookahead_steps=lookahead_steps)

    current_dir = target_xyz - current_xyz
    future_dir = lookahead_xyz - target_xyz

    current_dir_norm = _safe_normalize(current_dir)
    future_dir_norm = _safe_normalize(future_dir)

    cos_sim = torch.sum(current_dir_norm * future_dir_norm, dim=-1)
    cos_sim = torch.clamp(cos_sim, -1.0 + 1e-6, 1.0 - 1e-6)
    angle_diff = torch.acos(cos_sim)

    # 코너 강도: threshold 이하 0, 그 이상 선형 증가
    corner_strength = torch.relu(angle_diff - cornering_threshold_angle)
    corner_strength = corner_strength / max(math.pi - cornering_threshold_angle, 1e-6)

    speed_term = current_vel_norm / max(speed_ref, 1e-6)

    action_term = torch.zeros_like(speed_term)
    if hasattr(env, "action_manager"):
        if hasattr(env.action_manager, "action") and hasattr(env.action_manager, "prev_action"):
            action_diff = env.action_manager.action - env.action_manager.prev_action
            action_term = torch.norm(action_diff, dim=-1) * action_rate_scale

    penalty = penalty_scale * corner_strength * (speed_term + action_term)
    penalty = torch.clamp(penalty, min=0.0, max=5.0)

    return -penalty


# =========================================================================
# 4. Trajectory Tracking Penalty (Position & Rotation Error)
# =========================================================================
def trajectory_tracking_penalty(
    env: "ManagerBasedRLEnv",
    pos_sigma: float = 0.03,
    rot_sigma: float = 0.20,
    asset_name: str = "robot",
    body_name: str = "spindle_link",
) -> torch.Tensor:
    """
    경로 이탈 페널티
    - bounded penalty 형태
    - 원래 코드처럼 exp(+err^2)가 아니라 안정적으로 음수 범위에 머물도록 수정
    """
    current_xyz, current_quat, _ = _get_current_pose_and_velocity(
        env,
        asset_name=asset_name,
        body_name=body_name,
    )

    target_xyz, target_quat = _get_current_target_pose(env)

    pos_error_norm = torch.norm(current_xyz - target_xyz, dim=-1)
    rot_error_norm = _quat_angle_error(current_quat, target_quat)

    pos_penalty = -(1.0 - torch.exp(-torch.square(pos_error_norm / max(pos_sigma, 1e-6))))
    rot_penalty = -(1.0 - torch.exp(-torch.square(rot_error_norm / max(rot_sigma, 1e-6))))

    return pos_penalty + rot_penalty


# =========================================================================
# 5. Action Smoothness (Chattering Prevention)
# =========================================================================
def action_smoothness_penalty(
    env: "ManagerBasedRLEnv",
) -> torch.Tensor:
    """
    이전 Action과 현재 Action의 변화량(L2 Norm)에 대한 페널티
    """
    if not hasattr(env, "action_manager"):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    if not hasattr(env.action_manager, "action") or not hasattr(env.action_manager, "prev_action"):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    action_diff = env.action_manager.action - env.action_manager.prev_action
    return -torch.norm(action_diff, dim=-1)