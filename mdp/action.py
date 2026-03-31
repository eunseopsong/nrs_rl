# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import os
import h5py
import torch

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass

from ..utils import debug as local_debug


# =========================================================
# Math utils
# =========================================================
def normalize_quat(q: torch.Tensor) -> torch.Tensor:
    return q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1e-8)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[:, 1:] = -out[:, 1:]
    return out


def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    # q = [w, x, y, z]
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    q = normalize_quat(q)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    r = torch.zeros((q.shape[0], 3, 3), device=q.device, dtype=q.dtype)

    r[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    r[:, 0, 1] = 2.0 * (x * y - z * w)
    r[:, 0, 2] = 2.0 * (x * z + y * w)

    r[:, 1, 0] = 2.0 * (x * y + z * w)
    r[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    r[:, 1, 2] = 2.0 * (y * z - x * w)

    r[:, 2, 0] = 2.0 * (x * z - y * w)
    r[:, 2, 1] = 2.0 * (y * z + x * w)
    r[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)

    return r


def rpy_to_quat(rpy: torch.Tensor) -> torch.Tensor:
    roll = rpy[:, 0]
    pitch = rpy[:, 1]
    yaw = rpy[:, 2]

    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)

    # Rz(yaw) * Ry(pitch) * Rx(roll)
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = cy * sp * cr + sy * cp * sr
    z = sy * cp * cr - cy * sp * sr
    return normalize_quat(torch.stack([w, x, y, z], dim=-1))


def quat_to_rpy(q: torch.Tensor) -> torch.Tensor:
    q = normalize_quat(q)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1)


def orientation_error_world(cur_quat: torch.Tensor, des_quat: torch.Tensor) -> torch.Tensor:
    cur_quat = normalize_quat(cur_quat)
    des_quat = normalize_quat(des_quat)

    q_err = quat_multiply(des_quat, quat_conjugate(cur_quat))
    q_err = normalize_quat(q_err)

    sign = torch.where(q_err[:, 0:1] < 0.0, -1.0, 1.0)
    q_err = q_err * sign
    return 2.0 * q_err[:, 1:4]


# =========================================================
# Action Term
# =========================================================
class AdmittanceControlAction(ActionTerm):
    """
    Multi-env HDF5 pose path follower.

    Behavior:
    - ignores RL action input
    - loads target EE path from HDF5: (x, y, z, roll, pitch, yaw)
    - every env tracks the trajectory in its own local frame
    - current EE pose is converted from world frame to env-local frame
    - waypoint progression is managed independently per env
    - uses Jacobian-based damped least-squares IK for UR10e 6 joints
    - supports TCP/spindle length compensation in tool-local axis
    """

    cfg: "AdmittanceControlActionCfg"

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.cfg = cfg
        self.robot = self._env.scene[cfg.asset_name]
        self._num_envs_local = self._env.num_envs
        self._step_dt_local = self._env.step_dt

        body_ids = self.robot.find_bodies(self.cfg.body_name)[0]
        if len(body_ids) == 0:
            raise ValueError(
                f"[Action] body_name='{self.cfg.body_name}' not found. "
                f"Available bodies: {self.robot.body_names}"
            )
        self.ee_idx = body_ids[0]

        self._raw_actions = torch.zeros((self._num_envs_local, self.cfg.action_dim), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        traj_full = self._load_hdf5_positions(
            self.cfg.hdf5_file_path,
            self.cfg.position_dataset_key,
        )

        stride = max(1, int(self.cfg.waypoint_stride))
        self.traj_positions = traj_full[::stride].contiguous()
        self.traj_length = self.traj_positions.shape[0]

        # env-wise trajectory state
        self.path_index = torch.zeros(self._num_envs_local, dtype=torch.long, device=self.device)
        self.steps_at_waypoint = torch.zeros(self._num_envs_local, dtype=torch.long, device=self.device)
        self.path_done = torch.zeros(self._num_envs_local, dtype=torch.bool, device=self.device)

        self.des_pos = torch.zeros((self._num_envs_local, 3), device=self.device)
        self.des_quat = torch.zeros((self._num_envs_local, 4), device=self.device)
        self.des_quat[:, 0] = 1.0

        local_debug.print_action_init(
            hdf5_file_path=self.cfg.hdf5_file_path,
            position_dataset_key=self.cfg.position_dataset_key,
            traj_shape=tuple(traj_full.shape),
            stride=stride,
            used_traj_shape=tuple(self.traj_positions.shape),
            body_name=self.cfg.body_name,
            ee_idx=self.ee_idx,
            num_envs=self._num_envs_local,
            tcp_length_offset_m=self.cfg.tcp_length_offset_m,
            tcp_offset_axis=self.cfg.tcp_offset_axis,
        )

    @property
    def action_dim(self):
        return self.cfg.action_dim

    @property
    def raw_actions(self):
        return self._raw_actions

    @property
    def processed_actions(self):
        return self._processed_actions

    def _load_hdf5_positions(self, file_path: str, dataset_key: str) -> torch.Tensor:
        if not file_path:
            raise ValueError("[Action] hdf5_file_path is empty.")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[Action] HDF5 file not found: {file_path}")

        with h5py.File(file_path, "r") as f:
            if dataset_key in f:
                data = f[dataset_key][:]
            elif "target_positions" in f:
                data = f["target_positions"][:]
            elif "positions" in f:
                data = f["positions"][:]
            else:
                keys = list(f.keys())
                if len(keys) == 0:
                    raise KeyError("[Action] HDF5 file has no datasets.")
                data = f[keys[0]][:]

        data = torch.tensor(data, dtype=torch.float32, device=self.device)

        if data.ndim != 2:
            raise ValueError(f"[Action] expected [T, D], got {tuple(data.shape)}")
        if data.shape[1] < 6:
            raise ValueError(f"[Action] expected at least 6 columns, got {data.shape[1]}")

        return data[:, :6]

    def reset(self, env_ids=None):
        super().reset(env_ids)

        if env_ids is None:
            env_ids = torch.arange(self._num_envs_local, device=self.device)

        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0

        self.path_index[env_ids] = 0
        self.steps_at_waypoint[env_ids] = 0
        self.path_done[env_ids] = False

        des = self.traj_positions[0].unsqueeze(0).repeat(len(env_ids), 1)
        self.des_pos[env_ids] = des[:, 0:3]
        self.des_quat[env_ids] = rpy_to_quat(des[:, 3:6])

    def process_actions(self, actions: torch.Tensor):
        # RL action ignored; env-wise deterministic path follower
        self._raw_actions = torch.nan_to_num(actions.clone(), nan=0.0)
        self._processed_actions.zero_()

    def apply_actions(self):
        # -------------------------------------------------
        # 1) Current EE pose
        #    Convert world position -> env local position
        # -------------------------------------------------
        ee_pos_w = self.robot.data.body_pos_w[:, self.ee_idx, :]
        env_origins = self._env.scene.env_origins
        ee_pos = ee_pos_w - env_origins

        ee_quat = self.robot.data.body_quat_w[:, self.ee_idx, :]

        # -------------------------------------------------
        # 2) Current joints / Jacobian
        # -------------------------------------------------
        q_all = self.robot.data.joint_pos
        q = q_all[:, :6]

        jac_all = self.robot.root_physx_view.get_jacobians()
        jacobian = jac_all[:, self.ee_idx - 1, :, :6]

        # -------------------------------------------------
        # 3) Raw target from HDF5 (same path, independent per env)
        # -------------------------------------------------
        des = self.traj_positions[self.path_index]
        raw_des_pos = des[:, 0:3].clone()
        self.des_quat = rpy_to_quat(des[:, 3:6])

        # -------------------------------------------------
        # 4) TCP / spindle length compensation
        # -------------------------------------------------
        offset_local = self._get_local_tcp_offset(
            length=self.cfg.tcp_length_offset_m,
            axis=self.cfg.tcp_offset_axis,
            dtype=raw_des_pos.dtype,
        )
        rotm = quat_to_rotmat(self.des_quat)
        offset_world_like = torch.bmm(rotm, offset_local.unsqueeze(-1)).squeeze(-1)

        # Here target trajectory is already in env-local coordinates.
        # Since envs differ only by translation, the same rotated offset
        # can be added directly in each env-local frame.
        self.des_pos = raw_des_pos + offset_world_like

        # optional extra world/local z trim
        self.des_pos[:, 2] += self.cfg.z_target_offset

        # -------------------------------------------------
        # 5) IK error (all in env-local position + world orientation)
        # -------------------------------------------------
        pos_err = self.des_pos - ee_pos
        rot_err = orientation_error_world(ee_quat, self.des_quat)

        pos_err_norm = torch.linalg.norm(pos_err, dim=-1)
        rot_err_norm = torch.linalg.norm(rot_err, dim=-1)

        pos_err_clamped = torch.clamp(pos_err, -self.cfg.max_pos_err, self.cfg.max_pos_err)
        rot_err_clamped = torch.clamp(rot_err, -self.cfg.max_rot_err, self.cfg.max_rot_err)
        err_6d = torch.cat([pos_err_clamped, rot_err_clamped], dim=-1)

        dq = self._solve_dls_ik(jacobian, err_6d, self.cfg.dls_lambda)
        dq = torch.clamp(dq, -self.cfg.max_dq, self.cfg.max_dq)

        q_cmd_6 = q + self.cfg.ik_step_size * dq

        if self.cfg.joint_lower_limits is not None and self.cfg.joint_upper_limits is not None:
            q_min = torch.tensor(self.cfg.joint_lower_limits, device=self.device, dtype=q_cmd_6.dtype).unsqueeze(0)
            q_max = torch.tensor(self.cfg.joint_upper_limits, device=self.device, dtype=q_cmd_6.dtype).unsqueeze(0)
            q_cmd_6 = torch.clamp(q_cmd_6, q_min, q_max)

        q_cmd_all = q_all.clone()
        q_cmd_all[:, :6] = q_cmd_6
        q_cmd_all = torch.where(torch.isnan(q_cmd_all), q_all, q_cmd_all)

        self.robot.set_joint_position_target(q_cmd_all)

        # -------------------------------------------------
        # 6) Env-wise waypoint update
        # -------------------------------------------------
        self._update_waypoint_progress(pos_err_norm, rot_err_norm)

        # -------------------------------------------------
        # 7) Debug (env-local current pose)
        # -------------------------------------------------
        self._debug_print_status(ee_pos, ee_quat, raw_des_pos, pos_err_norm, rot_err_norm)

    def _get_local_tcp_offset(self, length: float, axis: str, dtype: torch.dtype) -> torch.Tensor:
        """
        Create tool-local offset vector for all envs.

        axis options:
          - local_x_pos / local_x_neg
          - local_y_pos / local_y_neg
          - local_z_pos / local_z_neg
        """
        offset = torch.zeros((self._num_envs_local, 3), device=self.device, dtype=dtype)

        if abs(length) < 1e-9:
            return offset

        if axis == "local_x_pos":
            offset[:, 0] = length
        elif axis == "local_x_neg":
            offset[:, 0] = -length
        elif axis == "local_y_pos":
            offset[:, 1] = length
        elif axis == "local_y_neg":
            offset[:, 1] = -length
        elif axis == "local_z_pos":
            offset[:, 2] = length
        elif axis == "local_z_neg":
            offset[:, 2] = -length
        else:
            raise ValueError(
                f"[Action] Unsupported tcp_offset_axis='{axis}'. "
                f"Use one of: local_x_pos, local_x_neg, local_y_pos, local_y_neg, local_z_pos, local_z_neg"
            )

        return offset

    def _update_waypoint_progress(self, pos_err_norm: torch.Tensor, rot_err_norm: torch.Tensor):
        reached = (pos_err_norm < self.cfg.waypoint_pos_tol) & (rot_err_norm < self.cfg.waypoint_rot_tol)
        timeout = self.steps_at_waypoint >= self.cfg.max_steps_per_waypoint
        advance = (reached | timeout) & (~self.path_done)

        next_index = self.path_index + advance.long()
        done_now = next_index >= (self.traj_length - 1)

        self.path_index = torch.clamp(next_index, max=self.traj_length - 1)
        self.path_done = self.path_done | done_now

        self.steps_at_waypoint = torch.where(
            advance,
            torch.zeros_like(self.steps_at_waypoint),
            self.steps_at_waypoint + 1,
        )

    def _debug_print_status(self, ee_pos, ee_quat, raw_des_pos, pos_err_norm, rot_err_norm):
        if not self.cfg.enable_debug_print:
            return

        global_step = int(self._env.episode_length_buf[0].item())
        if self.cfg.debug_print_interval > 0 and (global_step % self.cfg.debug_print_interval != 0):
            return

        env_id = min(self.cfg.debug_env_id, self._num_envs_local - 1)

        raw_target_xyz = raw_des_pos[env_id].detach().cpu()
        target_xyz = self.des_pos[env_id].detach().cpu()
        target_rpy = quat_to_rpy(self.des_quat[env_id:env_id + 1]).squeeze(0).detach().cpu()

        current_xyz = ee_pos[env_id].detach().cpu()
        current_rpy = quat_to_rpy(ee_quat[env_id:env_id + 1]).squeeze(0).detach().cpu()

        local_debug.print_action_debug_status(
            env_id=env_id,
            global_step=global_step,
            path_index=int(self.path_index[env_id].item()),
            traj_length=self.traj_length,
            waypoint_steps=int(self.steps_at_waypoint[env_id].item()),
            path_done=bool(self.path_done[env_id].item()),
            raw_target_xyz=raw_target_xyz,
            target_xyz=target_xyz,
            target_rpy=target_rpy,
            current_xyz=current_xyz,
            current_rpy=current_rpy,
            pos_err_norm=float(pos_err_norm[env_id].item()),
            rot_err_norm=float(rot_err_norm[env_id].item()),
        )

    def _solve_dls_ik(self, J: torch.Tensor, e: torch.Tensor, damping: float) -> torch.Tensor:
        n = J.shape[0]
        I = torch.eye(6, device=J.device, dtype=J.dtype).unsqueeze(0).repeat(n, 1, 1)
        JJt = J @ J.transpose(1, 2)
        A = JJt + (damping ** 2) * I
        e_col = e.unsqueeze(-1)
        x = torch.linalg.solve(A, e_col)
        dq = (J.transpose(1, 2) @ x).squeeze(-1)
        return dq


# =========================================================
# Config
# =========================================================
@configclass
class AdmittanceControlActionCfg(ActionTermCfg):
    class_type: type = AdmittanceControlAction

    asset_name: str = "robot"
    body_name: str = "spindle_link"

    hdf5_file_path: str = ""
    position_dataset_key: str = "target_positions"

    action_dim: int = 2

    # IK
    dls_lambda: float = 0.10
    ik_step_size: float = 0.60
    max_dq: float = 0.08

    max_pos_err: float = 0.05
    max_rot_err: float = 0.30

    # waypoint follower
    waypoint_stride: int = 100
    waypoint_pos_tol: float = 0.02
    waypoint_rot_tol: float = 0.20
    max_steps_per_waypoint: int = 120

    # TCP / spindle compensation
    tcp_length_offset_m: float = 0.20
    tcp_offset_axis: str = "local_z_neg"

    # extra trim
    z_target_offset: float = 0.0

    # debug
    enable_debug_print: bool = True
    debug_print_interval: int = 10
    debug_env_id: int = 0

    joint_lower_limits: tuple | None = (
        -2.0 * math.pi,
        -2.0 * math.pi,
        -math.pi,
        -2.0 * math.pi,
        -2.0 * math.pi,
        -2.0 * math.pi,
    )
    joint_upper_limits: tuple | None = (
        2.0 * math.pi,
        2.0 * math.pi,
        math.pi,
        2.0 * math.pi,
        2.0 * math.pi,
        2.0 * math.pi,
    )