# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import os
import h5py
import torch
import importlib

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass

from ..utils import debug as local_debug

local_ft_sensor = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.assets.assets.sensors.six_axis_ft_sensor"
)

from y2_control_py import (
    ForceCon1DMode5,
    CONTEXT_NAF_MDGRADI_CKPT,
    CONTROL_PERIOD as Y2_CONTROL_PERIOD,
)


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
    Multi-env HDF5 pose+force path follower.

    - Reads desired pose trajectory from HDF5 key: "position"  -> (x,y,z,r,p,yaw)
    - Reads desired force trajectory from HDF5 key: "force"    -> (fx,fy,fz)
    - Uses ForceCon1DMode5 from y2_control_py on one Cartesian axis (default: z)
    - Uses batched DLS IK for 6-DoF UR10e
    - RL action input is ignored
    """

    cfg: "AdmittanceControlActionCfg"

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.cfg = cfg
        self.robot = self._env.scene[cfg.asset_name]
        self._num_envs_local = self._env.num_envs
        self._step_dt_local = float(self._env.step_dt)

        body_ids = self.robot.find_bodies(self.cfg.body_name)[0]
        if len(body_ids) == 0:
            raise ValueError(
                f"[Action] body_name='{self.cfg.body_name}' not found. "
                f"Available bodies: {self.robot.body_names}"
            )
        self.ee_idx = body_ids[0]

        self._raw_actions = torch.zeros((self._num_envs_local, self.cfg.action_dim), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        traj_pos_full, traj_force_full = self._load_hdf5_pose_force(
            self.cfg.hdf5_file_path,
            self.cfg.position_dataset_key,
            self.cfg.force_dataset_key,
            self.cfg.position_scale,
        )

        stride = max(1, int(self.cfg.waypoint_stride))
        self.traj_positions = traj_pos_full[::stride].contiguous()  # [T, 6]
        self.traj_forces = traj_force_full[::stride].contiguous()   # [T, 3]
        self.traj_length = self.traj_positions.shape[0]

        if self.traj_forces.shape[0] != self.traj_length:
            raise ValueError(
                f"[Action] position/force trajectory length mismatch: "
                f"{self.traj_positions.shape[0]} vs {self.traj_forces.shape[0]}"
            )

        self.path_index = torch.zeros(self._num_envs_local, dtype=torch.long, device=self.device)
        self.steps_at_waypoint = torch.zeros(self._num_envs_local, dtype=torch.long, device=self.device)
        self.path_done = torch.zeros(self._num_envs_local, dtype=torch.bool, device=self.device)

        self.des_pos = torch.zeros((self._num_envs_local, 3), device=self.device)
        self.des_quat = torch.zeros((self._num_envs_local, 4), device=self.device)
        self.des_quat[:, 0] = 1.0
        self.des_force = torch.zeros((self._num_envs_local, 3), device=self.device)

        # one force controller per env
        self.force_controllers = [
            ForceCon1DMode5(
                model_path=self.cfg.force_model_path,
                dt=float(self.cfg.force_dt),
                threads=int(self.cfg.force_threads),
                device=self.cfg.force_device,
                md_ratio=float(self.cfg.force_md_ratio),
                fc_fext=float(self.cfg.force_fc_fext),
                free_mass=float(self.cfg.force_free_mass),
                free_damping=float(self.cfg.force_free_damping),
                free_stiffness=float(self.cfg.force_free_stiffness),
                contact_stiffness=float(self.cfg.force_contact_stiffness),
                recovery_tau=float(self.cfg.force_recovery_tau),
                action_low=list(self.cfg.force_action_low),
                action_high=list(self.cfg.force_action_high),
                mass_min=float(self.cfg.force_mass_min),
                mass_max=float(self.cfg.force_mass_max),
                alpha_min=float(self.cfg.force_alpha_min),
                alpha_max=float(self.cfg.force_alpha_max),
                alpha_rate_up=float(self.cfg.force_alpha_rate_up),
                alpha_rate_down=float(self.cfg.force_alpha_rate_down),
            )
            for _ in range(self._num_envs_local)
        ]

        local_debug.print_action_init(
            hdf5_file_path=self.cfg.hdf5_file_path,
            position_dataset_key=self.cfg.position_dataset_key,
            traj_shape=tuple(traj_pos_full.shape),
            stride=stride,
            used_traj_shape=tuple(self.traj_positions.shape),
            body_name=self.cfg.body_name,
            ee_idx=self.ee_idx,
            num_envs=self._num_envs_local,
            tcp_length_offset_m=self.cfg.tcp_length_offset_m,
            tcp_offset_axis=self.cfg.tcp_offset_axis,
        )
        print(
            f"[Action] force trajectory loaded: shape={tuple(traj_force_full.shape)}, "
            f"used_shape={tuple(self.traj_forces.shape)}, force_axis={self.cfg.force_axis}"
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

    def _load_hdf5_pose_force(
        self,
        file_path: str,
        position_key: str,
        force_key: str,
        position_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not file_path:
            raise ValueError("[Action] hdf5_file_path is empty.")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[Action] HDF5 file not found: {file_path}")

        with h5py.File(file_path, "r") as f:
            if position_key not in f:
                raise KeyError(f"[Action] position dataset '{position_key}' not found. Available keys: {list(f.keys())}")
            if force_key not in f:
                raise KeyError(f"[Action] force dataset '{force_key}' not found. Available keys: {list(f.keys())}")

            pos_data = f[position_key][:]
            force_data = f[force_key][:]

        pos = torch.tensor(pos_data, dtype=torch.float32, device=self.device)
        force = torch.tensor(force_data, dtype=torch.float32, device=self.device)

        if pos.ndim != 2 or pos.shape[1] < 6:
            raise ValueError(f"[Action] expected position shape [T, 6], got {tuple(pos.shape)}")
        if force.ndim != 2 or force.shape[1] < 3:
            raise ValueError(f"[Action] expected force shape [T, 3], got {tuple(force.shape)}")
        if pos.shape[0] != force.shape[0]:
            raise ValueError(f"[Action] position/force length mismatch: {pos.shape[0]} vs {force.shape[0]}")

        # xyz from txt are mm-scale -> convert to meters for IsaacLab
        pos = pos[:, :6].clone()
        pos[:, :3] = pos[:, :3] * float(position_scale)

        force = force[:, :3].clone()

        return pos, force

    def reset(self, env_ids=None):
        super().reset(env_ids)

        if env_ids is None:
            env_ids = torch.arange(self._num_envs_local, device=self.device)

        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0

        self.path_index[env_ids] = 0
        self.steps_at_waypoint[env_ids] = 0
        self.path_done[env_ids] = False

        des_pos = self.traj_positions[0].unsqueeze(0).repeat(len(env_ids), 1)
        des_force = self.traj_forces[0].unsqueeze(0).repeat(len(env_ids), 1)

        self.des_pos[env_ids] = des_pos[:, 0:3]
        self.des_quat[env_ids] = rpy_to_quat(des_pos[:, 3:6])
        self.des_force[env_ids] = des_force[:, 0:3]

        env_ids_cpu = env_ids.detach().cpu().tolist()
        force_axis_idx = self._force_axis_to_index(self.cfg.force_axis)
        for env_id in env_ids_cpu:
            xd0 = float(self.des_pos[env_id, force_axis_idx].item())
            self.force_controllers[env_id].reset(xd0)

    def process_actions(self, actions: torch.Tensor):
        # RL action ignored
        self._raw_actions = torch.nan_to_num(actions.clone(), nan=0.0)
        self._processed_actions.zero_()

    def apply_actions(self):
        # -------------------------------------------------
        # 1) Current EE pose
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
        # 3) Desired pose / force from trajectory
        # -------------------------------------------------
        des_pose_row = self.traj_positions[self.path_index]   # [N, 6]
        des_force_row = self.traj_forces[self.path_index]     # [N, 3]

        raw_des_pos = des_pose_row[:, 0:3].clone()
        self.des_quat = rpy_to_quat(des_pose_row[:, 3:6])
        self.des_force = des_force_row[:, 0:3].clone()

        # -------------------------------------------------
        # 4) TCP / spindle compensation
        # -------------------------------------------------
        offset_local = self._get_local_tcp_offset(
            length=self.cfg.tcp_length_offset_m,
            axis=self.cfg.tcp_offset_axis,
            dtype=raw_des_pos.dtype,
        )
        rotm = quat_to_rotmat(self.des_quat)
        offset_world_like = torch.bmm(rotm, offset_local.unsqueeze(-1)).squeeze(-1)

        self.des_pos = raw_des_pos + offset_world_like
        self.des_pos[:, 2] += self.cfg.z_target_offset

        # -------------------------------------------------
        # 5) Measured force from sensor
        # -------------------------------------------------
        wrench = local_ft_sensor.get_6axis_ft_fixed_joint(
            env=self._env,
            asset_name=self.cfg.asset_name,
            fixed_joint_name=self.cfg.fixed_joint_name,
            joint_prim_relpath=self.cfg.joint_prim_relpath,
            verbose=False,
        )  # [N, 6]

        force_axis_idx = self._force_axis_to_index(self.cfg.force_axis)

        current_axis_pos = ee_pos[:, force_axis_idx]
        desired_axis_pos = self.des_pos[:, force_axis_idx]
        desired_axis_force = self.des_force[:, force_axis_idx]
        measured_axis_force = wrench[:, force_axis_idx]

        # -------------------------------------------------
        # 6) Per-env Mode5 force control on one axis
        # -------------------------------------------------
        corrected_axis_pos = desired_axis_pos.clone()
        fc_mass = torch.zeros((self._num_envs_local,), device=self.device, dtype=torch.float32)
        fc_alpha = torch.zeros((self._num_envs_local,), device=self.device, dtype=torch.float32)
        fc_damping = torch.zeros((self._num_envs_local,), device=self.device, dtype=torch.float32)
        fc_stiffness = torch.zeros((self._num_envs_local,), device=self.device, dtype=torch.float32)
        fc_filtered_fext = torch.zeros((self._num_envs_local,), device=self.device, dtype=torch.float32)

        for env_id in range(self._num_envs_local):
            out = self.force_controllers[env_id].step(
                xd=float(desired_axis_pos[env_id].item()),
                x=float(current_axis_pos[env_id].item()),
                fd=float(desired_axis_force[env_id].item()),
                fext=float(measured_axis_force[env_id].item()),
            )
            # [xc, mass, alpha, damping, stiffness, filtered_fext]
            corrected_axis_pos[env_id] = float(out[0])
            fc_mass[env_id] = float(out[1])
            fc_alpha[env_id] = float(out[2])
            fc_damping[env_id] = float(out[3])
            fc_stiffness[env_id] = float(out[4])
            fc_filtered_fext[env_id] = float(out[5])

        self.des_pos[:, force_axis_idx] = corrected_axis_pos

        # -------------------------------------------------
        # 7) IK error
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
        # 8) Env-wise waypoint update
        # -------------------------------------------------
        self._update_waypoint_progress(pos_err_norm, rot_err_norm)

        # -------------------------------------------------
        # 9) Debug
        # -------------------------------------------------
        self._debug_print_status(
            ee_pos=ee_pos,
            ee_quat=ee_quat,
            raw_des_pos=raw_des_pos,
            desired_force=self.des_force,
            measured_wrench=wrench,
            corrected_axis_pos=corrected_axis_pos,
            pos_err_norm=pos_err_norm,
            rot_err_norm=rot_err_norm,
            fc_mass=fc_mass,
            fc_alpha=fc_alpha,
            fc_damping=fc_damping,
            fc_stiffness=fc_stiffness,
            fc_filtered_fext=fc_filtered_fext,
        )

    def _force_axis_to_index(self, axis: str) -> int:
        axis = axis.lower()
        if axis == "x":
            return 0
        if axis == "y":
            return 1
        if axis == "z":
            return 2
        raise ValueError(f"[Action] Unsupported force_axis='{axis}'. Use x, y, or z.")

    def _get_local_tcp_offset(self, length: float, axis: str, dtype: torch.dtype) -> torch.Tensor:
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

    def _debug_print_status(
        self,
        ee_pos,
        ee_quat,
        raw_des_pos,
        desired_force,
        measured_wrench,
        corrected_axis_pos,
        pos_err_norm,
        rot_err_norm,
        fc_mass,
        fc_alpha,
        fc_damping,
        fc_stiffness,
        fc_filtered_fext,
    ):
        if not self.cfg.enable_debug_print:
            return

        global_step = int(self._env.episode_length_buf[0].item())
        if self.cfg.debug_print_interval > 0 and (global_step % self.cfg.debug_print_interval != 0):
            return

        env_id = min(self.cfg.debug_env_id, self._num_envs_local - 1)
        axis_idx = self._force_axis_to_index(self.cfg.force_axis)

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

        print(
            f"[ForceCon Debug] env={env_id} | step={global_step} | "
            f"fd_{self.cfg.force_axis}={float(desired_force[env_id, axis_idx].item()):+.4f} | "
            f"fext_{self.cfg.force_axis}={float(measured_wrench[env_id, axis_idx].item()):+.4f} | "
            f"xc_{self.cfg.force_axis}={float(corrected_axis_pos[env_id].item()):+.6f} | "
            f"M={float(fc_mass[env_id].item()):.4f}, "
            f"alpha={float(fc_alpha[env_id].item()):.4f}, "
            f"D={float(fc_damping[env_id].item()):.4f}, "
            f"K={float(fc_stiffness[env_id].item()):.4f}, "
            f"fext_filt={float(fc_filtered_fext[env_id].item()):+.4f}"
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
    position_dataset_key: str = "position"
    force_dataset_key: str = "force"

    # xyz in txt/h5 are mm -> Isaac uses m
    position_scale: float = 0.001

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

    # measured FT source
    fixed_joint_name: str = "tool0_to_spindle"
    joint_prim_relpath: str = "joints"

    # force-control axis
    force_axis: str = "z"

    # Mode5 force controller config
    force_model_path: str = CONTEXT_NAF_MDGRADI_CKPT
    force_dt: float = float(Y2_CONTROL_PERIOD)
    force_threads: int = 1
    force_device: str = "cpu"
    force_md_ratio: float = 1000.0
    force_fc_fext: float = 50.0

    force_free_mass: float = 2.0
    force_free_damping: float = 6000.0
    force_free_stiffness: float = 2000.0
    force_contact_stiffness: float = 0.0
    force_recovery_tau: float = 3.0

    force_action_low: tuple = (-0.25, -0.25)
    force_action_high: tuple = (0.25, 0.25)

    force_mass_min: float = 0.5
    force_mass_max: float = 5.0
    force_alpha_min: float = 0.5
    force_alpha_max: float = 3.0
    force_alpha_rate_up: float = 4.0
    force_alpha_rate_down: float = 4.0

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