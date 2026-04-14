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


def quat_conjugate_single(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


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


def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """
    R: [N, 3, 3]
    return q: [N, 4] = [w, x, y, z]
    """
    n = R.shape[0]
    q = torch.zeros((n, 4), device=R.device, dtype=R.dtype)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    mask = trace > 0.0
    if torch.any(mask):
        s = torch.sqrt(trace[mask] + 1.0) * 2.0
        q[mask, 0] = 0.25 * s
        q[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / s
        q[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / s
        q[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / s

    mask1 = (~mask) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if torch.any(mask1):
        s = torch.sqrt(1.0 + R[mask1, 0, 0] - R[mask1, 1, 1] - R[mask1, 2, 2]) * 2.0
        q[mask1, 0] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s
        q[mask1, 1] = 0.25 * s
        q[mask1, 2] = (R[mask1, 0, 1] + R[mask1, 1, 0]) / s
        q[mask1, 3] = (R[mask1, 0, 2] + R[mask1, 2, 0]) / s

    mask2 = (~mask) & (~mask1) & (R[:, 1, 1] > R[:, 2, 2])
    if torch.any(mask2):
        s = torch.sqrt(1.0 + R[mask2, 1, 1] - R[mask2, 0, 0] - R[mask2, 2, 2]) * 2.0
        q[mask2, 0] = (R[mask2, 0, 2] - R[mask2, 2, 0]) / s
        q[mask2, 1] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s
        q[mask2, 2] = 0.25 * s
        q[mask2, 3] = (R[mask2, 1, 2] + R[mask2, 2, 1]) / s

    mask3 = (~mask) & (~mask1) & (~mask2)
    if torch.any(mask3):
        s = torch.sqrt(1.0 + R[mask3, 2, 2] - R[mask3, 0, 0] - R[mask3, 1, 1]) * 2.0
        q[mask3, 0] = (R[mask3, 1, 0] - R[mask3, 0, 1]) / s
        q[mask3, 1] = (R[mask3, 0, 2] + R[mask3, 2, 0]) / s
        q[mask3, 2] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s
        q[mask3, 3] = 0.25 * s

    return normalize_quat(q)


def spatial_to_rotmat(spatial: torch.Tensor) -> torch.Tensor:
    """
    spatial: (N, 3) = [wx wy wz]  (rotation vector / axis-angle vector)
    return : (N, 3, 3)
    """
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


def spatial_to_quat(spatial: torch.Tensor) -> torch.Tensor:
    return rotmat_to_quat(spatial_to_rotmat(spatial))


def rotmat_to_spatial(R: torch.Tensor) -> torch.Tensor:
    """
    R: (N, 3, 3)
    return: (N, 3) = [wx wy wz]
    """
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    angle = torch.acos(cos_angle)

    eps = 1e-6
    pi = torch.tensor(math.pi, device=R.device, dtype=R.dtype)

    out = torch.zeros((R.shape[0], 3), device=R.device, dtype=R.dtype)

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


def orientation_error_world(cur_quat: torch.Tensor, des_quat: torch.Tensor) -> torch.Tensor:
    cur_quat = normalize_quat(cur_quat)
    des_quat = normalize_quat(des_quat)

    q_err = quat_multiply(des_quat, quat_conjugate(cur_quat))
    q_err = normalize_quat(q_err)

    sign = torch.where(q_err[:, 0:1] < 0.0, -1.0, 1.0)
    q_err = q_err * sign
    return 2.0 * q_err[:, 1:4]


# =========================================================
# Fixed orientation calibration
# user frame(home) = [0, 0, 1.5708]
# =========================================================
_HOME_Q = torch.tensor(
    [0.5585, -2.0949, -1.5711, -1.0472, 1.5708, 0.5585],
    dtype=torch.float32,
)


def _get_user_to_world_quat(device: torch.device) -> torch.Tensor:
    """
    Build constant quaternion q_uw such that:
      R_world = R(q_uw) * R_user
    where user-home is defined as [0, 0, 1.5708].
    """
    q_home_user = spatial_to_quat(
        torch.tensor([[0.0, 0.0, 1.5708]], dtype=torch.float32, device=device)
    )

    # world-home from Isaac raw EE quaternion convention
    # this is treated as the raw home orientation of spindle_link
    # using the same robot model convention as the simulator.
    # NOTE: action control uses only this constant transform.
    # It does not alter positions.
    #
    # Since the user requested the same convention as observation/home,
    # we use this fixed raw-home quaternion derived from the home pose
    # through observation/FK-consistent convention:
    #
    # Here we approximate using the known raw-home orientation observed
    # from the simulator-side home pose:
    # [w, x, y, z] equivalent to current home quaternion in raw world frame.
    #
    # Instead of hardcoding a numerical quaternion, the transform is kept as:
    # q_uw = q_world_home * conj(q_user_home)
    #
    # The raw world-home is obtained from the actual simulator at runtime
    # using env body_quat_w on the first apply.
    raise RuntimeError("q_user_to_world is initialized inside AdmittanceControlAction at runtime.")


# =========================================================
# Action Term
# =========================================================
class AdmittanceControlAction(ActionTerm):
    """
    Multi-env HDF5 pose path follower.

    Control:
    - current EE orientation is used in raw Isaac/world convention
    - target wx wy wz from HDF5 is converted from USER convention to WORLD convention
    - this keeps control stable while debug/output can still use USER convention

    HDF5:
    - position = [x, y, z, wx, wy, wz] with xyz in mm
    - force    = [fx, fy, fz]
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
        self.ee_idx = int(body_ids[0])

        self._raw_actions = torch.zeros((self._num_envs_local, self.cfg.action_dim), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        traj_full = self._load_hdf5_positions(
            self.cfg.hdf5_file_path,
            self.cfg.position_dataset_key,
        )
        force_full = self._load_hdf5_forces(
            self.cfg.hdf5_file_path,
            self.cfg.force_dataset_key,
            traj_full.shape[0],
        )

        stride = max(1, int(self.cfg.waypoint_stride))
        self.traj_positions = traj_full[::stride].contiguous()   # [T, 6], xyz in mm
        self.traj_forces = force_full[::stride].contiguous()     # [T, 3]
        self.traj_length = self.traj_positions.shape[0]

        self.path_index = torch.zeros(self._num_envs_local, dtype=torch.long, device=self.device)
        self.steps_at_waypoint = torch.zeros(self._num_envs_local, dtype=torch.long, device=self.device)
        self.path_done = torch.zeros(self._num_envs_local, dtype=torch.bool, device=self.device)

        # internal desired target for IK: meters + WORLD quaternion
        self.des_pos = torch.zeros((self._num_envs_local, 3), device=self.device)   # m
        self.des_quat_world = torch.zeros((self._num_envs_local, 4), device=self.device)
        self.des_quat_world[:, 0] = 1.0

        # raw target cache for debug
        self.des_pos_mm_raw = torch.zeros((self._num_envs_local, 3), device=self.device)
        self.des_wxyz_raw = torch.zeros((self._num_envs_local, 3), device=self.device)
        self.des_force = torch.zeros((self._num_envs_local, 3), device=self.device)

        # USER <-> WORLD fixed transform
        # q_user_to_world satisfies:
        #   q_world = q_user_to_world * q_user
        self.q_user_to_world = None
        self.q_world_to_user = None
        self._orientation_tf_ready = False

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


    def _load_hdf5_forces(self, file_path: str, dataset_key: str, expected_rows: int) -> torch.Tensor:
        if not file_path:
            raise ValueError("[Action] hdf5_file_path is empty.")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[Action] HDF5 file not found: {file_path}")

        with h5py.File(file_path, "r") as f:
            if dataset_key in f:
                data = f[dataset_key][:]
            else:
                data = torch.zeros((expected_rows, 3), dtype=torch.float32).cpu().numpy()

        data = torch.tensor(data, dtype=torch.float32, device=self.device)

        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError(f"[Action] expected force dataset [T, >=3], got {tuple(data.shape)}")

        data = data[:, :3]

        if data.shape[0] != expected_rows:
            raise ValueError(
                f"[Action] position/force row mismatch: {expected_rows} vs {data.shape[0]}"
            )

        return data

    def _init_orientation_transform_if_needed(self, ee_quat_world_raw: torch.Tensor):
        """
        Build USER -> WORLD fixed orientation transform from actual simulator home orientation.

        At home, the USER convention should be [0, 0, 1.5708].
        So:
            q_user_to_world = q_world_home * conj(q_user_home)
        """
        if self._orientation_tf_ready:
            return

        q_world_home = normalize_quat(ee_quat_world_raw[0:1].clone())
        q_user_home = spatial_to_quat(
            torch.tensor([[0.0, 0.0, 1.5708]], dtype=torch.float32, device=self.device)
        )

        self.q_user_to_world = normalize_quat(quat_multiply(q_world_home, quat_conjugate(q_user_home)))
        self.q_world_to_user = normalize_quat(quat_conjugate(self.q_user_to_world))
        self._orientation_tf_ready = True

    def _user_quat_to_world_quat(self, q_user: torch.Tensor) -> torch.Tensor:
        return normalize_quat(quat_multiply(self.q_user_to_world.repeat(q_user.shape[0], 1), q_user))

    def _world_quat_to_user_quat(self, q_world: torch.Tensor) -> torch.Tensor:
        return normalize_quat(quat_multiply(self.q_world_to_user.repeat(q_world.shape[0], 1), q_world))

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
        frc = self.traj_forces[0].unsqueeze(0).repeat(len(env_ids), 1)

        self.des_pos_mm_raw[env_ids] = des[:, 0:3]
        self.des_wxyz_raw[env_ids] = des[:, 3:6]
        self.des_force[env_ids] = frc

        self.des_pos[env_ids] = des[:, 0:3] * 0.001

        # des_quat_world is initialized later once world<->user transform is known
        self.des_quat_world[env_ids] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device
        )

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions = torch.nan_to_num(actions.clone(), nan=0.0)
        self._processed_actions.zero_()

    def apply_actions(self):
        # -------------------------------------------------
        # 1) Current EE pose (world/raw)
        # -------------------------------------------------
        ee_pos_w = self.robot.data.body_pos_w[:, self.ee_idx, :]
        env_origins = self._env.scene.env_origins
        ee_pos = ee_pos_w - env_origins       # m

        ee_quat_world = normalize_quat(self.robot.data.body_quat_w[:, self.ee_idx, :])

        # build fixed USER<->WORLD transform once
        self._init_orientation_transform_if_needed(ee_quat_world)

        # -------------------------------------------------
        # 2) Current joints / Jacobian
        # -------------------------------------------------
        q_all = self.robot.data.joint_pos
        q = q_all[:, :6]

        jac_all = self.robot.root_physx_view.get_jacobians()
        jacobian = jac_all[:, self.ee_idx - 1, :, :6]

        # -------------------------------------------------
        # 3) Raw target from HDF5
        #    raw_des_pos_mm : mm
        #    raw_des_wxyz   : USER rotation vector
        # -------------------------------------------------
        des = self.traj_positions[self.path_index]
        frc = self.traj_forces[self.path_index]

        raw_des_pos_mm = des[:, 0:3].clone()
        raw_des_wxyz = des[:, 3:6].clone()

        self.des_pos_mm_raw = raw_des_pos_mm
        self.des_wxyz_raw = raw_des_wxyz
        self.des_force = frc

        # USER target quaternion from dataset
        des_quat_user = spatial_to_quat(raw_des_wxyz)

        # WORLD quaternion for actual control
        self.des_quat_world = self._user_quat_to_world_quat(des_quat_user)

        # -------------------------------------------------
        # 4) TCP / spindle length compensation
        #    Keep original logic, but in meters internally
        # -------------------------------------------------
        raw_des_pos_m = raw_des_pos_mm * 0.001

        offset_local = self._get_local_tcp_offset(
            length=self.cfg.tcp_length_offset_m,
            axis=self.cfg.tcp_offset_axis,
            dtype=raw_des_pos_m.dtype,
        )
        rotm_world = quat_to_rotmat(self.des_quat_world)
        offset_world_like = torch.bmm(rotm_world, offset_local.unsqueeze(-1)).squeeze(-1)

        self.des_pos = raw_des_pos_m + offset_world_like
        self.des_pos[:, 2] += self.cfg.z_target_offset_m

        # -------------------------------------------------
        # 5) IK error in WORLD convention
        # -------------------------------------------------
        pos_err = self.des_pos - ee_pos
        rot_err = orientation_error_world(ee_quat_world, self.des_quat_world)

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
        # 7) Debug
        #    Show USER convention so home becomes [0,0,1.5708]
        # -------------------------------------------------
        self._debug_print_status(
            ee_pos=ee_pos,
            ee_quat_world=ee_quat_world,
            pos_err_norm=pos_err_norm,
            rot_err_norm=rot_err_norm,
        )

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

    def _debug_print_status(self, ee_pos, ee_quat_world, pos_err_norm, rot_err_norm):
        if not self.cfg.enable_debug_print:
            return

        global_step = int(self._env.episode_length_buf[0].item())
        if self.cfg.debug_print_interval > 0 and (global_step % self.cfg.debug_print_interval != 0):
            return

        env_id = min(self.cfg.debug_env_id, self._num_envs_local - 1)

        raw_target_xyz = self.des_pos_mm_raw[env_id].detach().cpu()
        target_xyz = (self.des_pos[env_id] * 1000.0).detach().cpu()
        target_wxyz = self.des_wxyz_raw[env_id].detach().cpu()

        current_xyz = (ee_pos[env_id] * 1000.0).detach().cpu()

        current_quat_user = self._world_quat_to_user_quat(ee_quat_world[env_id:env_id + 1])
        current_wxyz = rotmat_to_spatial(quat_to_rotmat(current_quat_user)).squeeze(0).detach().cpu()

        local_debug.print_action_debug_status(
            env_id=env_id,
            global_step=global_step,
            path_index=int(self.path_index[env_id].item()),
            traj_length=self.traj_length,
            waypoint_steps=int(self.steps_at_waypoint[env_id].item()),
            path_done=bool(self.path_done[env_id].item()),
            raw_target_xyz=raw_target_xyz,
            raw_target_force=self.des_force[env_id].detach().cpu(),
            target_xyz=target_xyz,
            target_wxyz=target_wxyz,
            target_force=self.des_force[env_id].detach().cpu(),
            current_xyz=current_xyz,
            current_wxyz=current_wxyz,
            pos_err_norm=float((pos_err_norm[env_id] * 1000.0).item()),   # mm display
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
    position_dataset_key: str = "position"
    force_dataset_key: str = "force"

    action_dim: int = 2

    # IK
    dls_lambda: float = 0.10
    ik_step_size: float = 0.60
    max_dq: float = 0.08

    max_pos_err: float = 0.05      # m
    max_rot_err: float = 0.30

    # waypoint follower
    waypoint_stride: int = 100
    waypoint_pos_tol: float = 0.02     # m
    waypoint_rot_tol: float = 0.20
    max_steps_per_waypoint: int = 120

    # TCP / spindle compensation
    tcp_length_offset_m: float = 0.20
    tcp_offset_axis: str = "local_z_neg"

    # extra trim
    z_target_offset_m: float = 0.0

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