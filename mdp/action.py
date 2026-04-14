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


# =========================================================
# pybind import
# =========================================================
y2_cfg = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.y2_control_pybind.y2_control_py.config"
)
y2_pb = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.y2_control_pybind.y2_control_py._y2_control_pybind"
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


def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
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


# =========================================================
# Fixed orientation calibration
# =========================================================
_HOME_Q = torch.tensor(
    [0.5585, -2.0949, -1.5711, -1.0472, 1.5708, 0.5585],
    dtype=torch.float32,
)


# =========================================================
# Action Term
# =========================================================
class AdmittanceControlAction(ActionTerm):
    """
    Multi-env HDF5 pose path follower using pybind FK/Jacobian inner-loop IK.

    Unit convention:
    - position    : mm
    - orientation : rad
    - force       : N

    Important behavior:
    - one apply_actions() call -> one HDF5 index increment
    - if callback/control rate is 125 Hz, trajectory is consumed at 125 index/sec
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
        self.traj_positions = traj_full[::stride].contiguous()
        self.traj_forces = force_full[::stride].contiguous()
        self.traj_length = self.traj_positions.shape[0]

        # next index to consume
        self.path_index = torch.zeros(self._num_envs_local, dtype=torch.long, device=self.device)
        # index actually used in the current/last command
        self.current_target_index = torch.zeros(self._num_envs_local, dtype=torch.long, device=self.device)

        self.steps_at_waypoint = torch.zeros(self._num_envs_local, dtype=torch.long, device=self.device)
        self.path_done = torch.zeros(self._num_envs_local, dtype=torch.bool, device=self.device)

        self.des_pos_mm_raw = torch.zeros((self._num_envs_local, 3), device=self.device)
        self.des_wxyz_raw = torch.zeros((self._num_envs_local, 3), device=self.device)
        self.des_force = torch.zeros((self._num_envs_local, 3), device=self.device)

        self.prev_q_cmd_6 = torch.zeros((self._num_envs_local, 6), device=self.device)
        self.prev_valid = torch.zeros((self._num_envs_local,), dtype=torch.bool, device=self.device)

        self.kin = y2_pb.UR10eKinematics(
            dt=float(getattr(y2_cfg, "CONTROL_PERIOD", self._step_dt_local)),
            ee2tcp=getattr(y2_cfg, "EE2TCP", [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]),
        )

        self.R_offset = self._get_orientation_offset_rotm(self.device)

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

    def _get_orientation_offset_rotm(self, device: torch.device) -> torch.Tensor:
        T_home = self.kin.forward_kinematics(_HOME_Q.detach().cpu().tolist())
        T_home = torch.tensor(T_home, dtype=torch.float32, device=device)

        R_home_fk = T_home[:3, :3]
        desired_home_spatial = torch.tensor([[0.0, 0.0, 1.5708]], dtype=torch.float32, device=device)
        R_home_desired = spatial_to_rotmat(desired_home_spatial).squeeze(0)

        return R_home_desired @ R_home_fk.T

    def _fk_pose_pybind_corrected(self, q6: torch.Tensor):
        T = self.kin.forward_kinematics(q6.detach().cpu().to(torch.float64).tolist())
        T = torch.tensor(T, dtype=torch.float32, device=self.device)

        pos_mm = T[:3, 3]
        R_fk = T[:3, :3]
        R_corr = self.R_offset @ R_fk

        quat_corr = rotmat_to_quat(R_corr.unsqueeze(0)).squeeze(0)
        wxyz_corr = rotmat_to_spatial(R_corr.unsqueeze(0)).squeeze(0)

        return pos_mm, quat_corr, wxyz_corr, R_corr

    def _solve_pybind_iterative_ik(
        self,
        q_seed: torch.Tensor,
        target_pos_mm: torch.Tensor,
        target_rotm: torch.Tensor,
        inner_iters: int = 5,
    ):
        q_iter = q_seed.clone()

        last_pos_err_norm_mm = torch.tensor(0.0, device=self.device)
        last_rot_err_norm_rad = torch.tensor(0.0, device=self.device)
        last_dq_norm = 0.0

        q_min = None
        q_max = None
        if self.cfg.joint_lower_limits is not None and self.cfg.joint_upper_limits is not None:
            q_min = torch.tensor(self.cfg.joint_lower_limits, device=self.device, dtype=torch.float32)
            q_max = torch.tensor(self.cfg.joint_upper_limits, device=self.device, dtype=torch.float32)

        for _ in range(inner_iters):
            T = self.kin.forward_kinematics(q_iter.detach().cpu().to(torch.float64).tolist())
            T = torch.tensor(T, dtype=torch.float32, device=self.device)

            pos_cur_mm = T[:3, 3]
            R_fk = T[:3, :3]
            R_cur = self.R_offset @ R_fk

            pos_err_mm = target_pos_mm - pos_cur_mm
            rot_err_rad = rotmat_to_spatial((target_rotm @ R_cur.T).unsqueeze(0)).squeeze(0)

            last_pos_err_norm_mm = torch.linalg.norm(pos_err_mm)
            last_rot_err_norm_rad = torch.linalg.norm(rot_err_rad)

            pos_err_mm = torch.clamp(
                pos_err_mm,
                -self.cfg.max_pos_err * 1000.0,
                self.cfg.max_pos_err * 1000.0,
            )
            rot_err_rad = torch.clamp(
                rot_err_rad,
                -self.cfg.max_rot_err,
                self.cfg.max_rot_err,
            )

            err_6 = torch.cat([pos_err_mm, rot_err_rad], dim=0)

            J = self.kin.calculate_jacobian(q_iter.detach().cpu().to(torch.float64).tolist())
            J = torch.tensor(J, dtype=torch.float32, device=self.device)

            I = torch.eye(6, device=self.device, dtype=torch.float32)
            dq = J.T @ torch.linalg.solve(J @ J.T + (self.cfg.dls_lambda ** 2) * I, err_6.unsqueeze(-1))
            dq = dq.squeeze(-1)

            dq = torch.clamp(dq, -self.cfg.max_dq, self.cfg.max_dq)
            last_dq_norm = float(torch.linalg.norm(dq).item())

            q_iter = q_iter + self.cfg.ik_step_size * dq

            if q_min is not None and q_max is not None:
                q_iter = torch.clamp(q_iter, q_min, q_max)

        return q_iter, last_pos_err_norm_mm, last_rot_err_norm_rad, last_dq_norm

    def reset(self, env_ids=None):
        super().reset(env_ids)

        if env_ids is None:
            env_ids = torch.arange(self._num_envs_local, device=self.device)

        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0

        self.path_index[env_ids] = 0
        self.current_target_index[env_ids] = 0
        self.steps_at_waypoint[env_ids] = 0
        self.path_done[env_ids] = False

        des = self.traj_positions[0].unsqueeze(0).repeat(len(env_ids), 1)
        frc = self.traj_forces[0].unsqueeze(0).repeat(len(env_ids), 1)

        self.des_pos_mm_raw[env_ids] = des[:, 0:3]
        self.des_wxyz_raw[env_ids] = des[:, 3:6]
        self.des_force[env_ids] = frc

        self.prev_q_cmd_6[env_ids] = 0.0
        self.prev_valid[env_ids] = False

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions = torch.nan_to_num(actions.clone(), nan=0.0)
        self._processed_actions.zero_()

    def apply_actions(self):
        q_all = self.robot.data.joint_pos
        q = q_all[:, :6]

        # consume exactly one index per apply_actions call
        cmd_index = self.path_index.clone()
        self.current_target_index = cmd_index.clone()

        des = self.traj_positions[cmd_index]
        frc = self.traj_forces[cmd_index]

        self.des_pos_mm_raw = des[:, 0:3].clone()
        self.des_wxyz_raw = des[:, 3:6].clone()
        self.des_force = frc

        q_cmd_all = q_all.clone()
        pos_err_norm_mm = torch.zeros((self._num_envs_local,), device=self.device)
        rot_err_norm_rad = torch.zeros((self._num_envs_local,), device=self.device)

        pybind_called = False
        pybind_success = False
        pybind_dq_norm = 0.0

        for env_id in range(self._num_envs_local):
            q_seed = self.prev_q_cmd_6[env_id] if self.prev_valid[env_id] else q[env_id]

            target_pos_mm = self.des_pos_mm_raw[env_id].clone()
            target_pos_mm[2] += self.cfg.z_target_offset_m * 1000.0

            target_rotm = spatial_to_rotmat(self.des_wxyz_raw[env_id : env_id + 1]).squeeze(0)

            pybind_called = True
            q_cmd6, pos_e_mm, rot_e_rad, dq_norm = self._solve_pybind_iterative_ik(
                q_seed=q_seed,
                target_pos_mm=target_pos_mm,
                target_rotm=target_rotm,
                inner_iters=5,
            )
            pybind_success = True
            pybind_dq_norm = dq_norm

            pos_err_norm_mm[env_id] = pos_e_mm
            rot_err_norm_rad[env_id] = rot_e_rad

            self.prev_q_cmd_6[env_id] = q_cmd6
            self.prev_valid[env_id] = True

            q_cmd_all[env_id, :6] = q_cmd6

        self.robot.set_joint_position_target(q_cmd_all)

        # no waypoint reaching logic:
        # exactly +1 index per callback until the last row
        can_advance = ~self.path_done
        next_index = torch.where(
            can_advance,
            torch.clamp(self.path_index + 1, max=self.traj_length - 1),
            self.path_index,
        )
        done_now = self.path_index >= (self.traj_length - 1)

        self.path_done = self.path_done | done_now
        self.path_index = next_index
        self.steps_at_waypoint.zero_()

        if self.cfg.enable_debug_print:
            global_step = int(self._env.episode_length_buf[0].item())
            if self.cfg.debug_print_interval <= 0 or global_step % self.cfg.debug_print_interval == 0:
                env_id = min(self.cfg.debug_env_id, self._num_envs_local - 1)

                cur_pos_mm, _, cur_wxyz, _ = self._fk_pose_pybind_corrected(q[env_id])

                print("=" * 100)
                print(
                    f"[Pybind IK   ] called={pybind_called} success={pybind_success} "
                    f"inner_iters=5 dq_norm={pybind_dq_norm:.6f}"
                )
                print(
                    f"[Action Debug ] env={env_id} | step={global_step} "
                    f"| h5_index={int(self.current_target_index[env_id].item())}/{self.traj_length} "
                    f"| next_index={int(self.path_index[env_id].item())}/{self.traj_length} "
                    f"| done={bool(self.path_done[env_id].item())} "
                    f"| pos_err_norm={float(pos_err_norm_mm[env_id].item()):.6f} "
                    f"| rot_err_norm={float(rot_err_norm_rad[env_id].item()):.6f}"
                )
                print(
                    f"[Current Pose ] x={float(cur_pos_mm[0]): .6f}, "
                    f"y={float(cur_pos_mm[1]): .6f}, "
                    f"z={float(cur_pos_mm[2]): .6f}, "
                    f"wx={float(cur_wxyz[0]): .6f}, "
                    f"wy={float(cur_wxyz[1]): .6f}, "
                    f"wz={float(cur_wxyz[2]): .6f}"
                )
                print(
                    f"[Target Pose  ] x={float(self.des_pos_mm_raw[env_id, 0]): .6f}, "
                    f"y={float(self.des_pos_mm_raw[env_id, 1]): .6f}, "
                    f"z={float(self.des_pos_mm_raw[env_id, 2] + self.cfg.z_target_offset_m * 1000.0): .6f}, "
                    f"wx={float(self.des_wxyz_raw[env_id, 0]): .6f}, "
                    f"wy={float(self.des_wxyz_raw[env_id, 1]): .6f}, "
                    f"wz={float(self.des_wxyz_raw[env_id, 2]): .6f}"
                )
                print(
                    f"[Target Force ] Fx={float(self.des_force[env_id, 0]): .6f}, "
                    f"Fy={float(self.des_force[env_id, 1]): .6f}, "
                    f"Fz={float(self.des_force[env_id, 2]): .6f}"
                )
                print(
                    f"[Joint Cmd    ] q_now={q[env_id].detach().cpu().numpy()} | "
                    f"q_cmd={q_cmd_all[env_id, :6].detach().cpu().numpy()}"
                )
                print("=" * 100)


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

    dls_lambda: float = 0.10
    ik_step_size: float = 0.60
    max_dq: float = 0.08

    max_pos_err: float = 0.05
    max_rot_err: float = 0.30

    waypoint_stride: int = 1
    waypoint_pos_tol: float = 0.02
    waypoint_rot_tol: float = 0.20
    max_steps_per_waypoint: int = 120

    tcp_length_offset_m: float = 0.20
    tcp_offset_axis: str = "local_z_neg"

    z_target_offset_m: float = 0.0

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