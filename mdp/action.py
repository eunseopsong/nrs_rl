# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""
================================================================================
Force-aware Variable Path-Speed Action
================================================================================

[Goal]
- Keep behavior identical to the current baseline.
- Reduce runtime overhead / memory pressure where possible.
- Keep:
    * original ForceCon parameters
    * variable index-speed scheduler
    * debug capability

[Optimizations in this version]
- Pre-create joint limit tensors once in __init__
- Avoid repeated tensor allocations where possible
- Wrap reset/process/apply with torch.no_grad()
- Do not build expensive debug payloads unless debug print is actually needed
- Remove q_now/q_cmd debug payload generation path

Units:
- position: mm
- orientation: rad
- force: N
================================================================================
"""

import math
import os
import h5py
import torch
import importlib

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass

from ..utils import debug as local_debug

y2_cfg = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.y2_control_pybind.y2_control_py.config"
)
y2_pb = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.y2_control_pybind.y2_control_py._y2_control_pybind"
)
local_ft_sensor = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.assets.assets.sensors.six_axis_ft_sensor"
)


def normalize_quat(q: torch.Tensor) -> torch.Tensor:
    return q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1e-8)


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


_HOME_Q = torch.tensor(
    [0.5585, -2.0949, -1.5711, -1.0472, 1.5708, 0.5585],
    dtype=torch.float32,
)


@configclass
class OriginalControllerForceConCfg:
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


@configclass
class ActionIntegrationCfg:
    body_name: str = "spindle_link"
    fixed_joint_name: str = "tool0_to_spindle"
    joint_prim_relpath: str = "joints"

    hdf5_file_path: str = ""
    position_dataset_key: str = "position"
    force_dataset_key: str = "force"

    action_dim: int = 1

    target_mrr_n_mm_s: float = 2500.0
    speed_action_scale: float = 0.35
    base_index_rate: float = 48.0
    min_index_rate: float = 1.0
    max_index_rate: float = 96.0
    progress_rate_ema_beta: float = 0.55
    force_eps_n: float = 1.0
    force_tracking_ready_ratio: float = 0.8
    min_force_rate_scale: float = 0.25
    force_error_slowdown_ratio: float = 0.35
    min_force_error_rate_scale: float = 0.35
    force_normal_push_sign: float = -1.0
    force_normal_kp_mm_per_n: float = 0.26
    force_normal_ki_mm_per_n_s: float = 4.00
    force_normal_release_kp_mm_per_n: float = 0.35
    force_normal_max_step_mm: float = 2.00
    force_normal_retract_max_step_mm: float = 6.00
    force_normal_offset_limit_mm: float = 28.0
    force_admittance_delta_limit_mm: float = 1.5
    force_total_normal_delta_limit_mm: float = 30.0
    force_normal_deadband_n: float = 0.35
    force_band_min_n: float = 8.0
    force_band_max_n: float = 12.0
    force_band_index_rate_limit: float = 0.05
    force_band_saturated_min_n: float = 7.5
    force_steady_error_band_n: float = 5.0
    force_overload_ratio: float = 1.5
    force_overload_rate_scale: float = 0.02
    path_tracking_slowdown_start_mm: float = 2.0
    path_tracking_stop_mm: float = 8.0
    path_tracking_min_rate_scale: float = 0.0
    path_projection_window: int = 160
    path_projection_max_advance_index: float = 0.0
    path_lookahead_min_index: float = 0.0
    path_lookahead_max_index: float = 8.0
    path_lookahead_time_s: float = 0.015
    path_command_max_xy_step_mm: float = 0.0
    path_command_max_z_step_mm: float = 4.0
    approach_interpolation_enabled: bool = True
    approach_duration_s: float = 2.0

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


@configclass
class AdmittanceControlActionCfg(ActionTermCfg):
    class_type: type | None = None
    asset_name: str = "robot"

    original_forcecon: OriginalControllerForceConCfg = OriginalControllerForceConCfg()
    integration: ActionIntegrationCfg = ActionIntegrationCfg()


class AdmittanceControlAction(ActionTerm):
    cfg: AdmittanceControlActionCfg

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.cfg = cfg
        self.fc_cfg = cfg.original_forcecon
        self.int_cfg = cfg.integration

        self.robot = self._env.scene[cfg.asset_name]
        self._num_envs_local = self._env.num_envs
        self._step_dt_local = float(self._env.step_dt)
        self._control_period = float(getattr(y2_cfg, "CONTROL_PERIOD", self._step_dt_local))

        body_ids = self.robot.find_bodies(self.int_cfg.body_name)[0]
        if len(body_ids) == 0:
            raise ValueError(f"[Action] body_name='{self.int_cfg.body_name}' not found.")
        self.ee_idx = int(body_ids[0])

        self._raw_actions = torch.zeros((self._num_envs_local, self.int_cfg.action_dim), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        traj_full = self._load_hdf5_positions(self.int_cfg.hdf5_file_path, self.int_cfg.position_dataset_key)
        force_full = self._load_hdf5_forces(self.int_cfg.hdf5_file_path, self.int_cfg.force_dataset_key, traj_full.shape[0])

        self.traj_positions = traj_full.contiguous()
        self.traj_forces = force_full.contiguous()
        self.traj_length = self.traj_positions.shape[0]
        segment_lengths = torch.linalg.norm(
            self.traj_positions[1:, 0:3] - self.traj_positions[:-1, 0:3],
            dim=-1,
        )
        self.traj_segment_lengths_mm = torch.empty((self.traj_length,), dtype=torch.float32, device=self.device)
        self.traj_segment_lengths_mm[:-1] = segment_lengths
        self.traj_segment_lengths_mm[-1] = segment_lengths[-1] if segment_lengths.numel() > 0 else 1.0
        self.traj_segment_lengths_mm = torch.clamp(self.traj_segment_lengths_mm, min=1.0e-6)

        self.path_cursor = torch.zeros((self._num_envs_local,), dtype=torch.float32, device=self.device)
        self.path_index = torch.zeros((self._num_envs_local,), dtype=torch.long, device=self.device)
        self.current_target_index = torch.zeros((self._num_envs_local,), dtype=torch.long, device=self.device)
        self.path_done = torch.zeros((self._num_envs_local,), dtype=torch.bool, device=self.device)
        self.current_index_delta = torch.zeros((self._num_envs_local,), dtype=torch.float32, device=self.device)
        self.current_sliding_velocity_mm_s = torch.zeros((self._num_envs_local,), dtype=torch.float32, device=self.device)
        self.current_abs_fz = torch.zeros((self._num_envs_local,), dtype=torch.float32, device=self.device)
        self.current_mrr_n_mm_s = torch.zeros((self._num_envs_local,), dtype=torch.float32, device=self.device)
        self.cumulative_removal = torch.zeros((self._num_envs_local,), dtype=torch.float32, device=self.device)
        self.current_path_tracking_error_mm = torch.zeros((self._num_envs_local,), dtype=torch.float32, device=self.device)
        self.force_normal_offset_mm = torch.zeros((self._num_envs_local,), dtype=torch.float32, device=self.device)
        self.force_normal_error_i = torch.zeros((self._num_envs_local,), dtype=torch.float32, device=self.device)
        self.approach_active = torch.zeros((self._num_envs_local,), dtype=torch.bool, device=self.device)
        self.approach_step = torch.zeros((self._num_envs_local,), dtype=torch.long, device=self.device)
        self.approach_start_pos_mm = torch.zeros((self._num_envs_local, 3), dtype=torch.float32, device=self.device)
        self.approach_start_wxyz = torch.zeros((self._num_envs_local, 3), dtype=torch.float32, device=self.device)
        self.approach_total_steps = max(
            1,
            int(round(float(self.int_cfg.approach_duration_s) / max(self._step_dt_local, 1.0e-8))),
        )

        self.progress_rate_filtered = torch.full(
            (self._num_envs_local,),
            float(self.int_cfg.base_index_rate),
            dtype=torch.float32,
            device=self.device,
        )

        self.des_pos_mm_raw = torch.zeros((self._num_envs_local, 3), device=self.device)
        self.des_wxyz_raw = torch.zeros((self._num_envs_local, 3), device=self.device)
        self.des_force = torch.zeros((self._num_envs_local, 3), device=self.device)
        self.cmd_target_xyz_mm = torch.zeros((self._num_envs_local, 3), dtype=torch.float32, device=self.device)

        self.prev_q_cmd_6 = torch.zeros((self._num_envs_local, 6), device=self.device)
        self.prev_valid = torch.zeros((self._num_envs_local,), dtype=torch.bool, device=self.device)

        if self.int_cfg.joint_lower_limits is not None:
            self._q_min = torch.tensor(self.int_cfg.joint_lower_limits, device=self.device, dtype=torch.float32)
        else:
            self._q_min = None
        if self.int_cfg.joint_upper_limits is not None:
            self._q_max = torch.tensor(self.int_cfg.joint_upper_limits, device=self.device, dtype=torch.float32)
        else:
            self._q_max = None

        self.kin = y2_pb.UR10eKinematics(
            dt=self._control_period,
            ee2tcp=getattr(y2_cfg, "EE2TCP", [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]),
        )

        self.R_offset = self._get_orientation_offset_rotm(self.device)

        model_path = getattr(y2_cfg, "CONTEXT_NAF_MDGRADI_CKPT", None) or getattr(y2_cfg, "FORCECON_MODEL_PATH", None)
        if not model_path:
            raise RuntimeError("ForceCon model path not found in y2_control_pybind config.")

        self.force_controllers = []
        for _ in range(self._num_envs_local):
            self.force_controllers.append(
                y2_pb.ForceCon1DMode5(
                    model_path,
                    self._step_dt_local,
                    1,
                    "cpu",
                    float(self.fc_cfg.force_md_ratio),
                    float(self.fc_cfg.force_fc_fext),
                    float(self.fc_cfg.force_free_mass),
                    float(self.fc_cfg.force_free_damping),
                    float(self.fc_cfg.force_free_stiffness),
                    float(self.fc_cfg.force_contact_stiffness),
                    float(self.fc_cfg.force_recovery_tau),
                    list(self.fc_cfg.force_action_low),
                    list(self.fc_cfg.force_action_high),
                    float(self.fc_cfg.force_mass_min),
                    float(self.fc_cfg.force_mass_max),
                    float(self.fc_cfg.force_alpha_min),
                    float(self.fc_cfg.force_alpha_max),
                    float(self.fc_cfg.force_alpha_rate_up),
                    float(self.fc_cfg.force_alpha_rate_down),
                )
            )

        local_debug.print_action_init(
            hdf5_file_path=self.int_cfg.hdf5_file_path,
            position_dataset_key=self.int_cfg.position_dataset_key,
            traj_shape=tuple(traj_full.shape),
            body_name=self.int_cfg.body_name,
            ee_idx=self.ee_idx,
            num_envs=self._num_envs_local,
        )

    @property
    def action_dim(self):
        return self.int_cfg.action_dim

    @property
    def raw_actions(self):
        return self._raw_actions

    @property
    def processed_actions(self):
        return self._processed_actions

    def _load_hdf5_positions(self, file_path: str, dataset_key: str) -> torch.Tensor:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[Action] HDF5 file not found: {file_path}")
        with h5py.File(file_path, "r") as f:
            data = f[dataset_key][:] if dataset_key in f else f[list(f.keys())[0]][:]
        data = torch.tensor(data, dtype=torch.float32, device=self.device)
        return data[:, :6]

    def _load_hdf5_forces(self, file_path: str, dataset_key: str, expected_rows: int) -> torch.Tensor:
        with h5py.File(file_path, "r") as f:
            data = f[dataset_key][:] if dataset_key in f else torch.zeros((expected_rows, 3), dtype=torch.float32).cpu().numpy()
        data = torch.tensor(data, dtype=torch.float32, device=self.device)
        return data[:, :3]

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

    def _solve_pybind_single_step_ik(self, q_seed: torch.Tensor, target_pos_mm: torch.Tensor, target_rotm: torch.Tensor):
        T = self.kin.forward_kinematics(q_seed.detach().cpu().to(torch.float64).tolist())
        T = torch.tensor(T, dtype=torch.float32, device=self.device)

        pos_cur_mm = T[:3, 3]
        R_fk = T[:3, :3]
        R_cur = self.R_offset @ R_fk

        pos_err_mm = target_pos_mm - pos_cur_mm
        rot_err_rad = rotmat_to_spatial((target_rotm @ R_cur.T).unsqueeze(0)).squeeze(0)

        pos_err_norm_mm = torch.linalg.norm(pos_err_mm)
        rot_err_norm_rad = torch.linalg.norm(rot_err_rad)

        err_6 = torch.cat([pos_err_mm, rot_err_rad], dim=0)

        J = self.kin.calculate_jacobian(q_seed.detach().cpu().to(torch.float64).tolist())
        J = torch.tensor(J, dtype=torch.float32, device=self.device)

        dq = torch.linalg.lstsq(J, err_6.unsqueeze(-1)).solution.squeeze(-1)
        dq_norm = float(torch.linalg.norm(dq).item())

        q_next = q_seed + dq

        if self._q_min is not None and self._q_max is not None:
            q_next = torch.clamp(q_next, self._q_min, self._q_max)

        return q_next, pos_err_norm_mm, rot_err_norm_rad, dq_norm

    def _reset_force_controller_for_env(self, env_id: int, xd_mm: float):
        self.force_controllers[env_id].reset(float(xd_mm) / 1000.0)

    def _compute_progress_rate(self, env_id: int, idx: int, abs_fz: float) -> float:
        denom = max(abs_fz, self.int_cfg.force_eps_n)
        target_velocity_mm_s = float(self.int_cfg.target_mrr_n_mm_s) / denom
        segment_length_mm = float(self.traj_segment_lengths_mm[idx].item())
        raw_rate = target_velocity_mm_s * self._step_dt_local / max(segment_length_mm, 1.0e-6)

        raw_rate = max(self.int_cfg.min_index_rate, min(self.int_cfg.max_index_rate, raw_rate))

        target_abs_fz = abs(float(self.traj_forces[idx, 2].item()))
        if target_abs_fz > self.int_cfg.force_eps_n:
            steady_band = float(getattr(self.int_cfg, "force_steady_error_band_n", 5.0))
            if abs_fz > target_abs_fz + steady_band:
                error_ratio = (abs_fz - target_abs_fz) / max(target_abs_fz, self.int_cfg.force_eps_n)
                error_scale = max(
                    self.int_cfg.min_force_error_rate_scale,
                    1.0 - (error_ratio - self.int_cfg.force_error_slowdown_ratio),
                )
                raw_rate *= error_scale
                raw_rate *= 0.35

        beta = self.int_cfg.progress_rate_ema_beta
        prev = float(self.progress_rate_filtered[env_id].item())
        filt = beta * raw_rate + (1.0 - beta) * prev
        filt = max(self.int_cfg.min_index_rate, min(self.int_cfg.max_index_rate, filt))

        self.progress_rate_filtered[env_id] = filt
        return filt

    def _smoothstep(self, x: float) -> float:
        x = max(0.0, min(1.0, x))
        return x * x * (3.0 - 2.0 * x)

    def _path_tracking_rate_scale(self, tracking_error_mm: float) -> float:
        start = float(self.int_cfg.path_tracking_slowdown_start_mm)
        stop = float(self.int_cfg.path_tracking_stop_mm)
        min_scale = float(self.int_cfg.path_tracking_min_rate_scale)
        if stop <= start:
            return 1.0
        if tracking_error_mm <= start:
            return 1.0
        if tracking_error_mm >= stop:
            return min_scale
        ratio = (tracking_error_mm - start) / (stop - start)
        return 1.0 - ratio * (1.0 - min_scale)

    def _project_cursor_to_path(self, env_id: int, current_pos_mm: torch.Tensor) -> float:
        window = max(8, int(self.int_cfg.path_projection_window))
        cursor = float(self.path_cursor[env_id].item())
        center = int(max(0, min(self.traj_length - 1, round(cursor))))
        start = max(0, center - window // 3)
        end = min(self.traj_length, center + window)
        if end <= start:
            return float(center)

        xy = self.traj_positions[start:end, 0:2]
        delta = xy - current_pos_mm[0:2].unsqueeze(0)
        dist2 = torch.sum(delta * delta, dim=1)
        local_idx = int(torch.argmin(dist2).item())
        projected = float(start + local_idx)
        max_advance = max(0.0, float(self.int_cfg.path_projection_max_advance_index))
        projected = min(projected, cursor + max_advance)
        return max(cursor - float(window), projected)

    def _compute_lookahead_indices(self, base_rate: float) -> float:
        lookahead = base_rate * float(self.int_cfg.path_lookahead_time_s) / max(self._step_dt_local, 1.0e-8)
        return max(
            float(self.int_cfg.path_lookahead_min_index),
            min(float(self.int_cfg.path_lookahead_max_index), lookahead),
        )

    def _limit_command_step(self, current_pos_mm: torch.Tensor, target_pos_mm: torch.Tensor) -> torch.Tensor:
        limited = target_pos_mm.clone()

        xy_delta = target_pos_mm[0:2] - current_pos_mm[0:2]
        xy_norm = float(torch.linalg.norm(xy_delta).item())
        max_xy = float(self.int_cfg.path_command_max_xy_step_mm)
        if xy_norm > max_xy > 0.0:
            limited[0:2] = current_pos_mm[0:2] + xy_delta * (max_xy / max(xy_norm, 1.0e-6))

        z_delta = float((target_pos_mm[2] - current_pos_mm[2]).item())
        max_z = float(self.int_cfg.path_command_max_z_step_mm)
        if max_z > 0.0:
            z_delta = max(-max_z, min(max_z, z_delta))
            limited[2] = current_pos_mm[2] + z_delta

        return limited

    def _update_force_normal_offset(
        self,
        env_id: int,
        target_abs_fz: float,
        measured_fz: float,
    ) -> float:
        if target_abs_fz <= self.int_cfg.force_eps_n:
            self.force_normal_offset_mm[env_id] = 0.0
            self.force_normal_error_i[env_id] = 0.0
            return 0.0

        abs_fz = abs(measured_fz)
        error_n = abs_fz - target_abs_fz
        if abs(error_n) < float(self.int_cfg.force_normal_deadband_n):
            error_n = 0.0

        push_sign = 1.0 if float(self.int_cfg.force_normal_push_sign) >= 0.0 else -1.0
        retract_sign = -push_sign

        prev_offset = float(self.force_normal_offset_mm[env_id].item())

        if error_n > 0.0:
            self.force_normal_error_i[env_id] = torch.clamp(self.force_normal_error_i[env_id], max=0.0)
            self.force_normal_error_i[env_id] -= float(error_n) * self._step_dt_local
            i_limit = float(self.int_cfg.force_normal_offset_limit_mm) / max(
                float(self.int_cfg.force_normal_ki_mm_per_n_s),
                1.0e-6,
            )
            self.force_normal_error_i[env_id] = torch.clamp(
                self.force_normal_error_i[env_id],
                -i_limit,
                i_limit,
            )
            target_offset = retract_sign * (
                float(self.int_cfg.force_normal_kp_mm_per_n) * error_n
            )
        else:
            underforce_n = max(0.0, target_abs_fz - abs_fz - float(self.int_cfg.force_normal_deadband_n))
            self.force_normal_error_i[env_id] += float(underforce_n) * self._step_dt_local
            i_limit = float(self.int_cfg.force_normal_offset_limit_mm) / max(
                float(self.int_cfg.force_normal_ki_mm_per_n_s),
                1.0e-6,
            )
            self.force_normal_error_i[env_id] = torch.clamp(
                self.force_normal_error_i[env_id],
                -i_limit,
                i_limit,
            )
            target_offset = push_sign * (
                float(self.int_cfg.force_normal_release_kp_mm_per_n) * underforce_n
                + float(self.int_cfg.force_normal_ki_mm_per_n_s) * float(self.force_normal_error_i[env_id].item())
            )

        limit = float(self.int_cfg.force_normal_offset_limit_mm)
        target_offset = max(-limit, min(limit, target_offset))

        if error_n > 0.0:
            max_step = float(getattr(self.int_cfg, "force_normal_retract_max_step_mm", self.int_cfg.force_normal_max_step_mm))
        else:
            max_step = float(self.int_cfg.force_normal_max_step_mm)
        delta = max(-max_step, min(max_step, target_offset - prev_offset))
        next_offset = prev_offset + delta
        self.force_normal_offset_mm[env_id] = next_offset
        return next_offset

    @torch.no_grad()
    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self._num_envs_local, device=self.device)

        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0

        self.path_cursor[env_ids] = 0.0
        self.path_index[env_ids] = 0
        self.current_target_index[env_ids] = 0
        self.path_done[env_ids] = False
        self.current_index_delta[env_ids] = 0.0
        self.current_sliding_velocity_mm_s[env_ids] = 0.0
        self.current_abs_fz[env_ids] = 0.0
        self.current_mrr_n_mm_s[env_ids] = 0.0
        self.cumulative_removal[env_ids] = 0.0
        self.current_path_tracking_error_mm[env_ids] = 0.0
        self.force_normal_offset_mm[env_ids] = 0.0
        self.force_normal_error_i[env_ids] = 0.0
        self.progress_rate_filtered[env_ids] = float(self.int_cfg.base_index_rate)
        self.approach_step[env_ids] = 0
        self.approach_active[env_ids] = bool(self.int_cfg.approach_interpolation_enabled)

        des = self.traj_positions[0].unsqueeze(0).repeat(len(env_ids), 1)
        frc = self.traj_forces[0].unsqueeze(0).repeat(len(env_ids), 1)

        self.des_pos_mm_raw[env_ids] = des[:, 0:3]
        self.des_wxyz_raw[env_ids] = des[:, 3:6]
        self.des_force[env_ids] = frc
        self.cmd_target_xyz_mm[env_ids] = des[:, 0:3]

        self.prev_q_cmd_6[env_ids] = 0.0
        self.prev_valid[env_ids] = False

        q_all = self.robot.data.joint_pos
        q = q_all[:, :6]
        for env_id in env_ids.tolist():
            pos_mm, _, wxyz, _ = self._fk_pose_pybind_corrected(q[env_id])
            self.approach_start_pos_mm[env_id] = pos_mm
            self.approach_start_wxyz[env_id] = wxyz

        xd0 = float(self.traj_positions[0, 2].item())
        for env_id in env_ids.tolist():
            self._reset_force_controller_for_env(env_id, xd0)

    @torch.no_grad()
    def process_actions(self, actions: torch.Tensor):
        self._raw_actions = torch.nan_to_num(actions.clone(), nan=0.0)
        self._processed_actions[:] = torch.clamp(self._raw_actions, -1.0, 1.0)

    @torch.no_grad()
    def apply_actions(self):
        q_all = self.robot.data.joint_pos
        q = q_all[:, :6]

        wrench6 = local_ft_sensor.get_6axis_ft_fixed_joint(
            env=self._env,
            asset_name=self.cfg.asset_name,
            fixed_joint_name=self.int_cfg.fixed_joint_name,
            joint_prim_relpath=self.int_cfg.joint_prim_relpath,
            verbose=False,
        )

        q_cmd_all = q_all.clone()
        pos_err_norm_mm = torch.zeros((self._num_envs_local,), device=self.device)
        rot_err_norm_rad = torch.zeros((self._num_envs_local,), device=self.device)

        debug_needed = False
        debug_env_id = 0
        global_step = 0
        if self.int_cfg.enable_debug_print:
            global_step = int(self._env.episode_length_buf[0].item())
            if self.int_cfg.debug_print_interval <= 0 or global_step % self.int_cfg.debug_print_interval == 0:
                debug_needed = True
                debug_env_id = min(self.int_cfg.debug_env_id, self._num_envs_local - 1)

        pybind_called = False
        pybind_success = False
        pybind_dq_norm = 0.0

        for env_id in range(self._num_envs_local):
            q_seed = self.prev_q_cmd_6[env_id] if self.prev_valid[env_id] else q[env_id]

            if bool(self.approach_active[env_id].item()):
                step = int(self.approach_step[env_id].item())
                alpha = self._smoothstep(float(step + 1) / float(self.approach_total_steps))

                target_pos_mm = self.approach_start_pos_mm[env_id] + alpha * (
                    self.traj_positions[0, 0:3] - self.approach_start_pos_mm[env_id]
                )
                target_wxyz = self.approach_start_wxyz[env_id] + alpha * (
                    self.traj_positions[0, 3:6] - self.approach_start_wxyz[env_id]
                )
                target_rotm = spatial_to_rotmat(target_wxyz.view(1, 3)).squeeze(0)

                self.path_index[env_id] = 0
                self.current_target_index[env_id] = 0
                self.des_pos_mm_raw[env_id] = target_pos_mm
                self.des_wxyz_raw[env_id] = target_wxyz
                self.des_force[env_id] = self.traj_forces[0]
                self.cmd_target_xyz_mm[env_id] = target_pos_mm
                self.current_index_delta[env_id] = 0.0
                self.current_sliding_velocity_mm_s[env_id] = 0.0
                self.current_abs_fz[env_id] = abs(float(wrench6[env_id, 2].item()))
                self.current_mrr_n_mm_s[env_id] = 0.0
                self.current_path_tracking_error_mm[env_id] = float(torch.linalg.norm(target_pos_mm[0:2] - self.approach_start_pos_mm[env_id, 0:2]).item())
                self.force_normal_offset_mm[env_id] = 0.0
                self.force_normal_error_i[env_id] = 0.0

                pybind_called = True
                q_cmd6, pos_e_mm, rot_e_rad, dq_norm = self._solve_pybind_single_step_ik(
                    q_seed, target_pos_mm, target_rotm
                )
                pybind_success = True
                pybind_dq_norm = dq_norm

                pos_err_norm_mm[env_id] = pos_e_mm
                rot_err_norm_rad[env_id] = rot_e_rad

                self.prev_q_cmd_6[env_id] = q_cmd6
                self.prev_valid[env_id] = True
                q_cmd_all[env_id, :6] = q_cmd6

                self.approach_step[env_id] += 1
                if int(self.approach_step[env_id].item()) >= self.approach_total_steps:
                    self.approach_active[env_id] = False
                    self.path_cursor[env_id] = 0.0
                    self.progress_rate_filtered[env_id] = float(self.int_cfg.base_index_rate)
                    self._reset_force_controller_for_env(env_id, float(self.traj_positions[0, 2].item()))
                continue

            cur_pos_mm, _, _, cur_rotm = self._fk_pose_pybind_corrected(q[env_id])

            projected_cursor = self._project_cursor_to_path(env_id, cur_pos_mm)
            self.path_cursor[env_id] = min(max(float(self.path_cursor[env_id].item()), projected_cursor), float(self.traj_length - 1))
            idx_for_rate = int(min(max(0, round(float(self.path_cursor[env_id].item()))), self.traj_length - 1))
            base_rate_for_lookahead = self._compute_progress_rate(env_id, idx_for_rate, float(self.current_abs_fz[env_id].item()))
            lookahead_idx = self._compute_lookahead_indices(base_rate_for_lookahead)
            target_cursor = min(float(self.path_cursor[env_id].item()) + lookahead_idx, float(self.traj_length - 1))

            idx = int(target_cursor)
            if idx >= self.traj_length:
                idx = self.traj_length - 1
                self.path_done[env_id] = True

            self.path_index[env_id] = int(self.path_cursor[env_id].item())
            self.current_target_index[env_id] = idx

            measured_force_base = wrench6[env_id, 0:3]
            target_fz = float(self.traj_forces[idx, 2].item())

            self.des_pos_mm_raw[env_id] = self.traj_positions[idx, 0:3]
            self.des_wxyz_raw[env_id] = self.traj_positions[idx, 3:6]
            self.des_force[env_id] = self.traj_forces[idx]

            nominal_pos_mm = self.traj_positions[idx, 0:3].clone()
            target_rotm = spatial_to_rotmat(self.traj_positions[idx, 3:6].view(1, 3)).squeeze(0)

            target_pos_mm = nominal_pos_mm.clone()
            normal_axis = cur_rotm[:, 2]
            measured_fz = float(torch.dot(normal_axis, measured_force_base).item())
            abs_fz = abs(measured_fz)
            path_tracking_error_mm = float(torch.linalg.norm(cur_pos_mm[0:2] - nominal_pos_mm[0:2]).item())
            self.current_path_tracking_error_mm[env_id] = path_tracking_error_mm
            nominal_tcp_z_m = float(torch.dot(normal_axis, nominal_pos_mm).item()) / 1000.0
            current_tcp_z_m = float(torch.dot(normal_axis, cur_pos_mm).item()) / 1000.0

            fc_out = self.force_controllers[env_id].step(
                nominal_tcp_z_m,
                current_tcp_z_m,
                float(target_fz),
                float(measured_fz),
            )
            normal_delta_mm = (float(fc_out[0]) - nominal_tcp_z_m) * 1000.0
            admittance_limit = float(self.int_cfg.force_admittance_delta_limit_mm)
            normal_delta_mm = max(-admittance_limit, min(admittance_limit, normal_delta_mm))
            normal_delta_mm += self._update_force_normal_offset(env_id, abs(target_fz), measured_fz)
            total_limit = float(self.int_cfg.force_total_normal_delta_limit_mm)
            normal_delta_mm = max(-total_limit, min(total_limit, normal_delta_mm))

            target_pos_mm = nominal_pos_mm + normal_axis * normal_delta_mm
            target_pos_mm = self._limit_command_step(cur_pos_mm, target_pos_mm)
            self.cmd_target_xyz_mm[env_id] = target_pos_mm

            pybind_called = True
            q_cmd6, pos_e_mm, rot_e_rad, dq_norm = self._solve_pybind_single_step_ik(
                q_seed, target_pos_mm, target_rotm
            )
            pybind_success = True
            pybind_dq_norm = dq_norm

            pos_err_norm_mm[env_id] = pos_e_mm
            rot_err_norm_rad[env_id] = rot_e_rad

            self.prev_q_cmd_6[env_id] = q_cmd6
            self.prev_valid[env_id] = True
            q_cmd_all[env_id, :6] = q_cmd6

            if not bool(self.path_done[env_id].item()):
                base_rate = self._compute_progress_rate(env_id, idx_for_rate, abs_fz)
                action_scale = 1.0 + float(self.int_cfg.speed_action_scale) * float(self._processed_actions[env_id, 0].item())
                action_scale = max(0.05, action_scale)
                final_rate = base_rate * action_scale
                final_rate = max(self.int_cfg.min_index_rate, min(self.int_cfg.max_index_rate, final_rate))
                target_abs_fz = abs(target_fz)
                force_rate_limit = None
                if target_abs_fz > self.int_cfg.force_eps_n:
                    overload_force = float(self.int_cfg.force_overload_ratio) * target_abs_fz
                    if abs_fz > overload_force:
                        force_rate_limit = float(self.int_cfg.force_overload_rate_scale)
                    band_min = float(getattr(self.int_cfg, "force_band_min_n", 0.0))
                    band_max = float(getattr(self.int_cfg, "force_band_max_n", 0.0))
                    if band_min > 0.0 and band_max > band_min and (abs_fz < band_min or abs_fz > band_max):
                        saturated_min = float(getattr(self.int_cfg, "force_band_saturated_min_n", band_min))
                        mild_underforce = abs_fz < band_min and abs_fz >= saturated_min
                        if not mild_underforce:
                            band_limit = float(getattr(self.int_cfg, "force_band_index_rate_limit", 0.05))
                            force_rate_limit = band_limit if force_rate_limit is None else min(force_rate_limit, band_limit)
                tracking_scale = self._path_tracking_rate_scale(path_tracking_error_mm)
                final_rate *= tracking_scale
                if tracking_scale <= 0.0:
                    final_rate = 0.0
                else:
                    final_rate = max(self.int_cfg.min_index_rate, min(self.int_cfg.max_index_rate, final_rate))
                if force_rate_limit is not None:
                    final_rate = min(final_rate, max(0.0, force_rate_limit))

                segment_length_mm = float(self.traj_segment_lengths_mm[idx].item())
                sliding_velocity_mm_s = final_rate * segment_length_mm / max(self._step_dt_local, 1.0e-8)

                self.current_index_delta[env_id] = final_rate
                self.current_sliding_velocity_mm_s[env_id] = sliding_velocity_mm_s
                self.current_abs_fz[env_id] = abs_fz
                self.current_mrr_n_mm_s[env_id] = abs_fz * sliding_velocity_mm_s
                self.cumulative_removal[env_id] += self.current_mrr_n_mm_s[env_id] * self._step_dt_local
                self.path_cursor[env_id] = min(float(self.path_cursor[env_id].item()) + final_rate, target_cursor)

                if self.path_cursor[env_id] >= float(self.traj_length - 1):
                    self.path_cursor[env_id] = float(self.traj_length - 1)
                    self.path_done[env_id] = True

        self.robot.set_joint_position_target(q_cmd_all)

        if debug_needed:
            current_index = int(self.path_index[debug_env_id].item())
            target_index = int(self.current_target_index[debug_env_id].item())
            progress_pct = 100.0 * float(self.path_cursor[debug_env_id].item()) / max(float(self.traj_length - 1), 1.0)
            episode_number = int(getattr(self._env, "_ep_curriculum", 0)) + 1
            cur_dbg_xyz_mm, _, cur_dbg_wxyz, _ = self._fk_pose_pybind_corrected(q[debug_env_id])
            tgt_dbg_xyz_mm = self.des_pos_mm_raw[debug_env_id]
            tgt_dbg_wxyz = self.des_wxyz_raw[debug_env_id]
            cmd_dbg_xyz_mm = self.cmd_target_xyz_mm[debug_env_id]
            local_debug.print_info(
                f"\n[Polishing Live] ep{episode_number} step={global_step} env={debug_env_id} "
                f"| hdf5_index={current_index}/{self.traj_length - 1} ({progress_pct:.1f}%) "
                f"| target_index={target_index} "
                f"| cursor={float(self.path_cursor[debug_env_id].item()):.3f}\n"
                f"  current xyz/wxyz = "
                f"({float(cur_dbg_xyz_mm[0].item()):.3f}, {float(cur_dbg_xyz_mm[1].item()):.3f}, {float(cur_dbg_xyz_mm[2].item()):.3f}) / "
                f"({float(cur_dbg_wxyz[0].item()):.4f}, {float(cur_dbg_wxyz[1].item()):.4f}, {float(cur_dbg_wxyz[2].item()):.4f})\n"
                f"  target  xyz/wxyz = "
                f"({float(tgt_dbg_xyz_mm[0].item()):.3f}, {float(tgt_dbg_xyz_mm[1].item()):.3f}, {float(tgt_dbg_xyz_mm[2].item()):.3f}) / "
                f"({float(tgt_dbg_wxyz[0].item()):.4f}, {float(tgt_dbg_wxyz[1].item()):.4f}, {float(tgt_dbg_wxyz[2].item()):.4f})\n"
                f"  command xyz      = "
                f"({float(cmd_dbg_xyz_mm[0].item()):.3f}, {float(cmd_dbg_xyz_mm[1].item()):.3f}, {float(cmd_dbg_xyz_mm[2].item()):.3f})\n"
                f"  force/speed      = "
                f"| target_force_N={abs(float(self.des_force[debug_env_id, 2].item())):.4f} "
                f"| normal_force_N={float(self.current_abs_fz[debug_env_id].item()):.4f} "
                f"| sliding_velocity_mm_s={float(self.current_sliding_velocity_mm_s[debug_env_id].item()):.4f} "
                f"| removal_rate_N_mm_s={float(self.current_mrr_n_mm_s[debug_env_id].item()):.4f} "
                f"| cumulative_removal={float(self.cumulative_removal[debug_env_id].item()):.4f}\n"
                f"  control          = "
                f"| fn_offset_mm={float(self.force_normal_offset_mm[debug_env_id].item()):.4f} "
                f"| action={float(self._processed_actions[debug_env_id, 0].item()):.4f} "
                f"| index_rate={float(self.current_index_delta[debug_env_id].item()):.4f} "
                f"| path_err_xy_mm={float(self.current_path_tracking_error_mm[debug_env_id].item()):.3f}\n"
            )


AdmittanceControlActionCfg.class_type = AdmittanceControlAction
