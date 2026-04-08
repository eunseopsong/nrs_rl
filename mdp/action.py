# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import h5py
import importlib
import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

local_obs = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.mdp.observation"
)
local_debug = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.utils.debug"
)
local_ft_sensor = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.assets.assets.sensors.six_axis_ft_sensor"
)

y2_cfg = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.y2_control_pybind.y2_control_py.config"
)
y2_pb = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.y2_control_pybind.y2_control_py._y2_control_pybind"
)


def _spatial_angle_to_rotmat_torch(spatial: torch.Tensor) -> torch.Tensor:
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


def _build_target_htm_torch(pos_mm: torch.Tensor, spatial: torch.Tensor) -> torch.Tensor:
    n = pos_mm.shape[0]
    device = pos_mm.device
    dtype = pos_mm.dtype

    T = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(n, 1, 1)
    T[:, :3, :3] = _spatial_angle_to_rotmat_torch(spatial)
    T[:, :3, 3] = pos_mm
    return T


class _PerEnvForceControllers:
    def __init__(self, cfg):
        self.fx = y2_pb.ForceCon1DMode5(
            model_path=cfg.force_model_path,
            dt=cfg.force_dt,
            threads=cfg.force_threads,
            device=cfg.force_device,
            md_ratio=cfg.force_md_ratio,
            fc_fext=cfg.force_fc_fext,
            free_mass=cfg.force_free_mass,
            free_damping=cfg.force_free_damping,
            free_stiffness=cfg.force_free_stiffness,
            contact_stiffness=cfg.force_contact_stiffness,
            recovery_tau=cfg.force_recovery_tau,
            action_low=list(cfg.force_action_low),
            action_high=list(cfg.force_action_high),
            mass_min=cfg.force_mass_min,
            mass_max=cfg.force_mass_max,
            alpha_min=cfg.force_alpha_min,
            alpha_max=cfg.force_alpha_max,
            alpha_rate_up=cfg.force_alpha_rate_up,
            alpha_rate_down=cfg.force_alpha_rate_down,
        )
        self.fy = y2_pb.ForceCon1DMode5(
            model_path=cfg.force_model_path,
            dt=cfg.force_dt,
            threads=cfg.force_threads,
            device=cfg.force_device,
            md_ratio=cfg.force_md_ratio,
            fc_fext=cfg.force_fc_fext,
            free_mass=cfg.force_free_mass,
            free_damping=cfg.force_free_damping,
            free_stiffness=cfg.force_free_stiffness,
            contact_stiffness=cfg.force_contact_stiffness,
            recovery_tau=cfg.force_recovery_tau,
            action_low=list(cfg.force_action_low),
            action_high=list(cfg.force_action_high),
            mass_min=cfg.force_mass_min,
            mass_max=cfg.force_mass_max,
            alpha_min=cfg.force_alpha_min,
            alpha_max=cfg.force_alpha_max,
            alpha_rate_up=cfg.force_alpha_rate_up,
            alpha_rate_down=cfg.force_alpha_rate_down,
        )
        self.fz = y2_pb.ForceCon1DMode5(
            model_path=cfg.force_model_path,
            dt=cfg.force_dt,
            threads=cfg.force_threads,
            device=cfg.force_device,
            md_ratio=cfg.force_md_ratio,
            fc_fext=cfg.force_fc_fext,
            free_mass=cfg.force_free_mass,
            free_damping=cfg.force_free_damping,
            free_stiffness=cfg.force_free_stiffness,
            contact_stiffness=cfg.force_contact_stiffness,
            recovery_tau=cfg.force_recovery_tau,
            action_low=list(cfg.force_action_low),
            action_high=list(cfg.force_action_high),
            mass_min=cfg.force_mass_min,
            mass_max=cfg.force_mass_max,
            alpha_min=cfg.force_alpha_min,
            alpha_max=cfg.force_alpha_max,
            alpha_rate_up=cfg.force_alpha_rate_up,
            alpha_rate_down=cfg.force_alpha_rate_down,
        )

    def reset(self, x_m: torch.Tensor):
        self.fx.reset(float(x_m[0].item()))
        self.fy.reset(float(x_m[1].item()))
        self.fz.reset(float(x_m[2].item()))


class AdmittanceControlAction(ActionTerm):
    cfg: "AdmittanceControlActionCfg"

    def __init__(self, cfg: "AdmittanceControlActionCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)

        self._device = env.device
        self._num_envs = env.num_envs
        self._robot = env.scene[cfg.asset_name]

        self._raw_actions = torch.zeros((self._num_envs, cfg.action_dim), device=self._device, dtype=torch.float32)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        with h5py.File(cfg.hdf5_file_path, "r") as f:
            pos = f[cfg.position_dataset_key][:]
            frc = f[cfg.force_dataset_key][:]

        self._traj_pos = torch.tensor(pos, device=self._device, dtype=torch.float32)
        self._traj_force = torch.tensor(frc, device=self._device, dtype=torch.float32)
        self._traj_len = int(pos.shape[0])

        self._ik = y2_pb.UR10eKinematics(
            dt=float(y2_cfg.CONTROL_PERIOD),
            ee2tcp=y2_cfg.EE2TCP,
        )

        self._q_des = self._robot.data.joint_pos[:, :6].clone()

        # 핵심 추가
        self._connected = torch.zeros((self._num_envs,), device=self._device, dtype=torch.bool)
        self._traj_row = torch.zeros((self._num_envs,), device=self._device, dtype=torch.long)

        self._fc = [_PerEnvForceControllers(cfg) for _ in range(self._num_envs)]

        self.path_done = torch.zeros((self._num_envs,), device=self._device, dtype=torch.bool)

    @property
    def action_dim(self) -> int:
        return int(self.cfg.action_dim)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def reset(self, env_ids=None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)

        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._q_des[env_ids] = self._robot.data.joint_pos[env_ids, :6].clone()

        self._connected[env_ids] = False
        self._traj_row[env_ids] = 0
        self.path_done[env_ids] = False

        ee_pose = local_obs.get_ee_pose(self._env, asset_name=self.cfg.asset_name)
        current_xyz_m = ee_pose[:, :3] / 1000.0
        for i in env_ids.tolist():
            self._fc[i].reset(current_xyz_m[i])

        return {}

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions.to(self._device)
        self._processed_actions[:] = self._raw_actions

    def _solve_ik_batch(self, target_pose: torch.Tensor):
        target_htm = _build_target_htm_torch(target_pose[:, 0:3], target_pose[:, 3:6])

        q_des_list = []
        for i in range(self._num_envs):
            q_seed = self._q_des[i].detach().cpu().tolist()
            T_i = target_htm[i].detach().cpu().tolist()
            q_next = self._ik.solve_ik(q_seed, T_i)
            q_des_list.append(q_next)

        self._q_des = torch.tensor(q_des_list, device=self._device, dtype=torch.float32)
        self._robot.set_joint_position_target(self._q_des)

    def apply_actions(self):
        ee_pose = local_obs.get_ee_pose(self._env, asset_name=self.cfg.asset_name)
        wrench = local_ft_sensor.get_6axis_ft_fixed_joint(
            env=self._env,
            asset_name=self.cfg.asset_name,
            fixed_joint_name=self.cfg.fixed_joint_name,
            joint_prim_relpath=self.cfg.joint_prim_relpath,
            verbose=False,
        )

        current_xyz_mm = ee_pose[:, 0:3]
        current_wxyz = ee_pose[:, 3:6]
        current_xyz_m = current_xyz_mm / 1000.0
        current_force = wrench[:, 0:3]

        # --------------------------------------------------
        # CONNECT phase target = first row only
        # --------------------------------------------------
        first_pose = self._traj_pos[0].unsqueeze(0).repeat(self._num_envs, 1)
        pos_err_connect = torch.norm(first_pose[:, 0:3] - current_xyz_mm, dim=-1)
        rot_err_connect = torch.norm(first_pose[:, 3:6] - current_wxyz, dim=-1)

        newly_connected = (~self._connected) & \
                          (pos_err_connect < self.cfg.connect_pos_tol_mm) & \
                          (rot_err_connect < self.cfg.connect_rot_tol_rad)

        if torch.any(newly_connected):
            self._connected[newly_connected] = True
            for i in torch.where(newly_connected)[0].tolist():
                self._fc[i].reset(current_xyz_m[i])

        # --------------------------------------------------
        # target pose init
        # --------------------------------------------------
        target_pose = torch.zeros((self._num_envs, 6), device=self._device, dtype=torch.float32)
        target_force = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)

        # CONNECT: pure position control to first row
        not_connected = ~self._connected
        if torch.any(not_connected):
            target_pose[not_connected] = first_pose[not_connected]
            target_force[not_connected] = 0.0

        # TRACK: row = traj_row
        if torch.any(self._connected):
            idx = torch.clamp(self._traj_row, 0, self._traj_len - 1)
            target_pose[self._connected] = self._traj_pos[idx[self._connected]]
            target_force[self._connected] = self._traj_force[idx[self._connected]]

        # RL residuals: TRACK에서만 적용
        if self.cfg.action_dim >= 1:
            target_pose[self._connected, 2] += self._processed_actions[self._connected, 0] * self.cfg.position_scale
        if self.cfg.action_dim >= 2:
            target_force[self._connected, 2] += self._processed_actions[self._connected, 1] * self.cfg.force_scale

        target_pose[:, 2] += float(self.cfg.z_target_offset_mm)

        cmd_pose = target_pose.clone()

        # --------------------------------------------------
        # CONNECT: pure IK
        # --------------------------------------------------
        if torch.any(not_connected):
            self._solve_ik_batch(cmd_pose)

        # --------------------------------------------------
        # TRACK: force control only after connected
        # --------------------------------------------------
        if torch.any(self._connected):
            cmd_xyz_m = current_xyz_m.clone()

            for i in torch.where(self._connected)[0].tolist():
                xd_x = float(target_pose[i, 0].item() / 1000.0)
                xd_y = float(target_pose[i, 1].item() / 1000.0)
                xd_z = float(target_pose[i, 2].item() / 1000.0)

                x_x = float(current_xyz_m[i, 0].item())
                x_y = float(current_xyz_m[i, 1].item())
                x_z = float(current_xyz_m[i, 2].item())

                fd_x = float(target_force[i, 0].item())
                fd_y = float(target_force[i, 1].item())
                fd_z = float(target_force[i, 2].item())

                fext_x = float(current_force[i, 0].item())
                fext_y = float(current_force[i, 1].item())
                fext_z = float(current_force[i, 2].item())

                out_x = self._fc[i].fx.step(xd_x, x_x, fd_x, fext_x)
                out_y = self._fc[i].fy.step(xd_y, x_y, fd_y, fext_y)
                out_z = self._fc[i].fz.step(xd_z, x_z, fd_z, fext_z)

                cmd_xyz_m[i, 0] = float(out_x[0])
                cmd_xyz_m[i, 1] = float(out_y[0])
                cmd_xyz_m[i, 2] = float(out_z[0])

            cmd_pose[self._connected, 0:3] = cmd_xyz_m[self._connected] * 1000.0
            self._solve_ik_batch(cmd_pose)

            # TRACK에서만 row 증가
            self._traj_row[self._connected] += 1
            self._traj_row = torch.clamp(self._traj_row, 0, self._traj_len - 1)

        self.path_done = self._traj_row >= (self._traj_len - 1)

        if self.cfg.enable_debug_print:
            step = int(getattr(self._env, "common_step_counter", 0))
            if step % int(self.cfg.debug_print_interval) == 0:
                env_id = int(self.cfg.debug_env_id)
                env_id = max(0, min(env_id, self._num_envs - 1))

                pos_err_norm = torch.norm(cmd_pose[env_id, 0:3] - current_xyz_mm[env_id]).item()
                rot_err_norm = torch.norm(cmd_pose[env_id, 3:6] - current_wxyz[env_id]).item()

                h5_index = 0 if not self._connected[env_id] else int(self._traj_row[env_id].item())

                local_debug.print_action_debug_status(
                    env_id=env_id,
                    global_step=step,
                    path_index=h5_index,
                    traj_length=self._traj_len,
                    waypoint_steps=1,
                    path_done=bool(self.path_done[env_id].item()),
                    raw_target_xyz=target_pose[env_id, 0:3],
                    target_xyz=cmd_pose[env_id, 0:3],
                    target_wxyz=cmd_pose[env_id, 3:6],
                    current_xyz=current_xyz_mm[env_id, 0:3],
                    current_wxyz=current_wxyz[env_id, 0:3],
                    pos_err_norm=float(pos_err_norm),
                    rot_err_norm=float(rot_err_norm),
                    reward_total=None,
                    reward_score=None,
                    penalty_score=None,
                    dt=float(self.cfg.force_dt),
                )


@configclass
class AdmittanceControlActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = AdmittanceControlAction

    asset_name: str = "robot"
    hdf5_file_path: str = ""
    position_dataset_key: str = "position"
    force_dataset_key: str = "force"

    body_name: str = "spindle_link"
    fixed_joint_name: str = "tool0_to_spindle"
    joint_prim_relpath: str = "joints"

    action_dim: int = 2
    position_scale: float = 1.0
    force_scale: float = 1.0
    z_target_offset_mm: float = 0.0

    force_model_path: str = ""
    force_dt: float = 0.002
    force_threads: int = 1
    force_device: str = "cpu"
    force_md_ratio: float = 1000.0
    force_fc_fext: float = 50.0
    force_free_mass: float = 2.0
    force_free_damping: float = 6000.0
    force_free_stiffness: float = 2000.0
    force_contact_stiffness: float = 0.0
    force_recovery_tau: float = 3.0
    force_action_low: tuple[float, float] = (-0.25, -0.25)
    force_action_high: tuple[float, float] = (0.25, 0.25)
    force_mass_min: float = 0.5
    force_mass_max: float = 5.0
    force_alpha_min: float = 0.5
    force_alpha_max: float = 3.0
    force_alpha_rate_up: float = 4.0
    force_alpha_rate_down: float = 4.0

    connect_pos_tol_mm: float = 20.0
    connect_rot_tol_rad: float = 0.20

    enable_debug_print: bool = True
    debug_print_interval: int = 10
    debug_env_id: int = 0