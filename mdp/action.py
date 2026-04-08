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
    """
    spatial: (N, 3) = [wx wy wz]
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


def _build_target_htm_torch(pos_mm: torch.Tensor, spatial: torch.Tensor) -> torch.Tensor:
    """
    pos_mm : (N,3)
    spatial: (N,3)
    return : (N,4,4)
    """
    n = pos_mm.shape[0]
    device = pos_mm.device
    dtype = pos_mm.dtype

    T = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(n, 1, 1)
    T[:, :3, :3] = _spatial_angle_to_rotmat_torch(spatial)
    T[:, :3, 3] = pos_mm
    return T


class _PerEnvForceControllers:
    def __init__(self, cfg):
        common_kwargs = dict(
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
        self.fz = y2_pb.ForceCon1DMode5(**common_kwargs)

    def reset(self, z_m: float):
        self.fz.reset(float(z_m))


class AdmittanceControlAction(ActionTerm):
    """
    HDF5 tracking with state machine:

        APPROACH -> DESCEND -> TRACK

    TRACK:
        - x, y      : position tracking
        - z         : force control
        - wx,wy,wz  : orientation tracking
    """

    cfg: "AdmittanceControlActionCfg"

    def __init__(self, cfg: "AdmittanceControlActionCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)

        self._device = env.device
        self._num_envs = env.num_envs
        self._robot = env.scene[cfg.asset_name]

        self._raw_actions = torch.zeros((self._num_envs, cfg.action_dim), device=self._device, dtype=torch.float32)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        with h5py.File(cfg.hdf5_file_path, "r") as f:
            if cfg.position_dataset_key not in f:
                raise KeyError(
                    f"[Action] position dataset '{cfg.position_dataset_key}' not found. "
                    f"Available keys: {list(f.keys())}"
                )
            if cfg.force_dataset_key not in f:
                raise KeyError(
                    f"[Action] force dataset '{cfg.force_dataset_key}' not found. "
                    f"Available keys: {list(f.keys())}"
                )
            pos = f[cfg.position_dataset_key][:]
            frc = f[cfg.force_dataset_key][:]

        if pos.ndim != 2 or pos.shape[1] != 6:
            raise ValueError(f"[Action] expected position dataset (T,6), got {pos.shape}")
        if frc.ndim != 2 or frc.shape[1] != 3:
            raise ValueError(f"[Action] expected force dataset (T,3), got {frc.shape}")
        if pos.shape[0] != frc.shape[0]:
            raise ValueError(f"[Action] position/force length mismatch: {pos.shape[0]} vs {frc.shape[0]}")

        self._traj_pos = torch.tensor(pos, device=self._device, dtype=torch.float32)
        self._traj_force = torch.tensor(frc, device=self._device, dtype=torch.float32)
        self._traj_len = int(pos.shape[0])

        self._ik = y2_pb.UR10eKinematics(
            dt=float(y2_cfg.CONTROL_PERIOD),
            ee2tcp=y2_cfg.EE2TCP,
        )

        self._q_des = self._robot.data.joint_pos[:, :6].clone()

        # 1 = APPROACH, 2 = DESCEND, 3 = TRACK
        self._phase = torch.ones((self._num_envs,), device=self._device, dtype=torch.long)
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

        self._phase[env_ids] = 1
        self._traj_row[env_ids] = 0
        self.path_done[env_ids] = False

        current_pose = local_obs.get_ee_pose(self._env, asset_name=self.cfg.asset_name)
        current_z_m = current_pose[:, 2] / 1000.0
        for i in env_ids.tolist():
            self._fc[i].reset(float(current_z_m[i].item()))

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

    def _build_first_pose(self) -> torch.Tensor:
        return self._traj_pos[0].unsqueeze(0).repeat(self._num_envs, 1)

    def _build_approach_pose(self) -> torch.Tensor:
        pose = self._build_first_pose()
        pose[:, 2] = pose[:, 2] + float(self.cfg.approach_offset_mm)
        return pose

    def _pos_rot_error(self, current_pose: torch.Tensor, target_pose: torch.Tensor):
        pos_err = torch.norm(target_pose[:, 0:3] - current_pose[:, 0:3], dim=-1)
        rot_err = torch.norm(target_pose[:, 3:6] - current_pose[:, 3:6], dim=-1)
        return pos_err, rot_err

    def apply_actions(self):
        current_pose = local_obs.get_ee_pose(self._env, asset_name=self.cfg.asset_name)
        wrench = local_ft_sensor.get_6axis_ft_fixed_joint(
            env=self._env,
            asset_name=self.cfg.asset_name,
            fixed_joint_name=self.cfg.fixed_joint_name,
            joint_prim_relpath=self.cfg.joint_prim_relpath,
            verbose=False,
        )

        current_xyz_mm = current_pose[:, 0:3]
        current_wxyz = current_pose[:, 3:6]
        current_xyz_m = current_xyz_mm / 1000.0
        current_force = wrench[:, 0:3]

        first_pose = self._build_first_pose()
        approach_pose = self._build_approach_pose()

        approach_pos_err, approach_rot_err = self._pos_rot_error(current_pose, approach_pose)
        descend_pos_err, descend_rot_err = self._pos_rot_error(current_pose, first_pose)

        # APPROACH -> DESCEND
        approach_done = (
            (self._phase == 1)
            & (approach_pos_err < float(self.cfg.approach_pos_tol_mm))
            & (approach_rot_err < float(self.cfg.approach_rot_tol_rad))
        )
        if torch.any(approach_done):
            self._phase[approach_done] = 2

        # DESCEND -> TRACK
        descend_done = (
            (self._phase == 2)
            & (descend_pos_err < float(self.cfg.descend_pos_tol_mm))
            & (descend_rot_err < float(self.cfg.descend_rot_tol_rad))
        )
        if torch.any(descend_done):
            self._phase[descend_done] = 3
            for i in torch.where(descend_done)[0].tolist():
                self._fc[i].reset(float(current_xyz_m[i, 2].item()))

        target_pose = torch.zeros((self._num_envs, 6), device=self._device, dtype=torch.float32)
        target_force = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)

        # APPROACH
        approach_mask = self._phase == 1
        if torch.any(approach_mask):
            target_pose[approach_mask] = approach_pose[approach_mask]
            target_force[approach_mask] = 0.0

        # DESCEND
        descend_mask = self._phase == 2
        if torch.any(descend_mask):
            target_pose[descend_mask] = first_pose[descend_mask]
            target_force[descend_mask] = 0.0

        # TRACK
        track_mask = self._phase == 3
        if torch.any(track_mask):
            idx = torch.clamp(self._traj_row, 0, self._traj_len - 1)
            target_pose[track_mask] = self._traj_pos[idx[track_mask]]
            target_force[track_mask] = self._traj_force[idx[track_mask]]

            # optional RL residuals: z only
            if self.cfg.action_dim >= 1:
                target_pose[track_mask, 2] += self._processed_actions[track_mask, 0] * self.cfg.position_scale
            if self.cfg.action_dim >= 2:
                target_force[track_mask, 2] += self._processed_actions[track_mask, 1] * self.cfg.force_scale

        target_pose[:, 2] += float(self.cfg.z_target_offset_mm)

        cmd_pose = target_pose.clone()

        # APPROACH / DESCEND: pure IK
        pure_ik_mask = self._phase != 3
        if torch.any(pure_ik_mask):
            self._solve_ik_batch(cmd_pose)

        # TRACK: x/y/orientation position control, z only force control
        if torch.any(track_mask):
            cmd_xyz_mm = target_pose[:, 0:3].clone()

            for i in torch.where(track_mask)[0].tolist():
                # x, y: pure position tracking
                cmd_xyz_mm[i, 0] = target_pose[i, 0]
                cmd_xyz_mm[i, 1] = target_pose[i, 1]

                # z: force control only
                xd_z = float(target_pose[i, 2].item() / 1000.0)
                x_z = float(current_xyz_m[i, 2].item())

                fd_z = float(target_force[i, 2].item())
                fext_z = float(current_force[i, 2].item())

                out_z = self._fc[i].fz.step(xd_z, x_z, fd_z, fext_z)
                cmd_xyz_mm[i, 2] = float(out_z[0]) * 1000.0

            cmd_pose[track_mask, 0:3] = cmd_xyz_mm[track_mask]
            cmd_pose[track_mask, 3:6] = target_pose[track_mask, 3:6]

            self._solve_ik_batch(cmd_pose)

            # row advance only if z-force is reasonably close
            force_err = torch.abs(current_force[:, 2] - target_force[:, 2])
            allow_advance = track_mask & (force_err < float(self.cfg.track_force_tol_n))

            self._traj_row[allow_advance] += 1
            self._traj_row = torch.clamp(self._traj_row, 0, self._traj_len - 1)

        self.path_done = self._traj_row >= (self._traj_len - 1)

        if self.cfg.enable_debug_print:
            step = int(getattr(self._env, "common_step_counter", 0))
            if step % int(self.cfg.debug_print_interval) == 0:
                env_id = int(self.cfg.debug_env_id)
                env_id = max(0, min(env_id, self._num_envs - 1))

                pos_err_norm = torch.norm(cmd_pose[env_id, 0:3] - current_xyz_mm[env_id]).item()
                rot_err_norm = torch.norm(cmd_pose[env_id, 3:6] - current_wxyz[env_id]).item()

                h5_index = int(self._traj_row[env_id].item()) if self._phase[env_id].item() == 3 else 0

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
                    target_force=target_force[env_id],   # 추가
                    reward_total=None,
                    reward_score=None,
                    penalty_score=None,
                    dt=float(self.cfg.force_dt),
                )


@configclass
class AdmittanceControlActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = AdmittanceControlAction

    asset_name: str = "robot"
    body_name: str = "spindle_link"

    hdf5_file_path: str = ""
    position_dataset_key: str = "position"
    force_dataset_key: str = "force"

    fixed_joint_name: str = "tool0_to_spindle"
    joint_prim_relpath: str = "joints"

    action_dim: int = 2
    position_scale: float = 1.0   # z residual [mm]
    force_scale: float = 1.0      # z-force residual [N]
    z_target_offset_mm: float = 0.0

    approach_offset_mm: float = 120.0
    approach_pos_tol_mm: float = 20.0
    approach_rot_tol_rad: float = 0.20
    descend_pos_tol_mm: float = 10.0
    descend_rot_tol_rad: float = 0.10

    track_force_tol_n: float = 10.0

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

    enable_debug_print: bool = True
    debug_print_interval: int = 10
    debug_env_id: int = 0