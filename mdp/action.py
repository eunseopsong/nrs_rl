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


class AdmittanceControlAction(ActionTerm):
    """
    Z-axis poke mode for contact / FT-axis validation.

    Behavior:
    - Use HDF5 first row orientation as fixed orientation target
    - Lock x,y to current pose every step
    - Move z downward by constant step each call until contact threshold is reached
    - After contact threshold, hold current z
    - No force controller
    - Raw FT is only observed, not fed back into control

    This isolates:
    1) whether contact occurs
    2) how raw Fz rises
    3) whether Fx/Fy remain small or get mixed
    """

    cfg: "AdmittanceControlActionCfg"

    def __init__(self, cfg: "AdmittanceControlActionCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)

        self._device = env.device
        self._num_envs = env.num_envs
        self._robot = env.scene[cfg.asset_name]

        # keep skrl happy, though not actually used
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
        if pos.shape[0] < 1:
            raise ValueError("[Action] trajectory is empty.")

        self._traj_pos = torch.tensor(pos, device=self._device, dtype=torch.float32)
        self._traj_force = torch.tensor(frc, device=self._device, dtype=torch.float32)
        self._traj_len = int(pos.shape[0])

        self._fixed_target_pose = self._traj_pos[0].clone()     # (6,)
        self._fixed_target_force = self._traj_force[0].clone()  # (3,)

        self._ik = y2_pb.UR10eKinematics(
            dt=float(y2_cfg.CONTROL_PERIOD),
            ee2tcp=y2_cfg.EE2TCP,
        )

        self._q_des = self._robot.data.joint_pos[:, :6].clone()
        self._traj_row = torch.zeros((self._num_envs,), device=self._device, dtype=torch.long)
        self.path_done = torch.zeros((self._num_envs,), device=self._device, dtype=torch.bool)

        # per-env poke target z [mm]
        self._poke_z_mm = torch.zeros((self._num_envs,), device=self._device, dtype=torch.float32)
        self._contact_latched = torch.zeros((self._num_envs,), device=self._device, dtype=torch.bool)

        if cfg.enable_debug_print:
            body_ids = self._robot.find_bodies(cfg.body_name)[0]
            ee_idx = int(body_ids[0]) if len(body_ids) > 0 else 0
            local_debug.print_action_init(
                hdf5_file_path=cfg.hdf5_file_path,
                position_dataset_key=cfg.position_dataset_key,
                traj_shape=tuple(self._traj_pos.shape),
                stride=1,
                used_traj_shape=(1, self._traj_pos.shape[1]),
                body_name=cfg.body_name,
                ee_idx=ee_idx,
                num_envs=self._num_envs,
                tcp_length_offset_m=0.0,
                tcp_offset_axis="z",
            )
            print("[Action] poke mode enabled")
            print("[Action] fixed target pose (row 0):", self._fixed_target_pose.detach().cpu().numpy())
            print("[Action] fixed target force(row 0):", self._fixed_target_force.detach().cpu().numpy())
            print(f"[Action] poke_step_mm={cfg.poke_step_mm}, contact_stop_force_n={cfg.contact_stop_force_n}, initial_offset_mm={cfg.initial_z_offset_mm}")

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

        self._traj_row[env_ids] = 0
        self.path_done[env_ids] = False

        current_pose = local_obs.get_ee_pose(self._env, asset_name=self.cfg.asset_name)
        current_xyz_mm = current_pose[:, 0:3]

        # start from current z + initial offset
        self._poke_z_mm[env_ids] = current_xyz_mm[env_ids, 2] + float(self.cfg.initial_z_offset_mm)
        self._contact_latched[env_ids] = False

        return {}

    def process_actions(self, actions: torch.Tensor):
        # not used, but keep interface
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
        current_force = wrench[:, 0:3]

        raw_target_pose = self._fixed_target_pose.unsqueeze(0).repeat(self._num_envs, 1).clone()
        raw_target_force = self._fixed_target_force.unsqueeze(0).repeat(self._num_envs, 1).clone()

        cmd_pose = raw_target_pose.clone()
        cmd_xyz_mm = current_xyz_mm.clone()

        # orientation is fixed to first row orientation
        cmd_pose[:, 3:6] = raw_target_pose[:, 3:6]

        for i in range(self._num_envs):
            fz = float(current_force[i, 2].item())

            # x,y lock to current pose
            cmd_xyz_mm[i, 0] = current_xyz_mm[i, 0]
            cmd_xyz_mm[i, 1] = current_xyz_mm[i, 1]

            # if enough contact, latch and hold z
            if (not bool(self._contact_latched[i].item())) and (abs(fz) >= float(self.cfg.contact_stop_force_n)):
                self._contact_latched[i] = True
                self._poke_z_mm[i] = current_xyz_mm[i, 2]

            # if not yet in contact, keep poking downward
            if not bool(self._contact_latched[i].item()):
                self._poke_z_mm[i] = self._poke_z_mm[i] - float(self.cfg.poke_step_mm)

            cmd_xyz_mm[i, 2] = self._poke_z_mm[i]

        cmd_pose[:, 0:3] = cmd_xyz_mm

        self._solve_ik_batch(cmd_pose)

        self._traj_row[:] = 0
        self.path_done[:] = False

        if self.cfg.enable_debug_print:
            step = int(getattr(self._env, "common_step_counter", 0))
            if step % int(self.cfg.debug_print_interval) == 0:
                env_id = int(self.cfg.debug_env_id)
                env_id = max(0, min(env_id, self._num_envs - 1))

                pos_err_norm = torch.norm(cmd_pose[env_id, 0:3] - current_xyz_mm[env_id]).item()
                rot_err_norm = torch.norm(cmd_pose[env_id, 3:6] - current_wxyz[env_id]).item()

                local_debug.print_action_debug_status(
                    env_id=env_id,
                    global_step=step,
                    path_index=0,
                    traj_length=self._traj_len,
                    waypoint_steps=1,
                    path_done=False,
                    raw_target_xyz=raw_target_pose[env_id, 0:3],
                    raw_target_force=raw_target_force[env_id],
                    target_xyz=cmd_pose[env_id, 0:3],
                    target_wxyz=cmd_pose[env_id, 3:6],
                    target_force=raw_target_force[env_id],
                    current_xyz=current_xyz_mm[env_id, 0:3],
                    current_wxyz=current_wxyz[env_id, 0:3],
                    pos_err_norm=float(pos_err_norm),
                    rot_err_norm=float(rot_err_norm),
                    reward_total=None,
                    reward_score=None,
                    penalty_score=None,
                    dt=float(self.cfg.control_dt_debug),
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

    # keep skrl happy
    action_dim: int = 2

    # poke parameters
    initial_z_offset_mm: float = 0.0
    poke_step_mm: float = 0.05
    contact_stop_force_n: float = 5.0

    # debug only
    control_dt_debug: float = 0.008

    enable_debug_print: bool = True
    debug_print_interval: int = 10
    debug_env_id: int = 0