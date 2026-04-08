# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

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


class AdmittanceControlAction(ActionTerm):
    """
    Debug / validation mode:
    - Do NOT follow HDF5 trajectory
    - Do NOT run force control
    - Just hold the initial EE pose captured at reset
    """

    cfg: "AdmittanceControlActionCfg"

    def __init__(self, cfg: "AdmittanceControlActionCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)

        self._device = env.device
        self._num_envs = env.num_envs
        self._robot = env.scene[cfg.asset_name]

        self._raw_actions = torch.zeros((self._num_envs, cfg.action_dim), device=self._device, dtype=torch.float32)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        self._ik = y2_pb.UR10eKinematics(
            dt=float(y2_cfg.CONTROL_PERIOD),
            ee2tcp=y2_cfg.EE2TCP,
        )

        self._q_des = self._robot.data.joint_pos[:, :6].clone()

        # reset 시 저장할 hold target
        self._hold_pose = torch.zeros((self._num_envs, 6), device=self._device, dtype=torch.float32)

        # termination compatibility
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
            env_ids = slice(None)

        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._q_des[env_ids] = self._robot.data.joint_pos[env_ids, :6].clone()
        self.path_done[env_ids] = False

        # 현재 pose를 그대로 hold target으로 저장
        current_pose = local_obs.get_ee_pose(self._env, asset_name=self.cfg.asset_name)
        self._hold_pose[env_ids] = current_pose[env_ids].clone()

        return {}

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions.to(self._device)
        self._processed_actions[:] = self._raw_actions

    def apply_actions(self):
        current_pose = local_obs.get_ee_pose(self._env, asset_name=self.cfg.asset_name)
        wrench = local_ft_sensor.get_6axis_ft_fixed_joint(
            env=self._env,
            asset_name=self.cfg.asset_name,
            fixed_joint_name=self.cfg.fixed_joint_name,
            joint_prim_relpath=self.cfg.joint_prim_relpath,
            verbose=False,
        )

        current_xyz = current_pose[:, 0:3]
        current_wxyz = current_pose[:, 3:6]

        target_pose = self._hold_pose.clone()

        target_htm = _build_target_htm_torch(target_pose[:, 0:3], target_pose[:, 3:6])

        q_des_list = []
        for i in range(self._num_envs):
            q_seed = self._q_des[i].detach().cpu().tolist()
            T_i = target_htm[i].detach().cpu().tolist()
            q_next = self._ik.solve_ik(q_seed, T_i)
            q_des_list.append(q_next)

        self._q_des = torch.tensor(q_des_list, device=self._device, dtype=torch.float32)
        self._robot.set_joint_position_target(self._q_des)

        # hold mode에서는 종료 안 함
        self.path_done[:] = False

        if self.cfg.enable_debug_print:
            step = int(getattr(self._env, "common_step_counter", 0))
            if step % int(self.cfg.debug_print_interval) == 0:
                env_id = int(self.cfg.debug_env_id)
                env_id = max(0, min(env_id, self._num_envs - 1))

                pos_err_norm = torch.norm(target_pose[env_id, 0:3] - current_xyz[env_id]).item()
                rot_err_norm = torch.norm(target_pose[env_id, 3:6] - current_wxyz[env_id]).item()

                local_debug.print_action_debug_status(
                    env_id=env_id,
                    global_step=step,
                    path_index=0,
                    traj_length=1,
                    waypoint_steps=1,
                    path_done=False,
                    raw_target_xyz=target_pose[env_id, 0:3],
                    target_xyz=target_pose[env_id, 0:3],
                    target_wxyz=target_pose[env_id, 3:6],
                    current_xyz=current_xyz[env_id, 0:3],
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
    body_name: str = "spindle_link"

    # compatibility only
    hdf5_file_path: str = ""
    position_dataset_key: str = "position"
    force_dataset_key: str = "force"

    fixed_joint_name: str = "tool0_to_spindle"
    joint_prim_relpath: str = "joints"

    action_dim: int = 2
    position_scale: float = 1.0
    force_scale: float = 1.0
    z_target_offset_mm: float = 0.0

    # compatibility only
    connect_pos_tol_mm: float = 20.0
    connect_rot_tol_rad: float = 0.20

    # compatibility only
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