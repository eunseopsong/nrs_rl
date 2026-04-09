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
    Minimal single-point controller using C++ Admittance1D directly.

    cfg에 남기는 control parameter:
      - adm_dt
      - mass
      - damping
      - stiffness

    나머지 상수(contact threshold, approach step 등)는
    C++ 함수 파라미터가 아니므로 action 내부 상수로 고정.
    """

    cfg: "AdmittanceControlActionCfg"

    # -----------------------------
    # Internal fixed constants
    # -----------------------------
    TARGET_FZ_N = 10.0
    CONTACT_FORCE_THRESHOLD_N = 8.0
    CONTACT_DEBOUNCE_STEPS = 5
    APPROACH_STEP_MM = 0.05
    FORCE_LPF_ALPHA = 0.8

    Z_DOWN_LIMIT_MM = 150.0
    Z_UP_LIMIT_MM = 5.0

    ENABLE_DEBUG_PRINT = True
    DEBUG_PRINT_INTERVAL = 10
    DEBUG_ENV_ID = 0
    CONTROL_DT_DEBUG = 0.008

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

        # C++ Admittance1D per env
        self._adm_1d = [y2_pb.Admittance1D(float(cfg.adm_dt)) for _ in range(self._num_envs)]
        for adm in self._adm_1d:
            adm.set_mdk(float(cfg.mass), float(cfg.damping), float(cfg.stiffness))

        self._q_des = self._robot.data.joint_pos[:, :6].clone()

        self._reset_xyz_mm = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self._xy_anchor_mm = torch.zeros((self._num_envs, 2), device=self._device, dtype=torch.float32)
        self._ori_anchor_wxyz = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)

        self._xd_ref_mm = torch.zeros((self._num_envs,), device=self._device, dtype=torch.float32)
        self._xc_cmd_mm = torch.zeros((self._num_envs,), device=self._device, dtype=torch.float32)
        self._z_center_mm = torch.zeros((self._num_envs,), device=self._device, dtype=torch.float32)

        self._contact_latched = torch.zeros((self._num_envs,), device=self._device, dtype=torch.bool)
        self._contact_counter = torch.zeros((self._num_envs,), device=self._device, dtype=torch.long)
        self._fz_abs_lpf = torch.zeros((self._num_envs,), device=self._device, dtype=torch.float32)

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

        current_pose = local_obs.get_ee_pose(self._env, asset_name=self.cfg.asset_name)
        current_xyz_mm = current_pose[:, 0:3]
        current_wxyz = current_pose[:, 3:6]

        self._reset_xyz_mm[env_ids] = current_xyz_mm[env_ids]
        self._xy_anchor_mm[env_ids] = current_xyz_mm[env_ids, 0:2]
        self._ori_anchor_wxyz[env_ids] = current_wxyz[env_ids]

        self._z_center_mm[env_ids] = current_xyz_mm[env_ids, 2]
        self._xd_ref_mm[env_ids] = current_xyz_mm[env_ids, 2]
        self._xc_cmd_mm[env_ids] = current_xyz_mm[env_ids, 2]

        self._contact_latched[env_ids] = False
        self._contact_counter[env_ids] = 0
        self._fz_abs_lpf[env_ids] = 0.0
        self.path_done[env_ids] = False

        env_ids_cpu = env_ids.detach().cpu().tolist()
        for i in env_ids_cpu:
            self._adm_1d[i].set_mdk(float(self.cfg.mass), float(self.cfg.damping), float(self.cfg.stiffness))
            self._adm_1d[i].reset(float(self._xd_ref_mm[i].item()))

        return {}

    def process_actions(self, actions: torch.Tensor):
        # RL action is intentionally unused
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

        raw_fz = current_force[:, 2]
        raw_fz_abs = torch.abs(raw_fz)

        alpha = float(self.FORCE_LPF_ALPHA)
        self._fz_abs_lpf = alpha * self._fz_abs_lpf + (1.0 - alpha) * raw_fz_abs

        cmd_pose = torch.zeros((self._num_envs, 6), device=self._device, dtype=torch.float32)
        cmd_pose[:, 0] = self._xy_anchor_mm[:, 0]
        cmd_pose[:, 1] = self._xy_anchor_mm[:, 1]
        cmd_pose[:, 3:6] = self._ori_anchor_wxyz

        for i in range(self._num_envs):
            fext = float(raw_fz[i].item())
            fext_abs_f = float(self._fz_abs_lpf[i].item())

            # contact detection
            if not bool(self._contact_latched[i].item()):
                if fext_abs_f >= float(self.CONTACT_FORCE_THRESHOLD_N):
                    self._contact_counter[i] += 1
                else:
                    self._contact_counter[i] = 0

                if int(self._contact_counter[i].item()) >= int(self.CONTACT_DEBOUNCE_STEPS):
                    self._contact_latched[i] = True
                    # contact 순간의 현재 z를 reference로 다시 잡음
                    self._xd_ref_mm[i] = current_xyz_mm[i, 2]
                    self._adm_1d[i].reset(float(self._xd_ref_mm[i].item()))
                else:
                    self._xd_ref_mm[i] = self._xd_ref_mm[i] - float(self.APPROACH_STEP_MM)

            fd = float(self.TARGET_FZ_N) if bool(self._contact_latched[i].item()) else 0.0

            z_min = float(self._z_center_mm[i].item()) - float(self.Z_DOWN_LIMIT_MM)
            z_max = float(self._z_center_mm[i].item()) + float(self.Z_UP_LIMIT_MM)
            self._xd_ref_mm[i] = torch.clamp(self._xd_ref_mm[i], min=z_min, max=z_max)

            xc = self._adm_1d[i].step(
                float(self._xd_ref_mm[i].item()),
                float(fd),
                float(fext),
            )

            xc = max(z_min, min(z_max, float(xc)))
            self._xc_cmd_mm[i] = float(xc)

        cmd_pose[:, 2] = self._xc_cmd_mm

        self._solve_ik_batch(cmd_pose)
        self.path_done[:] = False

        if self.ENABLE_DEBUG_PRINT:
            step = int(getattr(self._env, "common_step_counter", 0))
            if step % int(self.DEBUG_PRINT_INTERVAL) == 0:
                env_id = int(self.DEBUG_ENV_ID)
                env_id = max(0, min(env_id, self._num_envs - 1))

                z_min = float(self._z_center_mm[env_id].item()) - float(self.Z_DOWN_LIMIT_MM)
                z_max = float(self._z_center_mm[env_id].item()) + float(self.Z_UP_LIMIT_MM)
                phase = "force" if bool(self._contact_latched[env_id].item()) else "approach"

                pos_err_norm = torch.norm(cmd_pose[env_id, 0:3] - current_xyz_mm[env_id]).item()
                rot_err_norm = torch.norm(cmd_pose[env_id, 3:6] - current_wxyz[env_id]).item()

                print(
                    f"[ForceCtrl Internal] step={step} | env={env_id} | phase={phase} | "
                    f"latched={bool(self._contact_latched[env_id].item())} | "
                    f"counter={int(self._contact_counter[env_id].item())} | "
                    f"xd_ref={float(self._xd_ref_mm[env_id].item()):.6f} | "
                    f"xc_cmd={float(self._xc_cmd_mm[env_id].item()):.6f} | "
                    f"z_center={float(self._z_center_mm[env_id].item()):.6f} | "
                    f"z_min={z_min:.6f} | z_max={z_max:.6f} | "
                    f"raw_fz={float(raw_fz[env_id].item()):.6f} | "
                    f"filt_abs_fz={float(self._fz_abs_lpf[env_id].item()):.6f} | "
                    f"target_fz={float(self.TARGET_FZ_N):.6f}"
                )

                local_debug.print_action_debug_status(
                    env_id=env_id,
                    global_step=step,
                    path_index=0,
                    traj_length=1,
                    waypoint_steps=1,
                    path_done=False,
                    raw_target_xyz=cmd_pose[env_id, 0:3],
                    raw_target_force=torch.tensor([0.0, 0.0, self.TARGET_FZ_N], device=self._device),
                    target_xyz=cmd_pose[env_id, 0:3],
                    target_wxyz=cmd_pose[env_id, 3:6],
                    target_force=torch.tensor([0.0, 0.0, self.TARGET_FZ_N], device=self._device),
                    current_xyz=current_xyz_mm[env_id, 0:3],
                    current_wxyz=current_wxyz[env_id, 0:3],
                    pos_err_norm=float(pos_err_norm),
                    rot_err_norm=float(rot_err_norm),
                    reward_total=None,
                    reward_score=None,
                    penalty_score=None,
                    dt=float(self.CONTROL_DT_DEBUG),
                )


@configclass
class AdmittanceControlActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = AdmittanceControlAction

    # Isaac Lab 연결용 필수 항목
    asset_name: str = "robot"
    body_name: str = "spindle_link"
    fixed_joint_name: str = "tool0_to_spindle"
    joint_prim_relpath: str = "joints"
    action_dim: int = 2

    # C++ Admittance1D와 직접 대응되는 cfg만 유지
    adm_dt: float = 0.008
    mass: float = 2.0
    damping: float = 80.0
    stiffness: float = 0.0