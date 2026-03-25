import torch
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg  # <-- ActionTermCfg 추가
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.utils import configclass  # <-- configclass 추가


class AdmittanceControlAction(ActionTerm):
    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot = self._env.scene[cfg.asset_name]

        # --- Force control ---
        self.target_force = 15.0
        self.current_target_force = torch.zeros(self.num_envs, device=self.device)

        # --- Admittance params ---
        self.M = 1.0
        self.D = 40.0
        self.dt = self._env.step_dt

        self.adm_z_vel = torch.zeros(self.num_envs, device=self.device)

        # --- Stage control ---
        self.stage = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.stage_timer = torch.zeros(self.num_envs, device=self.device)

        # --- Targets ---
        self.target_pos_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_quat_cmd = torch.zeros(self.num_envs, 4, device=self.device)

        # --- Actions ---
        self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # --- IK ---
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls"
        )
        self.ik_controller = DifferentialIKController(
            ik_cfg, num_envs=self.num_envs, device=self.device
        )

        self.ee_idx = self.robot.find_bodies("wrist_3_link")[0][0]

    # ==========================================
    # IsaacLab 필수 요구 속성들 (이번에 추가된 부분)
    # ==========================================
    @property
    def action_dim(self):
        return 2

    @property
    def raw_actions(self):
        return self._raw_actions

    @property
    def processed_actions(self):
        return self._processed_actions
    # ==========================================

    def process_actions(self, actions):
        self._raw_actions = actions
        self._processed_actions = torch.nan_to_num(actions * 0.05, nan=0.0)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)

        self.stage[env_ids] = 0
        self.stage_timer[env_ids] = 0.0

        self.adm_z_vel[env_ids] = 0.0
        self.current_target_force[env_ids] = 0.0

    def apply_actions(self):
        ee_pos = self.robot.data.body_pos_w[:, self.ee_idx, :]
        ee_quat = self.robot.data.body_quat_w[:, self.ee_idx, :]
        q = self.robot.data.joint_pos

        jacobian = self.robot.root_physx_view.get_jacobians()[:, self.ee_idx - 1]

        # Stage 0: 초기 자세 캡처
        mask0 = self.stage == 0
        if mask0.any():
            self.target_pos_cmd[mask0] = ee_pos[mask0].clone()
            self.target_quat_cmd[mask0] = ee_quat[mask0].clone()
            self.stage[mask0] = 1

        # Stage 1: 0.5초 대기
        mask1 = self.stage == 1
        if mask1.any():
            self.stage_timer[mask1] += self.dt
            done = self.stage_timer > 0.5
            self.stage[mask1 & done] = 2

        # Stage 2: Admittance 하강 제어
        mask2 = self.stage == 2
        if mask2.any():
            force_step = 5.0 * self.dt
            self.current_target_force[mask2] = torch.clamp(
                self.current_target_force[mask2] + force_step,
                max=self.target_force
            )

            contact_sensor = self._env.scene.sensors["contact_forces"]
            F_ext = torch.nan_to_num(
                contact_sensor.data.net_forces_w[mask2, 0, 2],
                nan=0.0
            )

            F_target = self.current_target_force[mask2]

            F_error = F_ext - F_target
            F_error = torch.where(
                torch.abs(F_error) < 1.0,
                torch.zeros_like(F_error),
                F_error
            )

            adm_acc = (F_error - self.D * self.adm_z_vel[mask2]) / self.M
            
            self.adm_z_vel[mask2] += adm_acc * self.dt
            self.adm_z_vel[mask2] = torch.clamp(self.adm_z_vel[mask2], -0.05, 0.05)

            self.target_pos_cmd[mask2, 2] += self.adm_z_vel[mask2] * self.dt
            self.target_pos_cmd[mask2, 2] = torch.clamp(self.target_pos_cmd[mask2, 2], min=0.1)

            self.target_pos_cmd[mask2, 0] += self._processed_actions[mask2, 0] * self.dt
            self.target_pos_cmd[mask2, 1] += self._processed_actions[mask2, 1] * self.dt

        # IK solve
        pose = torch.cat([self.target_pos_cmd, self.target_quat_cmd], dim=-1)
        self.ik_controller.set_command(pose)

        q_cmd = self.ik_controller.compute(ee_pos, ee_quat, jacobian, q)
        q_cmd = torch.where(torch.isnan(q_cmd), q, q_cmd)

        self.robot.set_joint_position_target(q_cmd)

# ==========================================
# [26.03.25. 추가] Action Configuration 정의
# ==========================================
@configclass
class AdmittanceControlActionCfg(ActionTermCfg):
    class_type: type = AdmittanceControlAction
    asset_name: str = "robot"