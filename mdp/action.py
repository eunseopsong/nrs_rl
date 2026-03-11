import torch
import numpy as np
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg

class AdmittanceControlAction(ActionTerm):
    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        self.robot = self._env.scene[cfg.asset_name]
        
        # 1. 30Hz 시뮬레이션에 맞춘 AC 파라미터 
        self.target_force = 15.0
        self.M, self.D, self.K = 1.0, 100.0, 0.0 
        self.dt = self._env.step_dt
        
        self.adm_z_vel = torch.zeros(self.num_envs, device=self.device)
        
        # 💡 [핵심] 초기화 플래그 및 목표 좌표 버퍼
        self.needs_init = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.target_pos_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_quat_cmd = torch.zeros(self.num_envs, 4, device=self.device)
        
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        
        ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self.ik_controller = DifferentialIKController(ik_cfg, num_envs=self.num_envs, device=self.device)
        
        # 최적화: 매 스텝 검색하지 않도록 인덱스를 미리 찾아둠
        self.ee_idx = self.robot.find_bodies("wrist_3_link")[0][0]

    # ==========================================================
    # 🚨 [수정됨] 인공지능이 출력할 액션의 개수를 2개(X, Y)로 늘림!
    # ==========================================================
    @property
    def action_dim(self) -> int: 
        return 2

    @property
    def raw_actions(self) -> torch.Tensor: 
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor: 
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions = actions
        # AI가 내뱉은 값(-1 ~ 1)을 초당 최대 5cm(0.05m/s)의 속도로 변환
        self._processed_actions = actions * 0.05

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        
        # 0,0,0 좌표를 캡처하는 대신, "다음에 캡처해라"라고 플래그만 꽂아둠
        self.needs_init[env_ids] = True
        self.adm_z_vel[env_ids] = 0.0

    def apply_actions(self):
        ee_pos = self.robot.data.body_pos_w[:, self.ee_idx, :]
        ee_quat = self.robot.data.body_quat_w[:, self.ee_idx, :]
        
        # 물리 엔진이 로봇을 제대로 세워둔 '첫 스텝'에 목표 좌표 캡처!
        init_mask = self.needs_init.clone()
        if init_mask.any():
            self.target_pos_cmd[init_mask] = ee_pos[init_mask].clone()
            self.target_quat_cmd[init_mask] = ee_quat[init_mask].clone()
            self.needs_init[init_mask] = False
        
        # --- [1단계: 힘 피드백 및 AC 연산 (Z축)] ---
        contact_sensor = self._env.scene.sensors["contact_forces"]
        current_fz = contact_sensor.data.net_forces_w[:, 0, 2] 
        
        force_error = current_fz - self.target_force
        force_error = torch.where(torch.abs(force_error) < 0.5, torch.zeros_like(force_error), force_error)
        
        # 가속도 계산 (F = M*a + D*v)
        adm_acc = (force_error - self.D * self.adm_z_vel) / self.M
        
        self.adm_z_vel += adm_acc * self.dt
        self.adm_z_vel = torch.clamp(self.adm_z_vel, min=-0.05, max=0.05) # 속도 제한
        
        # --- [2단계: 목표 궤적 통합 업데이트 (X, Y, Z)] ---
        # 💡 1. 인공지능(AI)이 명령한 X, Y 속도를 적분하여 평면 이동!
        self.target_pos_cmd[:, 0] += self._processed_actions[:, 0] * self.dt
        self.target_pos_cmd[:, 1] += self._processed_actions[:, 1] * self.dt
        
        # 💡 2. 어드미턴스 제어기가 계산한 Z축 속도를 적분하여 높이 조절!
        self.target_pos_cmd[:, 2] += self.adm_z_vel * self.dt

        # --- [3단계: 역운동학(IK) 풀기 및 모터 구동] ---
        # 자코비안은 고정된 Base를 빼고 나오므로 인덱스에서 1을 빼줌
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self.ee_idx - 1, :, :]
        current_joint_pos = self.robot.data.joint_pos
        
        pose_command = torch.cat([self.target_pos_cmd, self.target_quat_cmd], dim=-1)
        self.ik_controller.set_command(pose_command)
        
        joint_pos_targets = self.ik_controller.compute(
            ee_pos, ee_quat, jacobian, current_joint_pos
        )
        
        self.robot.set_joint_position_target(joint_pos_targets)