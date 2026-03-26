import torch
import math
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg  # <-- ActionTermCfg 추가
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.utils import configclass  # <-- configclass 추가


class AdmittanceControlAction(ActionTerm):
    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot = self._env.scene[cfg.asset_name]
        self.cfg = cfg  # [26.03.26. 추가] 설정값 접근용 저장

        # --- Force control ---
        # [26.03.26. 수정] 하드코딩된 값을 Cfg에서 받아오도록 변경
        self.target_force = self.cfg.target_force
        self.current_target_force = torch.zeros(self.num_envs, device=self.device)

        # --- Admittance params ---
        # [26.03.26. 수정] M, D 값을 Cfg에서 받아오도록 변경
        self.M = self.cfg.M
        self.D = self.cfg.D
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
            ik_method="dls"  # [26.03.26. 팁] dls가 너무 느리다면 "pinv" (가상역행렬)로 변경 고려
        )
        self.ik_controller = DifferentialIKController(
            ik_cfg, num_envs=self.num_envs, device=self.device
        )

        # [26.03.26. 수정] 엔드이펙터 링크 이름을 Cfg에서 받아오도록 변경
        self.ee_idx = self.robot.find_bodies(self.cfg.body_name)[0][0]
        
        # [26.03.26. 추가] 스핀들 수직 고정을 위한 목표 쿼터니언 텐서
        self.vertical_quat_tensor = torch.tensor(self.cfg.vertical_quat, device=self.device)

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
            # [26.03.26. 수정] 엉거주춤한 현재 자세(ee_quat)를 복사하지 않고, 수직 쿼터니언을 강제 주입
            self.target_quat_cmd[mask0] = self.vertical_quat_tensor.repeat(mask0.sum(), 1)
            self.stage[mask0] = 1

        # Stage 1: [26.03.26. 수정] 0.5초 대기 대신 수직 자세 완벽 도달 전까지 절대 대기 (강제성 부여)
        mask1 = self.stage == 1
        if mask1.any():
            # 현재 쿼터니언과 목표 쿼터니언 간의 내적(Dot product)으로 각도 오차 계산
            quat_dot = torch.sum(ee_quat[mask1] * self.target_quat_cmd[mask1], dim=-1)
            # 오차를 라디안으로 변환
            angle_error = torch.acos(torch.clamp(torch.abs(quat_dot), -1.0, 1.0)) * 2.0
            
            # 오차가 0.05 라디안(약 2.8도) 미만일 때만 완료(done) 판정
            done = angle_error < 0.05
            
            # 완료된 환경들만 Stage 2로 진입 (안 된 애들은 계속 목표 자세만 추종하며 대기)
            self.stage[mask1 & done] = 2

        # Stage 2: Admittance 하강 제어 및 [26.03.26. 추가] 적응형 자세 보정
        mask2 = self.stage == 2
        if mask2.any():
            force_step = 5.0 * self.dt
            self.current_target_force[mask2] = torch.clamp(
                self.current_target_force[mask2] + force_step,
                max=self.target_force
            )

            # [26.03.26. 추가] 관측(Observations) 데이터에서 3D 힘 벡터 가져오기
            contact_sensor = self._env.scene.sensors["contact_forces"]
            F_ext_3d = torch.nan_to_num(contact_sensor.data.net_forces_w[mask2, 0, :], nan=0.0)
            
            # Z축 힘 (수직 방향 제어용)
            F_ext_z = F_ext_3d[:, 2]
            F_target = self.current_target_force[mask2]

            F_error = F_ext_z - F_target
            F_error = torch.where(
                torch.abs(F_error) < 1.0,
                torch.zeros_like(F_error),
                F_error
            )

            # 1. 병진(Z축) 어드미턴스
            adm_acc = (F_error - self.D * self.adm_z_vel[mask2]) / self.M
            
            self.adm_z_vel[mask2] += adm_acc * self.dt
            self.adm_z_vel[mask2] = torch.clamp(self.adm_z_vel[mask2], -0.05, 0.05)

            self.target_pos_cmd[mask2, 2] += self.adm_z_vel[mask2] * self.dt
            # [26.03.26. 수정] 로봇이 자기 무게를 못 이기고 바닥을 뚫고 눕는 것을 막기 위해 하한선 조율 (필요시 0.05 등으로 수정)
            self.target_pos_cmd[mask2, 2] = torch.clamp(self.target_pos_cmd[mask2, 2], min=0.1)

            self.target_pos_cmd[mask2, 0] += self._processed_actions[mask2, 0] * self.dt
            self.target_pos_cmd[mask2, 1] += self._processed_actions[mask2, 1] * self.dt

            # -----------------------------------------------------------------
            # 2. [26.03.26. 적응형 제어 추가] 접촉 면의 기울기를 반영한 미세 자세 조정
            # -----------------------------------------------------------------
            K_rot = self.cfg.K_rot  # 회전 어드미턴스 게인
            
            delta_rx = -F_ext_3d[:, 1] * K_rot * self.dt # Y축 힘 -> X축 기준 회전
            delta_ry =  F_ext_3d[:, 0] * K_rot * self.dt # X축 힘 -> Y축 기준 회전
            
            delta_quat = torch.zeros_like(self.target_quat_cmd[mask2])
            delta_quat[:, 0] = 1.0
            delta_quat[:, 1] = delta_rx * 0.5
            delta_quat[:, 2] = delta_ry * 0.5
            delta_quat = delta_quat / torch.linalg.norm(delta_quat, dim=-1, keepdim=True)

            q1_w, q1_xyz = self.target_quat_cmd[mask2, 0:1], self.target_quat_cmd[mask2, 1:]
            q2_w, q2_xyz = delta_quat[:, 0:1], delta_quat[:, 1:]
            
            new_w = q1_w * q2_w - torch.sum(q1_xyz * q2_xyz, dim=-1, keepdim=True)
            new_xyz = q1_w * q2_xyz + q2_w * q1_xyz + torch.cross(q1_xyz, q2_xyz, dim=-1)
            
            self.target_quat_cmd[mask2] = torch.cat([new_w, new_xyz], dim=-1)

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
    
    # [26.03.26. 추가] 하드코딩 탈피를 위한 파라미터 세팅
    body_name: str = "wrist_3_link" 
    vertical_quat: tuple = (0.0, 1.0, 0.0, 0.0) # UR 등 모델에 맞게 수직(w, x, y, z) 설정 필요 시 변경
    target_force: float = 15.0
    M: float = 1.0
    D: float = 40.0
    
    # [26.03.26. 적응형 자세 보정 파라미터 추가]
    # 값이 클수록 표면 굴곡에 발목이 유연하게 꺾임. 진동이 심하면 0.001 이하로 줄이세요.
    K_rot: float = 0.005