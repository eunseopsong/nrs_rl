import os
import torch
import math
import numpy as np
import h5py
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.utils import configclass


class AdmittanceControlAction(ActionTerm):
    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot = self._env.scene[cfg.asset_name]
        self.cfg = cfg  

        # --- Force control ---
        self.target_force = self.cfg.target_force
        self.current_target_force = torch.zeros(self.num_envs, device=self.device)

        # --- Admittance params ---
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
            ik_method="dls"  
        )
        self.ik_controller = DifferentialIKController(
            ik_cfg, num_envs=self.num_envs, device=self.device
        )

        self.ee_idx = self.robot.find_bodies(self.cfg.body_name)[0][0]
        
        # ✅ [26.03.28. 추가] 스핀들 수직 고정을 위한 목표 쿼터니언 텐서
        self.vertical_quat_tensor = torch.tensor(self.cfg.vertical_quat, device=self.device)

        # -----------------------------------------------------------
        # ✅ [26.03.28. 추가] HDF5를 읽어 베이스(Local) 기준 워크스페이스 한계치 설정
        # -----------------------------------------------------------
        self.base_x_min, self.base_x_max = self.cfg.workspace_x_limits
        self.base_y_min, self.base_y_max = self.cfg.workspace_y_limits
        
        if hasattr(self.cfg, 'hdf5_file_path') and self.cfg.hdf5_file_path:
            if os.path.exists(self.cfg.hdf5_file_path):
                try:
                    with h5py.File(self.cfg.hdf5_file_path, 'r') as f:
                        dataset_name = 'positions' if 'positions' in f else list(f.keys())[0]
                        traj_data = f[dataset_name][:]
                        margin = self.cfg.workspace_margin
                        
                        self.base_x_min = float(np.min(traj_data[:, 0])) - margin
                        self.base_x_max = float(np.max(traj_data[:, 0])) + margin
                        self.base_y_min = float(np.min(traj_data[:, 1])) - margin
                        self.base_y_max = float(np.max(traj_data[:, 1])) + margin
                        print(f"✅ [Action] HDF5 로드 완료! Local Limits: X[{self.base_x_min:.3f}, {self.base_x_max:.3f}], Y[{self.base_y_min:.3f}, {self.base_y_max:.3f}]")
                except Exception as e:
                    print(f"⚠️ [Action] HDF5 로드 실패. Cfg 기본 Limit 사용: {e}")

        # -----------------------------------------------------------
        # ✅ [26.03.28. 추가] 각 환경별(Global) 워크스페이스를 담아둘 텐서 준비
        # -----------------------------------------------------------
        self.env_x_min = torch.zeros(self.num_envs, device=self.device)
        self.env_x_max = torch.zeros(self.num_envs, device=self.device)
        self.env_y_min = torch.zeros(self.num_envs, device=self.device)
        self.env_y_max = torch.zeros(self.num_envs, device=self.device)

    @property
    def action_dim(self):
        return 2

    @property
    def raw_actions(self):
        return self._raw_actions

    @property
    def processed_actions(self):
        return self._processed_actions

    def process_actions(self, actions):
        self._raw_actions = actions
        scaled_actions = actions * self.cfg.action_scale
        self._processed_actions = torch.nan_to_num(
            torch.clamp(scaled_actions, min=-self.cfg.max_xy_vel, max=self.cfg.max_xy_vel),
            nan=0.0
        )

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)

        self.stage[env_ids] = 0
        self.stage_timer[env_ids] = 0.0
        self.adm_z_vel[env_ids] = 0.0
        self.current_target_force[env_ids] = 0.0

        # ✅ [26.03.28. 추가] 에피소드 시작 시 각 로봇의 글로벌 Root 위치 측정 및 워크스페이스 고정
        root_pos = self.robot.data.root_pos_w[env_ids]
        
        self.env_x_min[env_ids] = root_pos[:, 0] + self.base_x_min
        self.env_x_max[env_ids] = root_pos[:, 0] + self.base_x_max
        self.env_y_min[env_ids] = root_pos[:, 1] + self.base_y_min
        self.env_y_max[env_ids] = root_pos[:, 1] + self.base_y_max

    def apply_actions(self):
        ee_pos = self.robot.data.body_pos_w[:, self.ee_idx, :]
        ee_quat = self.robot.data.body_quat_w[:, self.ee_idx, :]
        q = self.robot.data.joint_pos
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self.ee_idx - 1]

        # Stage 0: 초기 자세 캡처 및 수직 강제 주입
        mask0 = self.stage == 0
        if mask0.any():
            self.target_pos_cmd[mask0] = ee_pos[mask0].clone()
            self.target_quat_cmd[mask0] = self.vertical_quat_tensor.repeat(mask0.sum(), 1)
            self.stage[mask0] = 1

        # Stage 1: 수직 자세 도달 대기
        mask1 = self.stage == 1
        if mask1.any():
            self.stage_timer[mask1] += self.dt
            quat_dot = torch.sum(ee_quat[mask1] * self.target_quat_cmd[mask1], dim=-1)
            angle_error = torch.acos(torch.clamp(torch.abs(quat_dot), -1.0, 1.0)) * 2.0
            done = (angle_error < 0.1) | (self.stage_timer[mask1] > self.cfg.stage1_timeout)
            self.stage[mask1 & done] = 2

        # Stage 2: Admittance 하강 제어 및 조건부 적응형 자세 보정
        mask2 = self.stage == 2
        if mask2.any():
            force_step = 5.0 * self.dt
            self.current_target_force[mask2] = torch.clamp(
                self.current_target_force[mask2] + force_step,
                max=self.target_force
            )

            contact_sensor = self._env.scene.sensors["contact_forces"]
            # 계산의 편의를 위해 전체 환경에 대한 힘을 먼저 가져옵니다.
            F_ext_3d_all = torch.nan_to_num(contact_sensor.data.net_forces_w[:, 0, :], nan=0.0)
            F_ext_z_all = F_ext_3d_all[:, 2]

            F_target = self.current_target_force[mask2]
            F_error = F_ext_z_all[mask2] - F_target
            F_error = torch.where(torch.abs(F_error) < 1.0, torch.zeros_like(F_error), F_error)

            # 1. Z축 병진 어드미턴스 (하강)
            adm_acc = (F_error - self.D * self.adm_z_vel[mask2]) / self.M
            self.adm_z_vel[mask2] += adm_acc * self.dt
            self.adm_z_vel[mask2] = torch.clamp(self.adm_z_vel[mask2], -0.05, 0.05)
            self.target_pos_cmd[mask2, 2] += self.adm_z_vel[mask2] * self.dt

            # 2. X, Y 위치 제어 (RL Action)
            self.target_pos_cmd[mask2, 0] += self._processed_actions[mask2, 0] * self.dt
            self.target_pos_cmd[mask2, 1] += self._processed_actions[mask2, 1] * self.dt

            # ✅ [26.03.28. 추가] 글로벌 워크스페이스 한계치로 즉시 클리핑
            self.target_pos_cmd[mask2, 0] = torch.clamp(self.target_pos_cmd[mask2, 0], min=self.env_x_min[mask2], max=self.env_x_max[mask2])
            self.target_pos_cmd[mask2, 1] = torch.clamp(self.target_pos_cmd[mask2, 1], min=self.env_y_min[mask2], max=self.env_y_max[mask2])
            
            root_pos = self.robot.data.root_pos_w[mask2]
            self.target_pos_cmd[mask2, 2] = torch.clamp(self.target_pos_cmd[mask2, 2], min=root_pos[:, 2] + self.cfg.z_min)

            # -----------------------------------------------------------------
            # ✅ [26.03.28. 수정] 동적 자세 제어 (표적 USD 감지 여부에 따라 분기)
            # -----------------------------------------------------------------
            # 표적에 닿았는지 판별: Z축 접촉 힘이 2.0N 이상 & Cfg 기능 켜짐
            is_contact_meaningful = F_ext_z_all > 2.0
            
            # 조건에 맞는 마스크 분리
            adaptive_mask = mask2 & is_contact_meaningful & self.cfg.enable_adaptive_orientation
            vertical_mask = mask2 & ~adaptive_mask

            # Case A: 표적이 없거나 허공이거나 기능이 꺼졌을 때 -> 수직 강제 고정
            if vertical_mask.any():
                self.target_quat_cmd[vertical_mask] = self.vertical_quat_tensor.repeat(vertical_mask.sum(), 1)

            # Case B: 표적이 감지되고 닿아있을 때 -> 적응형 회전 제어 (기울기 보정)
            if adaptive_mask.any():
                K_rot = self.cfg.K_rot 
                delta_rx = -F_ext_3d_all[adaptive_mask, 1] * K_rot * self.dt 
                delta_ry =  F_ext_3d_all[adaptive_mask, 0] * K_rot * self.dt 
                
                delta_quat = torch.zeros_like(self.target_quat_cmd[adaptive_mask])
                delta_quat[:, 0] = 1.0
                delta_quat[:, 1] = delta_rx * 0.5
                delta_quat[:, 2] = delta_ry * 0.5
                delta_quat = delta_quat / torch.linalg.norm(delta_quat, dim=-1, keepdim=True)

                q1_w, q1_xyz = self.target_quat_cmd[adaptive_mask, 0:1], self.target_quat_cmd[adaptive_mask, 1:]
                q2_w, q2_xyz = delta_quat[:, 0:1], delta_quat[:, 1:]
                
                new_w = q1_w * q2_w - torch.sum(q1_xyz * q2_xyz, dim=-1, keepdim=True)
                new_xyz = q1_w * q2_xyz + q2_w * q1_xyz + torch.cross(q1_xyz, q2_xyz, dim=-1)
                
                self.target_quat_cmd[adaptive_mask] = torch.cat([new_w, new_xyz], dim=-1)

        # IK solve
        pose = torch.cat([self.target_pos_cmd, self.target_quat_cmd], dim=-1)
        self.ik_controller.set_command(pose)

        q_cmd = self.ik_controller.compute(ee_pos, ee_quat, jacobian, q)
        q_cmd = torch.where(torch.isnan(q_cmd), q, q_cmd)

        self.robot.set_joint_position_target(q_cmd)

# ==========================================
# Action Configuration
# ==========================================
@configclass
class AdmittanceControlActionCfg(ActionTermCfg):
    class_type: type = AdmittanceControlAction
    asset_name: str = "robot"
    body_name: str = "wrist_3_link" 
    
    # -----------------------------------------------------------
    # ✅ [26.03.28. 추가] 적응형 제어 스위치
    # USD 표적이 없을 때는 False, 나중에 추가하시면 True로 바꾸세요!
    # -----------------------------------------------------------
    enable_adaptive_orientation: bool = False 
    
    hdf5_file_path: str = "" 
    workspace_margin: float = 0.05
    
    vertical_quat: tuple = (0.0, 1.0, 0.0, 0.0) 
    target_force: float = 15.0
    M: float = 1.0
    D: float = 60.0
    K_rot: float = 0.005
    z_min: float = 0.1 
    stage1_timeout: float = 2.0

    workspace_x_limits: tuple = (0.2, 0.8) 
    workspace_y_limits: tuple = (-0.5, 0.5) 
    action_scale: float = 0.04  
    max_xy_vel: float = 0.20    # ✅ [26.03.28. 수정] 기존 0.06 -> 0.20 (초당 최대 20cm 이동 허용)