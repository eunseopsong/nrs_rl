# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------
# Title: NRS_RL (UR10e + Spindle) Environment Config
# Author: Seungjun Song (NRS Lab)
# -----------------------------------------------------------------------------
"""
Manager-based Isaac Lab environment for the UR10e robot equipped with a spindle tool.

Key features:
- Horizon-based joint & position trajectory tracking
- Exponential-shaped reward for position tracking
- End-effector 6D pose observation via FKSolver
- Contact/camera sensor integration (optional)

NOTE:
- This file is placed in:
  nrs_rl/tasks/manager_based/nrs_rl/nrs_rl_env_cfg.py
- We keep ONE task ("Template-Nrs-Rl-v0") and just swap its env cfg to UR10e spindle.
"""

from __future__ import annotations

from dataclasses import MISSING
import importlib
import isaaclab.sim as sim_utils

from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    ActionTermCfg as ActionTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    EventTermCfg as EventTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import ContactSensorCfg, CameraCfg

# Reach manipulation utilities
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp

# [26/03/05. 추가] 접촉 센서에서 힘 데이터를 가져오는 유틸 함수
from pxr import UsdPhysics
import omni.usd

# -----------------------------------------------------------------------------
# Local modules (dynamic import)  ✅ nrs_lab2 -> nrs_rl 로 수정
# -----------------------------------------------------------------------------
local_obs = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.mdp.observations"
)
local_rewards = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.mdp.rewards"
)
local_action = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.mdp.action"
)

# -----------------------------------------------------------------------------
# Robot asset
# -----------------------------------------------------------------------------
from nrs_rl.tasks.manager_based.nrs_rl.assets.assets.robots.ur10e_w_spindle import UR10E_W_SPINDLE_CFG


# -----------------------------------------------------------------------------
# Scene Configuration
# -----------------------------------------------------------------------------
@configclass
class SpindleSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    robot: AssetBaseCfg = MISSING

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

    # workpiece = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Workpiece",
    #     spawn=sim_utils.UsdFileCfg(
    #         # ✅ [26.03.24. Fixed by eunseop; Change the workpiece USD]
    #         usd_path="/home/eunseop/isaac/isaac_save/surface/surface_assembly.usd",
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(
    #         pos=(0.0, 0.0, 0.0),
    #         rot=(1.0, 0.0, 0.0, 0.0),
    #     ),
    # )

    # -----------------------------------------------------------
    # ✅ [26.03.01. 추가] 힘 제어를 위한 접촉 센서 활성화
    # -----------------------------------------------------------
    # ✅ [26.03.05. Fixed by eunseop]
    # -----------------------------------------------------------
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Robot/wrist_3_link",
        update_period=0.0,
        history_length=3,
        track_air_time=False,
    )


# -----------------------------------------------------------------------------
# Actions ✅ [26.03.25. 추가] 정의한 설정 클래스를 바로 호출)
# -----------------------------------------------------------------------------
@configclass
class ActionsCfg:
    # -----------------------------------------------------------
    # ✅ [26.03.28. 수정] HDF5 경로 파일을 읽어 동적 워크스페이스 설정
    # -----------------------------------------------------------
    arm_action = local_action.AdmittanceControlActionCfg(
        asset_name="robot",
        hdf5_file_path="/home/eunseop/nrs_rl/source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/datasets/flat_g_recording.h5",
        workspace_margin=0.05
    )

# -----------------------------------------------------------------------------
# Observations
# -----------------------------------------------------------------------------
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 기본 joint 관측
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        actions = ObsTerm(func=mdp.last_action)

        # EE pose (x,y,z,r,p,yaw)
        ee_pose = ObsTerm(
            func=local_obs.get_ee_pose,
            params={"asset_name": "robot"},
        )

        # ✅ [추가] 에이전트가 현재 접촉하고 있는 힘의 크기를 알 수 있도록 관측값에 추가
        contact_forces = ObsTerm(
            func=local_obs.get_contact_forces,
            params={"sensor_name": "contact_forces"},
        )

        # HDF5 horizon trajectory
        target_joints = ObsTerm(
            func=local_obs.get_hdf5_target_joints,
            params={"horizon": 5},
        )
        target_positions = ObsTerm(
            func=local_obs.get_hdf5_target_positions,
            params={"horizon": 5},
        )

        def __post_init__(self):#adk
            self.enable_corruption = True
            self.concatenate_terms = True  # dict -> tensor
    policy: PolicyCfg = PolicyCfg()


# -----------------------------------------------------------------------------
# Events
# -----------------------------------------------------------------------------
@configclass
class EventCfg:
    """Episode 시작 시 trajectory (joints / positions)를 로드"""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )

    # Joint trajectory 로드
    load_hdf5_joints = EventTerm(
        func=local_obs.load_hdf5_joints,
        mode="reset",
        params={
            "file_path": "/home/eunseop/nrs_rl/source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/datasets/joint_recording_filtered.h5",
            "dataset_key": "target_joints",
        },
    )

    # Position trajectory 로드
    load_hdf5_positions = EventTerm(
        func=local_obs.load_hdf5_positions,
        mode="reset",
        params={
            "file_path": "/home/eunseop/nrs_rl/source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/datasets/flat_g_recording.h5",
            "dataset_key": "target_positions",
        },
    )


# -----------------------------------------------------------------------------
# Rewards
# -----------------------------------------------------------------------------
@configclass
class RewardsCfg:
    # -----------------------------------------------------------
    # ✅ [26.03.28. 수정] 기존의 개별 위치/수직 보상을 지우고, 통합 퀄리티 보상으로 대체!
    # 위치 오차 + 수직(Roll/Pitch) 유지 + 접촉 여부를 동시에 평가합니다.
    # -----------------------------------------------------------
    perfect_polishing_quality = RewTerm(
        func=local_rewards.perfect_polishing_quality_reward,
        weight=10.0, # 에이전트가 가장 최우선으로 달성해야 할 핵심 목표
    )

    # 2. 기존 힘 제어 보상 (15N의 목표 압력을 일정하게 유지하는 세밀한 튜닝용)
    force_tracking_reward = RewTerm(
        func=local_rewards.force_tracking_reward,
        weight=2.0,
        params={"target_force": 15.0}
    )

    # 3. 기존 부드러운 움직임 페널티 (덜덜거림 방지)
    action_smoothness = RewTerm(
        func=local_rewards.action_smoothness_penalty,
        weight=-0.1,
    )

    # 4. [26.03.08. 추가] 표면 이탈 시 강력한 감점
    # weight를 50.0으로 주어 경로를 벗어나면 총 점수가 확 깎이게 설정.
    off_surface = RewTerm(
        func=local_rewards.off_surface_penalty, 
        weight=50.0
    )   

# -----------------------------------------------------------------------------
# Terminations
# -----------------------------------------------------------------------------
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


# -----------------------------------------------------------------------------
# Environment Configuration
# -----------------------------------------------------------------------------
@configclass
class NrsRlEnvCfg(ManagerBasedRLEnvCfg):
    """Template-Nrs-Rl-v0 가 이 cfg를 가리키면 UR10e spindle env가 됨."""
    # ✅ [26.03.05. Fixed by eunseop; Reduce the number of environments for preventing GPU OOM]
    scene: SpindleSceneCfg = SpindleSceneCfg(num_envs=128, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):

        # sim / episode
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 30.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.dt = 1.0 / 30.0

        # ==========================================================
        # ✅ [수정됨] PhysX GPU 버퍼 강제 확장 (메모리 폭발 에러 완전 해결)
        # ==========================================================
        self.sim.physx.gpu_max_rigid_patch_count = 1024 * 1024 * 16
        self.sim.physx.gpu_max_rigid_contact_count = 1024 * 1024 * 16
        self.sim.physx.gpu_temp_buffer_capacity = 32 * 1024 * 1024
        
        # 🚨 [새로 추가된 핵심 코드] 에러에서 178M 이상을 요구한 충돌 스택 메모리를 약 268MB(2^28)로 대폭 늘림
        self.sim.physx.gpu_collision_stack_size = 2**28 
        self.sim.physx.gpu_found_lost_pairs_capacity = 1024 * 1024 * 16
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 16
        # ==========================================================

        # robot
        self.scene.robot = UR10E_W_SPINDLE_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )