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

    workpiece = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workpiece",
        spawn=sim_utils.UsdFileCfg(
            # ✅ [26.03.05. Fixed by eunseop; Add the collider model to the workpiece USD]
            usd_path="/home/eunseop/isaac/isaac_save/surface/flat_surface_w_collider.usd",
            # usd_path="/home/eunseop/isaac/isaac_save/surface/concave_surface_w_collider.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # -----------------------------------------------------------
    # ✅ [26.03.01. 추가] 힘 제어를 위한 접촉 센서 활성화
    # -----------------------------------------------------------
    # 접촉 센서
    # -----------------------------------------------------------
    # contact_forces = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/.*wrist_3_link",
    #     update_period=0.0,
    #     history_length=3,
    #     track_air_time=False,
    # )
    # -----------------------------------------------------------
    # ✅ [26.03.05. Fixed by eunseop]
    # -----------------------------------------------------------
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Robot/wrist_3_link",
        update_period=0.0,
        history_length=3,
        track_air_time=False,
    )
    # -----------------------------------------------------------
    # ✅ [26.03.08 . 추가] 지형 파악을 위한 Depth & Normal 카메라 센서를 로봇의 손목에 추가
    # 표면까지의 거리 및 굴곡 정보를 스캔
    # -----------------------------------------------------------

    # camera = CameraCfg(
    #     # 손목 링크의 하위 트리로 카메라를 생성 (경로는 USD에 맞게 조정 필요)
    #     prim_path="{ENV_REGEX_NS}/Robot/Robot/wrist_3_link/camera",
    #     update_period=0.0,
    #     height=64,  # 연산량을 줄이기 위해 해상도를 작게 설정 (평균값만 쓸 것이므로 충분함)
    #     width=64,         # observations.py에서 요구하는 데이터 타입 활성화
    #     data_types=["distance_to_image_plane", "normals"], 
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 10.0)
    #     ),

    #     # 카메라가 툴 끝(Workpiece 쪽)을 바라보도록 위치와 회전(Offset) 설정
    #     # (주의: 실제 로봇 USD 축 방향에 따라 rot 값을 조절해야 할 수 있습니다)

    #     offset=CameraCfg.OffsetCfg(

    #         pos=(0.0, 0.0, 0.1), # Z축으로 10cm 앞쪽

    #         rot=(1.0, 0.0, 0.0, 0.0), # Quaternion (w, x, y, z)

    #         convention="ros"

    #     ),

    # )


# -----------------------------------------------------------------------------
# Actions
# -----------------------------------------------------------------------------
@configclass
class ActionsCfg:
    arm_action: ActionTerm = MISSING


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
        # -----------------------------------------------------------
        # ✅ [26.03.08. 추가] 카메라를 통한 지형 인지 관측 (적응형 제어용)
        # -----------------------------------------------------------
        # 1. 툴과 표면 사이의 거리 (움푹 패였는지, 튀어나왔는지 파악)
        # camera_distance = ObsTerm(
        #     func=local_obs.get_camera_distance,
        #     params={"sensor_name": "camera", "debug_interval": 100}, # 100스텝마다 콘솔에 거리 출력
        # )

        # # 2. 표면의 법선 벡터 (평평한지, 경사가 졌는지 파악)
        # camera_normals = ObsTerm(
        #     func=local_obs.get_camera_normals,
        #     params={"sensor_name": "camera"},
        # )
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
            "file_path": "/home/eunseop/nrs_lab2/datasets/joint_recording_filtered.h5",
            "dataset_key": "target_joints",
        },
    )

    # Position trajectory 로드
    load_hdf5_positions = EventTerm(
        func=local_obs.load_hdf5_positions,
        mode="reset",
        params={
            "file_path": "/home/eunseop/nrs_lab2/datasets/hand_g_recording.h5",
            "dataset_key": "target_positions",
        },
    )


# -----------------------------------------------------------------------------
# Rewards
# -----------------------------------------------------------------------------
@configclass
class RewardsCfg:
    # 1. 기존 위치 추종 보상
    position_tracking_reward = RewTerm(
        func=local_rewards.position_tracking_reward,
        weight=1.0,
    )

    # 2. 기존 힘 제어 보상
    force_tracking_reward = RewTerm(
        func=local_rewards.force_tracking_reward,
        weight=2.0,
        params={"target_force": 15.0}
    )

    # 3. 기존 부드러운 움직임 페널티
    action_smoothness = RewTerm(
        func=local_rewards.action_smoothness_penalty,
        weight=-0.1,
    )

    # 4. [26.03.08. 추가] 표면 이탈 시 강력한 감점
    # weight를 50.0으로 주어 경로를 벗어나면 총 점수가 확 깎이게 설정했습니다.
    off_surface = RewTerm(
        func=local_rewards.off_surface_penalty, 
        weight=50.0
    )   

    # 5. [26.03.08. 추가] 수직 유지 시 가산점
    # 스핀들이 표면과 90도를 잘 유지할수록 추가 점수를 줍니다.
    perpendicular_align = RewTerm(
        func=local_rewards.perpendicular_alignment_reward, 
        weight=10.0
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

        # ✅ [26.02.24. 추가] PhysX GPU 버퍼 강제 확장 (Patch buffer overflow 에러 해결)
        self.sim.physx.gpu_max_rigid_patch_count = 1024 * 1024
        self.sim.physx.gpu_max_rigid_contact_count = 2048 * 1024
        self.sim.physx.gpu_temp_buffer_capacity = 32 * 1024 * 1024

        # robot
        self.scene.robot = UR10E_W_SPINDLE_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        # action
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=0.2,
            use_default_offset=True,
        )