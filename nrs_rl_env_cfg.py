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


# -----------------------------------------------------------------------------
# Local modules (dynamic import)  ✅ nrs_lab2 -> nrs_rl 로 수정
# -----------------------------------------------------------------------------
# 여기서 말하는 "local_obs/local_rewards"는 아래 경로에 존재해야 함:
#   nrs_rl/tasks/manager_based/nrs_rl/mdp/observations.py
#   nrs_rl/tasks/manager_based/nrs_rl/mdp/rewards.py
local_obs = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.mdp.observations"
)
local_rewards = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.mdp.rewards"
)


# -----------------------------------------------------------------------------
# Robot asset
# -----------------------------------------------------------------------------
# ✅ 이 import가 되려면 nrs_rl 실행 환경에서 `assets` 패키지가 PYTHONPATH에 잡혀 있어야 함.
# (nrs_lab2에서 쓰던 그대로 유지)
# from nrs_rl.assets.assets.robots.ur10e_w_spindle import UR10E_W_SPINDLE_CFG
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
            usd_path="/home/eunseop/isaac/isaac_save/concave_surface.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # (옵션) contact sensor / camera sensor를 쓰려면 여기서 scene에 붙이면 됨.
    # 지금은 네 코드 그대로 "import만" 해둔 상태라,
    # 실제로 쓸 때만 아래 예시처럼 추가하면 됨.
    #
    # contact_forces = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/**/spindle_contact_sensor",
    #     update_period=0.0,
    #     history_length=1,
    # )
    #
    # camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/**/camera",
    #     update_period=0.0,
    #     height=240,
    #     width=320,
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

        # HDF5 horizon trajectory
        target_joints = ObsTerm(
            func=local_obs.get_hdf5_target_joints,
            params={"horizon": 5},
        )
        target_positions = ObsTerm(
            func=local_obs.get_hdf5_target_positions,
            params={"horizon": 5},
        )

        def __post_init__(self):
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
            # ✅ nrs_lab2 -> nrs_rl 로 바꾸는 게 맞으면 여기 경로도 바꿔
            #    (데이터는 그대로 nrs_lab2 밑에 있을 수도 있으니 일단 원본 유지)
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
    # joint_tracking_reward = RewTerm(
    #     func=local_rewards.joint_tracking_reward,
    #     weight=1.0,
    # )

    position_tracking_reward = RewTerm(
        func=local_rewards.position_tracking_reward,
        weight=1.0,
    )


# -----------------------------------------------------------------------------
# Terminations
# -----------------------------------------------------------------------------
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


# -----------------------------------------------------------------------------
# Environment Configuration  ✅ (기존 Template task가 이 cfg를 바라보게 됨)
# -----------------------------------------------------------------------------
@configclass
class NrsRlEnvCfg(ManagerBasedRLEnvCfg):
    """Template-Nrs-Rl-v0 가 이 cfg를 가리키면 UR10e spindle env가 됨."""
    scene: SpindleSceneCfg = SpindleSceneCfg(num_envs=1024, env_spacing=2.5)
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

        # robot
        self.scene.robot = UR10E_W_SPINDLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # action
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=0.2,
            use_default_offset=True,
        )
