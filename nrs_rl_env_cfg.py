# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
import importlib

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    TerminationTermCfg as DoneTerm,
    EventTermCfg as EventTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp

local_obs = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.mdp.observation"
)
local_action = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.mdp.action"
)
local_terms = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.mdp.terminations"
)
local_ft_sensor = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.assets.assets.sensors.six_axis_ft_sensor"
)

from nrs_rl.tasks.manager_based.nrs_rl.assets.assets.robots.ur10e_w_spindle import (
    UR10E_W_SPINDLE_HIGH_PD_CFG,
)


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
            usd_path="/home/eunseop/isaac/isaac_save/surface/workpiece_standard.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    arm_action = local_action.AdmittanceControlActionCfg(
        asset_name="robot",
        body_name="spindle_link",
        hdf5_file_path="/home/eunseop/nrs_rl/source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/datasets/target_positions.h5",
        position_dataset_key="target_positions",
        action_dim=2,

        # IK
        dls_lambda=0.10,
        ik_step_size=0.60,
        max_dq=0.08,
        max_pos_err=0.05,
        max_rot_err=0.30,

        # waypoint follower
        waypoint_stride=100,
        waypoint_pos_tol=0.02,
        waypoint_rot_tol=0.20,
        max_steps_per_waypoint=120,

        # spindle / TCP compensation
        tcp_length_offset_m=0.10,
        tcp_offset_axis="local_z_neg",
        z_target_offset=0.0,

        # debug
        enable_debug_print=True,
        debug_print_interval=10,
        debug_env_id=0,
    )


@configclass
class ObservationCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        actions = ObsTerm(func=mdp.last_action)

        ee_pose = ObsTerm(
            func=local_obs.get_ee_pose,
            params={"asset_name": "robot"},
        )

        target_positions = ObsTerm(
            func=local_obs.get_hdf5_target_positions,
            params={"horizon": 5},
        )

        ft_6axis = ObsTerm(
            func=local_ft_sensor.get_6axis_ft_fixed_joint,
            params={
                "asset_name": "robot",
                "fixed_joint_name": "tool0_to_spindle",
                "joint_prim_relpath": "joints",
                "verbose": False,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class DebugCfg(ObsGroup):
        processed_polishing_target = ObsTerm(
            func=local_obs.get_processed_polishing_target,
            params={
                "asset_name": "robot",
                "body_name": "spindle_link",
                "fixed_joint_name": "tool0_to_spindle",
                "joint_prim_relpath": "joints",
                "contact_force_threshold": 10.0,
                "removal_gain": 0.001,
                "offset_axis": 2,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    debug: DebugCfg = DebugCfg()


@configclass
class EventCfg:
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )

    load_hdf5_positions = EventTerm(
        func=local_obs.load_hdf5_positions,
        mode="reset",
        params={
            "file_path": "/home/eunseop/nrs_rl/source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/datasets/target_positions.h5",
            "dataset_key": "target_positions",
        },
    )


@configclass
class RewardsCfg:
    pass


@configclass
class TerminationsCfg:
    trajectory_finished = DoneTerm(
        func=local_terms.trajectory_finished,
        params={"action_term_name": "arm_action"},
    )

    # 필요하면 안전장치로 매우 큰 timeout 하나 남겨도 됨
    # safety_time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class NrsRlEnvCfg(ManagerBasedRLEnvCfg):
    scene: SpindleSceneCfg = SpindleSceneCfg(num_envs=64, env_spacing=2.5)
    observations: ObservationCfg = ObservationCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.sim.render_interval = self.decimation

        # episode 종료는 h5 완료 기준
        self.episode_length_s = 9999.0

        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.dt = 1.0 / 60.0

        self.sim.physx.gpu_max_rigid_patch_count = 1024 * 1024 * 16
        self.sim.physx.gpu_max_rigid_contact_count = 1024 * 1024 * 16
        self.sim.physx.gpu_temp_buffer_capacity = 32 * 1024 * 1024
        self.sim.physx.gpu_collision_stack_size = 2**28
        self.sim.physx.gpu_found_lost_pairs_capacity = 1024 * 1024 * 16
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 16

        self.scene.robot = UR10E_W_SPINDLE_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )