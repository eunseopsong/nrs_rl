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
    RewardTermCfg as RewTerm,
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
local_rewards = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.mdp.rewards"
)

from nrs_rl.tasks.manager_based.nrs_rl.assets.assets.robots.ur10e_w_spindle import (
    UR10E_W_SPINDLE_HIGH_PD_CFG,
)

HDF5_TRAJ_PATH = "/home/eunseop/nrs_rl/source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/datasets/cmd_continue9D.h5"


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
            usd_path="/home/eunseop/isaac/isaac_save/surface/workpiece_standard_8.usd",
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

        hdf5_file_path=HDF5_TRAJ_PATH,
        position_dataset_key="position",
        force_dataset_key="force",
        position_scale=0.001,   # txt/h5 xyz is mm -> m

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

        # FT source
        fixed_joint_name="tool0_to_spindle",
        joint_prim_relpath="joints",

        # force control axis
        force_axis="z",

        # Mode5 force controller
        force_model_path="/home/eunseop/nrs_rl/source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/y2_control_pybind/checkpoints/ContextNAF_MDGradi/contextNAF_mdGradi_policy_script.pt",
        force_dt=0.002,
        force_threads=1,
        force_device="cpu",
        force_md_ratio=1000.0,
        force_fc_fext=50.0,

        force_free_mass=2.0,
        force_free_damping=6000.0,
        force_free_stiffness=2000.0,
        force_contact_stiffness=0.0,
        force_recovery_tau=3.0,

        force_action_low=(-0.25, -0.25),
        force_action_high=(0.25, 0.25),

        force_mass_min=0.5,
        force_mass_max=5.0,
        force_alpha_min=0.5,
        force_alpha_max=3.0,
        force_alpha_rate_up=4.0,
        force_alpha_rate_down=4.0,

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

        target_forces = ObsTerm(
            func=local_obs.get_hdf5_target_forces,
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
            "file_path": HDF5_TRAJ_PATH,
            "position_dataset_key": "position",
            "force_dataset_key": "force",
        },
    )


@configclass
class RewardsCfg:
    uniform_mrr = RewTerm(
        func=local_rewards.uniform_mrr_reward,
        weight=2.0,
        params={
            "target_force": 20.0,
            "target_velocity": 0.0002,
            "mrr_sigma": 0.002,
            "asset_name": "robot",
            "body_name": "spindle_link",
            "fixed_joint_name": "tool0_to_spindle",
            "joint_prim_relpath": "joints",
        },
    )

    force_track = RewTerm(
        func=local_rewards.force_tracking_reward,
        weight=1.0,
        params={
            "target_force": 20.0,
            "force_sigma": 5.0,
            "asset_name": "robot",
            "fixed_joint_name": "tool0_to_spindle",
            "joint_prim_relpath": "joints",
        },
    )

    cornering = RewTerm(
        func=local_rewards.lookahead_cornering_penalty,
        weight=0.3,
        params={
            "cornering_threshold_angle": 0.5,
            "penalty_scale": 0.5,
            "lookahead_steps": 5,
            "speed_ref": 0.002,
            "action_rate_scale": 0.1,
            "asset_name": "robot",
            "body_name": "spindle_link",
        },
    )

    traj_track = RewTerm(
        func=local_rewards.trajectory_tracking_penalty,
        weight=1.0,
        params={
            "pos_sigma": 0.03,
            "rot_sigma": 0.20,
            "asset_name": "robot",
            "body_name": "spindle_link",
        },
    )

    action_smooth = RewTerm(
        func=local_rewards.action_smoothness_penalty,
        weight=0.05,
    )


@configclass
class TerminationsCfg:
    trajectory_finished = DoneTerm(
        func=local_terms.trajectory_finished,
        params={"action_term_name": "arm_action"},
    )


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