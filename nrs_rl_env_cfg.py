# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
import importlib

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import RewardTermCfg
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
import nrs_rl.tasks.manager_based.nrs_rl.mdp.rewards as custom_rewards
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
local_vis = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.utils.visualization"
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
            usd_path="/home/eunseop/isaac/isaac_save/workpiece_8_v2.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    arm_action = local_action.AdmittanceControlActionCfg(
        class_type=local_action.AdmittanceControlAction,
        asset_name="robot",
        original_forcecon=local_action.OriginalControllerForceConCfg(
            force_md_ratio=1000.0,
            force_fc_fext=50.0,
            force_free_mass=2.0,
            force_free_damping=6000.0,
            force_free_stiffness=2000.0,
            force_contact_stiffness=0.0,
            force_recovery_tau=0.2,
            force_action_low=(-0.25, -0.25),
            force_action_high=(0.25, 0.25),
            force_mass_min=0.5,
            force_mass_max=5.0,
            force_alpha_min=0.5,
            force_alpha_max=3.0,
            force_alpha_rate_up=4.0,
            force_alpha_rate_down=4.0,
        ),
        integration=local_action.ActionIntegrationCfg(
            body_name="spindle_link",
            fixed_joint_name="tool0_to_spindle",
            joint_prim_relpath="joints",

            hdf5_file_path=HDF5_TRAJ_PATH,
            position_dataset_key="position",
            force_dataset_key="force",

            action_dim=1,

            target_mrr_n_mm_s=400.0,
            speed_action_scale=0.12,
            base_index_rate=30.0,
            min_index_rate=1.0,
            max_index_rate=96.0,
            progress_rate_ema_beta=0.55,
            command_rate_ema_beta=0.10,
            command_rate_max_delta_up=2.0,
            command_rate_max_delta_down=2.0,
            command_velocity_ema_beta=0.30,
            command_velocity_max_delta_up_mm_s=3.0,
            command_velocity_max_delta_down_mm_s=3.0,
            command_velocity_spike_delta_mm_s=3.0,
            command_velocity_spike_return_ratio=0.10,
            command_velocity_hard_stop_decay=0.98,
            command_velocity_max_mm_s=46.0,
            force_filter_beta=0.04,
            force_spike_delta_n=0.35,
            force_spike_hold_steps=20,
            force_spike_velocity_decay=0.98,
            force_velocity_compensation=1.0,
            command_mrr_ema_beta=0.30,
            command_mrr_max_delta_up_n_mm_s=55.0,
            command_mrr_max_delta_down_n_mm_s=55.0,
            command_mrr_min_ratio=0.75,
            command_mrr_max_ratio=1.05,
            force_eps_n=1.0,
            force_tracking_ready_ratio=0.8,
            min_force_rate_scale=0.25,
            force_error_slowdown_ratio=0.35,
            min_force_error_rate_scale=0.35,
            force_normal_kp_mm_per_n=0.18,
            force_normal_release_kp_mm_per_n=0.65,
            force_normal_ki_mm_per_n_s=3.50,
            force_normal_max_step_mm=2.50,
            force_normal_retract_max_step_mm=2.00,
            force_normal_offset_limit_mm=60.0,
            force_total_normal_delta_limit_mm=60.0,
            force_normal_deadband_n=0.30,
            force_band_min_n=9.0,
            force_band_max_n=11.0,
            force_band_index_rate_limit=6.0,
            force_band_saturated_min_n=9.0,
            force_band_low_speed_scale=0.25,
            force_band_high_speed_scale=0.25,
            force_band_hold_progress=False,
            path_tracking_slowdown_start_mm=2.0,
            path_tracking_stop_mm=8.0,
            path_tracking_min_rate_scale=0.0,
            path_projection_window=160,
            path_projection_max_advance_index=0.0,
            path_lookahead_min_index=0.0,
            path_lookahead_max_index=8.0,
            path_lookahead_time_s=0.015,
            path_command_max_xy_step_mm=0.0,
            path_command_max_z_step_mm=8.0,
            approach_interpolation_enabled=True,
            approach_duration_s=2.0,

            enable_debug_print=True,
            debug_print_interval=50,
            debug_env_id=0,
        ),
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

        visualization_step = ObsTerm(
            func=local_vis.rl_step_hook,
            params={
                "action_term_name": "arm_action",
                "asset_name": "robot",
                "fixed_joint_name": "tool0_to_spindle",
                "joint_prim_relpath": "joints",
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

    load_hdf5_trajectory = EventTerm(
        func=local_obs.load_hdf5_trajectory,
        mode="reset",
        params={
            "file_path": HDF5_TRAJ_PATH,
            "position_dataset_key": "position",
            "force_dataset_key": "force",
        },
    )

    finalize_visualization_episode = EventTerm(
        func=local_vis.on_episode_reset,
        mode="reset",
        params={},
        # ⚠️ visualization.py의 on_episode_reset(env, env_ids)가
        # env를 첫 번째 인자로 자동 수신합니다.
        # params={}로 두면 Isaac Lab이 env를 자동으로 넘겨줍니다.
    )


@configclass
class RewardsCfg:
    """Reward terms for the RL environment."""

    removal_rate_reward = RewardTermCfg(
        func=custom_rewards.removal_rate_reward,
        weight=2.0,
        params={
            "saturation_mrr": 500.0,
            "min_contact_force": 1.0,
        },
    )

    removal_constancy_reward = RewardTermCfg(
        func=custom_rewards.removal_constancy_reward,
        weight=4.0,
        params={
            "delta_tau": 25.0,
            "min_contact_force": 1.0,
            "min_active_mrr": 80.0,
        },
    )

    removal_instability_penalty = RewardTermCfg(
        func=custom_rewards.removal_instability_penalty,
        weight=20.0,
        params={
            "spike_delta": 35.0,
            "spike_tau": 20.0,
            "dip_mrr": 220.0,
            "dip_tau": 35.0,
            "dip_weight": 2.0,
            "min_contact_force": 1.0,
            "min_prev_mrr": 180.0,
        },
    )

    surface_uniformity_reward = RewardTermCfg(
        func=custom_rewards.surface_uniformity_reward,
        weight=8.0,
        params={
            "cv_tau": 0.35,
            "total_tau": 5500.0,
        },
    )

@configclass
class TerminationsCfg:
    trajectory_finished = DoneTerm(
        func=local_terms.trajectory_finished,
    )


@configclass
class VisualizationCfg:
    enable_visualizer: bool = True
    save_interval_episodes: int = 1
    force_threshold: float = 0.5
    speed_threshold: float = 0.1


@configclass
class NrsRlEnvCfg(ManagerBasedRLEnvCfg):
    scene: SpindleSceneCfg = SpindleSceneCfg(num_envs=64, env_spacing=2.5)
    observations: ObservationCfg = ObservationCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    visualization: VisualizationCfg = VisualizationCfg()

    def __post_init__(self):
        self.decimation = 1
        self.sim.render_interval = self.decimation

        self.episode_length_s = 9999.0

        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.dt = 1.0 / 125.0

        self.sim.physx.gpu_max_rigid_patch_count = 1024 * 1024 * 16
        self.sim.physx.gpu_max_rigid_contact_count = 1024 * 1024 * 16
        self.sim.physx.gpu_temp_buffer_capacity = 32 * 1024 * 1024
        self.sim.physx.gpu_collision_stack_size = 2**28
        self.sim.physx.gpu_found_lost_pairs_capacity = 1024 * 1024 * 16
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 16

        self.scene.robot = UR10E_W_SPINDLE_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
