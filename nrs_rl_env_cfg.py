# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause
#
from __future__ import annotations

from dataclasses import MISSING
from pathlib import Path
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
    UR10_W_SPINDLE_HIGH_PD_CFG,
)

HDF5_TRAJ_PATH = "/home/eunseop/nrs_rl/source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/datasets/cmd_continue9D_flat.h5"
#HDF5_TRAJ_PATH = "/home/eunseop/nrs_rl/source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/datasets/cmd_continue9D_convex_2.h5"
#HDF5_TRAJ_PATH = "/home/eunseop/nrs_rl/source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/datasets/cmd_continue9D_tail_lamp_proxy_run2.h5"
local_vis.configure_run_log_dir(HDF5_TRAJ_PATH)


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
            usd_path=str(Path(__file__).parent / "assets/assets/workpiece_8_training.usda"),
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
        integration=local_action.ActionIntegrationCfg(
            body_name="spindle_link",
            fixed_joint_name="tool0_to_spindle",
            joint_prim_relpath="joints",

            hdf5_file_path=HDF5_TRAJ_PATH,
            position_dataset_key="position",
            force_dataset_key="force",

            action_dim=1,

            # Direct learned feed-speed multiplier; constant baseline shares
            # the same 20 N Mode-3 controller and native speed limiter.
            nominal_speed_mm_s=6.0,
            residual_speed_fraction=0.50,
            target_normal_force_n=20.0,
            target_mrr_n_mm_s=120.0,
            force_rate_compensation=False,
            tool_diameter_mm=30.0,
            spindle_rpm=None,
            policy_signal_tau_s=0.08,
            min_speed_mm_s=1.0,
            max_speed_mm_s=12.0,
            action_filter_tau_s=0.08,
            action_slew_per_s=4.0,
            force_overload_ratio=1.6,
            tracking_stop_mm=10.0,
            contact_force_n=1.5,
            projection_window=200,
            surface_bins=256,
            approach_duration_s=2.0,
            force_scale_range=(0.90, 1.10),
            force_bias_range_n=(-0.75, 0.75),
            max_action_delay_steps=3,
            enable_debug_print=True,
            debug_print_interval=50,
            debug_env_id=0,
        ),
    )


@configclass
class ObservationCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        adaptive_velocity_state = ObsTerm(func=local_obs.adaptive_velocity_observation)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


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
    # Preston F*v at useful throughput is the requested training objective.
    # K is unknown and constant in this proxy, so it cancels in the rate CV.
    # Spatial accumulated depth/volume is independently evaluated on a fixed
    # surface ROI; temporal rate CV is not a physical depth-uniformity claim.
    # A throughput-only
    # reward favors the maximum speed, while a trend penalty misses slow drift.
    realized_removal = RewTerm(func=local_rewards.realized_removal_reward, weight=0.0)
    removal_rate_tracking = RewTerm(
        func=local_rewards.removal_rate_tracking_penalty, weight=4.0,
    )
    force_tracking = RewTerm(func=local_rewards.force_tracking_reward, weight=0.1)
    force_overshoot = RewTerm(func=local_rewards.force_overshoot_penalty, weight=0.5)
    # F*ds integrated in equal path bins is almost independent of feed speed.
    # It cannot supply credit for the requested temporal F*v uniformity.
    spatial_uniformity = RewTerm(func=local_rewards.spatial_uniformity_reward, weight=0.0)
    removal_variation = RewTerm(func=local_rewards.removal_variation_penalty, weight=0.0)
    # Penalizing consecutive raw Gaussian samples penalizes exploration even
    # when the applied command is smooth. Use the applied derivatives below.
    action_rate = RewTerm(func=local_rewards.action_rate_penalty, weight=0.0)
    command_acceleration = RewTerm(func=local_rewards.command_acceleration_penalty, weight=0.1)
    command_jerk = RewTerm(func=local_rewards.command_jerk_penalty, weight=0.05)
    safety_shield = RewTerm(func=local_rewards.safety_shield_penalty, weight=2.0)
    completion_quality = RewTerm(func=local_rewards.completion_rate_quality_reward, weight=1.0)

@configclass
class TerminationsCfg:
    control_failed = DoneTerm(func=local_terms.control_failed)
    polishing_timeout = DoneTerm(func=local_terms.polishing_timeout, time_out=True)
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
    scene: SpindleSceneCfg = SpindleSceneCfg(num_envs=16, env_spacing=2.5)
    observations: ObservationCfg = ObservationCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    visualization: VisualizationCfg = VisualizationCfg()

    def __post_init__(self):
        # Resolve stiff contact at 500 Hz while holding UR10 CB3 targets for
        # four substeps. The action term executes Mode 3 / IK only at 125 Hz.
        self.decimation = 4
        self.sim.render_interval = self.decimation

        self.episode_length_s = 9999.0

        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.dt = 1.0 / 500.0

        self.sim.physx.gpu_max_rigid_patch_count = 1024 * 1024 * 16
        self.sim.physx.gpu_max_rigid_contact_count = 1024 * 1024 * 16
        self.sim.physx.gpu_temp_buffer_capacity = 32 * 1024 * 1024
        self.sim.physx.gpu_collision_stack_size = 2**28
        self.sim.physx.gpu_found_lost_pairs_capacity = 1024 * 1024 * 16
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 16

        self.scene.robot = UR10_W_SPINDLE_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
