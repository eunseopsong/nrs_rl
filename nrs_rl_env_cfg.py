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
            force_recovery_tau=3.0,
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

            action_dim=2,

            base_index_rate=10.0,
            min_index_rate=3.0,
            max_index_rate=16.0,
            progress_rate_ema_beta=0.3,
            force_eps_n=1.0,

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

    # ── 1. 가공량(MRR) 맞추기 ─────────────────────────────────────────
    # 가장 중요한 보상. sigma가 에피소드마다 좁아져 후기 ep이 더 정밀해야 보상.
    # [수정] mrr_sigma → mrr_sigma_start / mrr_sigma_end / curriculum_ramp
    adaptive_mrr_reward = RewardTermCfg(
        func=custom_rewards.adaptive_mrr_reward,
        weight=5.0,
        params={
            "target_mrr": 0.1,
            "mrr_sigma_start": 1.2,    # 초기: 넓은 허용 → 탐색 장려
            "mrr_sigma_end": 0.3,      # 후기: 좁은 허용 → 정밀 제어 강제
            "curriculum_ramp": 200,    # 200 에피소드에 걸쳐 sigma 축소
            "min_contact_force": 1.0,
            "min_velocity": 1e-3,
            "gate_sharpness": 8.0,     # soft gate 선명도
        },
    )

    # ── 2. 반비례 관계 학습 보너스 ────────────────────────────────────
    # 힘 강하면 속도 낮추는 역비례 패턴을 직접 가르침.
    # 05_force_vel_correlation.png에서 쌍곡선 정렬로 시각화됨.
    # [수정] bonus_sigma → bonus_sigma_start / bonus_sigma_end / curriculum_ramp
    inverse_fv_bonus = RewardTermCfg(
        func=custom_rewards.inverse_fv_bonus,
        weight=2.0,
        params={
            "target_mrr": 0.1,
            "bonus_sigma_start": 0.8,  # 초기: 넓게
            "bonus_sigma_end": 0.25,   # 후기: 좁게
            "curriculum_ramp": 200,
            "min_contact_force": 1.0,
            "gate_sharpness": 8.0,
        },
    )

    # ── 3. 경로 추종 ──────────────────────────────────────────────────
    # 목표점에 가까우면 보상 + 움직이고 있으면 추가 보너스.
    # vel_bonus_scale=0.3: 속도 유도는 보조 역할 (멈춤 방지)
    trajectory_tracking_reward = RewardTermCfg(
        func=custom_rewards.trajectory_tracking_reward,
        weight=3.0,
        params={
            "pos_sigma": 0.05,
            "vel_bonus_scale": 0.3,       # 움직임 유도 가중치
            "min_tracking_velocity": 5e-4,
            "gate_sharpness": 6.0,
        },
    )

    # ── 4. 액션 스무스니스 페널티 ─────────────────────────────────────
    # rewards.py 내부에서 이미 음수(-) 반환 → weight는 양수로 유지
    action_smoothness_penalty = RewardTermCfg(
        func=custom_rewards.action_smoothness_penalty,
        weight=0.2,
    )

    # ── 5. 물리 엔진 보호용 하드 리밋 ────────────────────────────────
    machining_safety_penalty = RewardTermCfg(
        func=custom_rewards.machining_safety_penalty,
        weight=1.0,
        params={
            "max_force": 50.0,
        },
    )

    surface_uniformity_reward = RewardTermCfg(
        func=custom_rewards.surface_uniformity_reward,
        weight=4.0,
    )
    
    force_stability_reward = RewardTermCfg(
        func=custom_rewards.force_stability_reward,
        weight=1.5,
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