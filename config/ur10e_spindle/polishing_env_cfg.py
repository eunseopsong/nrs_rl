# SPDX-License-Identifier: BSD-3-Clause
"""
UR10e + Spindle: Joint-Hold / Pose-Hold 환경 설정 (trajectory 로드)
- Loads joints / positions from HDF5 via reset events
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm

# ✅ base env: 기존 nrs_rl task 환경 설정을 그대로 상속
from nrs_rl.tasks.manager_based.nrs_rl.nrs_rl_env_cfg import NrsRlEnvCfg

# ✅ local mdp observations (nrs_rl 경로)
from nrs_rl.tasks.manager_based.nrs_rl.mdp import observations as local_obs


@configclass
class PolishingPoseHoldEnvCfg(NrsRlEnvCfg):
    """Pose trajectory (6D)만 로드해서 홀드/추종하는 설정"""

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------------------------------------
        # Simulation settings
        # ------------------------------------------------------------
        self.sim.dt = 1.0 / 60.0
        self.sim.physics_dt = 1.0 / 60.0
        self.sim.substeps = 1
        self.sim.use_gpu_pipeline = True

        # ------------------------------------------------------------
        # Env settings
        # ------------------------------------------------------------
        self.actions.arm_action.scale = 0.2
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 60.0

        # ------------------------------------------------------------
        # HDF5 trajectory loader (positions)
        # ------------------------------------------------------------
        # NOTE: NrsRlEnvCfg에서 events가 이미 존재한다고 가정.
        # 없으면 NrsRlEnvCfg 쪽에 events: EventCfg = EventCfg() 들어가 있어야 함.
        self.events.load_hdf5_positions = EventTerm(
            func=local_obs.load_hdf5_positions,
            mode="reset",
            params={
                "file_path": "/home/eunseop/nrs_lab2/datasets/hand_g_recording.h5",
                "dataset_key": "target_positions",
            },
        )

        # ------------------------------------------------------------
        # (Optional) joints trajectory loader
        # ------------------------------------------------------------
        # self.events.load_hdf5_joints = EventTerm(
        #     func=local_obs.load_hdf5_joints,
        #     mode="reset",
        #     params={
        #         "file_path": "/home/eunseop/nrs_lab2/datasets/joint_recording_filtered.h5",
        #         "dataset_key": "target_joints",
        #     },
        # )


@configclass
class PolishingPoseHoldEnvCfg_PLAY(PolishingPoseHoldEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

