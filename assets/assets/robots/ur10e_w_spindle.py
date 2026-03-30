# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for UR10e with spindle tool (local USD scene).

- Loads local USD:
    /home/eunseop/isaac/isaac_save/ur10e_only_v2.usd
- Uses the articulation defined inside the USD
- End-effector frame for downstream control:
    spindle_link

USD structure (updated)
-----------------------
Expected arm joint chain:
    shoulder_pan_joint
    shoulder_lift_joint
    elbow_joint
    wrist_1_joint
    wrist_2_joint
    wrist_3_joint

Expected link chain near the end-effector:
    wrist_3_link
      └── tool0_to_spindle (fixed joint)
            └── spindle_link

Notes
-----
* Since the USD now explicitly includes `spindle_link`, downstream action /
  observation / IK code should use `spindle_link` as the EE frame instead of
  `wrist_3_link`.
* The actuator still controls only the 6 UR10e arm joints.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# -----------------------------------------------------------------------------
# User paths and EE frame
# -----------------------------------------------------------------------------
UR10E_USD_PATH = "/home/eunseop/isaac/isaac_save/ur10e_only_v2.usd"
EE_FRAME_NAME = "spindle_link"

# -----------------------------------------------------------------------------
# Home pose (rad)
# -----------------------------------------------------------------------------
UR10E_HOME_DICT = {
    "shoulder_pan_joint": -0.5939,
    "shoulder_lift_joint": -1.2795,
    "elbow_joint": -2.2452,
    "wrist_1_joint": -1.1892,
    "wrist_2_joint": 1.5708,
    "wrist_3_joint": 0.1915,
}
# v20 baseline

# -----------------------------------------------------------------------------
# Controlled arm joints (must exactly match USD joint names)
# -----------------------------------------------------------------------------
UR10E_ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# -----------------------------------------------------------------------------
# Base articulation configuration
# -----------------------------------------------------------------------------
UR10E_W_SPINDLE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=UR10E_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        # Enable if needed later for more stable contact tuning
        # collision_props=sim_utils.CollisionPropertiesCfg(
        #     contact_offset=0.005,
        #     rest_offset=0.0,
        # ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos=UR10E_HOME_DICT,
    ),
    actuators={
        "ur10e_arm": ImplicitActuatorCfg(
            joint_names_expr=UR10E_ARM_JOINTS,
            effort_limit_sim=150.0,
            stiffness=120.0,
            damping=8.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

# Expose EE frame name for downstream controllers / tasks
UR10E_W_SPINDLE_CFG.ee_frame_name = EE_FRAME_NAME

# -----------------------------------------------------------------------------
# High-PD variant
# Helpful for task-space / differential IK style control
# -----------------------------------------------------------------------------
UR10E_W_SPINDLE_HIGH_PD_CFG = UR10E_W_SPINDLE_CFG.copy()
UR10E_W_SPINDLE_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
UR10E_W_SPINDLE_HIGH_PD_CFG.actuators["ur10e_arm"].stiffness = 400.0
UR10E_W_SPINDLE_HIGH_PD_CFG.actuators["ur10e_arm"].damping = 60.0
UR10E_W_SPINDLE_HIGH_PD_CFG.ee_frame_name = EE_FRAME_NAME

# -----------------------------------------------------------------------------
# Exports
# -----------------------------------------------------------------------------
__all__ = [
    "UR10E_W_SPINDLE_CFG",
    "UR10E_W_SPINDLE_HIGH_PD_CFG",
    "EE_FRAME_NAME",
]