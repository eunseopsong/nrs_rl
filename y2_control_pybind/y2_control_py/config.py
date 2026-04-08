# y2_control_py/config.py

"""
Configuration mirrored from y2_ur10skku_control/setup_parameters.hpp.

Units:
- Length: mm
- Angle: rad
- Time: s
"""

ROBOT_KINEMATICS = 2          # 0: KUKA_IIWA, 1: UR10, 2: UR10e
NUMBER_OF_JOINTS = 6

PACKAGE_BUNDLE_DIR = "/home/eunseop/dev_ws/src/y2_ur10skku_control"

TRAJECTORY_MODE = 2           # 0: cmd_6D, 1: cmd_9D, 2: cmd_continue9D

# 0: Base Coordinate, 1: TCP Coordinate
FORCE_CON_COORDINATE = 0

CONTROL_PERIOD = 0.002        # seconds
ROBOT_NAME = "ur10skku"

TEST_MODE = 0
REMAPPING_ENABLED = 1
REMAP_STATE_TOPIC = "/joint_states"
REMAP_COMMAND_TOPIC = "/forward_position_controller/commands"

# Joint names (UR10e)
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# ----------------------------------------------------------------------
# EE -> TCP homogeneous transform
#
# Original setup_parameters.hpp used:
# [
#   [-1,  0,  0,   0],
#   [ 0,  1,  0,   0],
#   [ 0,  0, -1, 111],
#   [ 0,  0,  0,   1],
# ]
#
# To keep config максимально identical to the original project,
# keep the same rotation and update only TCP length (z translation).
# Current tcp length: 146 mm
# ----------------------------------------------------------------------
TCP_LENGTH_MM = 146.0

EE2TCP = [
    [-1.0,  0.0,  0.0,   0.0],
    [ 0.0,  1.0,  0.0,   0.0],
    [ 0.0,  0.0, -1.0, TCP_LENGTH_MM],
    [ 0.0,  0.0,  0.0,   1.0],
]