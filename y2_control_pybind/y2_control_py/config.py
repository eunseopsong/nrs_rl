from pathlib import Path

"""
Configuration mirrored from y2_ur10skku_control/setup_parameters.hpp.

Units:
- Length: mm
- Angle: rad
- Time: s
"""

ROBOT_KINEMATICS = 2
NUMBER_OF_JOINTS = 6

PACKAGE_BUNDLE_DIR = "/home/eunseop/dev_ws/src/y2_ur10skku_control"

TRAJECTORY_MODE = 2
FORCE_CON_COORDINATE = 0

CONTROL_PERIOD = 0.002
ROBOT_NAME = "ur10skku"

TEST_MODE = 0
REMAPPING_ENABLED = 1
REMAP_STATE_TOPIC = "/joint_states"
REMAP_COMMAND_TOPIC = "/forward_position_controller/commands"

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

TCP_LENGTH_MM = 111.0

EE2TCP = [
    [-1.0,  0.0,  0.0,   0.0],
    [ 0.0,  1.0,  0.0,   0.0],
    [ 0.0,  0.0, -1.0, TCP_LENGTH_MM],
    [ 0.0,  0.0,  0.0,   1.0],
]

# ----------------------------------------------------------------------
# ForceCon Mode 5 checkpoint path
# ----------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent

CHECKPOINT_DIR = _PROJECT_ROOT / "checkpoints"
CONTEXT_NAF_MDGRADI_CKPT = str(
    CHECKPOINT_DIR / "ContextNAF_MDGradi" / "contextNAF_mdGradi_policy_script.pt"
)

# ----------------------------------------------------------------------
# Recommended default parameters for Mode 5 (mirroring robot_motion.cpp)
# Position axes example defaults used in original Y2RobMotion:
#   FC_MASS      = {2, 2, 2, ...}
#   FC_DAMPER    = {6000, 6000, 6000, ...}
#   FC_STIFFNESS = {2000, 2000, 2000, ...}
# Contact branch sets K=0.0 in latest RL mode.
# ----------------------------------------------------------------------
FORCECON_MODE5_MD_RATIO = 1000.0
FORCECON_MODE5_FC_FEXT = 50.0

FORCECON_MODE5_FREE_MASS = 2.0
FORCECON_MODE5_FREE_DAMPING = 6000.0
FORCECON_MODE5_FREE_STIFFNESS = 2000.0
FORCECON_MODE5_CONTACT_STIFFNESS = 0.0
FORCECON_MODE5_RECOVERY_TAU = 3.0

FORCECON_MODE5_ACTION_LOW = [-0.25, -0.25]
FORCECON_MODE5_ACTION_HIGH = [0.25, 0.25]

FORCECON_MODE5_MASS_MIN = 0.5
FORCECON_MODE5_MASS_MAX = 5.0
FORCECON_MODE5_ALPHA_MIN = 0.5
FORCECON_MODE5_ALPHA_MAX = 3.0
FORCECON_MODE5_ALPHA_RATE_UP = 4.0
FORCECON_MODE5_ALPHA_RATE_DOWN = 4.0