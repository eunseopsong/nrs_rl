import torch  # noqa: F401 - resolve libtorch symbols before loading the extension

try:
    from ._y2_control_pybind import Mode3ForceController, RobotKinematics
except ImportError as exc:  # gives config-only tools a useful import path before rebuild
    Mode3ForceController = None
    RobotKinematics = None
    _EXTENSION_IMPORT_ERROR = exc
from .config import (
    CONTROL_PERIOD,
    EE2TCP,
    FORCE_CON_COORDINATE,
    FORCE_SWITCH_ACTUAL_FORCE_THRESHOLD,
    FORCE_SWITCH_DESIRED_FORCE_THRESHOLD,
    FORCE_SWITCH_PRECONTACT_FORCE_HOLD,
    FORCE_SWITCH_RETURN_TAU,
    JOINT_NAMES,
    NAF_MDGRADI_CKPT,
    NUMBER_OF_JOINTS,
    ROBOT_CONFIG_PATH,
    ROBOT_KINEMATICS,
    TCP_LENGTH_MM,
    Y2_CONTROL_SOURCE_DIR,
)

__all__ = [
    "Mode3ForceController", "RobotKinematics", "CONTROL_PERIOD", "EE2TCP",
    "FORCE_CON_COORDINATE", "FORCE_SWITCH_ACTUAL_FORCE_THRESHOLD",
    "FORCE_SWITCH_DESIRED_FORCE_THRESHOLD", "FORCE_SWITCH_PRECONTACT_FORCE_HOLD",
    "FORCE_SWITCH_RETURN_TAU", "JOINT_NAMES", "NAF_MDGRADI_CKPT",
    "NUMBER_OF_JOINTS", "ROBOT_CONFIG_PATH", "ROBOT_KINEMATICS",
    "TCP_LENGTH_MM", "Y2_CONTROL_SOURCE_DIR",
]
