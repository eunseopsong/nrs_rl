"""Single source of truth for robot/simulator controller parameters.

Values are loaded from the ROS controller YAML at import time.  Training now
fails early if the production configuration is missing instead of silently
falling back to the former UR10e + 144 mm training geometry.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml


Y2_CONTROL_SOURCE_DIR = Path(
    os.environ.get("Y2_CONTROL_SOURCE_DIR", "/home/eunseop/dev_ws/src/y2_ur10skku_control")
).resolve()
ROBOT_CONFIG_PATH = Path(
    os.environ.get(
        "Y2_ROBOT_CONFIG",
        str(Y2_CONTROL_SOURCE_DIR / "Y2RobMotion/config/setup_parameters.yaml"),
    )
).resolve()

if not ROBOT_CONFIG_PATH.is_file():
    raise FileNotFoundError(f"Production robot configuration not found: {ROBOT_CONFIG_PATH}")

with ROBOT_CONFIG_PATH.open("r", encoding="utf-8") as stream:
    _cfg = yaml.safe_load(stream)

ROBOT_KINEMATICS = str(_cfg["ROBOT_KINEMATICS"])
NUMBER_OF_JOINTS = int(_cfg["NUMBER_OF_JOINTS"])
CONTROL_PERIOD = float(_cfg["CONTROL_PERIOD"])
FORCE_CON_COORDINATE = int(_cfg["Force_Con_Coordinate"])
EE2TCP = [[float(value) for value in row] for row in _cfg["EE2TCP"]]
TCP_LENGTH_MM = float(EE2TCP[2][3])
JOINT_NAMES = list(_cfg["JOINT_NAMES"])

FORCE_SWITCH_DESIRED_FORCE_THRESHOLD = float(
    _cfg["FORCE_SWITCH_DESIRED_FORCE_THRESHOLD"]
)
FORCE_SWITCH_ACTUAL_FORCE_THRESHOLD = float(
    _cfg["FORCE_SWITCH_ACTUAL_FORCE_THRESHOLD"]
)
FORCE_SWITCH_PRECONTACT_FORCE_HOLD = float(
    _cfg["FORCE_SWITCH_PRECONTACT_FORCE_HOLD"]
)
FORCE_SWITCH_RETURN_TAU = float(_cfg["FORCE_SWITCH_RETURN_TAU_M"])

NAF_MDGRADI_CKPT = str(
    Y2_CONTROL_SOURCE_DIR
    / "Y2ForceCon/src/checkpoints/NAF_MDGradi/NAF_mdGradi_policy_script.pt"
)
if not Path(NAF_MDGRADI_CKPT).is_file():
    raise FileNotFoundError(f"Production NAF MD-gradient checkpoint not found: {NAF_MDGRADI_CKPT}")

# Backward-compatible name for older scripts.  It now points to the actual
# Mode-3 checkpoint intentionally.
CONTEXT_NAF_MDGRADI_CKPT = NAF_MDGRADI_CKPT
