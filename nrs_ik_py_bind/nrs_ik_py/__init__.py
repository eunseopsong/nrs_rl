# nrs_ik_py/__init__.py
"""
Re-export IK/FK solvers.

Requires both native extensions to be built:
- nrs_ik_core  (IKSolver, PoseRPY)
- nrs_fk_core  (FKSolver)
"""

import numpy as np
import nrs_ik_core as _ik
import nrs_fk_core as _fk

# --- Re-exports ---
IKSolver = _ik.IKSolver
PoseRPY  = _ik.PoseRPY
FKSolver = _fk.FKSolver

# --- Python convenience wrappers (항상 제공) ---
def fk_transform(q, tool_z: float = 0.0, use_degrees: bool = False):
    """
    Return 4x4 TCP homogeneous transform.
    q: iterable of 6 (deg if use_degrees=True else rad)
    """
    solver = FKSolver(tool_z=tool_z, use_degrees=use_degrees)
    q = np.asarray(q, dtype=float).reshape(6)
    return solver.transform(q)

def fk_pose_rpy(q, tool_z: float = 0.0, use_degrees: bool = False, as_degrees: bool = False):
    """
    Return [x, y, z, roll, pitch, yaw] (ZYX).
    """
    solver = FKSolver(tool_z=tool_z, use_degrees=use_degrees)
    q = np.asarray(q, dtype=float).reshape(6)
    ok, pose = solver.compute(q, as_degrees=as_degrees)
    if not ok:
        raise RuntimeError("FK failed to compute pose.")
    return pose

__all__ = ["IKSolver", "PoseRPY", "FKSolver", "fk_transform", "fk_pose_rpy"]
