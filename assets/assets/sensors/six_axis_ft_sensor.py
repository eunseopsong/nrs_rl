# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import re
import torch
import omni.usd
from pxr import UsdPhysics

from ....utils import debug as local_debug


def _to_scalar_index(idx_obj):
    if isinstance(idx_obj, int):
        return idx_obj
    if isinstance(idx_obj, torch.Tensor):
        if idx_obj.numel() == 0:
            raise RuntimeError("Empty tensor index.")
        return int(idx_obj.reshape(-1)[0].item())
    if isinstance(idx_obj, (list, tuple)):
        if len(idx_obj) == 0:
            raise RuntimeError("Empty list/tuple index.")
        first = idx_obj[0]
        if isinstance(first, (list, tuple)) and len(first) > 0:
            return _to_scalar_index(first[0])
        return _to_scalar_index(first)
    raise RuntimeError(f"Unsupported index type: {type(idx_obj)}")


def _resolve_env0_robot_prim_path(robot) -> str:
    """
    Convert regex-style prim path into a concrete env_0 path for USD stage lookup.
    """
    prim_path = robot.cfg.prim_path

    if "{ENV_REGEX_NS}" in prim_path:
        return prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")

    # e.g. /World/envs/env_.*/Robot -> /World/envs/env_0/Robot
    prim_path = re.sub(r"/env_\.\*", "/env_0", prim_path)

    return prim_path


def _find_existing_joint_prim_path(
    stage,
    robot_prim_path_env0: str,
    joint_prim_relpath: str,
    fixed_joint_name: str,
) -> str:
    """
    Try multiple possible prim layouts.

    Common candidates:
      /World/envs/env_0/Robot/joints/tool0_to_spindle
      /World/envs/env_0/Robot/Robot/joints/tool0_to_spindle
    """
    candidates = [
        f"{robot_prim_path_env0}/{joint_prim_relpath}/{fixed_joint_name}",
        f"{robot_prim_path_env0}/Robot/{joint_prim_relpath}/{fixed_joint_name}",
        f"{robot_prim_path_env0}/robot/{joint_prim_relpath}/{fixed_joint_name}",
    ]

    for p in candidates:
        joint = UsdPhysics.Joint.Get(stage, p)
        if joint:
            return p

    raise RuntimeError(
        "[get_6axis_ft_fixed_joint] Joint prim not found. "
        f"Tried: {candidates}"
    )


def _init_fixed_joint_ft_cache(
    env: "ManagerBasedRLEnv",
    asset_name: str,
    fixed_joint_name: str,
    joint_prim_relpath: str = "joints",
    verbose: bool = False,
):
    if not hasattr(env, "_ft6_fixed_cache"):
        env._ft6_fixed_cache = {}

    cache_key = (asset_name, fixed_joint_name, joint_prim_relpath)
    if cache_key in env._ft6_fixed_cache:
        return env._ft6_fixed_cache[cache_key]

    robot = env.scene[asset_name]
    stage = omni.usd.get_context().get_stage()

    robot_prim_path_env0 = _resolve_env0_robot_prim_path(robot)
    joint_prim_path = _find_existing_joint_prim_path(
        stage=stage,
        robot_prim_path_env0=robot_prim_path_env0,
        joint_prim_relpath=joint_prim_relpath,
        fixed_joint_name=fixed_joint_name,
    )

    joint = UsdPhysics.Joint.Get(stage, joint_prim_path)
    if not joint:
        raise RuntimeError(
            f"[get_6axis_ft_fixed_joint] Joint prim not found even after resolve: {joint_prim_path}"
        )

    body1_targets = joint.GetBody1Rel().GetTargets()
    if len(body1_targets) == 0:
        raise RuntimeError(
            f"[get_6axis_ft_fixed_joint] body1 target missing: {joint_prim_path}"
        )

    child_link_path = str(body1_targets[0])
    child_link_name = child_link_path.split("/")[-1]

    body_ids = robot.find_bodies(child_link_name)[0]
    if len(body_ids) == 0:
        raise RuntimeError(
            f"[get_6axis_ft_fixed_joint] Child link '{child_link_name}' not found. "
            f"Available bodies: {robot.body_names}"
        )

    child_link_index = _to_scalar_index(body_ids)

    cache = {
        "robot": robot,
        "joint_prim_path": joint_prim_path,
        "child_link_name": child_link_name,
        "child_link_index": child_link_index,
    }
    env._ft6_fixed_cache[cache_key] = cache

    if verbose:
        local_debug.print_fixed_joint_ft_cache(
            robot_prim_path_env0=robot_prim_path_env0,
            joint_prim_path=joint_prim_path,
            child_link_name=child_link_name,
            child_link_index=child_link_index,
        )

    return cache


def get_6axis_ft_fixed_joint(
    env: "ManagerBasedRLEnv",
    asset_name: str = "robot",
    fixed_joint_name: str = "tool0_to_spindle",
    joint_prim_relpath: str = "joints",
    verbose: bool = False,
) -> torch.Tensor:
    """
    Read 6-axis FT from a fixed joint by mapping:
      fixed joint prim -> child link -> measured joint force row

    Returns:
      [num_envs, 6] = [Fx, Fy, Fz, Tx, Ty, Tz]
    """
    try:
        cache = _init_fixed_joint_ft_cache(
            env=env,
            asset_name=asset_name,
            fixed_joint_name=fixed_joint_name,
            joint_prim_relpath=joint_prim_relpath,
            verbose=verbose,
        )

        robot = cache["robot"]
        child_link_index = cache["child_link_index"]

        physx_view = getattr(robot, "root_physx_view", None)
        if physx_view is None:
            raise RuntimeError(
                "[get_6axis_ft_fixed_joint] robot.root_physx_view is missing"
            )

        forces = None

        # 1) preferred path
        if hasattr(physx_view, "get_link_incoming_joint_force"):
            forces = physx_view.get_link_incoming_joint_force()

        # 2) robot wrapper fallback
        elif hasattr(robot, "get_measured_joint_forces"):
            forces = robot.get_measured_joint_forces()

        # 3) physx view fallback
        elif hasattr(physx_view, "get_measured_joint_forces"):
            forces = physx_view.get_measured_joint_forces()

        else:
            raise RuntimeError(
                "[get_6axis_ft_fixed_joint] No available FT API. "
                "Checked: root_physx_view.get_link_incoming_joint_force(), "
                "robot.get_measured_joint_forces(), "
                "root_physx_view.get_measured_joint_for_view.get_measured_joint_forces()"
            )

        if forces is None:
            return torch.zeros((env.num_envs, 6), device=env.device, dtype=torch.float32)

        if not isinstance(forces, torch.Tensor):
            forces = torch.tensor(forces, device=env.device, dtype=torch.float32)
        else:
            forces = forces.to(device=env.device, dtype=torch.float32)

        # expected:
        #   (num_envs, num_links, 6)
        #   (num_envs, num_joints, 6)
        #   (num_links, 6)
        if forces.ndim == 2:
            wrench = forces[child_link_index, :].unsqueeze(0)
        elif forces.ndim == 3:
            wrench = forces[:, child_link_index, :]
        else:
            raise RuntimeError(
                f"[get_6axis_ft_fixed_joint] Unexpected force tensor shape: {tuple(forces.shape)}"
            )

        if wrench.shape[-1] != 6:
            raise RuntimeError(
                f"[get_6axis_ft_fixed_joint] Expected last dim=6, got {tuple(wrench.shape)}"
            )

        local_debug.print_ft_sensor_debug(int(env.common_step_counter), wrench[0])

        return wrench

    except Exception as e:
        local_debug.print_fixed_joint_ft_failed(e)
        return torch.zeros((env.num_envs, 6), device=env.device, dtype=torch.float32)