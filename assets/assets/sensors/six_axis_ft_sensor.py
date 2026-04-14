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


# ============================================================
# FT preprocessing config
# ============================================================
# If your contact should be positive but raw sensor reports negative,
# keep this as -1.0. If the opposite happens, change to +1.0.
FT_FZ_SIGN = -1.0

# Number of initial samples used to estimate static bias.
FT_BIAS_INIT_SAMPLES = 100

# EMA for Fz only (main signal used by force control).
FT_USE_FZ_EMA = True
FT_FZ_EMA_ALPHA = 0.07

# Small-signal deadbands.
FT_FORCE_DEADBAND = 0.05   # N
FT_TORQUE_DEADBAND = 0.005 # Nm (or sensor torque unit)

# Whether to also estimate/remove static bias for all 6 channels.
FT_USE_FULL_WRENCH_BIAS = True


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
    prim_path = robot.cfg.prim_path

    if "{ENV_REGEX_NS}" in prim_path:
        return prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")

    prim_path = re.sub(r"/env_\.\*", "/env_0", prim_path)
    return prim_path


def _find_existing_joint_prim_path(
    stage,
    robot_prim_path_env0: str,
    joint_prim_relpath: str,
    fixed_joint_name: str,
) -> str:
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
        "[FT] Joint prim not found. "
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
        raise RuntimeError(f"[FT] Joint prim not found after resolve: {joint_prim_path}")

    body1_targets = joint.GetBody1Rel().GetTargets()
    if len(body1_targets) == 0:
        raise RuntimeError(f"[FT] body1 target missing: {joint_prim_path}")

    child_link_path = str(body1_targets[0])
    child_link_name = child_link_path.split("/")[-1]

    body_ids = robot.find_bodies(child_link_name)[0]
    if len(body_ids) == 0:
        raise RuntimeError(
            f"[FT] Child link '{child_link_name}' not found. "
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


def _maybe_reset_filter_state(env: "ManagerBasedRLEnv", cache_key, num_envs: int, device):
    if not hasattr(env, "_ft6_filter_state"):
        env._ft6_filter_state = {}

    need_init = cache_key not in env._ft6_filter_state
    if need_init:
        env._ft6_filter_state[cache_key] = {
            "bias_accum": torch.zeros((num_envs, 6), device=device, dtype=torch.float32),
            "bias_count": torch.zeros((num_envs,), device=device, dtype=torch.long),
            "bias": torch.zeros((num_envs, 6), device=device, dtype=torch.float32),
            "bias_ready": torch.zeros((num_envs,), device=device, dtype=torch.bool),
            "fz_ema": torch.zeros((num_envs,), device=device, dtype=torch.float32),
            "fz_ema_ready": torch.zeros((num_envs,), device=device, dtype=torch.bool),
        }
        return env._ft6_filter_state[cache_key]

    state = env._ft6_filter_state[cache_key]

    # Episode reset handling
    if hasattr(env, "episode_length_buf"):
        reset_mask = (env.episode_length_buf == 0)
        if torch.any(reset_mask):
            state["bias_accum"][reset_mask] = 0.0
            state["bias_count"][reset_mask] = 0
            state["bias"][reset_mask] = 0.0
            state["bias_ready"][reset_mask] = False
            state["fz_ema"][reset_mask] = 0.0
            state["fz_ema_ready"][reset_mask] = False

    return state


def _apply_deadband(wrench: torch.Tensor) -> torch.Tensor:
    out = wrench.clone()

    # force deadband
    out[:, 0:3] = torch.where(
        torch.abs(out[:, 0:3]) < FT_FORCE_DEADBAND,
        torch.zeros_like(out[:, 0:3]),
        out[:, 0:3],
    )

    # torque deadband
    out[:, 3:6] = torch.where(
        torch.abs(out[:, 3:6]) < FT_TORQUE_DEADBAND,
        torch.zeros_like(out[:, 3:6]),
        out[:, 3:6],
    )
    return out


def _preprocess_wrench(env: "ManagerBasedRLEnv", cache_key, wrench: torch.Tensor) -> torch.Tensor:
    """
    Preprocess fixed-joint FT wrench.

    Applied operations:
    1) static bias initialization/removal
    2) Fz sign convention
    3) Fz EMA
    4) deadband
    """
    state = _maybe_reset_filter_state(env, cache_key, wrench.shape[0], wrench.device)

    out = wrench.clone()

    # --------------------------------------------------------
    # 1) Static bias init / removal
    # --------------------------------------------------------
    bias_ready = state["bias_ready"]
    not_ready = ~bias_ready

    if torch.any(not_ready):
        state["bias_accum"][not_ready] += out[not_ready]
        state["bias_count"][not_ready] += 1

        enough = state["bias_count"] >= FT_BIAS_INIT_SAMPLES
        newly_ready = enough & (~state["bias_ready"])
        if torch.any(newly_ready):
            counts = state["bias_count"][newly_ready].to(dtype=torch.float32).unsqueeze(-1)
            state["bias"][newly_ready] = state["bias_accum"][newly_ready] / counts
            state["bias_ready"][newly_ready] = True
            state["fz_ema_ready"][newly_ready] = False

    if FT_USE_FULL_WRENCH_BIAS:
        ready_mask = state["bias_ready"]
        if torch.any(ready_mask):
            out[ready_mask] = out[ready_mask] - state["bias"][ready_mask]
    else:
        ready_mask = state["bias_ready"]
        if torch.any(ready_mask):
            out[ready_mask, 2] = out[ready_mask, 2] - state["bias"][ready_mask, 2]

    # --------------------------------------------------------
    # 2) Fz sign convention
    # --------------------------------------------------------
    out[:, 2] = FT_FZ_SIGN * out[:, 2]

    # --------------------------------------------------------
    # 3) Fz EMA
    # --------------------------------------------------------
    if FT_USE_FZ_EMA:
        fz = out[:, 2].clone()
        ema_ready = state["fz_ema_ready"]

        init_mask = ~ema_ready
        if torch.any(init_mask):
            state["fz_ema"][init_mask] = fz[init_mask]
            state["fz_ema_ready"][init_mask] = True

        alpha = FT_FZ_EMA_ALPHA
        state["fz_ema"] = alpha * fz + (1.0 - alpha) * state["fz_ema"]
        out[:, 2] = state["fz_ema"]

    # --------------------------------------------------------
    # 4) deadband
    # --------------------------------------------------------
    out = _apply_deadband(out)

    return out


def get_6axis_ft_fixed_joint(
    env: "ManagerBasedRLEnv",
    asset_name: str = "robot",
    fixed_joint_name: str = "tool0_to_spindle",
    joint_prim_relpath: str = "joints",
    verbose: bool = False,
) -> torch.Tensor:
    """
    Read 6-axis wrench entering the child link through the fixed joint.
    Physically this is the reaction wrench transmitted by `fixed_joint_name`
    onto the child link (body1 side).

    Returned wrench is preprocessed with:
    - bias removal
    - Fz sign convention
    - Fz EMA
    - deadband
    """
    try:
        cache_key = (asset_name, fixed_joint_name, joint_prim_relpath)

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
            raise RuntimeError("[FT] robot.root_physx_view is missing")

        if not hasattr(physx_view, "get_link_incoming_joint_force"):
            raise RuntimeError("[FT] root_physx_view.get_link_incoming_joint_force() is missing")

        forces = physx_view.get_link_incoming_joint_force()
        if forces is None:
            return torch.zeros((env.num_envs, 6), device=env.device, dtype=torch.float32)

        if not isinstance(forces, torch.Tensor):
            forces = torch.tensor(forces, device=env.device, dtype=torch.float32)
        else:
            forces = forces.to(device=env.device, dtype=torch.float32)

        if forces.ndim == 2:
            wrench = forces[child_link_index, :].unsqueeze(0)
        elif forces.ndim == 3:
            wrench = forces[:, child_link_index, :]
        else:
            raise RuntimeError(f"[FT] Unexpected force tensor shape: {tuple(forces.shape)}")

        if wrench.shape[-1] != 6:
            raise RuntimeError(f"[FT] Expected last dim=6, got {tuple(wrench.shape)}")

        wrench = _preprocess_wrench(env, cache_key, wrench)

        local_debug.print_ft_sensor_debug(int(env.common_step_counter), wrench[0])
        return wrench

    except Exception as e:
        local_debug.print_fixed_joint_ft_failed(e)
        return torch.zeros((env.num_envs, 6), device=env.device, dtype=torch.float32)