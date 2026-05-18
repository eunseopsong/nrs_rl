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
FT_FX_SIGN = 1.0
FT_FY_SIGN = 1.0
FT_TX_SIGN = 1.0
FT_TY_SIGN = 1.0
FT_TZ_SIGN = 1.0
FT_FORCE_SCALE = 1.0
FT_TORQUE_SCALE = 1.0

FT_MOV_SIZE = 2

# Number of initial samples used to estimate static bias.
FT_BIAS_INIT_SAMPLES = 50
FT_USE_BIAS = True
FT_TOOL_MASS_KG = 1.6
FT_TOOL_COG_M = (0.0, 0.0, -0.149303)

# EMA for Fz only (main signal used by force control).
FT_USE_FZ_EMA = True
FT_FZ_EMA_ALPHA = 0.065

# Small-signal deadbands.
FT_FORCE_DEADBAND = 0.05   # N
FT_TORQUE_DEADBAND = 0.005  # Nm (or sensor torque unit)

# Whether to also estimate/remove static bias for all 6 channels.
FT_USE_FULL_WRENCH_BIAS = True


def _quat_wxyz_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    quat = quat / torch.clamp(torch.linalg.norm(quat, dim=-1, keepdim=True), min=1.0e-8)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    rot = torch.empty((quat.shape[0], 3, 3), device=quat.device, dtype=quat.dtype)
    rot[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rot[:, 0, 1] = 2.0 * (x * y - z * w)
    rot[:, 0, 2] = 2.0 * (x * z + y * w)
    rot[:, 1, 0] = 2.0 * (x * y + z * w)
    rot[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rot[:, 1, 2] = 2.0 * (y * z - x * w)
    rot[:, 2, 0] = 2.0 * (x * z - y * w)
    rot[:, 2, 1] = 2.0 * (y * z + x * w)
    rot[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rot


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
    force_row_index = child_link_index
    force_row_source = "body1_link_index"

    if hasattr(robot, "find_joints"):
        try:
            joint_ids = robot.find_joints(fixed_joint_name)[0]
            joint_index = _to_scalar_index(joint_ids)
            force_row_index = joint_index + 1
            force_row_source = "joint_index_plus_one"
        except Exception as e:
            if verbose:
                local_debug.print_info(f"[FT] find_joints fallback to body1 link index: {repr(e)}")

    cache = {
        "robot": robot,
        "joint_prim_path": joint_prim_path,
        "child_link_name": child_link_name,
        "child_link_index": child_link_index,
        "force_row_index": force_row_index,
        "force_row_source": force_row_source,
    }
    env._ft6_fixed_cache[cache_key] = cache

    if verbose:
        local_debug.print_fixed_joint_ft_cache(
            robot_prim_path_env0=robot_prim_path_env0,
            joint_prim_path=joint_prim_path,
            child_link_name=child_link_name,
            child_link_index=child_link_index,
        )
    local_debug.print_info(
        f"[get_6axis_ft_fixed_joint] force_row_index    : {force_row_index} ({force_row_source})"
    )

    return cache


def _maybe_reset_filter_state(env: "ManagerBasedRLEnv", cache_key, num_envs: int, device):
    if not hasattr(env, "_ft6_filter_state"):
        env._ft6_filter_state = {}

    need_init = cache_key not in env._ft6_filter_state
    if need_init:
        env._ft6_filter_state[cache_key] = {
            "mov_buffer": torch.zeros((num_envs, FT_MOV_SIZE, 6), device=device, dtype=torch.float32),
            "mov_count": torch.zeros((num_envs,), device=device, dtype=torch.long),
            "mov_cursor": torch.zeros((num_envs,), device=device, dtype=torch.long),
            "g_init": torch.zeros((num_envs, 3), device=device, dtype=torch.float32),
            "g_init_ready": torch.zeros((num_envs,), device=device, dtype=torch.bool),
            "bias_accum": torch.zeros((num_envs,), device=device, dtype=torch.float32),
            "bias_count": torch.zeros((num_envs,), device=device, dtype=torch.long),
            "bias": torch.zeros((num_envs,), device=device, dtype=torch.float32),
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
            state["mov_buffer"][reset_mask] = 0.0
            state["mov_count"][reset_mask] = 0
            state["mov_cursor"][reset_mask] = 0
            state["g_init"][reset_mask] = 0.0
            state["g_init_ready"][reset_mask] = False
            state["bias_accum"][reset_mask] = 0.0
            state["bias_count"][reset_mask] = 0
            state["bias"][reset_mask] = 0.0
            state["bias_ready"][reset_mask] = False
            state["fz_ema"][reset_mask] = 0.0
            state["fz_ema_ready"][reset_mask] = False

    return state


def _bridge_moving_average(state, wrench: torch.Tensor) -> torch.Tensor:
    if FT_MOV_SIZE <= 1:
        return wrench

    out = torch.empty_like(wrench)
    for env_id in range(wrench.shape[0]):
        cursor = int(state["mov_cursor"][env_id].item())
        count = int(state["mov_count"][env_id].item())
        state["mov_buffer"][env_id, cursor] = wrench[env_id]
        count = min(count + 1, FT_MOV_SIZE)
        state["mov_count"][env_id] = count
        state["mov_cursor"][env_id] = (cursor + 1) % FT_MOV_SIZE
        out[env_id] = state["mov_buffer"][env_id, :count].mean(dim=0)
    return out


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


def _preprocess_wrench(env: "ManagerBasedRLEnv", cache, cache_key, wrench: torch.Tensor) -> torch.Tensor:
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
    sign_scale = torch.tensor(
        [
            FT_FX_SIGN * FT_FORCE_SCALE,
            FT_FY_SIGN * FT_FORCE_SCALE,
            FT_FZ_SIGN * FT_FORCE_SCALE,
            FT_TX_SIGN * FT_TORQUE_SCALE,
            FT_TY_SIGN * FT_TORQUE_SCALE,
            FT_TZ_SIGN * FT_TORQUE_SCALE,
        ],
        device=out.device,
        dtype=out.dtype,
    )
    out = out * sign_scale

    out = _bridge_moving_average(state, out)

    sensor_to_tcp = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        device=out.device,
        dtype=out.dtype,
    )
    sensor_force = out[:, 0:3]
    sensor_moment = out[:, 3:6]
    tcp_force = torch.matmul(sensor_force, sensor_to_tcp.T)
    tcp_moment = torch.matmul(sensor_moment, sensor_to_tcp.T)

    robot = cache["robot"]
    child_link_index = int(cache["child_link_index"])
    quat_w = robot.data.body_quat_w[:, child_link_index, :].to(device=out.device, dtype=out.dtype)
    rot_base_to_tcp = _quat_wxyz_to_rotmat(quat_w)

    if FT_USE_BIAS and FT_TOOL_MASS_KG > 0.0:
        g_base = torch.tensor([0.0, 0.0, -9.81], device=out.device, dtype=out.dtype).expand(out.shape[0], 3)
        g_tcp = torch.bmm(rot_base_to_tcp.transpose(1, 2), g_base.unsqueeze(-1)).squeeze(-1)

        init_mask = ~state["g_init_ready"]
        if torch.any(init_mask):
            state["g_init"][init_mask] = g_tcp[init_mask]
            state["g_init_ready"][init_mask] = True

        gravity_force = (g_tcp - state["g_init"]) * float(FT_TOOL_MASS_KG)
        tool_cog = torch.tensor(FT_TOOL_COG_M, device=out.device, dtype=out.dtype).expand_as(gravity_force)
        gravity_moment = torch.cross(tool_cog, gravity_force, dim=1)
        tcp_force = tcp_force - gravity_force
        tcp_moment = tcp_moment - gravity_moment

    base_force = torch.bmm(rot_base_to_tcp, tcp_force.unsqueeze(-1)).squeeze(-1)
    base_moment = torch.bmm(rot_base_to_tcp, tcp_moment.unsqueeze(-1)).squeeze(-1)
    out = torch.cat([base_force, base_moment], dim=1)

    if FT_USE_BIAS:
        not_ready = ~state["bias_ready"]
        if torch.any(not_ready):
            state["bias_accum"][not_ready] += out[not_ready, 2]
            state["bias_count"][not_ready] += 1
            enough = state["bias_count"] >= FT_BIAS_INIT_SAMPLES
            newly_ready = enough & (~state["bias_ready"])
            if torch.any(newly_ready):
                counts = state["bias_count"][newly_ready].to(dtype=torch.float32)
                state["bias"][newly_ready] = state["bias_accum"][newly_ready] / counts
                state["bias_ready"][newly_ready] = True
                state["fz_ema_ready"][newly_ready] = False
            out[not_ready, 2] = 0.0

        ready_mask = state["bias_ready"]
        if torch.any(ready_mask):
            out[ready_mask, 2] = out[ready_mask, 2] - state["bias"][ready_mask]

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

    out = _apply_deadband(out)

    return out


def _get_step_cached_wrench(env: "ManagerBasedRLEnv", cache_key):
    if not hasattr(env, "_ft6_step_output_cache"):
        env._ft6_step_output_cache = {}
        return None

    cached = env._ft6_step_output_cache.get(cache_key)
    if cached is None:
        return None

    step = int(env.common_step_counter)
    if cached.get("step") != step:
        return None

    return cached["wrench"]


def _set_step_cached_wrench(env: "ManagerBasedRLEnv", cache_key, wrench: torch.Tensor):
    if not hasattr(env, "_ft6_step_output_cache"):
        env._ft6_step_output_cache = {}

    env._ft6_step_output_cache[cache_key] = {
        "step": int(env.common_step_counter),
        "wrench": wrench,
    }


def _fmt_wrench_env0(wrench: torch.Tensor) -> str:
    vals = wrench[0].detach().cpu().reshape(-1).tolist()
    vals = vals + [0.0] * max(0, 6 - len(vals))
    return (
        f"Fx={vals[0]: .4f}, Fy={vals[1]: .4f}, Fz={vals[2]: .4f}, "
        f"Tx={vals[3]: .4f}, Ty={vals[4]: .4f}, Tz={vals[5]: .4f}"
    )


def _print_ft_runtime_debug(env: "ManagerBasedRLEnv", cache, cache_key, raw_wrench: torch.Tensor, wrench: torch.Tensor):
    step = int(env.common_step_counter)
    if step % 50 != 0:
        return

    state = getattr(env, "_ft6_filter_state", {}).get(cache_key, {})
    bias_count = int(state.get("bias_count", torch.zeros(1, device=wrench.device, dtype=torch.long))[0].item())
    bias_ready = bool(state.get("bias_ready", torch.zeros(1, device=wrench.device, dtype=torch.bool))[0].item())
    bias = float(state.get("bias", torch.zeros(1, device=wrench.device, dtype=torch.float32))[0].item())

    local_debug.print_info(
        "\n[FT Runtime] "
        f"step={step} row={int(cache['force_row_index'])} source={cache['force_row_source']} "
        f"bias_count={bias_count}/{FT_BIAS_INIT_SAMPLES} bias_ready={bias_ready} bias_fz={bias:.4f}\n"
        f"  raw       = {_fmt_wrench_env0(raw_wrench)}\n"
        f"  processed = {_fmt_wrench_env0(wrench)}\n"
    )


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

        cached_wrench = _get_step_cached_wrench(env, cache_key)
        if cached_wrench is not None:
            local_debug.print_ft_sensor_debug(int(env.common_step_counter), cached_wrench[0])
            return cached_wrench

        cache = _init_fixed_joint_ft_cache(
            env=env,
            asset_name=asset_name,
            fixed_joint_name=fixed_joint_name,
            joint_prim_relpath=joint_prim_relpath,
            verbose=verbose,
        )

        robot = cache["robot"]
        force_row_index = int(cache["force_row_index"])

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
            wrench = forces[force_row_index, :].unsqueeze(0)
        elif forces.ndim == 3:
            wrench = forces[:, force_row_index, :]
        else:
            raise RuntimeError(f"[FT] Unexpected force tensor shape: {tuple(forces.shape)}")

        if wrench.shape[-1] != 6:
            raise RuntimeError(f"[FT] Expected last dim=6, got {tuple(wrench.shape)}")

        raw_wrench = wrench.clone()
        wrench = _preprocess_wrench(env, cache, cache_key, wrench)
        _set_step_cached_wrench(env, cache_key, wrench)
        _print_ft_runtime_debug(env, cache, cache_key, raw_wrench, wrench)

        local_debug.print_ft_sensor_debug(int(env.common_step_counter), wrench[0])
        return wrench

    except Exception as e:
        local_debug.print_fixed_joint_ft_failed(e)
        return torch.zeros((env.num_envs, 6), device=env.device, dtype=torch.float32)
