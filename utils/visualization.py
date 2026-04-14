# SPDX-License-Identifier: BSD-3-Clause
"""
Unified RL + Polishing Removal + Visualization Module

역할:
1. 저장 경로:
   nrs_rl/logs/polishing_results/<timestamp>/epN/
2. visualization 기록 및 episode 종료 시 plot 저장
3. debug 출력은 담당하지 않음
"""

from __future__ import annotations

import numpy as np
import datetime
from pathlib import Path
import importlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import torch

local_obs = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.mdp.observation"
)
local_ft_sensor = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.assets.assets.sensors.six_axis_ft_sensor"
)
local_debug = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.utils.debug"
)

_run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CURRENT_FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT_DIR = CURRENT_FILE_PATH.parent.parent

BASE_LOG_DIR = PROJECT_ROOT_DIR / "logs" / "polishing_results"
RUN_LOG_DIR = BASE_LOG_DIR / _run_timestamp
RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)

local_debug.print_info(f"[Init] Polishing log directory created: {RUN_LOG_DIR}")

_rl_time_buffer = []
_rl_state_buffer = []
_rl_force_buffer = []
_rl_start_time = None

_episode_counter = 1
_has_seen_any_step = False


def record_step(env_ids, state6, force3, sim_time):
    global _rl_time_buffer, _rl_state_buffer, _rl_force_buffer
    global _rl_start_time, _has_seen_any_step

    _has_seen_any_step = True

    if _rl_start_time is None:
        _rl_start_time = float(sim_time)

    try:
        if isinstance(env_ids, torch.Tensor):
            target_mask = (env_ids == 0)
            if bool(target_mask.any()):
                idx = target_mask.nonzero(as_tuple=True)[0][0].item()
            else:
                idx = 0
        else:
            idx = 0

        _rl_time_buffer.append(float(sim_time) - float(_rl_start_time))

        s_val = state6[idx].detach().cpu().numpy() if hasattr(state6, "detach") else np.asarray(state6[idx])
        f_val = force3[idx].detach().cpu().numpy() if hasattr(force3, "detach") else np.asarray(force3[idx])

        _rl_state_buffer.append(s_val.copy())
        _rl_force_buffer.append(f_val.copy())

    except Exception as e:
        local_debug.print_exception("Visualization record_step failed", e)


def process_episode():
    global _rl_time_buffer, _rl_state_buffer, _rl_force_buffer
    global _rl_start_time, _episode_counter

    if len(_rl_time_buffer) < 10:
        _rl_time_buffer.clear()
        _rl_state_buffer.clear()
        _rl_force_buffer.clear()
        _rl_start_time = None
        return 0.0

    t = np.array(_rl_time_buffer, dtype=float)
    s_arr = np.array(_rl_state_buffer, dtype=float)
    f_arr = np.array(_rl_force_buffer, dtype=float)

    xyz = s_arr[:, :3]
    dt = np.diff(t, prepend=t[0] - 1e-3)

    vxyz = np.zeros_like(xyz)
    for i in range(3):
        vxyz[:, i] = np.gradient(xyz[:, i], t)

    speed = np.linalg.norm(vxyz[:, :2], axis=1)
    fn = np.maximum(f_arr[:, 2], 0.0)
    removal_rate = np.where((fn > 0.5) & (speed > 0.1), fn * speed, 0.0)
    dremoval = removal_rate * dt

    ep_dir = RUN_LOG_DIR / f"ep{_episode_counter}"
    save_plots(ep_dir, t, s_arr, f_arr, dremoval, removal_rate, vxyz)

    local_debug.print_info(f"\n[STAMP] Episode {_episode_counter} results saved to: {ep_dir}\n")

    _episode_counter += 1

    _rl_time_buffer.clear()
    _rl_state_buffer.clear()
    _rl_force_buffer.clear()
    _rl_start_time = None

    return float(np.sum(dremoval))


def save_plots(out_dir, t, state6, force3, dremoval, removal_rate, vxyz):
    out_dir.mkdir(parents=True, exist_ok=True)
    xyz = state6[:, :3]

    plt.figure(figsize=(6, 5))
    plt.hexbin(xyz[:, 0], xyz[:, 1], C=dremoval, gridsize=30, cmap="jet")
    plt.colorbar(label="Removal")
    plt.title("Removal Heatmap")
    plt.savefig(out_dir / "01_removal_heatmap.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(t, removal_rate)
    plt.grid(True, alpha=0.3)
    plt.title("Removal Rate vs Time")
    plt.xlabel("Time [s]")
    plt.ylabel("Removal Rate")
    plt.savefig(out_dir / "02_heatmap_value_vs_time.png")
    plt.close()

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    v_norm_raw = np.linalg.norm(vxyz, axis=1)
    v_norm = savgol_filter(v_norm_raw, 11, 3) if len(t) > 11 else v_norm_raw

    axs[0].plot(t, v_norm, color="g")
    axs[0].set_title("Velocity Magnitude")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(t, force3[:, 2], color="purple")
    axs[1].set_title("Normal Force (Fz)")
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "03_signals_subplot.png")
    plt.close()

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=t, cmap="viridis", s=2)
    ax.set_title("3D Path")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig(out_dir / "04_3d_path_w.png")
    plt.close()


def rl_step(env_ids, state6, force3, sim_time):
    record_step(env_ids, state6, force3, sim_time)


def rl_episode_done():
    return process_episode()


def _safe_get_action_term(env, action_term_name: str = "arm_action"):
    am = env.action_manager
    if hasattr(am, "get_term"):
        try:
            return am.get_term(action_term_name)
        except Exception:
            pass

    if hasattr(am, "_terms"):
        terms = am._terms
        if isinstance(terms, dict) and action_term_name in terms:
            return terms[action_term_name]

    if hasattr(am, action_term_name):
        return getattr(am, action_term_name)

    raise RuntimeError(f"[Visualization] action term '{action_term_name}' not found.")


def rl_step_hook(
    env,
    action_term_name: str = "arm_action",
    asset_name: str = "robot",
    fixed_joint_name: str = "tool0_to_spindle",
    joint_prim_relpath: str = "joints",
):
    try:
        num_envs = env.num_envs
        device = env.device

        env_ids = torch.arange(num_envs, device=device, dtype=torch.long)

        state6 = local_obs.get_ee_pose(env, asset_name=asset_name)
        wrench6 = local_ft_sensor.get_6axis_ft_fixed_joint(
            env=env,
            asset_name=asset_name,
            fixed_joint_name=fixed_joint_name,
            joint_prim_relpath=joint_prim_relpath,
            verbose=False,
        )
        force3 = wrench6[:, :3]

        _ = _safe_get_action_term(env, action_term_name=action_term_name)

        sim_time = float(getattr(env, "common_step_counter", 0)) * float(getattr(env, "step_dt", 0.02))

        rl_step(
            env_ids=env_ids,
            state6=state6,
            force3=force3,
            sim_time=sim_time,
        )

    except Exception as e:
        local_debug.print_exception("Visualization rl_step_hook failed", e)

    return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)


def on_episode_reset(env, env_ids=None):
    global _has_seen_any_step

    try:
        if not _has_seen_any_step:
            return

        if len(_rl_time_buffer) > 0:
            rl_episode_done()

    except Exception as e:
        local_debug.print_exception("Visualization on_episode_reset failed", e)