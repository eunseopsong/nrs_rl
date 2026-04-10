# SPDX-License-Identifier: BSD-3-Clause
"""
Unified RL + Polishing Removal + Visualization + Debug Module
하나의 파일에서 데이터 기록, 그래프 저장, 화면 출력을 모두 처리함.

[수정 사항]
- env cfg의 ObservationTerm에서 매 스텝 rl_step_hook() 호출 가능
- env cfg의 EventTerm(reset)에서 on_episode_reset() 호출 가능
- action.py 직접 호출 없이 episode 종료 저장 가능
"""

from __future__ import annotations

import os
import numpy as np
import datetime
from pathlib import Path
import importlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import torch

# ============================================================
# Optional imports (lazy-safe style)
# ============================================================
local_obs = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.mdp.observation"
)
local_ft_sensor = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.assets.assets.sensors.six_axis_ft_sensor"
)

# ============================================================
# GLOBAL CONFIG & PATHS
# ============================================================
_run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CURRENT_FILE_PATH = Path(__file__).resolve()
BASE_LOG_DIR = CURRENT_FILE_PATH.parent / "logs"

RUN_LOG_DIR = BASE_LOG_DIR / _run_timestamp
RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
print(f"[Init] Log directory created: {RUN_LOG_DIR}")

# 데이터 버퍼 (env0 전용)
_rl_time_buffer = []
_rl_state_buffer = []
_rl_force_buffer = []
_rl_start_time = None

# 저장용 episode counter
_episode_counter = 1

# debug 출력용 episode counter도 여기 기준으로 통일
_debug_episode_counter = 1

# 초기 reset에서 저장이 트리거되지 않도록 하는 플래그
_has_seen_any_step = False

# ============================================================
# ------------------ DEBUG PRINT LOGIC -----------------------
# ============================================================

def print_polishing_status(
    env_id,
    step,
    path_index,
    traj_len,
    done,
    pos_err,
    target_force,
    current_force,
    reward_info=None,
):
    global _debug_episode_counter

    if env_id != 0:
        return

    print("\n" + "=" * 80)
    print(f"[Episode {_debug_episode_counter}] Step: {step} | H5_Idx: {path_index}/{traj_len} | Done: {done}")
    print(f"[Status] Pos_Err: {pos_err:.6f} m")

    fz = float(current_force[2]) if len(current_force) > 2 else 0.0
    tfz = float(target_force[2]) if len(target_force) > 2 else 0.0
    print(f"[Force ] Target_Fz: {tfz:.2f}N | Current_Fz: {fz:.2f}N")

    if reward_info:
        print(f"[Reward] Total: {reward_info.get('total', 0):.4f}")

    print("=" * 80)

# ============================================================
# ------------------ RECORDING & PROCESSING ------------------
# ============================================================

def record_step(env_ids, state6, force3, sim_time):
    """env0 데이터 기록"""
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
        print(f"[Visualization] record_step failed: {e}")


def process_episode():
    """에피소드 종료 시 호출: 그래프 저장 및 번호 증가"""
    global _rl_time_buffer, _rl_state_buffer, _rl_force_buffer
    global _rl_start_time, _episode_counter, _debug_episode_counter

    if len(_rl_time_buffer) < 10:
        # 데이터가 너무 적으면 저장하지 않음
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

    print(f"\n[STAMP] Episode {_episode_counter} Results Saved to: {ep_dir}\n")

    _episode_counter += 1
    _debug_episode_counter += 1

    _rl_time_buffer.clear()
    _rl_state_buffer.clear()
    _rl_force_buffer.clear()
    _rl_start_time = None

    return float(np.sum(dremoval))


def save_plots(out_dir, t, state6, force3, dremoval, removal_rate, vxyz):
    """그래프 저장 로직"""
    out_dir.mkdir(parents=True, exist_ok=True)
    xyz = state6[:, :3]

    # 1. Heatmap
    plt.figure(figsize=(6, 5))
    plt.hexbin(xyz[:, 0], xyz[:, 1], C=dremoval, gridsize=30, cmap="jet")
    plt.colorbar(label="Removal")
    plt.savefig(out_dir / "01_removal_heatmap.png")
    plt.close()

    # 2. Rate vs Time
    plt.figure(figsize=(10, 4))
    plt.plot(t, removal_rate)
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "02_heatmap_value_vs_time.png")
    plt.close()

    # 3. Signals
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    v_norm_raw = np.linalg.norm(vxyz, axis=1)
    v_norm = savgol_filter(v_norm_raw, 11, 3) if len(t) > 11 else v_norm_raw
    axs[0].plot(t, v_norm, color="g")
    axs[0].set_title("Velocity Magnitude")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(t, force3[:, 2], color="purple")
    axs[1].set_title("Normal Force (Z)")
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "03_signals_subplot.png")
    plt.close()

    # 4. 3D Path
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=t, cmap="viridis", s=2)
    plt.savefig(out_dir / "04_3d_path_w.png")
    plt.close()

# ============================================================
# ------------------ RL INTERFACE ----------------------------
# ============================================================

def rl_step(env_ids, state6, force3, sim_time, debug_info=None):
    """
    매 스텝 호출됨: 기록 + 화면 출력
    """
    record_step(env_ids, state6, force3, sim_time)

    if debug_info:
        print_polishing_status(
            env_id=0,
            step=debug_info["step"],
            path_index=debug_info["path_idx"],
            traj_len=debug_info["traj_len"],
            done=debug_info["done"],
            pos_err=debug_info["pos_err"],
            target_force=debug_info["target_force"],
            current_force=force3[0],
            reward_info=debug_info.get("reward_dict"),
        )


def rl_episode_done():
    """에피소드 종료 시 호출"""
    return process_episode()

# ============================================================
# -------- Hooks for env_cfg: observation-step / reset -------
# ============================================================

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
    """
    ObservationTerm으로 매 step 호출되는 훅.
    반환값은 dummy tensor이며, side-effect로 visualization 기록/출력을 수행한다.
    """
    try:
        num_envs = env.num_envs
        device = env.device

        env_ids = torch.arange(num_envs, device=device, dtype=torch.long)
        state6 = local_obs.get_ee_pose(env, asset_name=asset_name)  # (N,6)
        wrench6 = local_ft_sensor.get_6axis_ft_fixed_joint(
            env=env,
            asset_name=asset_name,
            fixed_joint_name=fixed_joint_name,
            joint_prim_relpath=joint_prim_relpath,
            verbose=False,
        )
        force3 = wrench6[:, :3]

        term = _safe_get_action_term(env, action_term_name=action_term_name)

        step = int(env.episode_length_buf[0].item()) if hasattr(env, "episode_length_buf") else 0
        sim_time = float(getattr(env, "common_step_counter", 0)) * float(getattr(env, "step_dt", 0.02))

        path_idx = int(term.path_index[0].item()) if hasattr(term, "path_index") else 0
        traj_len = int(getattr(term, "traj_length", 0))
        done = bool(term.path_done[0].item()) if hasattr(term, "path_done") else False

        if hasattr(term, "des_pos"):
            current_xyz_m = state6[0, :3] * 0.001  # state6 is mm
            target_xyz_m = term.des_pos[0]
            pos_err = float(torch.linalg.norm(target_xyz_m - current_xyz_m).item())
        else:
            pos_err = 0.0

        if hasattr(term, "des_force"):
            target_force = term.des_force.detach().cpu().numpy()
        else:
            target_force = np.zeros((num_envs, 3), dtype=float)

        debug_info = {
            "step": step,
            "path_idx": path_idx,
            "traj_len": traj_len,
            "done": done,
            "pos_err": pos_err,
            "target_force": target_force[0] if len(target_force) > 0 else np.zeros(3),
            "reward_dict": None,
        }

        rl_step(
            env_ids=env_ids,
            state6=state6,
            force3=force3,
            sim_time=sim_time,
            debug_info=debug_info,
        )

    except Exception as e:
        print(f"[Visualization] rl_step_hook failed: {e}")

    # dummy observation term output
    return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)


def on_episode_reset(env, env_ids=None):
    """
    EventTerm(mode='reset')에서 호출되는 종료 훅.
    초기 reset에서는 저장하지 않고, 실제 에피소드가 한 번이라도 진행된 후 reset될 때만 저장.
    """
    global _has_seen_any_step

    try:
        if not _has_seen_any_step:
            return

        if len(_rl_time_buffer) > 0:
            rl_episode_done()

    except Exception as e:
        print(f"[Visualization] on_episode_reset failed: {e}")