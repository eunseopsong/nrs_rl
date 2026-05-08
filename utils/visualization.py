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
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
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
    
    # 데이터 파싱
    xyz = state6[:, 0:3]
    # state6의 3~5번 인덱스를 각속도 혹은 방향 벡터(wx, wy, wz)로 간주
    wxyz = state6[:, 3:6] if state6.shape[1] >= 6 else np.zeros_like(xyz)

    # ==========================================
    # 01. Removal Heatmap (레퍼런스 코드와 동일한 디자인)
    # ==========================================
    x = xyz[:, 0]
    y = xyz[:, 1]

    margin = 10.0
    x_min, x_max = np.min(x) - margin, np.max(x) + margin
    y_min, y_max = np.min(y) - margin, np.max(y) + margin
    extent = [x_min, x_max, y_min, y_max]

    grid_bins = 150
    grid_removal, _, _ = np.histogram2d(
        x, y, bins=grid_bins, range=[[x_min, x_max], [y_min, y_max]], weights=dremoval
    )
    grid_removal = grid_removal.T 
    grid_removal = gaussian_filter(grid_removal, sigma=2.5)

    mean_rem = float(np.mean(grid_removal))
    std_rem = float(np.std(grid_removal))
    samples = len(x)
    contact_samples = int(np.sum(force3[:, 2] > 0.5))

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111)
    
    im = ax.imshow(grid_removal, origin="lower", extent=extent, aspect="auto", cmap="viridis")
    ax.set_title("Removal Heatmap")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Cell removal [a.u.]")

    text_str = (
        f"mean = {mean_rem:.6f} [a.u.]\n"
        f"std  = {std_rem:.6f} [a.u.]\n"
        f"samples = {samples}\n"
        f"contact = {contact_samples}"
    )
    ax.text(
        0.02, 0.98, text_str,
        transform=ax.transAxes,
        va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )
    fig.tight_layout()
    fig.savefig(out_dir / "01_removal_heatmap.png", dpi=220)
    plt.close(fig)

    # ==========================================
    # 02. Heatmap Value vs Time (평균선 포함)
    # ==========================================
    rate_mean = float(np.mean(removal_rate))
    
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    ax.plot(t, removal_rate, linewidth=1.5)
    ax.axhline(rate_mean, linestyle="--", linewidth=1.2, label="mean rate", color="orange")
    ax.set_title("Removal Rate at Current Position vs Time")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("Removal Rate [a.u.]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "02_heatmap_value_vs_time.png", dpi=220)
    plt.close(fig)

    # ==========================================
    # 03. Signals Subplot (3x3 Grid)
    # ==========================================
    labels = [
        (xyz[:, 0], "x [mm]"),
        (xyz[:, 1], "y [mm]"),
        (xyz[:, 2], "z [mm]"),
        (wxyz[:, 0], "wx"),
        (wxyz[:, 1], "wy"),
        (wxyz[:, 2], "wz"),
        (force3[:, 0], "fx [N]"),
        (force3[:, 1], "fy [N]"),
        (force3[:, 2], "fz [N]"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(14, 9), sharex=True)
    for ax, (y_data, ylabel) in zip(axes.ravel(), labels):
        ax.plot(t, y_data, linewidth=1.3)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    for ax in axes[-1, :]:
        ax.set_xlabel("time [s]")
        
    fig.suptitle("Recorded Signals: x y z wx wy wz fx fy fz")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(out_dir / "03_signals_subplot.png", dpi=220)
    plt.close(fig)

    # ==========================================
    # 04. 3D Path with Direction Arrows
    # ==========================================
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], linewidth=1.6, label="xyz path")

    # 데이터 개수에 맞춰 화살표 간격(stride) 자동 조절
    w_arrow_stride = max(int(len(t) // 60), 1)
    w_arrow_length_mm = 5.0

    idx = np.arange(0, xyz.shape[0], w_arrow_stride)
    if idx.size > 0:
        xyz_s = xyz[idx]
        w_s = wxyz[idx]
        w_norm = np.linalg.norm(w_s, axis=1, keepdims=True)
        valid = w_norm[:, 0] > 1e-12
        if np.any(valid):
            dirs = np.zeros_like(w_s)
            dirs[valid] = w_s[valid] / w_norm[valid]
            ax.quiver(
                xyz_s[valid, 0], xyz_s[valid, 1], xyz_s[valid, 2],
                dirs[valid, 0], dirs[valid, 1], dirs[valid, 2],
                length=float(w_arrow_length_mm),
                normalize=False,
                linewidth=0.8,
                arrow_length_ratio=0.25,
            )

    ax.set_title("3D Path with Angular-Velocity Direction (wx, wy, wz)")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.legend(loc="best")

    # 3축 비율 동일하게 맞추기 (Aspect ratio)
    x_range = float(np.ptp(xyz[:, 0])) if xyz.shape[0] > 0 else 1.0
    y_range = float(np.ptp(xyz[:, 1])) if xyz.shape[0] > 0 else 1.0
    z_range = float(np.ptp(xyz[:, 2])) if xyz.shape[0] > 0 else 1.0
    max_range = max(x_range, y_range, z_range, 1.0)
    
    x_mid = float(np.mean([np.min(xyz[:, 0]), np.max(xyz[:, 0])]))
    y_mid = float(np.mean([np.min(xyz[:, 1]), np.max(xyz[:, 1])]))
    z_mid = float(np.mean([np.min(xyz[:, 2]), np.max(xyz[:, 2])]))
    
    ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
    ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
    ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)

    fig.tight_layout()
    fig.savefig(out_dir / "04_3d_path_w.png", dpi=220)
    plt.close(fig)

    # ==========================================
    # 05. Force vs Velocity Correlation (가장 중요한 학습 증명 지표!)
    # ==========================================
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    fz_abs = np.abs(force3[:, 2])
    v_norm_raw = np.linalg.norm(vxyz, axis=1)
    
    # 실제 에이전트가 찍은 힘-속도 산점도
    ax.scatter(fz_abs, v_norm_raw, alpha=0.5, c='b', s=10, label="Agent Data (F vs V)")
    
    # 이상적인 반비례 곡선 (Target MRR Curve)
    # 현재 데이터의 평균 MRR을 기준으로 이상적인 y = C/x 곡선을 그립니다.
    mean_mrr = np.mean(fz_abs * v_norm_raw)
    f_ideal = np.linspace(max(0.1, np.min(fz_abs)), np.max(fz_abs), 100)
    v_ideal = mean_mrr / f_ideal
    ax.plot(f_ideal, v_ideal, 'r--', linewidth=2, label=f"Ideal Inverse Curve\n(V = {mean_mrr:.4f} / F)")
    
    ax.set_title("Force vs Velocity: Inverse Correlation Check")
    ax.set_xlabel("Normal Force |Fz| [N]")
    ax.set_ylabel("Velocity ||V|| [m/s]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "05_force_vel_correlation.png", dpi=220)
    plt.close(fig)

    # ==========================================
    # 06. MRR Error Convergence (가공량 오차 수렴도)
    # ==========================================
    fig = plt.figure(figsize=(9, 4))
    ax = fig.add_subplot(111)
    
    current_mrr_array = fz_abs * v_norm_raw
    # 목표 MRR 선 (평균으로 가정하거나, 설정한 Target MRR 값을 넣어도 됨)
    target_mrr_line = np.full_like(t, rate_mean) 
    
    ax.plot(t, current_mrr_array, linewidth=1.5, label="Current MRR (F x V)")
    ax.plot(t, target_mrr_line, 'r--', linewidth=1.5, label="Target MRR")
    
    # 두 선 사이의 면적을 칠해서 '오차(Error)'를 시각적으로 강조
    ax.fill_between(t, current_mrr_array, target_mrr_line, color='red', alpha=0.2, label="MRR Error")
    
    ax.set_title("MRR Tracking Performance Over Time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("MRR")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "06_mrr_tracking_error.png", dpi=220)
    plt.close(fig)

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