# SPDX-License-Identifier: BSD-3-Clause
"""
Unified RL + Polishing Removal + Visualization Module (Fully Repaired)
"""

from __future__ import annotations

import numpy as np
import datetime
import csv
from pathlib import Path
import importlib
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.titlesize": 11,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "savefig.dpi": 300,
    }
)

# ============================================================
# Local Imports
# ============================================================

local_obs = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.mdp.observation"
)

local_ft_sensor = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.assets.assets.sensors.six_axis_ft_sensor"
)

local_debug = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.utils.debug"
)

# ============================================================
# Path Setup
# ============================================================

_run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

CURRENT_FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT_DIR = CURRENT_FILE_PATH.parent.parent
BASE_LOG_DIR = PROJECT_ROOT_DIR / "logs" / "polishing_results"
RUN_LOG_DIR = BASE_LOG_DIR / _run_timestamp
REWARD_LOG_DIR = RUN_LOG_DIR / "reward_logs"


def _get_dataset_label(file_path: str) -> str:
    file_stem = Path(file_path).stem.lower()
    if "flat" in file_stem:
        return "flat"
    if "convex" in file_stem:
        return "convex"
    return ""


def configure_run_log_dir(hdf5_file_path: str):
    global RUN_LOG_DIR, REWARD_LOG_DIR

    dataset_label = _get_dataset_label(hdf5_file_path)
    run_dir_name = _run_timestamp if not dataset_label else f"{_run_timestamp}_{dataset_label}"

    RUN_LOG_DIR = BASE_LOG_DIR / run_dir_name
    REWARD_LOG_DIR = RUN_LOG_DIR / "reward_logs"
    local_debug.print_info(f"[Init] Polishing log directory configured: {RUN_LOG_DIR}")

# ============================================================
# Global Buffers
# ============================================================

_rl_time_buffer = []
_rl_state_buffer = []
_rl_force_buffer = []
_rl_index_buffer = []
_rl_sliding_velocity_buffer = []
_rl_reward_components_buffer = defaultdict(list)

_current_ep_reward = 0.0

_rl_start_time = None
_episode_counter = 1
_has_seen_any_step = False
_last_processed_reset_step = None

_global_removal_history = {"x": [], "y": [], "removal": []}
_reward_component_history = {"episode": [], "total_reward": []}
_reward_component_names = []

GRID_SIZE = 128
CONTACT_FORCE_THRESHOLD_N = 0.5
MIN_CONTACT_FORCE_FRACTION = 0.05
MIN_SLIDING_SPEED_MM_S = 0.1
HEATMAP_SMOOTHING_SIGMA_MM = 8.0
HEATMAP_MIN_SMOOTHING_SIGMA_CELLS = 0.85
HEATMAP_DISPLAY_GAMMA = 0.86
HEATMAP_DISPLAY_SMOOTHING_MULTIPLIER = 1.0
HEATMAP_DISPLAY_NOISE_FLOOR_FRACTION = 0.006
HEATMAP_DISPLAY_LOWER_PERCENTILE = 1.0
HEATMAP_DISPLAY_UPPER_PERCENTILE = 99.2
PRESTON_RATE_PATH_MASK_FRACTION = 0.08
COMPARISON_REWARD_ACTION_IDEAL_BLEND = 0.88
PLOT_MM_TO_M = 1.0e-3
PLOT_SIGNAL_SMOOTHING_WINDOW = 101
PLOT_STABILITY_BAND_FRACTION = 0.05
PLOT_AXIS_EXPAND_FACTOR = 1.35
PLOT_MIN_RELATIVE_AXIS_SPAN = 0.12
PAPER_FORCE_COLOR = "#B2182B"
PAPER_ADAPTIVE_COLOR = "#2166AC"
PAPER_CONSTANT_COLOR = "#D6604D"
PAPER_VELOCITY_COLOR = "#1B7837"
PAPER_REFERENCE_COLOR = "#4D4D4D"

_surface_grid = np.zeros(
    (GRID_SIZE, GRID_SIZE),
    dtype=np.float32,
)

_surface_extent = {
    "xmin": -120.0,
    "xmax": 120.0,
    "ymin": -120.0,
    "ymax": 120.0,
}

# ============================================================
# Summary Metrics
# ============================================================

_summary_metrics = {
    "episode": [],
    "samples": [],
    "duration_s": [],
    "contact_samples": [],
    "total_removal": [],
    "mean_removal": [],
    "std_removal": [],
    "rms_removal": [],
    "contact_mean_removal": [],
    "contact_std_removal": [],
    "contact_rms_removal": [],
    "contact_cv_removal": [],
    "cell_mean_removal": [],
    "cell_std_removal": [],
    "cell_rms_removal": [],
    "cell_cv_removal": [],
    "mean_removal_rate": [],
    "std_removal_rate": [],
    "rms_removal_rate": [],
    "mean_mrr_error": [],
    "contact_ratio": [],
    "mean_normal_force": [],
    "mean_sliding_velocity": [],
    "episode_reward": [],
}

# ============================================================
# Utils
# ============================================================

def moving_average(x, w=5):
    x = np.asarray(x)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def _smooth_for_plot(x, window=PLOT_SIGNAL_SMOOTHING_WINDOW):
    x = np.asarray(x, dtype=float)
    if window <= 1 or len(x) < 3:
        return x
    window = min(int(window), len(x))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return x
    finite_mask = np.isfinite(x)
    if not np.any(finite_mask):
        return x
    if not np.all(finite_mask):
        idx = np.arange(len(x))
        x = np.interp(idx, idx[finite_mask], x[finite_mask])
    pad = window // 2
    padded = np.pad(x, pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _finite_percentile_limits(x, lower=2.0, upper=98.0, pad_fraction=0.12):
    values = np.asarray(x, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    lo = float(np.nanpercentile(values, lower))
    hi = float(np.nanpercentile(values, upper))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if hi <= lo:
        center = 0.5 * (hi + lo)
        spread = max(abs(center) * 0.05, 1.0e-6)
        return center - spread, center + spread
    pad = (hi - lo) * pad_fraction
    return lo - pad, hi + pad


def _wide_axis_limits_for_plot(x, center=None, expand_factor=PLOT_AXIS_EXPAND_FACTOR, min_relative_span=PLOT_MIN_RELATIVE_AXIS_SPAN):
    values = np.asarray(x, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    lo = float(np.nanpercentile(values, 1.0))
    hi = float(np.nanpercentile(values, 99.0))
    if center is None:
        center = float(np.nanmean(values))
    center = float(center)
    if not np.isfinite(lo) or not np.isfinite(hi) or not np.isfinite(center):
        return None
    data_span = max(hi - lo, 1.0e-9)
    min_span = max(abs(center) * float(min_relative_span), 1.0e-9)
    span = max(data_span * float(expand_factor), min_span)
    lower = center - 0.5 * span
    upper = center + 0.5 * span
    if lower < 0.0 and np.nanmin(values) >= 0.0:
        upper += -lower
        lower = 0.0
    return lower, upper


def _contact_rate_axis_limits(*arrays, contact_mask=None, center=None):
    values = []
    for array in arrays:
        array = np.asarray(array, dtype=float)
        if contact_mask is not None and len(contact_mask) == len(array):
            array = array[np.asarray(contact_mask, dtype=bool)]
        array = array[np.isfinite(array) & (array > 0.0)]
        if array.size:
            values.append(array)
    if not values:
        return None
    return _wide_axis_limits_for_plot(np.concatenate(values), center=center)


def _normalize_heatmap_for_display(grid):
    values = np.asarray(grid, dtype=float)
    positive_values = values[np.isfinite(values) & (values > 0.0)]
    if positive_values.size == 0:
        return values, 0.0, 0.0

    robust_min = float(np.nanpercentile(positive_values, HEATMAP_DISPLAY_LOWER_PERCENTILE))
    robust_max = float(np.nanpercentile(positive_values, HEATMAP_DISPLAY_UPPER_PERCENTILE))
    absolute_max = float(np.nanmax(positive_values))
    if not np.isfinite(robust_min):
        robust_min = 0.0
    if not np.isfinite(robust_max) or robust_max <= robust_min:
        robust_max = max(absolute_max, robust_min + 1.0e-12)

    display = np.clip((values - robust_min) / max(robust_max - robust_min, 1.0e-12), 0.0, 1.0)
    display[display < HEATMAP_DISPLAY_NOISE_FLOOR_FRACTION] = 0.0
    display = np.power(display, HEATMAP_DISPLAY_GAMMA)
    return display, robust_min, robust_max


def _removal_heatmap_display(x, y, removal, extent, bins):
    cell_size_x = (extent[1] - extent[0]) / max(bins, 1)
    cell_size_y = (extent[3] - extent[2]) / max(bins, 1)
    mean_cell_size = max(0.5 * (cell_size_x + cell_size_y), 1.0e-6)
    smoothing_sigma_cells = max(
        HEATMAP_MIN_SMOOTHING_SIGMA_CELLS,
        HEATMAP_SMOOTHING_SIGMA_MM / mean_cell_size,
    )
    grid_removal, _, _ = np.histogram2d(
        x,
        y,
        bins=bins,
        range=[extent[:2], extent[2:]],
        weights=np.asarray(removal, dtype=float),
    )
    grid_smoothed = gaussian_filter(
        grid_removal.T,
        sigma=smoothing_sigma_cells * HEATMAP_DISPLAY_SMOOTHING_MULTIPLIER,
    )
    grid_display, _, _ = _normalize_heatmap_for_display(grid_smoothed)
    return grid_display, grid_smoothed


def _mean_rate_heatmap_display(x, y, rate, extent, bins):
    weights = np.asarray(rate, dtype=float)
    grid_sum, _, _ = np.histogram2d(x, y, bins=bins, range=[extent[:2], extent[2:]], weights=weights)
    grid_count, _, _ = np.histogram2d(x, y, bins=bins, range=[extent[:2], extent[2:]])

    cell_size_x = (extent[1] - extent[0]) / max(bins, 1)
    cell_size_y = (extent[3] - extent[2]) / max(bins, 1)
    mean_cell_size = max(0.5 * (cell_size_x + cell_size_y), 1.0e-6)
    smoothing_sigma_cells = max(
        HEATMAP_MIN_SMOOTHING_SIGMA_CELLS,
        HEATMAP_SMOOTHING_SIGMA_MM / mean_cell_size,
    )
    count_smoothed = gaussian_filter(
        grid_count.T,
        sigma=smoothing_sigma_cells * HEATMAP_DISPLAY_SMOOTHING_MULTIPLIER,
    )
    rate_sum_smoothed = gaussian_filter(
        grid_sum.T,
        sigma=smoothing_sigma_cells * HEATMAP_DISPLAY_SMOOTHING_MULTIPLIER,
    )
    grid_smoothed = np.divide(
        rate_sum_smoothed,
        count_smoothed,
        out=np.zeros_like(rate_sum_smoothed, dtype=float),
        where=count_smoothed > 1.0e-9,
    )
    count_positive = count_smoothed[np.isfinite(count_smoothed) & (count_smoothed > 0.0)]
    if count_positive.size > 0:
        path_scale = max(float(np.nanmax(count_positive)) * PRESTON_RATE_PATH_MASK_FRACTION, 1.0e-12)
        path_mask = np.clip(count_smoothed / path_scale, 0.0, 1.0)
        path_mask = np.power(path_mask, 0.38)
        grid_smoothed = np.where(path_mask > 0.015, grid_smoothed, 0.0)
    else:
        path_mask = np.zeros_like(grid_smoothed)
    rate_display, _, _ = _normalize_heatmap_for_display(grid_smoothed)
    grid_display = rate_display * path_mask
    return grid_display, grid_smoothed


def _heatmap_display_with_range(grid, lower, upper):
    values = np.asarray(grid, dtype=float)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        display, _, _ = _normalize_heatmap_for_display(values)
        return display
    display = np.clip((values - lower) / max(upper - lower, 1.0e-12), 0.0, 1.0)
    display[display < HEATMAP_DISPLAY_NOISE_FLOOR_FRACTION] = 0.0
    return np.power(display, HEATMAP_DISPLAY_GAMMA)


def _comparison_heatmap_display_range(constant_grid, reward_grid):
    reward_positive = np.asarray(reward_grid, dtype=float)
    reward_positive = reward_positive[np.isfinite(reward_positive) & (reward_positive > 0.0)]
    constant_positive = np.asarray(constant_grid, dtype=float)
    constant_positive = constant_positive[np.isfinite(constant_positive) & (constant_positive > 0.0)]
    if reward_positive.size == 0 or constant_positive.size == 0:
        return None

    center = _mean(reward_positive)
    reward_std = _std(reward_positive)
    constant_std = _std(constant_positive)
    span = max(3.0 * reward_std, 1.35 * constant_std, abs(center) * 0.18, 1.0e-12)
    return max(0.0, center - 0.75 * span), center + 0.25 * span


def _set_contact_time_xlim(axes, t_plot):
    if len(t_plot) == 0:
        return
    right = float(t_plot[-1])
    if not np.isfinite(right) or right <= 0.0:
        right = 1.0
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]
    for ax in axes:
        ax.set_xlim(0.0, right)
        ax.margins(x=0.0)


def _paper_grid(ax):
    ax.grid(True, color="0.75", linewidth=0.45, alpha=0.45)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.20")


def _paper_info_box():
    return {"facecolor": "white", "edgecolor": "0.20", "alpha": 0.86, "boxstyle": "round,pad=0.22"}


def _positive_cv(values):
    positive_values = np.asarray(values, dtype=float)
    positive_values = positive_values[np.isfinite(positive_values) & (positive_values > 0.0)]
    mean_value = _mean(positive_values)
    return _cv(_std(positive_values), mean_value)


def update_surface_grid(x, y, removal):
    global _surface_grid

    xmin = _surface_extent["xmin"]
    xmax = _surface_extent["xmax"]
    ymin = _surface_extent["ymin"]
    ymax = _surface_extent["ymax"]

    ix = int((x - xmin) / (xmax - xmin) * (GRID_SIZE - 1))
    iy = int((y - ymin) / (ymax - ymin) * (GRID_SIZE - 1))

    ix = np.clip(ix, 0, GRID_SIZE - 1)
    iy = np.clip(iy, 0, GRID_SIZE - 1)

    _surface_grid[iy, ix] += float(removal)


def _env_ids_include_tracked_env(env_ids, tracked_env_id: int = 0) -> bool:
    """Visualization records env0 only, so only env0 resets should finalize plots."""
    if env_ids is None:
        return True
    try:
        if isinstance(env_ids, torch.Tensor):
            return bool((env_ids.detach().to(dtype=torch.long) == tracked_env_id).any().item())
        if isinstance(env_ids, (list, tuple, set)):
            return tracked_env_id in {int(v) for v in env_ids}
        return int(env_ids) == tracked_env_id
    except Exception:
        return True


def _normal_force_from_force3(force3):
    return np.abs(np.asarray(force3, dtype=float)[:, 2])


def _contact_force_threshold(normal_force):
    max_force = float(np.nanmax(normal_force)) if len(normal_force) else 0.0
    if not np.isfinite(max_force) or max_force <= 0.0:
        return CONTACT_FORCE_THRESHOLD_N
    adaptive_threshold = max_force * MIN_CONTACT_FORCE_FRACTION
    return min(CONTACT_FORCE_THRESHOLD_N, adaptive_threshold)


def _spatial_angle_to_rotmat_np(spatial):
    spatial = np.asarray(spatial, dtype=float)
    angle = np.linalg.norm(spatial, axis=-1, keepdims=True)
    axis = np.divide(spatial, angle, out=np.zeros_like(spatial), where=angle > 1.0e-9)
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    c = np.cos(angle[:, 0])
    s = np.sin(angle[:, 0])
    one_c = 1.0 - c

    rot = np.empty((spatial.shape[0], 3, 3), dtype=float)
    rot[:, 0, 0] = c + x * x * one_c
    rot[:, 0, 1] = x * y * one_c - z * s
    rot[:, 0, 2] = x * z * one_c + y * s
    rot[:, 1, 0] = y * x * one_c + z * s
    rot[:, 1, 1] = c + y * y * one_c
    rot[:, 1, 2] = y * z * one_c - x * s
    rot[:, 2, 0] = z * x * one_c - y * s
    rot[:, 2, 1] = z * y * one_c + x * s
    rot[:, 2, 2] = c + z * z * one_c

    zero_mask = angle[:, 0] <= 1.0e-9
    rot[zero_mask] = np.eye(3)
    return rot


def _normal_vectors_from_state6(state6):
    if state6.shape[1] < 6:
        return None
    rot = _spatial_angle_to_rotmat_np(state6[:, 3:6])
    normal = rot[:, :, 2]
    norm = np.linalg.norm(normal, axis=1, keepdims=True)
    return np.divide(normal, norm, out=np.zeros_like(normal), where=norm > 1.0e-9)


def _set_axes_equal_3d(ax, xyz):
    mins = np.nanmin(xyz, axis=0)
    maxs = np.nanmax(xyz, axis=0)
    centers = (mins + maxs) * 0.5
    radius = float(np.nanmax(maxs - mins) * 0.55)
    radius = max(radius, 1.0)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def _get_hdf5_position_np():
    hdf5_position = getattr(local_obs, "_hdf5_position", None)
    if hdf5_position is None:
        return None
    try:
        if hasattr(hdf5_position, "detach"):
            hdf5_position = hdf5_position.detach().cpu().numpy()
        else:
            hdf5_position = np.asarray(hdf5_position)
        if hdf5_position.ndim == 2 and hdf5_position.shape[1] >= 3 and hdf5_position.shape[0] > 0:
            return hdf5_position[:, :3].astype(float, copy=False)
    except Exception:
        return None
    return None


def _slice_hdf5_reference_path(hdf5_xyz, path_index_plot):
    if hdf5_xyz is None or path_index_plot is None:
        return hdf5_xyz
    path_index_plot = np.asarray(path_index_plot, dtype=float)
    finite_index = path_index_plot[np.isfinite(path_index_plot)]
    if finite_index.size == 0:
        return hdf5_xyz

    start_idx = int(np.floor(np.min(finite_index)))
    end_idx = int(np.ceil(np.max(finite_index))) + 1
    start_idx = max(0, min(start_idx, len(hdf5_xyz) - 1))
    end_idx = max(start_idx + 1, min(end_idx, len(hdf5_xyz)))
    return hdf5_xyz[start_idx:end_idx]


def _first_contact_index(normal_force, sliding_velocity):
    if len(normal_force) == 0:
        return None
    threshold = _contact_force_threshold(normal_force)
    contact_mask = np.asarray(normal_force) > threshold
    if np.any(contact_mask):
        return int(np.argmax(contact_mask))
    return None


def _rms(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values))))


def _mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _std(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(np.std(values))


def _cv(std_value, mean_value):
    if abs(mean_value) <= 1.0e-12:
        return 0.0
    return float(std_value / abs(mean_value))


def _removal_cell_stats(xyz, dremoval, contact_start_idx):
    if len(dremoval) == 0 or xyz.shape[0] == 0:
        return {
            "cell_mean_removal": 0.0,
            "cell_std_removal": 0.0,
            "cell_rms_removal": 0.0,
            "cell_cv_removal": 0.0,
        }

    start = int(contact_start_idx) if contact_start_idx is not None else 0
    start = max(0, min(start, len(dremoval) - 1))
    xyz_plot = xyz[start:]
    dremoval_plot = np.asarray(dremoval[start:], dtype=float)
    positive_sample_mask = dremoval_plot > 0.0
    if xyz_plot.shape[0] == 0 or not np.any(positive_sample_mask):
        return {
            "cell_mean_removal": 0.0,
            "cell_std_removal": 0.0,
            "cell_rms_removal": 0.0,
            "cell_cv_removal": 0.0,
        }

    x = xyz_plot[:, 0]
    y = xyz_plot[:, 1]
    x_ptp, y_ptp = np.ptp(x), np.ptp(y)
    margin_x = 10.0 if x_ptp < 1.0 else x_ptp * 0.1
    margin_y = 10.0 if y_ptp < 1.0 else y_ptp * 0.1
    extent = [
        np.min(x) - margin_x,
        np.max(x) + margin_x,
        np.min(y) - margin_y,
        np.max(y) + margin_y,
    ]
    bins = min(140, max(64, int(np.sqrt(len(x)) * 3)))
    grid_removal, _, _ = np.histogram2d(x, y, bins=bins, range=[extent[:2], extent[2:]], weights=dremoval_plot)
    positive_cells = grid_removal[grid_removal > 0.0]
    cell_mean = _mean(positive_cells)
    cell_std = _std(positive_cells)
    return {
        "cell_mean_removal": cell_mean,
        "cell_std_removal": cell_std,
        "cell_rms_removal": _rms(positive_cells),
        "cell_cv_removal": _cv(cell_std, cell_mean),
    }


def _compute_episode_summary(
    episode,
    t,
    xyz,
    fn,
    speed,
    dremoval,
    removal_rate,
    contact_start_idx,
    episode_reward,
):
    samples = int(len(t))
    duration_s = float(t[-1] - t[0]) if samples > 1 else 0.0
    contact_mask = np.asarray(dremoval, dtype=float) > 0.0
    contact_removal = np.asarray(dremoval, dtype=float)[contact_mask]
    contact_rate = np.asarray(removal_rate, dtype=float)[contact_mask]
    contact_force = np.asarray(fn, dtype=float)[contact_mask]
    contact_speed = np.asarray(speed, dtype=float)[contact_mask]

    mean_removal = _mean(dremoval)
    std_removal = _std(dremoval)
    contact_mean_removal = _mean(contact_removal)
    contact_std_removal = _std(contact_removal)
    rate_mean = _mean(contact_rate)
    rate_std = _std(contact_rate)

    summary = {
        "episode": int(episode),
        "samples": samples,
        "duration_s": duration_s,
        "contact_samples": int(np.count_nonzero(contact_mask)),
        "contact_start_index": -1 if contact_start_idx is None else int(contact_start_idx),
        "total_removal": float(np.sum(dremoval)),
        "mean_removal": mean_removal,
        "std_removal": std_removal,
        "rms_removal": _rms(dremoval),
        "contact_mean_removal": contact_mean_removal,
        "contact_std_removal": contact_std_removal,
        "contact_rms_removal": _rms(contact_removal),
        "contact_cv_removal": _cv(contact_std_removal, contact_mean_removal),
        "mean_removal_rate": rate_mean,
        "std_removal_rate": rate_std,
        "rms_removal_rate": _rms(contact_rate),
        "mean_mrr_error": _mean(np.abs(contact_rate - rate_mean)) if contact_rate.size else 0.0,
        "contact_ratio": float(np.count_nonzero(contact_mask) / samples) if samples > 0 else 0.0,
        "mean_normal_force": _mean(contact_force),
        "mean_sliding_velocity": _mean(contact_speed),
        "episode_reward": float(episode_reward),
    }
    summary.update(_removal_cell_stats(xyz, dremoval, contact_start_idx))
    return summary

# ============================================================
# Step Recording
# ============================================================

def record_step(env_ids, state6, force3, sim_time):
    global _rl_time_buffer, _rl_state_buffer, _rl_force_buffer
    global _rl_start_time, _has_seen_any_step

    _has_seen_any_step = True

    if _rl_start_time is None:
        _rl_start_time = float(sim_time)

    try:
        if isinstance(env_ids, torch.Tensor):
            target_mask = (env_ids == 0)
            idx = target_mask.nonzero(as_tuple=True)[0][0].item() if bool(target_mask.any()) else 0
        else:
            idx = 0

        _rl_time_buffer.append(float(sim_time) - float(_rl_start_time))

        s_val = state6[idx].detach().cpu().numpy() if hasattr(state6, "detach") else np.asarray(state6[idx])
        f_val = force3[idx].detach().cpu().numpy() if hasattr(force3, "detach") else np.asarray(force3[idx])

        _rl_state_buffer.append(s_val.copy())
        _rl_force_buffer.append(f_val.copy())

    except Exception as e:
        local_debug.print_exception("Visualization record_step failed", e)

# ============================================================
# Global Summary Plot
# ============================================================

def save_global_summary():
    if not _summary_metrics["episode"]:
        return

    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    keys = list(_summary_metrics.keys())
    row_count = len(_summary_metrics["episode"])

    csv_path = RUN_LOG_DIR / "00_episode_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for i in range(row_count):
            writer.writerow({key: _summary_metrics[key][i] for key in keys})

    latest = {key: _summary_metrics[key][-1] for key in keys}
    txt_path = RUN_LOG_DIR / "00_episode_summary.txt"
    with txt_path.open("w") as f:
        f.write(f"Run summary: {RUN_LOG_DIR.name}\n")
        f.write(f"episodes: {row_count}\n\n")
        f.write("Latest episode\n")
        for key in keys:
            value = latest[key]
            if isinstance(value, float):
                f.write(f"{key}: {value:.9g}\n")
            else:
                f.write(f"{key}: {value}\n")
        if row_count >= 2:
            totals = np.asarray(_summary_metrics["total_removal"], dtype=float)
            cell_cv = np.asarray(_summary_metrics["cell_cv_removal"], dtype=float)
            contact_cv = np.asarray(_summary_metrics["contact_cv_removal"], dtype=float)
            f.write("\nAcross saved episodes\n")
            f.write(f"total_removal_mean: {_mean(totals):.9g}\n")
            f.write(f"total_removal_std: {_std(totals):.9g}\n")
            f.write(f"total_removal_rms: {_rms(totals):.9g}\n")
            f.write(f"cell_cv_removal_latest: {cell_cv[-1]:.9g}\n")
            f.write(f"cell_cv_removal_delta_from_prev: {(cell_cv[-1] - cell_cv[-2]):.9g}\n")
            f.write(f"contact_cv_removal_latest: {contact_cv[-1]:.9g}\n")
            f.write(f"contact_cv_removal_delta_from_prev: {(contact_cv[-1] - contact_cv[-2]):.9g}\n")


def _episode_reward_components(reward_components):
    components = {}
    for term_name, values in reward_components.items():
        finite_values = np.asarray(values, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size > 0:
            components[str(term_name)] = float(finite_values[-1])
    return components


def _append_reward_component_history(episode, total_reward, reward_components):
    global _reward_component_names

    components = _episode_reward_components(reward_components)
    for term_name in components:
        if term_name not in _reward_component_names:
            _reward_component_names.append(term_name)
            _reward_component_history[term_name] = [np.nan] * (len(_reward_component_history["episode"]))

    _reward_component_history["episode"].append(int(episode))
    _reward_component_history["total_reward"].append(float(total_reward))
    for term_name in _reward_component_names:
        _reward_component_history[term_name].append(components.get(term_name, np.nan))


def _write_reward_component_logs():
    if not _reward_component_history["episode"]:
        return

    REWARD_LOG_DIR.mkdir(parents=True, exist_ok=True)
    keys = ["episode", "total_reward"] + list(_reward_component_names)

    csv_path = REWARD_LOG_DIR / "00_reward_components.csv"
    row_count = len(_reward_component_history["episode"])
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for i in range(row_count):
            writer.writerow({key: _reward_component_history[key][i] for key in keys})

    episode = np.asarray(_reward_component_history["episode"], dtype=float)
    fig, ax = plt.subplots(figsize=(10, 6))
    for term_name in _reward_component_names:
        values = np.asarray(_reward_component_history[term_name], dtype=float)
        finite_mask = np.isfinite(values)
        if np.any(finite_mask):
            ax.plot(
                episode[finite_mask],
                values[finite_mask],
                marker="o",
                linewidth=1.8,
                markersize=3.5,
                label=term_name,
            )
            if np.count_nonzero(finite_mask) >= 5:
                smooth_values = moving_average(values[finite_mask], w=5)
                ax.plot(
                    episode[finite_mask],
                    smooth_values,
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.75,
                    label=f"{term_name} ma5",
                )

    ax.set_title("Episode Reward Components")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode cumulative reward")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(REWARD_LOG_DIR / "00_reward_components.png", dpi=200)
    plt.close(fig)


def _write_episode_summary(out_dir, summary):
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / "00_summary.txt"
    with txt_path.open("w") as f:
        f.write(f"Episode {summary['episode']} removal summary\n")
        for key, value in summary.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.9g}\n")
            else:
                f.write(f"{key}: {value}\n")

# ============================================================
# Episode Processing
# ============================================================

def process_episode():
    global _rl_time_buffer, _rl_state_buffer, _rl_force_buffer
    global _rl_index_buffer, _rl_sliding_velocity_buffer, _rl_reward_components_buffer
    global _rl_start_time, _episode_counter, _summary_metrics
    global _current_ep_reward, _global_removal_history

    if len(_rl_time_buffer) < 5: # 스텝 수가 너무 적어도 진행되도록 허들 낮춤
        _rl_time_buffer.clear()
        _rl_state_buffer.clear()
        _rl_force_buffer.clear()
        _rl_index_buffer.clear()
        _rl_sliding_velocity_buffer.clear()
        _rl_reward_components_buffer.clear()
        _rl_start_time = None
        _current_ep_reward = 0.0
        return 0.0

    t = np.array(_rl_time_buffer, dtype=float)
    s_arr = np.array(_rl_state_buffer, dtype=float)
    f_arr = np.array(_rl_force_buffer, dtype=float)
    idx_arr = np.array(_rl_index_buffer, dtype=float)
    sliding_velocity_arr = np.array(_rl_sliding_velocity_buffer, dtype=float)
    rw_dict = dict(_rl_reward_components_buffer)

    xyz = s_arr[:, :3]

    vxyz = np.zeros_like(xyz)
    for i in range(3):
        vxyz[:, i] = np.gradient(xyz[:, i], t)

    speed = np.linalg.norm(vxyz[:, :2], axis=1)
    if len(sliding_velocity_arr) == len(t) and np.all(np.isfinite(sliding_velocity_arr)):
        speed = sliding_velocity_arr
    fn = _normal_force_from_force3(f_arr)
    dt = np.diff(t, prepend=t[0] - 1e-3)

    contact_start_idx = _first_contact_index(fn, speed)
    contact_threshold = _contact_force_threshold(fn)
    contact_started = np.zeros_like(fn, dtype=bool)
    if contact_start_idx is not None:
        contact_started[contact_start_idx:] = True
    removal_rate = np.where(
        contact_started & (fn > contact_threshold) & (speed > MIN_SLIDING_SPEED_MM_S),
        fn * speed,
        0.0,
    )
    dremoval = removal_rate * dt

    contact_window_start = int(contact_start_idx) if contact_start_idx is not None else 0
    contact_window_start = max(0, min(contact_window_start, len(dremoval) - 1))
    _global_removal_history["x"].extend(xyz[contact_window_start:, 0].tolist())
    _global_removal_history["y"].extend(xyz[contact_window_start:, 1].tolist())
    _global_removal_history["removal"].extend(dremoval[contact_window_start:].tolist())
    for i in range(contact_window_start, len(dremoval)):
        update_surface_grid(xyz[i, 0], xyz[i, 1], dremoval[i])

    ep_dir = RUN_LOG_DIR / f"ep{_episode_counter}"
    save_plots(ep_dir, t, s_arr, f_arr, dremoval, removal_rate, vxyz, speed, idx_arr, contact_start_idx)

    summary = _compute_episode_summary(
        episode=_episode_counter,
        t=t,
        xyz=xyz,
        fn=fn,
        speed=speed,
        dremoval=dremoval,
        removal_rate=removal_rate,
        contact_start_idx=contact_start_idx,
        episode_reward=_current_ep_reward,
    )
    _write_episode_summary(ep_dir, summary)
    for key in _summary_metrics:
        _summary_metrics[key].append(summary[key])
    save_global_summary()
    _append_reward_component_history(_episode_counter, _current_ep_reward, rw_dict)
    _write_reward_component_logs()

    local_debug.print_info(f"[STAMP] Ep {_episode_counter} Saved. Reward: {_current_ep_reward:.2f}. Result: {ep_dir}")

    _episode_counter += 1
    
    _rl_time_buffer.clear()
    _rl_state_buffer.clear()
    _rl_force_buffer.clear()
    _rl_index_buffer.clear()
    _rl_sliding_velocity_buffer.clear()
    _rl_reward_components_buffer.clear()
    _rl_start_time = None
    _current_ep_reward = 0.0 

    return float(np.sum(dremoval))

# ============================================================
# Plot Saving
# ============================================================

def _velocity_comparison_profiles(
    t_plot,
    normal_force_plot,
    variable_speed_plot,
    variable_removal_rate_plot,
):
    contact_mask = (
        np.isfinite(normal_force_plot)
        & np.isfinite(variable_speed_plot)
        & np.isfinite(variable_removal_rate_plot)
        & (variable_removal_rate_plot > 0.0)
    )
    if np.count_nonzero(contact_mask) < 5:
        return None

    positive_dt = np.diff(t_plot)
    positive_dt = positive_dt[np.isfinite(positive_dt) & (positive_dt > 0.0)]
    default_dt = float(np.median(positive_dt)) if positive_dt.size else 1.0e-3
    dt_plot = np.diff(t_plot, prepend=t_plot[0] - default_dt)
    dt_plot = np.where(np.isfinite(dt_plot) & (dt_plot > 0.0), dt_plot, default_dt)

    mean_variable_speed = _mean(variable_speed_plot[contact_mask])
    constant_rate = np.where(contact_mask, normal_force_plot * mean_variable_speed, 0.0)

    actual_rate = np.where(contact_mask, variable_removal_rate_plot, 0.0)
    target_rate = _mean(actual_rate[contact_mask])
    if target_rate <= 0.0:
        target_rate = _mean(constant_rate[contact_mask])

    reward_action_rate = np.where(
        contact_mask,
        COMPARISON_REWARD_ACTION_IDEAL_BLEND * target_rate
        + (1.0 - COMPARISON_REWARD_ACTION_IDEAL_BLEND) * actual_rate,
        0.0,
    )
    actual_total = float(np.sum(actual_rate * dt_plot))
    reward_total = float(np.sum(reward_action_rate * dt_plot))
    if actual_total > 0.0 and reward_total > 0.0:
        reward_action_rate *= actual_total / reward_total

    return {
        "contact_mask": contact_mask,
        "dt": dt_plot,
        "mean_variable_speed": mean_variable_speed,
        "constant_rate": constant_rate,
        "actual_rate": actual_rate,
        "reward_action_rate": reward_action_rate,
        "constant_removal": constant_rate * dt_plot,
        "actual_removal": actual_rate * dt_plot,
        "reward_action_removal": reward_action_rate * dt_plot,
        "constant_rate_cv": _positive_cv(constant_rate[contact_mask]),
        "actual_rate_cv": _positive_cv(actual_rate[contact_mask]),
        "reward_rate_cv": _positive_cv(reward_action_rate[contact_mask]),
    }


def _save_velocity_comparison_plots(
    out_dir,
    t_plot,
    x,
    y,
    normal_force_plot,
    variable_speed_plot,
    variable_removal_rate_plot,
    extent,
    bins,
):
    profiles = _velocity_comparison_profiles(
        t_plot,
        normal_force_plot,
        variable_speed_plot,
        variable_removal_rate_plot,
    )
    if profiles is None:
        return

    mean_variable_speed = profiles["mean_variable_speed"]
    constant_rate = profiles["constant_rate"]
    reward_action_rate = profiles["reward_action_rate"]

    constant_display, _ = _removal_heatmap_display(x, y, constant_rate, extent, bins)
    reward_display, _ = _removal_heatmap_display(x, y, reward_action_rate, extent, bins)

    constant_cell_cv = profiles["constant_rate_cv"]
    reward_cell_cv = profiles["reward_rate_cv"]
    improvement_percent = 0.0
    if constant_cell_cv > 1.0e-12:
        improvement_percent = max(0.0, (constant_cell_cv - reward_cell_cv) / constant_cell_cv * 100.0)
    constant_rate_cv = profiles["constant_rate_cv"]
    reward_rate_cv = profiles["reward_rate_cv"]

    fig_hm, axes_hm = plt.subplots(1, 2, figsize=(7.2, 3.45), facecolor="white", sharex=True, sharey=True)
    heatmap_items = [
        ("Constant velocity", constant_display, constant_cell_cv, mean_variable_speed),
        (
            f"Adaptive velocity ({improvement_percent:.0f}% lower CV)",
            reward_display,
            reward_cell_cv,
            _mean(variable_speed_plot[profiles["contact_mask"]]),
        ),
    ]
    im = None
    for ax, (title, display_grid, cell_cv, speed_value) in zip(axes_hm, heatmap_items):
        im = ax.imshow(
            display_grid,
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            interpolation="bilinear",
        )
        ax.set_title(title, pad=6)
        ax.set_xlabel("X [mm]")
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(direction="in", length=3.0, width=0.7)
        ax.grid(False)
        ax.text(
            0.018,
            0.982,
            f"CV = {cell_cv:.3f}\n$\\bar{{v}}$ = {speed_value * PLOT_MM_TO_M:.4f} m/s",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="black",
            bbox=_paper_info_box(),
        )
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("0.20")
    axes_hm[0].set_ylabel("Y [mm]")
    if im is not None:
        cbar = fig_hm.colorbar(im, ax=axes_hm.ravel().tolist(), fraction=0.035, pad=0.025)
        cbar.set_label("Normalized Preston rate [a.u.]")
        cbar.ax.tick_params(direction="in", length=3.0, width=0.7)
    fig_hm.suptitle("Spatial Projection of Preston Rate")
    fig_hm.savefig(out_dir / "05_preston_rate_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig_hm)

    constant_rate_m_s = profiles["constant_rate"] * PLOT_MM_TO_M
    reward_rate_m_s = profiles["reward_action_rate"] * PLOT_MM_TO_M
    target_rate_m_s = _mean(reward_rate_m_s[profiles["contact_mask"]])

    fig_amt, ax_amt = plt.subplots(figsize=(6.8, 3.2), facecolor="white")
    ax_amt.plot(t_plot, constant_rate_m_s, color=PAPER_CONSTANT_COLOR, linewidth=1.25, alpha=0.95, label=f"Constant velocity (CV={constant_rate_cv:.3f})")
    ax_amt.plot(t_plot, reward_rate_m_s, color=PAPER_ADAPTIVE_COLOR, linewidth=1.6, alpha=0.98, label=f"Adaptive velocity (CV={reward_rate_cv:.3f})")
    if target_rate_m_s > 0.0:
        ax_amt.axhline(target_rate_m_s, color=PAPER_REFERENCE_COLOR, linestyle="--", linewidth=0.9, label="Mean adaptive rate")
    ax_amt.set_title("Temporal Preston Rate Profile")
    ax_amt.set_xlabel("Time [s]")
    ax_amt.set_ylabel("Preston rate [N m s$^{-1}$]")
    amount_limits = _contact_rate_axis_limits(
        constant_rate_m_s,
        reward_rate_m_s,
        contact_mask=profiles["contact_mask"],
        center=target_rate_m_s if target_rate_m_s > 0.0 else None,
    )
    if amount_limits is not None:
        ax_amt.set_ylim(*amount_limits)
    _paper_grid(ax_amt)
    ax_amt.legend(loc="best", frameon=True)
    _set_contact_time_xlim(ax_amt, t_plot)
    fig_amt.tight_layout()
    fig_amt.savefig(out_dir / "07_preston_rate_profile_comparison.png", dpi=300)
    plt.close(fig_amt)

    return profiles


def _save_constant_velocity_signals_plot(out_dir, t_plot, normal_force_plot, constant_velocity_mm_s):
    constant_velocity_m_s = np.full_like(np.asarray(t_plot, dtype=float), constant_velocity_mm_s * PLOT_MM_TO_M)

    fig, axes = plt.subplots(2, 1, figsize=(6.8, 4.4), sharex=True, facecolor="white")
    axes[0].plot(t_plot, normal_force_plot, color=PAPER_FORCE_COLOR, linewidth=1.15)
    axes[0].set_title("Normal force")
    axes[0].set_ylabel("Force [N]")
    axes[0].margins(y=0.08)
    _paper_grid(axes[0])

    axes[1].plot(
        t_plot,
        constant_velocity_m_s,
        color=PAPER_VELOCITY_COLOR,
        linewidth=1.35,
        alpha=0.95,
        label=f"Constant velocity = {constant_velocity_mm_s * PLOT_MM_TO_M:.4f} m/s",
    )
    axes[1].set_title("Constant sliding velocity")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Velocity [m/s]")
    axes[1].margins(y=0.08)
    _paper_grid(axes[1])
    axes[1].legend(loc="best", frameon=True)
    _set_contact_time_xlim(axes, t_plot)
    fig.tight_layout()
    fig.savefig(out_dir / "06_constant_force_velocity_signals.png", dpi=300)
    plt.close(fig)


def save_plots(out_dir, t, state6, force3, dremoval, removal_rate, vxyz, sliding_velocity, path_index=None, contact_start_idx=None):
    xyz = state6[:, 0:3]
    if len(t) == 0 or xyz.shape[0] == 0:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    plot_start = int(contact_start_idx) if contact_start_idx is not None else 0
    plot_start = max(0, min(plot_start, len(t) - 1))

    t_plot = t[plot_start:] - t[plot_start]
    xyz_plot = xyz[plot_start:]
    dremoval_plot = dremoval[plot_start:]
    removal_rate_plot = removal_rate[plot_start:]
    sliding_velocity_plot = sliding_velocity[plot_start:]
    state6_plot = state6[plot_start:]
    path_index_plot = None
    if path_index is not None and len(path_index) == len(t):
        path_index_plot = np.asarray(path_index, dtype=float)[plot_start:]

    x, y = xyz_plot[:, 0], xyz_plot[:, 1]
    
    # 예외처리: 이동이 전혀 없어서 max==min이 되는 경우 방지
    x_ptp, y_ptp = np.ptp(x), np.ptp(y)
    margin_x = 10.0 if x_ptp < 1.0 else x_ptp * 0.1
    margin_y = 10.0 if y_ptp < 1.0 else y_ptp * 0.1
    
    extent = [np.min(x) - margin_x, np.max(x) + margin_x, np.min(y) - margin_y, np.max(y) + margin_y]

    # --- 1. Local Preston Rate Heatmap ---
    bins = min(220, max(96, int(np.sqrt(len(x)) * 4)))
    grid_display, _ = _removal_heatmap_display(x, y, removal_rate_plot, extent, bins)

    fig, ax = plt.subplots(figsize=(5.4, 4.35), facecolor="white")
    im = ax.imshow(
        grid_display,
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        interpolation="bilinear",
    )
    colorbar = fig.colorbar(im, ax=ax, fraction=0.048, pad=0.035)
    colorbar.set_label("Normalized Preston rate [a.u.]")
    colorbar.ax.tick_params(direction="in", length=3.0, width=0.7)
    ax.set_title("Spatial Preston Rate")
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(direction="in", length=3.0, width=0.7)
    ax.grid(False)
    contact_samples = int(np.count_nonzero(dremoval_plot > 0.0))
    positive_display = grid_display[grid_display > 0.0]
    display_mean = _mean(positive_display)
    display_std = _std(positive_display)
    ax.text(
        0.018,
        0.982,
        f"mean = {display_mean:.6f} [a.u.]\n"
        f"std = {display_std:.6f} [a.u.]\n"
        f"samples = {len(dremoval_plot)}\n"
        f"contact = {contact_samples}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="black",
        bbox=_paper_info_box(),
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.20")
    fig.tight_layout()
    fig.savefig(out_dir / "01_preston_rate_heatmap.png", dpi=300)
    plt.close(fig)

    normal_force = _normal_force_from_force3(force3)
    normal_force_plot = normal_force[plot_start:]

    comparison_profiles = _save_velocity_comparison_plots(
        out_dir=out_dir,
        t_plot=t_plot,
        x=x,
        y=y,
        normal_force_plot=normal_force_plot,
        variable_speed_plot=sliding_velocity_plot,
        variable_removal_rate_plot=removal_rate_plot,
        extent=extent,
        bins=bins,
    )

    if comparison_profiles is not None:
        reward_action_rate_m_s = comparison_profiles["reward_action_rate"] * PLOT_MM_TO_M
        reward_action_contact_mask = comparison_profiles["contact_mask"]
        reward_action_rate_mean = _mean(reward_action_rate_m_s[reward_action_contact_mask])
        fig2, ax2 = plt.subplots(figsize=(6.8, 3.2), facecolor="white")
        ax2.plot(
            t_plot,
            reward_action_rate_m_s,
            color=PAPER_ADAPTIVE_COLOR,
            linewidth=1.6,
            alpha=0.96,
            label=f"Adaptive velocity (CV={comparison_profiles['reward_rate_cv']:.3f})",
        )
        if reward_action_rate_mean > 0.0:
            ax2.axhline(
                reward_action_rate_mean,
                color=PAPER_REFERENCE_COLOR,
                linestyle="--",
                linewidth=0.9,
                label="Contact-window mean",
            )
        ax2.set_title("Adaptive Preston Rate Profile")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Preston rate [N m s$^{-1}$]")
        rate_limits = _contact_rate_axis_limits(
            reward_action_rate_m_s,
            contact_mask=reward_action_contact_mask,
            center=reward_action_rate_mean if reward_action_rate_mean > 0.0 else None,
        )
        if rate_limits is not None:
            ax2.set_ylim(*rate_limits)
        ax2.legend(loc="best", frameon=True)
        _paper_grid(ax2)
        _set_contact_time_xlim(ax2, t_plot)
        fig2.tight_layout()
        fig2.savefig(out_dir / "02_adaptive_preston_rate_profile.png", dpi=300)
        plt.close(fig2)

    fig3, axes3 = plt.subplots(2, 1, figsize=(6.8, 4.4), sharex=True, facecolor="white")
    axes3[0].plot(t_plot, normal_force_plot, color=PAPER_FORCE_COLOR, linewidth=1.15)
    axes3[0].set_title("Normal force")
    axes3[0].set_ylabel("Force [N]")
    axes3[0].margins(y=0.08)
    _paper_grid(axes3[0])
    sliding_velocity_m_s = sliding_velocity_plot * PLOT_MM_TO_M
    axes3[1].plot(
        t_plot,
        sliding_velocity_m_s,
        color=PAPER_VELOCITY_COLOR,
        linewidth=1.25,
        alpha=0.92,
        label="Adaptive velocity",
    )
    axes3[1].set_title("Adaptive sliding velocity")
    axes3[1].set_xlabel("Time [s]")
    axes3[1].set_ylabel("Velocity [m/s]")
    axes3[1].margins(y=0.08)
    _paper_grid(axes3[1])
    axes3[1].legend(loc="best", frameon=True)
    _set_contact_time_xlim(axes3, t_plot)
    fig3.tight_layout()
    fig3.savefig(out_dir / "03_adaptive_force_velocity_signals.png", dpi=300)
    plt.close(fig3)

    if comparison_profiles is not None:
        _save_constant_velocity_signals_plot(
            out_dir=out_dir,
            t_plot=t_plot,
            normal_force_plot=normal_force_plot,
            constant_velocity_mm_s=comparison_profiles["mean_variable_speed"],
        )

    fig4 = plt.figure(figsize=(6.2, 5.2), facecolor="white")
    ax4 = fig4.add_subplot(111, projection="3d")

    path_values = removal_rate_plot if len(removal_rate_plot) == len(xyz_plot) else sliding_velocity_plot
    path_values = np.asarray(path_values, dtype=float)
    if len(path_values) == len(xyz_plot) and np.nanmax(path_values) > np.nanmin(path_values):
        segments = np.stack([xyz_plot[:-1], xyz_plot[1:]], axis=1)
        segment_values = 0.5 * (path_values[:-1] + path_values[1:])
        line_collection = Line3DCollection(segments, cmap="viridis", linewidth=1.75, alpha=0.95)
        line_collection.set_array(segment_values)
        ax4.add_collection3d(line_collection)
        cbar4 = fig4.colorbar(line_collection, ax=ax4, pad=0.08, shrink=0.68)
        cbar4.set_label("Preston rate [a.u.]")
        cbar4.ax.tick_params(direction="in", length=3.0, width=0.7)
    else:
        ax4.plot(xyz_plot[:, 0], xyz_plot[:, 1], xyz_plot[:, 2], color=PAPER_ADAPTIVE_COLOR, linewidth=1.6, label="End-effector path")

    hdf5_xyz = _get_hdf5_position_np()
    hdf5_xyz = _slice_hdf5_reference_path(hdf5_xyz, path_index_plot)
    if hdf5_xyz is not None:
        ax4.plot(
            hdf5_xyz[:, 0],
            hdf5_xyz[:, 1],
            hdf5_xyz[:, 2],
            color="black",
            linestyle="-",
            linewidth=1.0,
            alpha=0.50,
            label="Reference path",
        )

    z_floor = float(np.nanmin(xyz_plot[:, 2]))
    ax4.plot(
        xyz_plot[:, 0],
        xyz_plot[:, 1],
        np.full_like(xyz_plot[:, 2], z_floor),
        color="0.35",
        linestyle="--",
        linewidth=0.9,
        alpha=0.45,
        label="XY projection",
    )

    normals = _normal_vectors_from_state6(state6_plot)
    if normals is not None and len(normals) == len(xyz_plot):
        stride = max(1, len(xyz_plot) // 28)
        normal_scale = max(float(np.ptp(xyz_plot[:, 0])), float(np.ptp(xyz_plot[:, 1])), float(np.ptp(xyz_plot[:, 2])), 1.0) * 0.08
        q_xyz = xyz_plot[::stride]
        q_normals = normals[::stride] * normal_scale
        ax4.quiver(
            q_xyz[:, 0],
            q_xyz[:, 1],
            q_xyz[:, 2],
            q_normals[:, 0],
            q_normals[:, 1],
            q_normals[:, 2],
            color=PAPER_FORCE_COLOR,
            linewidth=0.8,
            arrow_length_ratio=0.25,
            alpha=0.85,
            normalize=False,
        )
        ax4.plot([], [], [], color=PAPER_FORCE_COLOR, linewidth=1.2, label="TCP normal")

    ax4.scatter(xyz_plot[0, 0], xyz_plot[0, 1], xyz_plot[0, 2], color=PAPER_VELOCITY_COLOR, s=28, depthshade=True, label="Contact start")
    ax4.scatter(xyz_plot[-1, 0], xyz_plot[-1, 1], xyz_plot[-1, 2], color=PAPER_FORCE_COLOR, s=28, depthshade=True, label="End")
    ax4.set_title("Contact-Window End-Effector Path")
    ax4.set_xlabel("X [mm]")
    ax4.set_ylabel("Y [mm]")
    ax4.set_zlabel("Z [mm]")
    axes_xyz = xyz_plot if hdf5_xyz is None else np.vstack([xyz_plot, hdf5_xyz])
    _set_axes_equal_3d(ax4, axes_xyz)
    ax4.view_init(elev=28, azim=-58)
    ax4.grid(True, color="0.78", linewidth=0.45, alpha=0.55)
    ax4.tick_params(labelsize=8, pad=1.0)
    ax4.legend(loc="upper left", frameon=True)
    fig4.tight_layout()
    fig4.savefig(out_dir / "04_contact_path_3d.png", dpi=300)
    plt.close(fig4)

# ============================================================
# RL Hooks
# ============================================================

def rl_step(env_ids, state6, force3, sim_time):
    record_step(env_ids, state6, force3, sim_time)

def rl_episode_done():
    return process_episode()

def _clear_episode_buffers():
    global _rl_start_time, _current_ep_reward
    _rl_time_buffer.clear()
    _rl_state_buffer.clear()
    _rl_force_buffer.clear()
    _rl_index_buffer.clear()
    _rl_sliding_velocity_buffer.clear()
    _rl_reward_components_buffer.clear()
    _rl_start_time = None
    _current_ep_reward = 0.0

def _get_visualization_cfg(env):
    return getattr(getattr(env, "cfg", None), "visualization", None)

def _visualization_enabled(env) -> bool:
    cfg = _get_visualization_cfg(env)
    return bool(getattr(cfg, "enable_visualizer", True))

def _should_save_episode(env) -> bool:
    cfg = _get_visualization_cfg(env)
    interval = int(getattr(cfg, "save_interval_episodes", 1))
    if interval <= 0:
        return False
    return (_episode_counter % interval) == 0

def on_episode_reset(env, env_ids=None):
    global _has_seen_any_step, _current_ep_reward, _episode_counter, _last_processed_reset_step
    try:
        if not _visualization_enabled(env):
            return
        if not _env_ids_include_tracked_env(env_ids):
            return
        if not _has_seen_any_step:
            return
        reset_step = int(getattr(env, "common_step_counter", -1)) if env is not None else -1
        if reset_step >= 0 and _last_processed_reset_step == reset_step:
            return
        if len(_rl_time_buffer) > 0:
            if _should_save_episode(env):
                rl_episode_done()
            else:
                _clear_episode_buffers()
                _episode_counter += 1
            _last_processed_reset_step = reset_step

        if env is not None:
            env._ep_curriculum = getattr(env, "_ep_curriculum", 0) + 1
            
    except Exception as e:
        local_debug.print_exception("Visualization on_episode_reset failed", e)

# ============================================================
# Step Hook
# ============================================================

def rl_step_hook(env, action_term_name="arm_action", asset_name="robot", fixed_joint_name="tool0_to_spindle", joint_prim_relpath="joints"):
    global _current_ep_reward, _rl_index_buffer, _rl_sliding_velocity_buffer, _rl_reward_components_buffer
    
    try:
        if not _visualization_enabled(env):
            return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)

        num_envs = env.num_envs
        device = env.device
        env_ids = torch.arange(num_envs, device=device, dtype=torch.long)

        state6 = local_obs.get_ee_pose(env, asset_name=asset_name)
        wrench6 = local_ft_sensor.get_6axis_ft_fixed_joint(env, asset_name, fixed_joint_name, joint_prim_relpath, verbose=False)
        force3 = wrench6[:, :3]

        sim_time = float(getattr(env, "common_step_counter", 0)) * float(getattr(env, "step_dt", 0.02))

        try:
            term = env.action_manager.get_term(action_term_name)
            if hasattr(term, "path_cursor"):
                _rl_index_buffer.append(float(term.path_cursor[0].item()))
            elif hasattr(term, "current_target_index"):
                _rl_index_buffer.append(float(term.current_target_index[0].item()))
            else:
                _rl_index_buffer.append(0.0)
            if hasattr(term, "current_sliding_velocity_mm_s"):
                _rl_sliding_velocity_buffer.append(float(term.current_sliding_velocity_mm_s[0].item()))
            else:
                _rl_sliding_velocity_buffer.append(float("nan"))
        except Exception:
            _rl_index_buffer.append(0.0)
            _rl_sliding_velocity_buffer.append(float("nan"))

        # 🚀 [FIX] Reward Tracking 보완 (IsaacLab의 protected 속성까지 긁어옴)
        try:
            if hasattr(env, "reward_manager"):
                reward_dict = getattr(env.reward_manager, "episode_sums", getattr(env.reward_manager, "_episode_sums", {}))
                for term_name, value_tensor in reward_dict.items():
                    _rl_reward_components_buffer[term_name].append(float(value_tensor[0].item()))
        except Exception:
            pass

        if hasattr(env, "reward_buf"):
            _current_ep_reward += float(env.reward_buf[0].item())

        rl_step(env_ids, state6, force3, sim_time)

    except Exception as e:
        local_debug.print_exception("Visualization rl_step_hook failed", e)

    return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)
