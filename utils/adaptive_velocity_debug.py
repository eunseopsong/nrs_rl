# SPDX-License-Identifier: BSD-3-Clause
"""e4efeeb console format; diagnostic adapters never alter the controller."""

from __future__ import annotations

from bisect import bisect_right
import sys


def write_debug(message: str) -> None:
    from tqdm import tqdm

    tqdm.write(message, file=sys.stdout)
    sys.stdout.flush()


def arc_to_index(distance_mm: float, arc_mm) -> float:
    """Convert the new millimetre cursor to the legacy fractional HDF5 index."""
    lower = min(max(bisect_right(arc_mm, distance_mm) - 1, 0), len(arc_mm) - 2)
    fraction = (distance_mm - arc_mm[lower]) / max(arc_mm[lower + 1] - arc_mm[lower], 1e-9)
    return lower + min(1.0, max(0.0, fraction))


def format_polishing_live(
    *, episode, step, env_id, current_index, last_index, target_index, cursor,
    current_pose, target_pose, command_pose, target_force, normal_force,
    sliding_velocity, removal_rate, cumulative_removal, fn_offset, action,
    index_rate, path_error_xy, reward_debug,
) -> str:
    """Same labels, layout and precision as e4efeeb:mdp/action.py."""
    progress = 100.0 * cursor / max(float(last_index), 1.0)

    def xyz(pose):
        return f"({pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f})"

    def wxyz(pose):
        return f"({pose[3]:.4f}, {pose[4]:.4f}, {pose[5]:.4f})"

    return (
        f"\n[Polishing Live] ep{episode} step={step} env={env_id} "
        f"| hdf5_index={current_index}/{last_index} ({progress:.1f}%) "
        f"| target_index={target_index} | cursor={cursor:.3f}\n"
        f"  current xyz/wxyz = {xyz(current_pose)} / {wxyz(current_pose)}\n"
        f"  target  xyz/wxyz = {xyz(target_pose)} / {wxyz(target_pose)}\n"
        f"  command xyz      = {xyz(command_pose)}\n"
        f"  force/speed      = | target_force_N={target_force:.4f} "
        f"| normal_force_N={normal_force:.4f} "
        f"| sliding_velocity_mm_s={sliding_velocity:.4f} "
        f"| removal_rate_N_mm_s={removal_rate:.4f} "
        f"| cumulative_removal={cumulative_removal:.4f}\n"
        f"  control          = | fn_offset_mm={fn_offset:.4f} "
        f"| action={action:.4f} | index_rate={index_rate:.4f} "
        f"| path_err_xy_mm={path_error_xy:.3f}\n"
        f"  rewards          = {reward_debug}\n"
    )


class EpisodeDebugPrinter:
    """Legacy zero-based cadence, with an independent selected-env episode count."""

    def __init__(self, env_id: int, interval: int, writer=write_debug):
        self.env_id = env_id
        self.interval = interval
        self.write = writer
        self.episode = 0
        self.step = 0

    def reset(self) -> None:
        if self.episode == 0 or self.step:
            self.episode += 1
        self.step = 0

    def tick(self) -> bool:
        due = self.interval <= 0 or self.step % self.interval == 0
        self.step += 1
        return due
