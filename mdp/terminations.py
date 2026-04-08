# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib
import torch

local_obs = importlib.import_module(
    "nrs_rl.tasks.manager_based.nrs_rl.mdp.observation"
)


def trajectory_finished(env) -> torch.Tensor:
    """
    Terminate when all rows in cmd_continue9D.h5 have been visited.

    Criterion:
        episode_length_buf >= (traj_len - 1)
    """
    traj_len = int(local_obs.get_hdf5_trajectory_length())

    if traj_len <= 0:
        return torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)

    if hasattr(env, "episode_length_buf"):
        idx = env.episode_length_buf.to(device=env.device, dtype=torch.long)
    else:
        idx = torch.zeros((env.num_envs,), device=env.device, dtype=torch.long)

    done = idx >= (traj_len - 1)
    return done.to(device=env.device, dtype=torch.bool)