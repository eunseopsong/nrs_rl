# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch


def _get_action_term(env, action_term_name: str):
    """
    Isaac Lab 버전 차이를 고려해서 action term 객체를 찾아주는 helper.
    """
    am = env.action_manager

    # case 1: public getter
    if hasattr(am, "get_term"):
        try:
            return am.get_term(action_term_name)
        except Exception:
            pass

    # case 2: dict-like private storage
    if hasattr(am, "_terms"):
        terms = am._terms
        if isinstance(terms, dict):
            if action_term_name in terms:
                return terms[action_term_name]

        # case 3: list + names
        if hasattr(am, "_term_names"):
            try:
                idx = am._term_names.index(action_term_name)
                return terms[idx]
            except Exception:
                pass

    # case 4: named attributes fallback
    if hasattr(am, action_term_name):
        return getattr(am, action_term_name)

    raise RuntimeError(
        f"[trajectory_finished] Could not find action term '{action_term_name}'. "
        f"Check your ActionsCfg name."
    )


def trajectory_finished(env, action_term_name: str = "arm_action") -> torch.Tensor:
    """
    Terminate episode when the HDF5 waypoint follower has consumed the full path.

    Expected in action term:
      - path_done: Bool tensor of shape [num_envs]
    """
    term = _get_action_term(env, action_term_name)

    if not hasattr(term, "path_done"):
        raise RuntimeError(
            f"[trajectory_finished] Action term '{action_term_name}' "
            f"does not have attribute 'path_done'."
        )

    done = term.path_done

    # safety: enforce bool tensor on env device
    if not isinstance(done, torch.Tensor):
        done = torch.tensor(done, device=env.device, dtype=torch.bool)
    else:
        done = done.to(device=env.device, dtype=torch.bool)

    return done