# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch


def _get_action_term(env, action_term_name: str = "arm_action"):
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

    raise RuntimeError(f"[trajectory_finished] action term '{action_term_name}' not found.")


def trajectory_finished(env, action_term_name: str = "arm_action") -> torch.Tensor:
    term = _get_action_term(env, action_term_name)

    if not hasattr(term, "path_done"):
        raise RuntimeError(f"[trajectory_finished] action term '{action_term_name}' has no path_done.")

    done = term.path_done
    if not isinstance(done, torch.Tensor):
        done = torch.tensor(done, device=env.device, dtype=torch.bool)
    else:
        done = done.to(device=env.device, dtype=torch.bool)

    return done