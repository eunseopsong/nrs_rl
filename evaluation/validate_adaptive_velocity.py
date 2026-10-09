# SPDX-License-Identifier: BSD-3-Clause
"""Bounded GPU regression for geometry, FT calibration and baseline contact.

This checks the environment with a constant speed residual. It does not certify
PPO quality or predict absolute material removal on the physical robot.
"""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import argparse
import json
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--check_debug_print", action="store_true", help="Also check console phases and partial resets.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 1500 or not 1 <= args.num_envs <= 16:
    parser.error("Use at least 1500 steps and between 1 and 16 environments.")
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab.utils.math import matrix_from_quat
from isaaclab_tasks.utils import parse_env_cfg
from pxr import UsdGeom, UsdPhysics
import omni.usd
import nrs_rl.tasks  # noqa: F401


def validate():
    cfg = parse_env_cfg("Template-Nrs-Rl-v0", device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    if args.check_debug_print:
        cfg.actions.arm_action.integration.enable_debug_print = True
        cfg.actions.arm_action.integration.debug_print_interval = 50
        cfg.actions.arm_action.integration.debug_env_id = args.num_envs - 1
    # These buffers suffice for a small regression scene. Training settings are
    # unchanged, and another running training job keeps its GPU allocation.
    cfg.sim.physx.gpu_max_rigid_contact_count = 2**18
    cfg.sim.physx.gpu_max_rigid_patch_count = 2**14
    cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**18
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**18
    cfg.sim.physx.gpu_collision_stack_size = 2**24
    env = gym.make("Template-Nrs-Rl-v0", cfg=cfg)
    try:
        env.reset(seed=args.seed)
        raw_env = env.unwrapped
        term = raw_env.action_manager.get_term("arm_action")
        debug_messages = []
        if args.check_debug_print:
            printer = term._debug_printer
            original_writer = printer.write

            def capture_debug(message):
                debug_messages.append(message)
                original_writer(message)

            printer.write = capture_debug
        stage = omni.usd.get_context().get_stage()
        scenes = [prim.GetPath() for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
        assert len(scenes) == 1, ("embedded physics scenes", scenes)
        cylinder = UsdGeom.Cylinder(stage.GetPrimAtPath(
            "/World/envs/env_0/Robot/spindle_link/collisions/polishing_tool"
        ))
        assert str(cylinder.GetRadiusAttr().GetTypeName()) == "double"
        assert str(cylinder.GetHeightAttr().GetTypeName()) == "double"
        assert abs(cylinder.GetHeightAttr().Get() - 0.114) < 1e-8
        extent = cylinder.GetExtentAttr().Get()
        assert abs(float(extent[1][2] - extent[0][2]) - 0.114) < 1e-6

        body = term.robot.body_names.index("spindle_link")
        tool_rotation = torch.diag(torch.tensor([-1., 1., -1.], device=raw_env.device))
        actions = torch.zeros(env.action_space.shape, device=raw_env.device)
        max_fk_error_mm = 0.0
        force_samples = []
        shield_samples = []
        preparation_steps = term._calibration_steps + term._approach_steps
        with torch.inference_mode():
            for step in range(args.steps):
                obs, rewards, terminated, truncated, info = env.step(actions)
                assert torch.isfinite(obs["policy"]).all(), (step, "nonfinite observations")
                assert torch.isfinite(rewards).all(), (step, "nonfinite rewards")
                assert not (terminated | truncated).any(), (step, "unexpected early termination")
                if step < preparation_steps:
                    assert not term.cumulative_removal.any(), "removal counted before polishing"
                if step == term._calibration_steps - 1:
                    state = next(iter(raw_env._ft6_filter_state.values()))
                    assert state["bias_ready"].all(), "FT calibration not completed"
                    assert state["bias"][:, :3].abs().max() < 0.1, state["bias"]
                if step % 50 == 0:
                    rotation = matrix_from_quat(term.robot.data.body_quat_w[:, body])
                    tip = term.robot.data.body_pos_w[:, body] - raw_env.scene.env_origins
                    tip = tip + rotation[:, :, 2] * 0.114
                    for env_id in range(args.num_envs):
                        _, tcp, tcp_rotation = term._fk(env_id, term.robot.data.joint_pos[env_id, :6])
                        error = torch.linalg.norm(tip[env_id] * 1000 - tcp).item()
                        max_fk_error_mm = max(max_fk_error_mm, error)
                        assert error < 0.02, (env_id, step, "FK position mismatch", error)
                        assert (tcp_rotation - rotation[env_id] @ tool_rotation).abs().max() < 1e-5
                if step >= args.steps - 250:
                    force_samples.append(term.current_abs_fz.clone())
                    shield_samples.append(term.safety_shield_active.clone())
                if step % 250 == 249:
                    print(f"Validated {step + 1}/{args.steps} steps", flush=True)
        force = torch.stack(force_samples)
        mean_force = force.mean(dim=0)
        assert ((mean_force > 5.0) & (mean_force < 15.0)).all(), ("contact force", mean_force)
        assert (term.path_cursor_mm > 20.0).all(), ("no path progress", term.path_cursor_mm)
        if args.check_debug_print:
            expected_steps = range(0, args.steps, 50)
            assert len(debug_messages) == len(expected_steps), "unexpected legacy debug cadence"
            assert all(f"env={args.num_envs - 1} " in msg for msg in debug_messages)
            assert "[Polishing Live] ep1 step=0" in debug_messages[0]
            assert "normal_force_N=nan" in debug_messages[0]
            assert "removal_rate_N_mm_s=" in debug_messages[-1]
            assert "rewards          =" in debug_messages[-1]
            assert printer.step == args.steps
        result = {
            "status": "PASS", "steps": args.steps, "num_envs": args.num_envs,
            "max_fk_error_mm": max_fk_error_mm,
            "last_250_steps_mean_observed_force_n": mean_force.cpu().tolist(),
            "last_250_steps_shield_fraction": torch.stack(shield_samples).float().mean(0).cpu().tolist(),
            "path_progress_mm": term.path_cursor_mm.cpu().tolist(),
            "debug_print_checked": args.check_debug_print,
            "note": "Constant residual with sensor randomization; not a trained policy evaluation.",
        }
        if args.check_debug_print:
            previous_episode, previous_count = printer.episode, len(debug_messages)
            # Isaac caches tensors created by the inference-mode rollout;
            # resetting those buffers must use the same inference context.
            with torch.inference_mode():
                if args.num_envs > 1:
                    # Resetting an unselected environment must not disturb the
                    # selected environment's episode number or console cadence.
                    term.reset(torch.tensor([0], device=raw_env.device))
                    assert printer.episode == previous_episode and printer.step == args.steps
                    assert len(debug_messages) == previous_count
                env.reset()
                assert len(debug_messages) == previous_count
                assert printer.episode == previous_episode + 1 and printer.step == 0
                env.step(actions)
                assert f"[Polishing Live] ep{previous_episode + 1} step=0" in debug_messages[-1]
                assert not term.cumulative_removal.any()
            print("Legacy console cadence and episode reset: PASS", flush=True)
        print(json.dumps(result, indent=2), flush=True)
    finally:
        env.close()


exit_code = 0
try:
    validate()
except Exception:
    traceback.print_exc()
    exit_code = 1
finally:
    # Notify Kit of the result before its fast shutdown terminates Python.
    # Merely raising after close() can otherwise mask a failed assertion.
    sys.stdout.flush()
    sys.stderr.flush()
    simulation_app.app.post_quit(exit_code)
    simulation_app.close()
sys.exit(exit_code)
