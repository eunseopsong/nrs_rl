# SPDX-License-Identifier: BSD-3-Clause
"""Bounded, policy-free regression across contact and repeated resets."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import argparse
import json
import math
import sys
import traceback
import tempfile

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cycles", type=int, default=3)
parser.add_argument("--steps-per-cycle", type=int, default=1500)
parser.add_argument("--diagnose", action="store_true")
parser.add_argument("--legacy-guards", action="store_true")
parser.add_argument("--fault-checks", action="store_true", help="Inject rejected IK commands in one environment")
parser.add_argument("--complete-path", action="store_true", help="Run maximum policy speed through full paths and auto-reset")
parser.add_argument("--start-mm", type=float, default=0.0, help="Diagnose a suffix of the path using a temporary HDF5")
parser.add_argument("--joint-stiffness", type=float, help="Diagnostic override of the simulated position servo")
parser.add_argument("--joint-damping", type=float, help="Diagnostic override of the simulated position servo")
parser.add_argument("--alternating-actions", action="store_true", help="Stress the speed filter with 125 Hz sign changes")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.cycles < 1 or args.steps_per_cycle < 750:
    parser.error("Use at least one cycle and 750 steps per cycle")
if args.legacy_guards and args.fault_checks:
    parser.error("Fault checks require the production safety limits")
app = AppLauncher(args).app

import gymnasium as gym
import h5py
import numpy as np
import torch
from isaaclab_tasks.utils import parse_env_cfg
import nrs_rl.tasks  # noqa: F401


def check_step(env, actions, *, allow_done=False):
    obs, reward, terminated, truncated, info = env.step(actions)
    assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
    term = env.unwrapped.action_manager.get_term("arm_action")
    valid = term.command_smoothness_valid
    assert (term.command_acceleration_mm_s2[valid].abs() <= term.int_cfg.max_speed_acceleration_mm_s2 + 0.01).all()
    assert (term.command_jerk_mm_s3[valid].abs() <= term.int_cfg.max_speed_jerk_mm_s3 + 0.5).all()
    if not allow_done:
        assert not (terminated | truncated).any(), "unexpected reset"
    return terminated, truncated, info


def check_faults(env, term, actions):
    """Exercise the actual command rejection and automatic subset reset path."""
    raw = env.unwrapped
    original_ik = term._command_ik

    def invalid_ik(env_id, seed, pose):
        result = original_ik(env_id, seed, pose)
        return result + 1.0 if env_id == 1 else result

    progress = term.path_cursor_mm.clone()
    removal = term.cumulative_removal.clone()
    term._command_ik = invalid_ik
    try:
        for step in range(term.int_cfg.fault_termination_steps):
            measured_q = term.robot.data.joint_pos[1, :6].clone()
            terminated, truncated, info = check_step(env, actions, allow_done=True)
            assert not terminated[0] and not truncated.any()
            if step + 1 < term.int_cfg.fault_termination_steps:
                assert not terminated.any()
                assert int(term._safety_fault_steps[1]) == step + 1
                assert not term.path_done[1]
                assert term.path_cursor_mm[1] == progress[1]
                assert term.cumulative_removal[1] == removal[1]
                assert term.current_mrr_n_mm_s[1] == 0
                torch.testing.assert_close(term.robot.data.joint_pos_target[1, :6], measured_q)
            else:
                assert terminated[1], "persistent IK fault did not terminate"
                assert raw.termination_manager.get_term("control_failed")[1]
                assert not raw.termination_manager.get_term("trajectory_finished")[1]
                assert float(info["log"]["Safety/failure_fraction"]) == 1.0
                assert float(info["log"]["Safety/fault_steps"]) == step + 1
                assert term.path_cursor_mm[1] == 0
                assert not term.safety_terminated[1]
        assert term.path_cursor_mm[0] > progress[0], "healthy environment was reset"
        print("FAULT_REJECTION_PASS", "ticks", term.int_cfg.fault_termination_steps, flush=True)
    finally:
        term._command_ik = original_ik
    for _ in range(1500):
        check_step(env, actions)
    assert (term.current_abs_fz > 5).all() and (term.current_abs_fz < 15).all()
    assert term.path_cursor_mm[1] > 20
    print("FAULT_AND_PARTIAL_RESET_PASS", flush=True)


def check_complete_paths(env, term, actions):
    """Cover the full geometry at the fastest command available to a policy."""
    raw = env.unwrapped
    env.reset(seed=42)
    actions.fill_(1.0)
    speed = min(term.int_cfg.max_speed_mm_s,
                term.int_cfg.nominal_speed_mm_s * (1 + term.int_cfg.residual_speed_fraction))
    max_steps = term._calibration_steps + term._approach_steps + math.ceil(
        1.5 * term.path_length_mm / (speed * term._step_dt_local))
    completed = torch.zeros_like(term.path_done)
    peak_force = torch.zeros_like(term.current_abs_fz)
    for step in range(max_steps):
        terminated, truncated, info = check_step(env, actions, allow_done=True)
        failed = env.unwrapped.termination_manager.get_term("control_failed")
        assert not failed.any(), (step, "full path control failure", info.get("log"))
        assert not truncated.any()
        completed |= terminated
        peak_force = torch.maximum(peak_force, term.current_abs_fz)
        if step % 500 == 0 or terminated.any():
            errors = []
            for i in range(raw.num_envs):
                pose, _, _ = term._fk(i, term.robot.data.joint_pos[i, :6])
                reference, _, _ = term._trajectory_at(float(term.path_cursor_mm[i]))
                errors.append({"measured_minus_reference": (pose[:3] - reference[:3]).tolist(),
                               "command_minus_reference": (term._previous_command_pose[i, :3] - reference[:3]).tolist()})
            print("FULL_PATH", json.dumps({"step": step, "progress": term.path_cursor_mm.tolist(),
                  "force": term.current_abs_fz.tolist(), "completed": completed.tolist(),
                  "tracking": term.current_path_tracking_error_mm.tolist(), "errors": errors}), flush=True)
        if completed.all():
            break
    assert completed.all(), "full paths did not finish within 150% of nominal duration"
    actions.zero_()
    for _ in range(1500):
        check_step(env, actions)
    assert (term.current_abs_fz > 5).all() and (term.current_abs_fz < 15).all()
    assert (term.path_cursor_mm > 20).all()
    print("FULL_PATH_AND_AUTO_RESET_PASS", "peak_force", peak_force.tolist(), flush=True)


def run():
    cfg = parse_env_cfg("Template-Nrs-Rl-v0", device=args.device, num_envs=2)
    cfg.seed = 42
    cfg.visualization.enable_visualizer = False
    if args.joint_stiffness is not None:
        cfg.scene.robot.actuators["ur10_arm"].stiffness = args.joint_stiffness
    if args.joint_damping is not None:
        cfg.scene.robot.actuators["ur10_arm"].damping = args.joint_damping
    integration = cfg.actions.arm_action.integration
    integration.enable_debug_print = False
    path_file = None
    if args.start_mm > 0:
        with h5py.File(integration.hdf5_file_path) as source:
            positions = source[integration.position_dataset_key][:]
            arc = np.r_[0.0, np.linalg.norm(np.diff(positions[:, :3], axis=0), axis=1).cumsum()]
            start = int(np.searchsorted(arc, args.start_mm))
            assert start < len(positions) - 1
            path_file = tempfile.NamedTemporaryFile(suffix=".h5", prefix="nrs_path_suffix_")
            with h5py.File(path_file.name, "w") as target:
                target[integration.position_dataset_key] = positions[start:]
                target[integration.force_dataset_key] = source[integration.force_dataset_key][start:]
        integration.hdf5_file_path = path_file.name
        print("DIAGNOSTIC_PATH_SUFFIX", args.start_mm, flush=True)
    if args.legacy_guards:
        for name in ("max_force_abort_n", "max_tracking_error_mm",
                     "max_command_position_step_mm", "max_command_angle_step_rad", "max_joint_step_rad"):
            setattr(integration, name, float("inf"))
    cfg.sim.physx.gpu_max_rigid_contact_count = 2**18
    cfg.sim.physx.gpu_max_rigid_patch_count = 2**14
    cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**18
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**18
    cfg.sim.physx.gpu_collision_stack_size = 2**24
    env = gym.make("Template-Nrs-Rl-v0", cfg=cfg)
    try:
        raw = env.unwrapped
        term = raw.action_manager.get_term("arm_action")
        robot = term.robot
        actions = torch.zeros(env.action_space.shape, device=raw.device)
        with torch.inference_mode():
            for cycle in range(args.cycles):
                env.reset(seed=42)
                print("RESET", cycle, "fixed_base", robot.is_fixed_base,
                      "root", robot.data.root_state_w.cpu().tolist(), flush=True)
                peak_force = torch.zeros(2, device=raw.device)
                peak_speed = torch.zeros(2, device=raw.device)
                forces = []
                early_done = 0
                for step in range(args.steps_per_cycle):
                    if args.alternating_actions:
                        actions.fill_(1.0 if step % 2 else -1.0)
                    obs, reward, terminated, truncated, _ = env.step(actions)
                    assert obs["policy"].shape == (2, 14)
                    # Catch accidentally advancing the 125 Hz controller once
                    # per 500 Hz physics substep, including after resets.
                    if step < term._calibration_steps:
                        assert (term._calibration_step == step + 1).all()
                        assert (term._approach_step == 0).all()
                    valid = term.command_smoothness_valid
                    assert (term.command_acceleration_mm_s2[valid].abs() <= integration.max_speed_acceleration_mm_s2 + 0.01).all()
                    assert (term.command_jerk_mm_s3[valid].abs() <= integration.max_speed_jerk_mm_s3 + 0.5).all()
                    if (terminated | truncated).any():
                        early_done += int((terminated | truncated).sum())
                        if not args.diagnose:
                            raise AssertionError((cycle, step, "unexpected reset"))
                    peak_force = torch.maximum(peak_force, term.current_abs_fz)
                    peak_speed = torch.maximum(peak_speed, term.current_sliding_velocity_mm_s)
                    if step >= args.steps_per_cycle - 250:
                        forces.append(term.current_abs_fz.clone())
                    if args.diagnose and (step in (0, 1, 249, 250, 499, 500) or step % 250 == 249):
                        print("STATE", json.dumps({
                            "cycle": cycle, "step": step,
                            "q": robot.data.joint_pos.cpu().tolist(),
                            "q_target": robot.data.joint_pos_target.cpu().tolist(),
                            "root": robot.data.root_state_w.cpu().tolist(),
                            "force": term.current_abs_fz.cpu().tolist(),
                            "speed": term.current_sliding_velocity_mm_s.cpu().tolist(),
                            "cursor": term.path_cursor_mm.cpu().tolist(),
                            "tracking": term.current_path_tracking_error_mm.cpu().tolist(),
                            "fault_steps": term._safety_fault_steps.cpu().tolist(),
                        }), flush=True)
                    assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
                mean_force = torch.stack(forces).mean(0)
                print("CYCLE", json.dumps({"cycle": cycle, "resets": early_done,
                      "peak_force": peak_force.cpu().tolist(), "peak_speed": peak_speed.cpu().tolist(),
                      "mean_force": mean_force.cpu().tolist(), "progress": term.path_cursor_mm.cpu().tolist()}), flush=True)
                if not args.diagnose:
                    assert ((mean_force > 5) & (mean_force < 15)).all()
                    assert (term.path_cursor_mm > 20).all()
            if args.fault_checks:
                actions.zero_()
                check_faults(env, term, actions)
            if args.complete_path:
                check_complete_paths(env, term, actions)
        print("DIAGNOSIS_COMPLETE" if args.diagnose else "RESET_REGRESSION_PASS", flush=True)
    finally:
        env.close()
        if path_file is not None:
            path_file.close()


status = 0
try:
    run()
except Exception:
    traceback.print_exc()
    status = 1
finally:
    sys.stdout.flush()
    sys.stderr.flush()
    app.app.post_quit(status)
    app.close()
sys.exit(status)
