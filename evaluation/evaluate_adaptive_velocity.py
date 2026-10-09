"""Deterministic policy versus independent constant-speed physics rollouts.

Both runs reset to the same seed, path, robot and sensor randomization. The
default baseline uses 6 mm/s; --match-mean-speed instead matches the adaptive
mean command. Measured durations and throughput are always reported.
"""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT as _REPO_ROOT
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import traceback
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
policy = parser.add_mutually_exclusive_group(required=True)
policy.add_argument("--checkpoint", type=str)
policy.add_argument("--constant-action", type=float, help="Scripted action for evaluator regression")
policy.add_argument("--tracking-feedback-gain", type=float,
                    help="Scripted causal tracking-error feedback; reports no RL claim")
parser.add_argument("--baseline-speed-mm-s", type=float, default=6.0,
                    help="Independent constant baseline (default: 6 mm/s)")
parser.add_argument("--match-mean-speed", action="store_true",
                    help="Match the baseline to the policy mean command instead")
parser.add_argument("--no-force-compensation", action="store_true",
                    help="Ablate the inverse-force process prior")
parser.add_argument("--require-improvement", action="store_true",
                    help="Fail unless raw processing CV improves with <=5%% cycle-time regression")
parser.add_argument("--min-cv-improvement-percent", type=float, default=0.0,
                    help="Minimum improvement over constant for the quality gate")
parser.add_argument("--compare-process-prior", action="store_true",
                    help="Also require improvement over an independent zero-residual process prior")
parser.add_argument("--diagnostic-only", action="store_true",
                    help="One scripted rollout for physics ablation; no policy performance claim")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--target-force-n", type=float,
                    help="Override scheduled normal contact force and set target F*v at nominal feed")
parser.add_argument("--contact-diameter-mm", type=float,
                    help="Spatial model contact diameter; default assumes full 30 mm tool face")
parser.add_argument("--surface-cell-mm", type=float, default=None,
                    help="Depth checkpoint: use its recorded grid; legacy evaluations: 2 mm")
parser.add_argument("--spindle-rpm", type=float,
                    help="Optional supplied RPM for an additional spinning-tool depth evaluation")
parser.add_argument("--max-steps", type=int, default=60000)
parser.add_argument("--path-length-mm", type=float, help="Temporary path prefix for a bounded diagnostic")
parser.add_argument("--output", type=Path, default=None)
parser.add_argument("--physics-substeps", type=int, default=None,
                    help="Physics integrations per 8 ms control tick")
parser.add_argument("--joint-stiffness", type=float)
parser.add_argument("--joint-damping", type=float)
parser.add_argument("--speed-limiter", choices=("legacy", "ruckig"),
                    help="Same limiter in adaptive and constant rollouts")
parser.add_argument("--profile-output", type=Path,
                    help="Save a cProfile of scripted rollout control steps")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 0.0 <= args.min_cv_improvement_percent < 100.0:
    parser.error("min-cv-improvement-percent must be in [0, 100)")
if args.diagnostic_only and (args.checkpoint or args.require_improvement):
    parser.error("diagnostic-only requires a scripted action and cannot test policy improvement")
app = AppLauncher(args).app

import gymnasium as gym
import h5py
import numpy as np
import torch
# The native control core also uses tiny CPU Torch operations. A large BLAS
# thread pool makes scripted baselines much slower than loaded-policy runs
# (whose inference loader already selects one thread).
torch.set_num_threads(1)
import matplotlib.pyplot as plt
from isaaclab_tasks.utils import parse_env_cfg
import nrs_rl.tasks  # noqa: F401
from nrs_rl.tasks.manager_based.nrs_rl.utils import visualization as vis
from nrs_rl.tasks.manager_based.nrs_rl.utils.velocity_policy import tracking_feedback_action
from nrs_rl.tasks.manager_based.nrs_rl.utils.velocity_evaluation import compare_rollouts, compare_surfaces, compare_rotation_surfaces
from nrs_rl.tasks.manager_based.nrs_rl.utils.preston_surface import PlanarPrestonSurface
from nrs_rl.tasks.manager_based.nrs_rl.mdp.action import spatial_to_rotmat
from nrs_rl.tasks.manager_based.nrs_rl.utils.analyze_rotation_depth import RotationDominatedSurface


def run():
    cfg = parse_env_cfg("Template-Nrs-Rl-v0", device=args.device, num_envs=1)
    cfg.seed = args.seed
    cfg.visualization.enable_visualizer = False
    cfg.actions.arm_action.integration.enable_debug_print = False
    if args.physics_substeps is not None:
        if args.physics_substeps < 1:
            raise ValueError("physics-substeps must be positive")
        cfg.decimation = args.physics_substeps
        cfg.sim.dt = 0.008 / args.physics_substeps
        cfg.sim.render_interval = cfg.decimation
    for name in ("stiffness", "damping"):
        value = getattr(args, "joint_" + name)
        if value is not None:
            setattr(cfg.scene.robot.actuators["ur10_arm"], name, value)
    cfg.sim.physx.gpu_max_rigid_contact_count = 2**18
    cfg.sim.physx.gpu_max_rigid_patch_count = 2**14
    cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**18
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**18
    cfg.sim.physx.gpu_collision_stack_size = 2**24
    root = _REPO_ROOT
    out = args.output or root / "logs/velocity_evaluation" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    path_file = None
    integration = cfg.actions.arm_action.integration
    if args.path_length_mm is not None:
        with h5py.File(integration.hdf5_file_path) as source:
            positions = source["position"][:]
            arc = np.r_[0., np.linalg.norm(np.diff(positions[:, :3], axis=0), axis=1).cumsum()]
            end = min(len(positions), max(2, int(np.searchsorted(arc, args.path_length_mm)) + 1))
            path_file = tempfile.NamedTemporaryFile(suffix=".h5", prefix="nrs_eval_path_")
            with h5py.File(path_file.name, "w") as target:
                target["position"] = positions[:end]
                target["force"] = source["force"][:end]
        integration.hdf5_file_path = path_file.name
    agent = None
    action_hold = None
    depth_objective = False
    if args.checkpoint:
        spec = importlib.util.spec_from_file_location("ppo_worker", root / "scripts/skrl/ppo_policy_node.py")
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        agent, _ = worker._build_agent(args.checkpoint)
        depth_objective = agent.velocity_contract['objective'] == 'preston_rotation_dominated_depth'
        if depth_objective and args.path_length_mm is not None:
            raise ValueError('An exported dwell profile must be evaluated on its full reference path')
        if depth_objective and agent.velocity_contract.get('reference_path_sha256') != hashlib.sha256(Path(integration.hdf5_file_path).read_bytes()).hexdigest():
            raise ValueError('Dwell policy must use its exact reference trajectory')
        action_hold = worker.PolicyActionHold(agent, agent.velocity_contract["policy_action_repeat"])
        # Restore the saved policy's speed/observation semantics even if the
        # repository defaults have changed since that checkpoint was trained.
        for name in worker.SCHEDULER_CONTRACT:
            setattr(integration, name, agent.velocity_contract[name])
        integration.tool_diameter_mm = agent.velocity_contract["tool_diameter_mm"]
        integration.spindle_rpm = agent.velocity_contract["spindle_rpm"]
    if args.target_force_n is not None:
        if args.target_force_n <= 0 or not np.isfinite(args.target_force_n):
            raise ValueError("target-force-n must be finite and positive")
        if agent is not None and integration.target_normal_force_n != args.target_force_n:
            raise ValueError("A checkpoint evaluation must keep its trained target force; use a scripted diagnostic")
        integration.target_normal_force_n = args.target_force_n
        integration.target_mrr_n_mm_s = args.target_force_n * integration.nominal_speed_mm_s
    if args.spindle_rpm is not None:
        integration.spindle_rpm = args.spindle_rpm
    if args.no_force_compensation:
        integration.force_rate_compensation = False
    if args.speed_limiter is not None:
        integration.speed_limiter_type = args.speed_limiter
    original_nominal = integration.nominal_speed_mm_s
    env = gym.make("Template-Nrs-Rl-v0", cfg=cfg)
    try:
        raw = env.unwrapped
        term = raw.action_manager.get_term("arm_action")
        if agent is not None and not np.isclose(agent.velocity_contract["physics_tool_diameter_mm"],
                                                term.physical_tool_diameter_mm):
            raise ValueError("Checkpoint collision geometry differs from the current tool; use its archived USD or retrain")
        if depth_objective and not np.isclose(agent.velocity_contract['reference_path_length_mm'], term.path_length_mm, atol=.1):
            raise ValueError('Dwell policy was planned for a different reference path length')
        endpoint_pose = term.traj_positions[-1].detach().cpu()
        endpoint_position = endpoint_pose[:3].numpy()
        endpoint_normal = spatial_to_rotmat(endpoint_pose[3:6].unsqueeze(0))[0, :, 2].numpy()
        surface_args = dict(tool_diameter_mm=integration.tool_diameter_mm,
                            contact_diameter_mm=args.contact_diameter_mm,
                            cell_size_mm=(args.surface_cell_mm if args.surface_cell_mm is not None
                                          else agent.velocity_contract['reference_roi']['cell_size_mm']
                                          if depth_objective else 2.))
        reference_xyz = term.traj_positions[:, :3].detach().cpu().numpy()
        if depth_objective:
            # Use the original reference precision, exactly as candidate training does.
            with h5py.File(integration.hdf5_file_path) as source:
                reference_xyz = source['position'][:, :3]
        surface = PlanarPrestonSurface(reference_xyz, endpoint_normal, **surface_args)
        rotation_surface = RotationDominatedSurface(reference_xyz, endpoint_normal, **surface_args) if depth_objective else None
        if depth_objective:
            roi = agent.velocity_contract['reference_roi']
            for key in ('contact_diameter_mm', 'cell_size_mm'):
                if not np.isclose(rotation_surface.metadata[key], roi[key]):
                    raise ValueError(f'Depth evaluation must preserve the trained ROI {key}')
            if rotation_surface.metadata['roi'] != roi['definition']:
                raise ValueError('Depth evaluation ROI differs from the training contract')
        spinning_surface = (PlanarPrestonSurface(reference_xyz, endpoint_normal, **surface_args,
                            velocity_model="rotating_disk", spindle_rpm=integration.spindle_rpm)
                            if integration.spindle_rpm is not None else None)
        traces = {}
        rows = []

        def capture(env_id, pose, normal_force):
            if env_id == 0:
                rows.append((pose.cpu().numpy().copy(), abs(float(normal_force)),
                             float(term.current_sliding_velocity_mm_s[0]),
                             float(term.current_mrr_n_mm_s[0]), float(term.realized_removal_step[0]),
                             vis.control_trace_snapshot(term), term.policy_state[0].cpu().numpy().copy(),
                             term._previous_command_pose[0].cpu().numpy().copy()))

        term._record_visualization = capture
        summary = {"seed": args.seed, "checkpoint": args.checkpoint,
                   "checkpoint_sha256": worker._sha256(args.checkpoint) if agent is not None else None,
                   "constant_action": args.constant_action,
                   "tracking_feedback_gain": args.tracking_feedback_gain, "dt_s": raw.step_dt,
                   "physics_device": str(raw.device),
                   "controller_device": str(term.device),
                   "policy_action_repeat": agent.velocity_contract["policy_action_repeat"] if agent else 1,
                   "physics_dt_s": cfg.sim.dt, "physics_substeps": cfg.decimation,
                   "observation_version": agent.velocity_contract["schema_version"] if agent else 3,
                   "observation_size": 14, "turn_preview_mm": integration.turn_preview_mm,
                   "force_rate_compensation": integration.force_rate_compensation,
                   "residual_speed_fraction": integration.residual_speed_fraction,
                   "target_normal_force_n": integration.target_normal_force_n,
                   "physics_tool_diameter_mm": term.physical_tool_diameter_mm,
                   "target_mrr_n_mm_s": integration.target_mrr_n_mm_s,
                   "preston_surface_model": surface.metadata,
                   "rotation_depth_model": rotation_surface.metadata if rotation_surface else None,
                   "reference_roi": agent.velocity_contract.get('reference_roi') if depth_objective else None,
                   "primary_objective": 'rotation_dominated_spatial_depth' if depth_objective else 'raw_force_tcp_speed_cv',
                   "spinning_surface_model": spinning_surface.metadata if spinning_surface else None,
                   "policy_signal_tau_s": integration.policy_signal_tau_s,
                   "speed_limiter_type": integration.speed_limiter_type,
                   "completion_criterion": "reference_path_cursor; endpoint settling is not simulated",
                   "endpoint_reference_position_mm": endpoint_position.tolist(),
                   "path_length_mm": term.path_length_mm,
                   "servo_stiffness": cfg.scene.robot.actuators["ur10_arm"].stiffness,
                   "servo_damping": cfg.scene.robot.actuators["ur10_arm"].damping,
                   "max_acceleration_mm_s2": integration.max_speed_acceleration_mm_s2,
                   "max_jerk_mm_s3": integration.max_speed_jerk_mm_s3}
        with torch.inference_mode():
            modes = ("adaptive",) if args.diagnostic_only else ("adaptive", "constant")
            if args.compare_process_prior and not args.diagnostic_only:
                modes += ("process_prior",)
            for mode in modes:
                if mode == "constant":
                    adaptive = traces["adaptive"]
                    active = adaptive["polishing_active"] > 0
                    baseline = args.baseline_speed_mm_s
                    if args.match_mean_speed:
                        baseline = float(adaptive["commanded_speed_mm_s"][active].mean())
                    if not 0 < baseline <= integration.max_speed_mm_s:
                        raise ValueError("baseline speed must be positive and within the command limit")
                    term.int_cfg.nominal_speed_mm_s = baseline
                    term.int_cfg.residual_speed_fraction = 0.0
                    term.int_cfg.force_rate_compensation = False
                    summary["baseline_speed_mm_s"] = baseline
                elif mode == "process_prior":
                    term.int_cfg.nominal_speed_mm_s = original_nominal
                    term.int_cfg.residual_speed_fraction = 0.0
                    term.int_cfg.force_rate_compensation = True
                obs, _ = env.reset(seed=args.seed)
                if action_hold is not None:
                    action_hold.reset()
                randomization = {
                    "force_scale": float(term._force_scale[0]),
                    "force_bias_n": float(term._force_bias[0]),
                    "action_delay_steps": int(term._action_delay[0]),
                }
                if mode != "adaptive":
                    assert randomization == summary["adaptive"]["randomization"], "paired randomization differs"
                rows.clear()
                completed = False
                rollout_started = time.monotonic()
                profiler = None
                if args.profile_output and mode == "adaptive":
                    import cProfile
                    profiler = cProfile.Profile()
                    profiler.enable()
                for step in range(args.max_steps):
                    if mode == "adaptive" and agent is not None:
                        action = action_hold.act(obs["policy"].cpu(), step).to(raw.device)
                    elif mode == "adaptive" and args.tracking_feedback_gain is not None:
                        process = term.process_states[0]
                        reference = process.reference_speed(
                            integration.nominal_speed_mm_s, integration.target_mrr_n_mm_s,
                            integration.contact_force_n, integration.force_rate_compensation)
                        value = tracking_feedback_action(
                            filtered_speed=process.speed, applied_speed=float(term.commanded_speed_mm_s[0]),
                            acceleration=term.speed_limiters[0].acceleration, reference_speed=reference,
                            residual_fraction=integration.residual_speed_fraction,
                            gain=args.tracking_feedback_gain, signal_tau=integration.policy_signal_tau_s)
                        if not process.valid or process.force < integration.contact_force_n:
                            value = 0.0
                        action = torch.full(env.action_space.shape, value, device=raw.device)
                    else:
                        value = args.constant_action if mode == "adaptive" else 0.0
                        action = torch.full(env.action_space.shape, value, device=raw.device)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
                    if step % 1000 == 0:
                        print("EVALUATION", mode, step, "cursor_mm", float(term.path_cursor_mm[0]), flush=True)
                    if (terminated | truncated).any():
                        completed = bool(raw.termination_manager.get_term("trajectory_finished")[0])
                        break
                if profiler is not None:
                    profiler.disable()
                    profiler.dump_stats(str(args.profile_output))
                trace = {
                    "time_s": np.arange(len(rows)) * raw.step_dt,
                    "tcp_pose": np.stack([r[0] for r in rows]),
                    "normal_force_n": np.array([r[1] for r in rows]),
                    "measured_speed_mm_s": np.array([r[2] for r in rows]),
                    "raw_mrr_n_mm_s": np.array([r[3] for r in rows]),
                    "removal_step_n_mm": np.array([r[4] for r in rows]),
                    "policy_observation": np.stack([r[6] for r in rows]),
                    "command_pose": np.stack([r[7] for r in rows]),
                }
                columns = np.stack([r[5] for r in rows])
                trace.update({name: columns[:, i] for i, name in enumerate(vis.CONTROL_TRACE_FIELDS)})
                traces[mode] = trace
                np.savez_compressed(out / f"{mode}_trace.npz", **trace)
                contact = np.flatnonzero(trace["removal_step_n_mm"] > 0)
                metrics = vis._compute_episode_summary(
                    1, trace["time_s"], trace["tcp_pose"][:, :3], trace["normal_force_n"],
                    trace["measured_speed_mm_s"], trace["removal_step_n_mm"], trace["raw_mrr_n_mm_s"],
                    int(contact[0]) if contact.size else None, 0., trace["polishing_active"] > 0)
                metrics["completed"] = completed
                metrics["preston_surface"] = surface.integrate_trace(trace)
                surface.save(out / f"{mode}_preston_surface.npz")
                if spinning_surface is not None:
                    metrics["spinning_preston_surface"] = spinning_surface.integrate_trace(trace)
                    spinning_surface.save(out / f"{mode}_spinning_preston_surface.npz")
                metrics["reference_path_completed"] = completed
                endpoint_error = endpoint_position - trace["tcp_pose"][-1, :3]
                normal_error = float(endpoint_error @ endpoint_normal)
                metrics["endpoint_tangential_error_mm"] = float(np.linalg.norm(
                    endpoint_error - normal_error * endpoint_normal))
                metrics["endpoint_normal_error_mm"] = normal_error
                metrics["endpoint_speed_mm_s"] = float(trace["measured_speed_mm_s"][-1])
                endpoint_command_error = endpoint_position - trace["command_pose"][-1, :3]
                metrics["endpoint_command_tangential_error_mm"] = float(np.linalg.norm(
                    endpoint_command_error - (endpoint_command_error @ endpoint_normal) * endpoint_normal))
                metrics["endpoint_tcp_to_command_error_mm"] = float(np.linalg.norm(
                    trace["command_pose"][-1, :3] - trace["tcp_pose"][-1, :3]))
                metrics["rollout_wall_seconds"] = time.monotonic() - rollout_started
                metrics["randomization"] = randomization
                active = trace["polishing_active"] > 0
                metrics["mean_applied_speed_mm_s"] = float(trace["commanded_speed_mm_s"][active].mean()) if active.any() else 0.
                metrics["shield_fraction"] = float(trace["safety_shield_active"][active].mean()) if active.any() else 0.
                metrics["fault_fraction"] = float((trace["safety_fault_reason"][active] != 0).mean()) if active.any() else 0.
                processing_start_s = float(trace["time_s"][active][0]) if active.any() else float("inf")
                for label, mask in (("processing", active), ("settled", active & (
                        trace["time_s"] >= processing_start_s + 5.0))):
                    rate = trace["raw_mrr_n_mm_s"][mask]
                    velocity = trace["measured_speed_mm_s"][mask]
                    command = trace["commanded_speed_mm_s"][mask]
                    metrics[label + "_samples"] = int(mask.sum())
                    metrics[label + "_raw_rate_cv"] = float(rate.std() / rate.mean()) if rate.size and rate.mean() > 0 else None
                    metrics[label + "_rate_target_rmse"] = float(np.sqrt(np.mean((rate - integration.target_mrr_n_mm_s)**2))) if rate.size else None
                    metrics[label + "_speed_std_mm_s"] = float(velocity.std()) if rate.size else None
                    metrics[label + "_command_std_mm_s"] = float(command.std()) if rate.size else None
                    metrics[label + "_tracking_speed_rmse_mm_s"] = float(np.sqrt(np.mean((velocity-command)**2))) if rate.size else None
                if rotation_surface is not None:
                    metrics['rotation_depth'] = rotation_surface.integrate_trace(trace)
                    rotation_surface.save(out / f'{mode}_rotation_surface.npz')
                summary[mode] = metrics
                (out / "summary.json").write_text(json.dumps(summary, indent=2))
                if not completed:
                    raise RuntimeError(f"{mode} did not complete; failure trace saved to {out}")
        if args.diagnostic_only:
            print("PHYSICS_DIAGNOSTIC_PASS", out, flush=True)
            return
        adaptive, constant = summary["adaptive"], summary["constant"]
        summary["comparison"] = {
            **compare_rollouts(adaptive, constant, summary.get("process_prior"),
                               args.min_cv_improvement_percent / 100.0),
            "scope": "single seed; reference path prefix" if args.path_length_mm else "single seed; full reference path",
            "primary_metric_scope": "all processing samples until reference completion; actual endpoint settling is not simulated",
            "adaptive_kind": ("learned residual" + (" + process prior" if summary["force_rate_compensation"] else ""))
                             if args.checkpoint else ("scripted tracking feedback; no learned policy"
                                 if args.tracking_feedback_gain is not None else "scripted residual; no learned policy"),
            "settling_exclusion_s": 5.0,
        }
        summary["surface_comparison"] = compare_surfaces(
            adaptive["preston_surface"], constant["preston_surface"])
        summary["comparison"]["rate_quality_passed"] = summary["comparison"]["passed"]
        summary["comparison"]["passed"] &= summary["surface_comparison"]["passed"]
        if depth_objective:
            summary['rotation_depth_comparison'] = compare_rotation_surfaces(
                adaptive['rotation_depth'], constant['rotation_depth'], args.min_cv_improvement_percent / 100.)
            summary['comparison']['passed'] = bool(
                summary['rotation_depth_comparison']['passed']
                and adaptive['completed'] and constant['completed']
                and adaptive['shield_fraction'] == 0 and adaptive['fault_fraction'] == 0
                and summary['comparison']['processing_duration_ratio'] <= 1.05)
            summary['comparison']['primary_metric_scope'] = 'same fixed swept ROI, all cells including zeros; reference cursor completion'
            summary['comparison']['primary_objective'] = 'rotation_dominated_spatial_depth'
            summary['comparison']['adaptive_kind'] = 'learned bounded feedback around a modeled dwell profile'
        summary["comparison"]["physical_depth_validated"] = False
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        for mode, trace in traces.items():
            active = trace["polishing_active"] > 0
            t = trace["time_s"][active]
            t = t - t[0]
            axes[0].plot(t, trace["raw_mrr_n_mm_s"][active], label=mode, linewidth=.8)
            axes[1].plot(t, trace["measured_speed_mm_s"][active], label=mode, linewidth=.8)
        axes[0].set_ylabel("Raw removal rate [N mm/s]")
        axes[1].set_ylabel("Measured speed [mm/s]")
        axes[1].set_xlabel("Polishing time [s]")
        for ax in axes:
            ax.legend()
            ax.grid(alpha=.25)
        fig.suptitle("Independent physics rollouts; all polishing samples")
        fig.tight_layout()
        fig.savefig(out / "comparison.png", dpi=180)
        plt.close(fig)
        if args.require_improvement and not summary["comparison"]["passed"]:
            raise RuntimeError(f"Policy quality comparison failed; see {out / 'summary.json'}")
        label = "EVALUATION_PASS" if summary["comparison"]["passed"] else "EVALUATION_COMPLETE_NO_IMPROVEMENT"
        print(label, out, json.dumps(summary), flush=True)
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
