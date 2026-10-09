"""CPU regressions for speed bounds, native controller parity and rewards."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT as _REPO_ROOT
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = _REPO_ROOT
TASK = ROOT / "source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl"
SCHEDULER_ROOT = Path(os.environ.get("NRS_Y2_SCHEDULER_ROOT", ROOT / "deployment/y2_speed_limiter"))


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


binding = load_file("_y2_control_pybind", next((TASK / "y2_control_pybind/y2_control_py").glob("_y2_control_pybind*.so")))
ruckig_path = next((TASK / "y2_control_pybind/y2_control_py").glob("_velocity_ruckig*.so"), None)
ruckig_binding = load_file("_velocity_ruckig", ruckig_path) if ruckig_path else None
rewards = load_file("smooth_rewards", TASK / "mdp/rewards.py")
process_model = load_file("velocity_policy", TASK / "utils/velocity_policy.py")
policy_worker = load_file("velocity_worker", ROOT / "scripts/skrl/ppo_policy_node.py")
evaluation = load_file("velocity_evaluation", TASK / "utils/velocity_evaluation.py")


class EvaluationTests(unittest.TestCase):
    @staticmethod
    def metrics(cv, samples=1000, rate=60.):
        return {"processing_rate_cv": cv, "processing_samples": samples,
                "processing_mean_rate": rate, "completed": True, "shield_fraction": 0.}

    def test_small_combined_gain_is_not_a_policy_gain_over_prior(self):
        comparison = evaluation.compare_rollouts(
            self.metrics(.128551), self.metrics(.139070), self.metrics(.127281))
        self.assertGreater(comparison["raw_cv_improvement_fraction"], .07)
        self.assertLess(comparison["cv_improvement_over_process_prior"], 0.)
        self.assertFalse(comparison["passed"])

    def test_quality_gate_enforces_requested_gain_and_productivity(self):
        constant, prior = self.metrics(.14), self.metrics(.127)
        self.assertTrue(evaluation.compare_rollouts(self.metrics(.10), constant, prior, .20)["passed"])
        self.assertFalse(evaluation.compare_rollouts(self.metrics(.12), constant, prior, .20)["passed"])
        self.assertFalse(evaluation.compare_rollouts(self.metrics(.10, samples=1100), constant, prior, .20)["passed"])
        self.assertFalse(evaluation.compare_rollouts(self.metrics(.10, rate=50.), constant, prior, .20)["passed"])

    def test_equal_perfect_signals_and_faults_cannot_pass_as_improvement(self):
        self.assertFalse(evaluation.compare_rollouts(self.metrics(0.), self.metrics(0.))["passed"])
        bad = {**self.metrics(.01), "fault_fraction": .001}
        self.assertFalse(evaluation.compare_rollouts(bad, self.metrics(.14))["passed"])


class SpeedLimiterTests(unittest.TestCase):
    @unittest.skipUnless(ruckig_binding, "optional Ruckig comparison module is not built")
    def test_ruckig_reduces_latency_without_relaxing_kinematic_limits(self):
        requests = np.r_[np.full(500, 6.), np.full(500, 12.), np.zeros(500),
                         np.random.default_rng(123).uniform(-10, 20, 20000)]
        settling = []
        for cls in (binding.PpoSpeedLimiter, ruckig_binding.PpoRuckigSpeedLimiter):
            limiter = cls()
            velocity = np.array([limiter.step(float(value)) for value in requests])
            acceleration = np.diff(np.r_[0., velocity]) / .008
            jerk = np.diff(np.r_[0., acceleration]) / .008
            self.assertGreaterEqual(velocity.min(), -1e-10)
            self.assertLessEqual(velocity.max(), 12 + 1e-10)
            self.assertLessEqual(np.abs(acceleration).max(), 16 + 1e-8)
            self.assertLessEqual(np.abs(jerk).max(), 160 + 1e-6)
            settling.append(np.flatnonzero(velocity[:500] >= 5.94)[0] * .008)
        self.assertLess(settling[1], .5 * settling[0])

    @unittest.skipUnless(ruckig_binding, "optional Ruckig comparison module is not built")
    def test_ruckig_stop_and_release_do_not_keep_old_acceleration(self):
        for enabled, request in ((False, 10.), (True, float("nan"))):
            limiter = ruckig_binding.PpoRuckigSpeedLimiter()
            for _ in range(40):
                limiter.step(10.)
            self.assertEqual(limiter.step(request, enabled), 0.)
            self.assertEqual(limiter.acceleration, 0.)
            fresh = ruckig_binding.PpoRuckigSpeedLimiter()
            for _ in range(200):
                self.assertEqual(limiter.step(6.), fresh.step(6.))

    def test_acceleration_jerk_bounds_and_no_overshoot(self):
        limiter = binding.PpoSpeedLimiter()
        rng = np.random.default_rng(14)
        requests = np.r_[np.full(700, 10.02), np.full(700, 1.98), np.zeros(700), rng.uniform(-20, 30, 3000)]
        previous_v = previous_a = 0.0
        for requested in requests:
            speed = limiter.step(float(requested))
            acceleration = (speed - previous_v) / .008
            jerk = (acceleration - previous_a) / .008
            self.assertGreaterEqual(speed, -1e-10)
            self.assertLessEqual(speed, 12 + 1e-10)
            self.assertLessEqual(abs(acceleration), 16 + 1e-8)
            self.assertLessEqual(abs(jerk), 160 + 1e-6)
            previous_v, previous_a = speed, acceleration
        limiter.reset()
        response = [limiter.step(6.0) for _ in range(700)]
        self.assertTrue(np.all(np.diff(response) >= -1e-12))
        self.assertAlmostEqual(response[-1], 6.0, places=8)

    def test_emergency_stop_and_release_reset_history(self):
        for stop in ("disabled", "nonfinite"):
            limiter = binding.PpoSpeedLimiter()
            for _ in range(500):
                limiter.step(10.02)
            self.assertEqual(limiter.step(10.02, False) if stop == "disabled" else limiter.step(float("nan")), 0)
            fresh = binding.PpoSpeedLimiter()
            for _ in range(200):
                self.assertEqual(limiter.step(10.02), fresh.step(10.02))

    def test_native_scheduler_matches_training_speed_path(self):
        canonical = TASK / "y2_control_pybind/cpp/include/y2_control_pybind/ppo_speed_limiter.hpp"
        deployed = SCHEDULER_ROOT / "include/Y2RobMotion/ppo_speed_limiter.hpp"
        self.assertEqual(canonical.read_bytes(), deployed.read_bytes())
        source = r'''
#include "Y2RobMotion/ppo_trajectory_scheduler.hpp"
#include <iostream>
#include <iomanip>
int main() {
    PpoTrajectoryScheduler scheduler(0.008);
    scheduler.setTrajectory({0,0,0,0,0,0,0,0,20,10000,0,0,0,0,0,0,0,20});
    Mode5StepInput input;
    input.control_tcp_rotation = {1,0,0,0,1,0,0,0,1};
    double action; int enabled;
    std::cout << std::setprecision(17);
    while (std::cin >> action >> enabled) {
        input.control_tcp_pose[0] = scheduler.cursor();
        input.wrench_base[2] = enabled ? 20 : 40;
        auto result = scheduler.step(input, action);
        std::cout << result.scheduled_index_delta / .008;
        for(double value : scheduler.buildObservation(input)) std::cout << ' ' << value;
        std::cout << '\n';
    }
}
'''
        with tempfile.TemporaryDirectory(prefix="nrs_speed_parity_") as directory:
            cpp, exe = Path(directory) / "main.cpp", Path(directory) / "parity"
            cpp.write_text(source)
            subprocess.run(["g++", "-std=c++17", "-O2", "-I" + str(SCHEDULER_ROOT / "include"),
                            str(cpp), str(SCHEDULER_ROOT / "src/ppo_trajectory_scheduler.cpp"),
                            "-o", str(exe)], check=True)
            rng = np.random.default_rng(83)
            actions = rng.uniform(-3, 3, 1500)
            enabled = np.ones(1500, dtype=bool)
            enabled[700:710] = False
            inputs = "".join(f"{a:.17g} {int(e)}\n" for a, e in zip(actions, enabled))
            native = np.fromstring(subprocess.run([str(exe)], input=inputs, text=True, capture_output=True, check=True).stdout, sep=" ").reshape(-1, 15)
        # The existing Torch action LPF followed by the shared native limiter.
        filtered = torch.zeros(1)
        limiter = binding.PpoSpeedLimiter()
        process = process_model.ProcessState()
        expected = []
        cursor = previous_speed = 0.0
        for action, running in zip(actions, enabled):
            target = filtered + (1 - np.exp(-.008 / .08)) * (np.clip(action, -1, 1) - filtered)
            filtered += (target - filtered).clamp(-4 * .008, 4 * .008)
            force = 20.0 if running else 40.0
            process.update(force, previous_speed, force * previous_speed)
            requested = np.clip(6. * (1 + .50 * float(filtered)), 1, 12)
            speed = limiter.step(requested, bool(running))
            cursor += speed * .008
            obs = process.observation(
                target_force=20., max_speed=12., tracking_error=0., tracking_stop=10.,
                filtered_action=float(filtered), applied_speed=speed, progress=cursor/10000.,
                target_rate=120., curvature=0., contact=True, shield=not running,
                acceleration=limiter.acceleration, max_acceleration=16., clipped_action=float(np.clip(action,-1,1)))
            expected.append([speed, *obs])
            previous_speed = speed
        np.testing.assert_allclose(native, expected, atol=3e-6, rtol=0)

    def test_native_preview_matches_python_before_and_after_a_corner(self):
        source = r'''
#include "Y2RobMotion/ppo_trajectory_scheduler.hpp"
#include <iostream>
#include <iomanip>
int main() {
    std::vector<double> path;
    for(int i=0; i<=200; ++i) {
        double x = i<=100 ? .1*i : 10.;
        double y = i<=100 ? 0. : .1*(i-100);
        path.insert(path.end(), {x,y,0,0,0,0,0,0,20});
    }
    PpoTrajectoryScheduler scheduler(.008);
    scheduler.setTrajectory(path);
    Mode5StepInput input;
    input.control_tcp_rotation={1,0,0,0,1,0,0,0,1};
    input.wrench_base[2]=20.;
    double x,y;
    std::cout << std::setprecision(17);
    while(std::cin >> x >> y) {
        input.control_tcp_pose[0]=x; input.control_tcp_pose[1]=y;
        scheduler.step(input,0.);
        std::cout << scheduler.buildObservation(input)[9] << '\n';
    }
}
'''
        points = np.array([[min(i, 100) * .1, max(0, i-100) * .1, 0.] for i in range(201)])
        arc = np.r_[0., np.linalg.norm(np.diff(points, axis=0), axis=1).cumsum()]
        preview = process_model.PathTurnPreview(points, arc, policy_worker.SCHEDULER_CONTRACT["turn_preview_mm"])
        with tempfile.TemporaryDirectory(prefix="nrs_preview_parity_") as directory:
            cpp, exe = Path(directory) / "main.cpp", Path(directory) / "parity"
            cpp.write_text(source)
            subprocess.run(["g++", "-std=c++17", "-O2", "-I" + str(SCHEDULER_ROOT / "include"),
                            str(cpp), str(SCHEDULER_ROOT / "src/ppo_trajectory_scheduler.cpp"),
                            "-o", str(exe)], check=True)
            inputs = "".join(f"{x:.17g} {y:.17g}\n" for x, y, _ in points)
            native = np.fromstring(subprocess.run([str(exe)], input=inputs, text=True,
                                                  capture_output=True, check=True).stdout, sep=" ")
        np.testing.assert_allclose(native, [preview.at(s) for s in arc], atol=1e-12, rtol=0)


class RewardTests(unittest.TestCase):
    def test_ppo_keeps_unclipped_gaussian_samples_for_likelihood_ratios(self):
        import yaml
        from gymnasium.spaces import Box
        from skrl.utils.model_instantiators.torch import gaussian_model
        cfg = yaml.safe_load((TASK / "agents/skrl_ppo_cfg.yaml").read_text())["models"]["policy"]
        cfg.pop("class")
        policy = gaussian_model(observation_space=Box(-np.inf, np.inf, (14,)),
                                action_space=Box(-1., 1., (1,)), device="cpu", **cfg)
        with torch.no_grad():
            policy.act({"states": torch.zeros((1, 14))})  # materialize lazy layers
            for parameter in policy.parameters():
                parameter.zero_()
            [layer for layer in policy.modules() if isinstance(layer, torch.nn.Linear)][-1].bias.fill_(2.)
            samples, log_prob, outputs = policy.act({"states": torch.zeros((64, 14))})
        self.assertTrue((samples > 1.).any())
        expected = torch.distributions.Normal(outputs["mean_actions"], np.exp(-1.)).log_prob(samples)
        torch.testing.assert_close(log_prob, expected)

    def setUp(self):
        self.term = SimpleNamespace(
            action_delta=torch.tensor([0., 1.]),
            command_smoothness_valid=torch.tensor([True, True]),
            command_acceleration_mm_s2=torch.tensor([0., 16.]),
            command_jerk_mm_s3=torch.tensor([0., 160.]),
            int_cfg=SimpleNamespace(max_speed_acceleration_mm_s2=16., max_speed_jerk_mm_s3=160., target_mrr_n_mm_s=60.),
            mrr_fluctuation_n_mm_s=torch.tensor([0., 30.]),
            polishing_active=torch.tensor([True, True]), safety_fault_active=torch.tensor([False, False]),
        )
        self.env = SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda _: self.term))

    def test_equal_throughput_prefers_smooth_commands_and_removal(self):
        for reward in (rewards.action_rate_penalty, rewards.command_acceleration_penalty,
                       rewards.command_jerk_penalty, rewards.removal_variation_penalty):
            result = reward(self.env)
            self.assertEqual(float(result[0]), 0.)
            self.assertLess(float(result[1]), 0.)

    def test_emergency_brake_is_not_penalized_as_policy_jerk(self):
        self.term.command_smoothness_valid[:] = False
        self.term.command_jerk_mm_s3[:] = 100000.
        self.assertTrue((rewards.command_jerk_penalty(self.env) == 0).all())

    def test_fixed_rate_target_rejects_slow_drift_and_stopping(self):
        # Same mean throughput: slowly alternating 20/100 must lose to 60.
        self.term.current_mrr_n_mm_s = torch.tensor([60., 60.])
        steady = rewards.removal_rate_tracking_penalty(self.env).mean()
        self.term.current_mrr_n_mm_s = torch.tensor([20., 100.])
        drifting = rewards.removal_rate_tracking_penalty(self.env).mean()
        self.assertLess(float(drifting), float(steady))
        self.term.current_mrr_n_mm_s.zero_()
        self.assertTrue((rewards.removal_rate_tracking_penalty(self.env) < drifting).all())
        self.term.polishing_active[:] = False
        self.assertTrue((rewards.removal_rate_tracking_penalty(self.env) == 0).all())

    def test_raw_variation_cannot_be_hidden_by_filtering(self):
        self.term.filtered_mrr_n_mm_s = torch.tensor([60., 60.])
        self.term.current_mrr_n_mm_s = torch.tensor([30., 90.])
        torch.testing.assert_close(rewards.removal_rate_tracking_penalty(self.env), torch.tensor([-.25, -.25]))

    def test_completion_bonus_uses_rate_quality_and_is_once_per_episode(self):
        self.term.rate_squared_error_sum = torch.tensor([10., 100.])
        self.term.polishing_steps = torch.tensor([100., 100.])
        self.term.path_done = torch.tensor([True, False])
        self.env.step_dt = .008
        bonus = rewards.completion_rate_quality_reward(self.env) * self.env.step_dt
        torch.testing.assert_close(bonus, torch.tensor([np.exp(-.1), 0.], dtype=torch.float32))

    def test_ppo_learning_rate_cannot_grow_above_configured_start(self):
        import yaml
        from skrl.resources.schedulers.torch import KLAdaptiveLR
        cfg = yaml.safe_load((TASK / "agents/skrl_ppo_cfg.yaml").read_text())["agent"]
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=cfg["learning_rate"])
        scheduler = KLAdaptiveLR(optimizer, **cfg["learning_rate_scheduler_kwargs"])
        for kl in [0.] * 30 + [1.] * 30 + [0.] * 30:
            scheduler.step(kl)
            self.assertLessEqual(optimizer.param_groups[0]["lr"], cfg["learning_rate"])


class ProcessStateTests(unittest.TestCase):
    def test_legacy_loading_restores_ten_newton_trajectory_semantics(self):
        import json
        with tempfile.TemporaryDirectory(prefix="nrs_policy_contract_") as directory:
            root = Path(directory)
            (root / "params").mkdir()
            checkpoint = root / "checkpoints/agent.pt"
            old = {"schema_version": 2, "observation_size": 14, "control_period_s": .008}
            (root / "params/velocity_policy.json").write_text(json.dumps(old))
            contract = policy_worker.load_policy_contract(str(checkpoint))
            self.assertIsNone(contract["target_normal_force_n"])
            self.assertEqual(contract["physics_tool_diameter_mm"], 56.0)
            self.assertEqual(contract["turn_preview_mm"], 0.)
            self.assertEqual(contract["policy_action_repeat"], 1)
            for invalid in (None, 0., float("nan"), True):
                bad = {**old, "schema_version": 3, "turn_preview_mm": invalid}
                (root / "params/velocity_policy.json").write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    policy_worker.load_policy_contract(str(checkpoint))

    def test_preview_exposes_an_upcoming_turn_before_arrival(self):
        preview = process_model.PathTurnPreview([[0, 0, 0], [10, 0, 0], [10, 10, 0]], [0, 10, 20], 6)
        self.assertEqual(preview.at(3.9), 0)
        self.assertAlmostEqual(preview.at(7), .5)
        self.assertAlmostEqual(preview.at(10), 1)
        self.assertEqual(preview.at(10.1), 0)

    def test_preview_precedes_command_turn_despite_contact_tracking_lag(self):
        # Observed 20 N regression: the physical TCP was 7-8 mm behind the
        # command, so the former 6 mm feature first appeared after the turn.
        # Cover even the tracking guard (10 mm) and 0.3 s command response
        # at the configured absolute speed bound (12 mm/s).
        points, arc = [[0, 0, 0], [100, 0, 0], [100, 100, 0]], [0, 100, 200]
        physical_distance_before_response = 100 - 10 - .3 * 12
        previous = process_model.PathTurnPreview(points, arc, 6.)
        current = process_model.PathTurnPreview(points, arc, policy_worker.SCHEDULER_CONTRACT["turn_preview_mm"])
        self.assertEqual(previous.at(physical_distance_before_response), 0.)
        self.assertGreater(current.at(physical_distance_before_response), .1)

    def test_surface_gate_cannot_hide_under_removal_or_uncovered_area(self):
        baseline = dict(roi_cells=100, roi_area_mm2=400., spatial_depth_cv=.4,
                        mean_depth_over_k=1., zero_depth_fraction=.01)
        better = {**baseline, "spatial_depth_cv": .3}
        self.assertTrue(evaluation.compare_surfaces(better, baseline)["passed"])
        self.assertFalse(evaluation.compare_surfaces({**better, "mean_depth_over_k": .5}, baseline)["passed"])
        self.assertFalse(evaluation.compare_surfaces({**better, "mean_depth_over_k": 1.2}, baseline)["passed"])
        self.assertFalse(evaluation.compare_surfaces({**better, "zero_depth_fraction": .05}, baseline)["passed"])
        self.assertFalse(evaluation.compare_surfaces({**better, "spatial_depth_cv": None}, baseline)["passed"])
        with self.assertRaises(ValueError):
            evaluation.compare_surfaces({**better, "roi_cells": 99}, baseline)

    def test_tracking_feedback_distinguishes_filter_lag_from_speed_deficit(self):
        feedback = process_model.tracking_feedback_action
        # At constant acceleration, the observation LPF lags by about tau*a.
        self.assertAlmostEqual(feedback(filtered_speed=5.84, applied_speed=6., acceleration=2.,
                                        reference_speed=6., gain=.5), 0.)
        self.assertGreater(feedback(filtered_speed=5., applied_speed=6., acceleration=0.,
                                    reference_speed=6., gain=.5), 0.)
        self.assertLess(feedback(filtered_speed=7., applied_speed=6., acceleration=0.,
                                 reference_speed=6., gain=.5), 0.)
        self.assertEqual(feedback(filtered_speed=0., applied_speed=6., acceleration=0.,
                                  reference_speed=6., gain=10.), 1.)

    def test_deployment_rejects_mismatched_speed_and_filter_contract(self):
        contract = dict(policy_worker.SCHEDULER_CONTRACT)
        policy_worker._validate_scheduler_contract(contract)
        for name, bad in (("residual_speed_fraction", .67), ("policy_signal_tau_s", .2),
                          ("force_rate_compensation", True), ("speed_limiter_type", "ruckig"),
                          ("target_normal_force_n", 10.0)):
            with self.assertRaises(ValueError):
                policy_worker._validate_scheduler_contract({**contract, name: bad})
        with self.assertRaises(ValueError):
            policy_worker._validate_scheduler_contract({**contract, "physics_tool_diameter_mm": 56.0})

    def test_compensation_is_bounded_and_contact_loss_has_no_inverse_gain(self):
        process = process_model.ProcessState()
        for force in (0., 1., 1.5, 8., 10., 12., 100.):
            process.reset()
            process.update(force, 6., force * 6.)
            reference = process.reference_speed(6., 60., 1.5)
            self.assertGreaterEqual(reference, 4.5)
            self.assertLessEqual(reference, 7.5)
            if force < 1.5:
                self.assertEqual(reference, 6.)
            self.assertEqual(process.reference_speed(6., 60., 1.5, False), 6.)

    def test_reset_clears_causal_history(self):
        process = process_model.ProcessState()
        for _ in range(100):
            process.update(12., 5., 60.)
        process.reset()
        process.update(8., 7.5, 60.)
        self.assertEqual((process.force, process.speed, process.rate, process.force_derivative), (8., 7.5, 60., 0.))


if __name__ == "__main__":
    unittest.main()
