"""On-policy PPO rollouts using the frozen current C++ force core and Isaac TCP motion."""
import copy
import json
from pathlib import Path
import sys
import numpy as np
import torch

from .agents.latest_force_ppo import CONTRACT, check_contract
from ..utils.rotary_geometry import FixedForceGeometry
from ..utils.removal_metrics import processing_mask
from ..paths import REFERENCE


class IsaacLatestForceEnv:
    def __init__(self, count, seed, native, controller_snapshot, output, reward_cell_mm=.5,
                 gui_scene=None, reward_forecast=None, nominal_sensors=False):
        import gymnasium as gym
        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab.utils.io import dump_yaml
        import nrs_rl.tasks
        from . import action as action_module
        from . import fixed_force_ppo_action as ppo_action

        self.count, self.device = count, torch.device('cpu')
        ppo_action.check_ppo_or_baseline = check_contract
        cfg = parse_env_cfg('Template-Nrs-Rl-v0', device='cpu', num_envs=count)
        cfg.seed = seed
        cfg.visualization.enable_visualizer = False
        cfg.actions.arm_action.class_type = ppo_action.FixedForcePPOAction
        self.scene_profile = None
        if gui_scene is not None:
            from .isaac_gui_scene import configure_scene
            self.scene_profile = configure_scene(cfg, gui_scene, randomize=not nominal_sensors)
        cfg.log_dir = str(output)
        integration = cfg.actions.arm_action.integration
        if nominal_sensors:
            integration.force_scale_range = (1., 1.)
            integration.force_bias_range_n = (0., 0.)
            integration.max_action_delay_steps = 0
        integration.enable_debug_print = False
        integration.hdf5_file_path = str(REFERENCE)
        integration.max_force_abort_n = 100.
        integration.shield_timeout_s = 1000.  # ROS has no separate overload residence timeout.
        dump_yaml(str(output / 'env.yaml'), cfg)
        sys.path.insert(0, str(native.resolve()))
        from _latest_force_mode3 import TangentialMode3
        validation = json.loads((native / 'validation.json').read_text())
        assert validation['tangent_damping_ns_m'] == 6000.
        assert validation['tangent_stiffness_n_m'] == 2000.
        original = action_module.y2_pb.Mode3ForceController
        def make_controller(*values):
            values = list(values)
            values[0] = str(controller_snapshot / 'Y2ForceCon/src/checkpoints/NAF_MDGradi/NAF_mdGradi_policy_script.pt')
            return TangentialMode3(*values, 6000., 2000.)
        action_module.y2_pb.Mode3ForceController = make_controller
        try:
            self.env = gym.make('Template-Nrs-Rl-v0', cfg=cfg)
        finally:
            action_module.y2_pb.Mode3ForceController = original
        self.raw = self.env.unwrapped
        assert self.raw.step_dt == .008
        self.term = self.raw.action_manager.get_term('arm_action')
        for i in range(count):
            self.term.configure_velocity(i, CONTRACT)
        assert np.all(self.term.half_range == 6.)
        if self.scene_profile is not None:
            (output / 'loaded_scene_drives.json').write_text(json.dumps({
                'stiffness': self.term.robot.data.default_joint_stiffness.tolist(),
                'damping': self.term.robot.data.default_joint_damping.tolist()}, indent=2))
        self.geometry = FixedForceGeometry(REFERENCE)
        self.fine_geometry = FixedForceGeometry(REFERENCE, cell_mm=reward_cell_mm)
        self.surface = [copy.deepcopy(self.geometry.surface) for _ in range(count)]
        self.fine = [copy.deepcopy(self.fine_geometry.surface) for _ in range(count)]
        self.residence = [np.zeros_like(surface.depth) for surface in self.surface]
        self.entry_depth = [np.zeros_like(surface.depth) for surface in self.fine]
        self.mask = processing_mask(self.fine_geometry.surface, self.fine_geometry.positions)
        fine_mask = self.mask.ravel()[self.fine_geometry.roi_flat]
        self.remaining = self.fine_geometry.remaining[:, fine_mask]
        self.forecast_arc = self.fine_geometry.arc
        if reward_forecast is not None:
            from ..utils.removal_metrics import sha
            metadata = json.loads(reward_forecast.with_name('calibration.json').read_text())
            if sha(reward_forecast) != metadata['forecast_sha256']:
                raise ValueError('Reward forecast hash mismatch')
            with np.load(reward_forecast) as archive:
                assert float(archive['cell_mm']) == reward_cell_mm
                np.testing.assert_array_equal(archive['processing_mask'], self.mask)
                self.remaining = archive['remaining']
                self.forecast_arc = archive['arc']
            assert self.remaining.shape == (len(self.forecast_arc)-1, int(self.mask.sum()))
            assert metadata['actor_teacher_used'] is False
            (output / 'reward_calibration.json').write_text(json.dumps(metadata, indent=2))
        self.baseline_processing = float(self.fine_geometry.nominal[fine_mask].sum())
        self.baseline_full = float(self.fine_geometry.nominal.sum())
        self.previous = [None] * count
        self.samples = np.zeros(count, dtype=np.int64)
        self.episode_reward = np.zeros(count)
        self.previous_action = np.zeros(count)
        self.potential = np.zeros(count)
        self.term._record_visualization = self.capture
        self.env.reset(seed=seed)
        for i in range(count):
            self.clear(i)

    def reset_batch(self):
        """Start fresh training episodes at a rollout boundary."""
        self.env.reset()
        for i in range(self.count):
            self.clear(i)
        return self.observation()

    def clear(self, i):
        self.surface[i].reset()
        self.fine[i].reset()
        self.residence[i].fill(0.)
        self.entry_depth[i].fill(0.)
        self.previous[i] = None
        self.samples[i] = 0
        self.episode_reward[i] = 0.
        self.previous_action[i] = 0.
        self.potential[i] = self.predicted(i)

    def capture(self, i, pose, normal_force):
        point = pose.detach().cpu().numpy().copy()
        previous = self.previous[i]
        self.previous[i] = point[:3].copy()
        if not bool(self.term.polishing_active[i]):
            return
        delta = np.zeros(3) if previous is None else point[:3] - previous
        delta[2] = 0.
        speed = float(self.term.current_sliding_velocity_mm_s[i])
        velocity = delta / max(np.linalg.norm(delta), 1.e-12) * speed
        midpoint = point[:3] - .5 * delta
        force = abs(float(normal_force))
        contact = force >= 1.5 and int(self.term.safety_fault_reason[i]) == 0
        for surface in (self.surface[i], self.fine[i]):
            surface.deposit(midpoint, velocity, force, .008, contact=contact)
        if contact:
            surface = self.surface[i]
            center = (midpoint - surface.origin) @ surface.basis.T
            rr, cc, _, _, weights, _ = surface._footprint(center)
            self.residence[i][rr, cc] += .008 * (weights > 0)
        if self.samples[i] * .008 < 2.:
            self.entry_depth[i][:] = self.fine[i].depth
        self.samples[i] += 1

    def predicted(self, i, terminal=False):
        values = np.maximum(self.fine[i].depth - self.entry_depth[i], 0.)[self.mask]
        cursor = float(self.term.path_cursor_mm[i])
        if not terminal and cursor < self.geometry.length:
            index = np.clip(np.searchsorted(self.forecast_arc, cursor, side='right') - 1,
                            0, len(self.remaining) - 1)
            values = values + self.remaining[index]
        mean = float(values.mean())
        return float(values.var() / max(mean * mean, 1.e-12))

    def metrics(self, i, completed):
        values = np.maximum(self.fine[i].depth - self.entry_depth[i], 0.)[self.mask]
        full = self.fine[i].depth[self.fine[i].roi]
        return {'cv': float(values.std() / max(float(values.mean()), 1.e-6)),
                'volume_ratio': float(values.sum() / self.baseline_processing),
                'full_ratio': float(full.sum() / self.baseline_full),
                'seconds': float(self.samples[i] * .008), 'completed': bool(completed)}

    def observation(self):
        t = self.term
        features = self.geometry.features(
            np.stack([s.depth.ravel()[self.geometry.roi_flat] for s in self.surface]),
            np.stack([r.ravel()[self.geometry.roi_flat] for r in self.residence]),
            t.path_cursor_mm.cpu().numpy(), np.array([r.frontier for r in t.return_states]),
            t.current_abs_fz.cpu().numpy(), t.current_sliding_velocity_mm_s.cpu().numpy(),
            self.samples * .008, np.array([r.reverse_mm for r in t.return_states]),
            t.target_force, np.zeros(self.count),
            tracking=t.current_path_tracking_error_mm.cpu().numpy(),
            shield=t.safety_shield_active.cpu().numpy())
        return torch.as_tensor(features, dtype=torch.float32, device=self.device)

    def step(self, action):
        previous = self.potential.copy()
        previous_action = self.previous_action.copy()
        done = np.zeros(self.count, dtype=bool)
        records = [None] * self.count
        terminal = np.zeros(self.count)
        episode_returns = np.zeros(self.count)
        for _ in range(10):
            _, _, terminated, truncated, _ = self.env.step(action[:, None])
            finished = (terminated | truncated).cpu().numpy() & ~done
            for i in np.flatnonzero(finished):
                completed = bool(self.raw.termination_manager.get_term('trajectory_finished')[i])
                records[i] = self.metrics(i, completed)
                terminal[i] = self.predicted(i, terminal=True)
                episode_returns[i] = self.episode_reward[i]
                self.clear(i)
            done |= finished
        potential = np.array([self.predicted(i) for i in range(self.count)])
        current = np.where(done, terminal, potential)
        reward = 100. * (previous - current) - .0001 * .08
        reward -= 1.e-5 * np.square(action.cpu().numpy() - previous_action)
        self.previous_action = action.cpu().numpy().copy()
        self.previous_action[done] = 0.
        for i in np.flatnonzero(done):
            row = records[i]
            violation = max(0., .5 - row['volume_ratio'], .5 - row['full_ratio'])
            reward[i] -= 100. * violation ** 2 + 20. * (not row['completed'])
            row['episode_reward'] = float(episode_returns[i] + reward[i])
        self.episode_reward += np.where(done, 0., reward)
        self.potential = potential
        return self.observation(), torch.as_tensor(reward, dtype=torch.float32), torch.as_tensor(done), records

    def close(self):
        self.env.close()
