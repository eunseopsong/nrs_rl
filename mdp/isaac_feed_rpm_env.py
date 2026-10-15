"""Measured Isaac contact/TCP motion with a separately modeled virtual spindle."""
import math

import numpy as np
import torch

from .isaac_latest_force_env import IsaacLatestForceEnv
from .rpm_control import RPMConfig, RateObjective, VirtualSpindle


class IsaacFeedRPMEnv(IsaacLatestForceEnv):
    def __init__(self, count, seed, native, controller_snapshot, output, config=None,
                 gui_scene=None, nominal_sensors=True, steady_gain=.2, record=False):
        self.rpm_config = config or RPMConfig()
        self.spindles = [VirtualSpindle(self.rpm_config) for _ in range(count)]
        self.rate_objectives = [RateObjective(self.rpm_config) for _ in range(count)]
        self.contact_started = np.zeros(count, dtype=bool)
        self.last_force = np.zeros(count)
        self.filtered_force = np.zeros(count)
        self.force_slew = np.zeros(count)
        self.last_rate = np.zeros(count)
        self.rate_step_reward = np.zeros(count)
        self.collect_rate = np.zeros(count, dtype=bool)
        self.records = [[] for _ in range(count)]
        self.record_enabled = record
        self.completed_records = [None] * count
        self.previous_joint_action = np.zeros((count, 2))
        self.ready = False
        super().__init__(count, seed, native, controller_snapshot, output, .5, gui_scene,
                         None, nominal_sensors, steady_gain=steady_gain)
        self.ready = True
        for i in range(count):
            self.clear(i)

    def predicted(self, i, terminal=False):
        # Full reference ROI, with initial contact and zero-depth cells included.
        g = self.fine_geometry
        values = self.fine[i].depth.ravel()[g.roi_flat].copy()
        if not terminal:
            index = min(int(g.index([float(self.term.path_cursor_mm[i])])[0]), len(g.remaining) - 1)
            values += g.remaining[index]
        mean = float(values.mean())
        return float(values.var() / max(mean**2, 1.e-12))

    def clear(self, i):
        super().clear(i)
        self.spindles[i].reset()
        self.rate_objectives[i].reset()
        self.contact_started[i] = False
        self.last_force[i] = self.filtered_force[i] = self.force_slew[i] = self.last_rate[i] = 0.
        self.records[i] = []
        self.previous_joint_action[i] = 0.
        for surface in (self.surface[i], self.fine[i]):
            surface.omega = self.rpm_config.nominal_rpm * 2. * math.pi / 60.

    def capture(self, i, pose, normal_force):
        if not self.ready:
            return super().capture(i, pose, normal_force)
        point = pose.detach().cpu().numpy().copy()
        previous = self.previous[i]
        delta = np.zeros(3) if previous is None else point[:3] - previous
        delta[2] = 0.
        midpoint = point[:3] - .5 * delta
        force = abs(float(normal_force))
        fault = int(self.term.safety_fault_reason[i])
        shield = bool(self.term.safety_shield_active[i])
        valid = force >= 1.5 and fault == 0 and not shield
        self.contact_started[i] |= force >= 1.5
        speed = float(self.term.current_sliding_velocity_mm_s[i])
        velocity = delta / max(np.linalg.norm(delta), 1.e-12) * speed
        before = self.fine[i].integrated_volume
        active = bool(self.term.polishing_active[i])
        super().capture(i, pose, normal_force)
        if not active and self.contact_started[i]:
            # The same measured initial approach contact belongs to both the
            # accumulated depth objective and the time-domain rate objective.
            # Keep the inherited 16D observation's active-motion history intact;
            # the fine evaluation/reward grid includes the initial contact too.
            self.fine[i].deposit(midpoint, velocity, force, .008,
                                 contact=bool(force >= 1.5 and fault == 0))
        self.last_rate[i] = max(0., (self.fine[i].integrated_volume - before) / .008)
        old_filtered = self.filtered_force[i]
        self.filtered_force[i] += -math.expm1(-.008 / .024) * (force - old_filtered)
        self.force_slew[i] = (self.filtered_force[i] - old_filtered) / .008
        self.last_force[i] = force
        if self.contact_started[i] and self.collect_rate[i]:
            self.rate_step_reward[i] += self.rate_objectives[i].update(
                self.last_rate[i], force, self.force_slew[i], valid)
        if self.record_enabled and self.contact_started[i]:
            self.records[i].append((point, force, speed, float(self.term.path_cursor_mm[i]),
                float(self.term.commanded_speed_mm_s[i]), self.spindles[i].rpm, self.spindles[i].commanded_rpm,
                self.last_rate[i], midpoint.copy(), velocity.copy(), fault, shield, active,
                self.force_slew[i]))

    def observation(self):
        base = super().observation()
        c = self.rpm_config
        extra = np.array([[math.log(s.rpm / c.nominal_rpm) / c.log_action_scale,
            math.log(s.commanded_rpm / c.nominal_rpm) / c.log_action_scale,
            self.force_slew[i] / 100., self.last_rate[i] / c.target_rate_over_k - 1.,
            min(1., self.rate_objectives[i].count * .008 / 3.), self.filtered_force[i] / 20.]
            for i, s in enumerate(self.spindles)], dtype=np.float32)
        return torch.cat((base, torch.as_tensor(extra)), -1)

    def metrics(self, i, completed):
        m = self.fine[i].metrics()
        return {'contact_cv': m['spatial_depth_cv'], 'contact_volume_over_k': m['roi_volume_over_k'],
                'completed': bool(completed), 'cursor_mm': float(self.term.path_cursor_mm[i]),
                **self.rate_objectives[i].metrics()}

    def step(self, action):
        if action.shape != (self.count, 2) or not torch.isfinite(action).all() or (action.abs() > 1.000001).any():
            raise ValueError('Expected bounded feed and log-RPM actions')
        previous = self.potential.copy()
        old_action = self.previous_joint_action.copy()
        done = np.zeros(self.count, dtype=bool)
        rows = [None] * self.count
        terminal = np.zeros(self.count)
        self.rate_step_reward.fill(0.)
        self.collect_rate.fill(True)
        for _ in range(self.rpm_config.policy_ticks):
            for i, spindle in enumerate(self.spindles):
                rpm = spindle.step(float(action[i, 1]), bool(self.contact_started[i]))
                for surface in (self.surface[i], self.fine[i]):
                    surface.omega = rpm * 2. * math.pi / 60.
            _, _, terminated, truncated, _ = self.env.step(action[:, :1])
            finished = (terminated | truncated).cpu().numpy() & ~done
            for i in np.flatnonzero(finished):
                complete = bool(self.raw.termination_manager.get_term('trajectory_finished')[i])
                rows[i] = self.metrics(i, complete)
                terminal[i] = self.predicted(i, terminal=True)
                if self.record_enabled:
                    self.completed_records[i] = {'trace': self.records[i],
                        'depth': self.fine[i].depth.copy(), 'metrics': rows[i]}
                self.collect_rate[i] = False
                self.clear(i)
            done |= finished
        potential = np.array([self.predicted(i) for i in range(self.count)])
        reward = self.rpm_config.spatial_cv_weight * (previous - np.where(done, terminal, potential))
        reward += self.rate_step_reward
        reward -= 1.e-4 * np.square(action.cpu().numpy() - old_action).sum(-1)
        self.previous_joint_action[:] = action.cpu().numpy()
        self.previous_joint_action[done] = 0.
        for i in np.flatnonzero(done):
            reward[i] -= self.rpm_config.incomplete_penalty * (not rows[i]['completed'])
            reward[i] -= 100. * max(0., .9 - rows[i]['mean_rate_ratio'])**2
        self.potential = potential
        return self.observation(), torch.as_tensor(reward, dtype=torch.float32), torch.as_tensor(done), rows
