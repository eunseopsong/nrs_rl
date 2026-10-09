"""Experimental nonrotating spatial-dose policy and bounded local retracing.

No ROS dependency. The fast model is a Preston/servo surrogate, not Isaac
physics. Force and feed are continuous actions; dwell is their measured
residence-time consequence. A third action requests one bounded local return.
"""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT as _REPO_ROOT
from pathlib import Path
import importlib.util
import math
import numpy as np
import torch

ROOT = _REPO_ROOT
from nrs_rl.tasks.manager_based.nrs_rl.utils.preston_surface import PlanarPrestonSurface
REFERENCE = ROOT / 'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/datasets/cmd_continue9D_flat.h5'
PARAMETERS = ['geometry_force_gain', 'online_force_gain', 'force_bias',
              'force_speed_coupling', 'rate_feedback_gain', 'speed_bias',
              'tracking_slowdown', 'return_gain', 'return_threshold']
LOW = np.array([0., 0., -.12, 0., 0., -.08, 0., 0., .025])
HIGH = np.array([5., 2., .12, 1.5, .15, .12, .5, 3., .6])
OBSERVATIONS = ['progress', 'force_over_20', 'speed_over_6',
                'centered_nominal_depth_deficit', 'centered_predicted_final_deficit',
                'behind_minus_current_deficit', 'local_residence_over_5s',
                'tracking_error_over_10mm', 'elapsed_over_nominal',
                'reverse_budget_fraction', 'contact', 'shield',
                'target_force_over_20', 'rate_error', 'return_allowed',
                'remaining_nominal_depth_over_target']
CONTRACT = {
    'backend': 'experimental_spatial_uniformity', 'schema_version': 1,
    'observation_names': OBSERVATIONS, 'action_names': ['feed', 'force', 'local_return'],
    'action_mapping': ['6+3*a0 mm/s', '20+6*a1 N', 'request local return when a2>0'],
    'rotation_contribution': False, 'spindle_rpm': 0.,
    'control_period_s': .008, 'policy_period_s': .08,
    'force_bounds_n': [14., 26.], 'force_slew_n_s': 5.,
    'forward_speed_bounds_mm_s': [3., 9.], 'reverse_speed_mm_s': 3.,
    'acceleration_limit_mm_s2': 16., 'jerk_limit_mm_s3': 160.,
    'local_return_mm': 5., 'return_cooldown_progress_mm': 20.,
    'reverse_distance_budget_fraction': .05, 'max_distance_ratio': 1.10,
    'second_complete_pass': False, 'static_dwell_removal': 0.,
    'deployment_compatible_with_current_mode5': False,
    'depth_units': 'h/K, uncalibrated',
}


def action_numpy(obs, parameters):
    p = np.asarray(parameters)
    target = np.clip(20. * (1. + p[..., 0]*obs[..., 3] + p[..., 1]*obs[..., 4] + p[..., 2]), 14., 26.)
    speed = np.clip(6. * (1. + p[..., 3]*(20./target-1.) - p[..., 4]*obs[..., 13]
                         + p[..., 5] - p[..., 6]*obs[..., 7]), 3., 9.)
    request = np.clip(p[..., 7]*obs[..., 5]-p[..., 8], -1., 1.)
    request = np.where((obs[..., 14] > .5) & (obs[..., 10] > .5) & (obs[..., 11] < .5), request, -1.)
    return np.stack(((speed-6.)/3., (target-20.)/6., request), axis=-1)


class SpatialUniformityActor(torch.nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.register_buffer('parameters_vector', torch.tensor(parameters, dtype=torch.float32))

    def forward(self, obs):
        p = self.parameters_vector
        target = torch.clamp(20.*(1.+p[0]*obs[:, 3]+p[1]*obs[:, 4]+p[2]), 14., 26.)
        speed = torch.clamp(6.*(1.+p[3]*(20./target-1.)-p[4]*obs[:, 13]+p[5]-p[6]*obs[:, 7]), 3., 9.)
        request = torch.clamp(p[7]*obs[:, 5]-p[8], -1., 1.)
        allowed = (obs[:, 14] > .5) & (obs[:, 10] > .5) & (obs[:, 11] < .5)
        request = torch.where(allowed, request, torch.full_like(request, -1.))
        return torch.stack(((speed-6.)/3., (target-20.)/6., request), dim=1)


from nrs_rl.tasks.manager_based.nrs_rl.utils.feed_limits import SignedSpeedLimiter


class LocalReturn:
    def __init__(self, length):
        self.length = float(length)
        self.reset()

    def reset(self):
        self.frontier = self.reverse_mm = self.next_allowed = 0.
        self.turn_target = None
        self.events = 0

    def allowed(self, cursor):
        return (self.turn_target is None and cursor >= max(20., self.next_allowed)
                and cursor >= self.frontier-.1 and cursor < self.length-10.
                and self.reverse_mm+7. < .05*self.length)

    def request(self, cursor, speed, return_action):
        if return_action > 0 and self.allowed(cursor):
            self.turn_target = cursor-5.
            self.next_allowed = self.frontier+20.
            self.events += 1
        # Brake before the turnaround to accommodate the signed speed filter.
        if self.turn_target is not None and cursor <= self.turn_target+1.:
            self.turn_target = None
        return -3. if self.turn_target is not None else speed

    def advance(self, cursor, delta):
        next_cursor = np.clip(cursor+delta, 0., self.length)
        self.reverse_mm += max(0., cursor-next_cursor)
        self.frontier = max(self.frontier, next_cursor)
        return float(next_cursor)


from nrs_rl.tasks.manager_based.nrs_rl.utils.spatial_geometry import DoseGeometry as SpatialGeometry


class DoseGeometry(SpatialGeometry):
    def fast_rollout(self, parameters, *, seed=42, allow_return=True, dt=.08, record=False):
        """Paired randomized force-lag surrogate; never reported as Isaac results."""
        p = np.atleast_2d(parameters)
        n = len(p)
        depth = np.zeros((n, len(self.roi_flat)), dtype=np.float64)
        residence = np.zeros_like(depth)
        cursor = np.zeros(n);frontier = np.zeros(n);reverse = np.zeros(n)
        elapsed = np.zeros(n);velocity = np.zeros(n);slew = np.zeros(n)
        force = np.full(n, 20.);target_force = force.copy()
        force_sum = np.zeros(n);rate_sum = np.zeros(n);rate2 = np.zeros(n);count = np.zeros(n)
        peak = np.zeros(n);travel = np.zeros(n);events = np.zeros(n, dtype=int)
        turn_target = np.full(n, np.nan);next_allowed = np.zeros(n)
        done = np.zeros(n, dtype=bool);history = []
        rng = np.random.default_rng(seed)
        gain, bias = rng.uniform(.97, 1.03), rng.uniform(-.25, .25)
        lag = rng.uniform(.06, .14)
        beta = min(1., 160.*dt/(2.*16.))
        for step in range(math.ceil(1.20*self.length/6./dt)):
            active = ~done
            allowed = (allow_return & np.isnan(turn_target) & (cursor >= np.maximum(20., next_allowed))
                       & (cursor >= frontier-.1) & (cursor < self.length-10.) & (reverse+7. < .05*self.length))
            obs = self.features(depth, residence, cursor, frontier, force, velocity,
                                elapsed, reverse, target_force, allowed)
            actions = action_numpy(obs, p)
            request_force = 20.+6.*actions[:, 1]
            target_force += np.clip(request_force-target_force, -5.*dt, 5.*dt)*active
            force += (1.-math.exp(-dt/lag))*(target_force*gain+bias-force)*active
            start_return = (actions[:, 2] > 0.) & allowed & active
            turn_target[start_return] = cursor[start_return]-5.
            next_allowed[start_return] = frontier[start_return]+20.
            events[start_return] += 1
            turn_target[cursor <= turn_target+1.] = np.nan
            requested_speed = np.where(np.isfinite(turn_target), -3., 6.+3.*actions[:, 0])
            slew += np.clip(requested_speed-slew, -16.*dt, 16.*dt)*active
            velocity += beta*(slew-velocity)*active
            next_cursor = np.clip(cursor+velocity*dt, 0., self.length)
            delta = (next_cursor-cursor)*active
            indices = self.index((cursor+next_cursor)*.5)
            ids, weights = self.indices[indices], self.weights[indices]
            batch = np.arange(n)[:, None]
            np.add.at(depth, (batch, ids), weights*force[:, None]*np.abs(delta[:, None]))
            np.add.at(residence, (batch, ids), (weights > 0)*dt*active[:, None])
            reverse += np.maximum(0., -delta)
            travel += np.abs(delta)
            frontier = np.maximum(frontier, next_cursor)
            rate = force*np.abs(delta)/dt
            force_sum += force*active;rate_sum += rate*active;rate2 += rate**2*active;count += active
            peak = np.maximum(peak, force*active);elapsed += dt*active
            cursor = np.where(active, next_cursor, cursor)
            done |= cursor >= self.length-1e-8
            if record:
                history.append(np.stack((elapsed, cursor, force, velocity, target_force, reverse), axis=1))
            if done.all():
                break
        mean = depth.mean(axis=1)
        cv = depth.std(axis=1)/np.maximum(mean, 1e-12)
        volume = depth.sum(axis=1)*self.surface.area
        rate_mean = rate_sum/np.maximum(count, 1)
        result = []
        for i in range(n):
            result.append({'spatial_cv': float(cv[i]), 'roi_volume_over_k': float(volume[i]),
                'time_s': float(elapsed[i]), 'force_mean_n': float(force_sum[i]/max(count[i], 1)),
                'force_peak_n': float(peak[i]), 'distance_mm': float(travel[i]),
                'reverse_mm': float(reverse[i]), 'return_events': int(events[i]), 'completed': bool(done[i]),
                'zero_fraction': float(np.mean(depth[i] == 0)),
                'rate_cv': float(np.sqrt(max(0., rate2[i]/max(count[i], 1)-rate_mean[i]**2))/max(rate_mean[i], 1e-12))})
        return result, depth, residence, np.asarray(history)



def assess(candidate, baseline):
    gains = {'spatial_cv_gain': 1.-candidate['spatial_cv']/baseline['spatial_cv'],
             'volume_ratio': candidate['roi_volume_over_k']/baseline['roi_volume_over_k'],
             'time_ratio': candidate['time_s']/baseline['time_s'],
             'raw_cv_ratio': candidate['rate_cv']/max(baseline['rate_cv'], 1e-9)}
    violations = {
        'incomplete': 0. if candidate['completed'] else 1.,
        'time': max(0., gains['time_ratio']-1.05),
        'volume_low': max(0., .95-gains['volume_ratio']),
        'volume_high': max(0., gains['volume_ratio']-1.05),
        'mean_force': max(0., candidate['force_mean_n']/21.-1.),
        'peak_force': max(0., candidate['force_peak_n']/32.-1.),
        'distance': max(0., candidate['distance_mm']/baseline['distance_mm']-1.10),
        'uncovered': max(0., candidate['zero_fraction']-baseline['zero_fraction']),
        'fault': float(candidate.get('fault_fraction', 0.) > 0.),
    }
    score = 3.*gains['spatial_cv_gain']-.10*max(0., gains['time_ratio']-1.)
    score -= 30.*sum(violations.values())
    return {'score': score, 'gains': gains, 'passed': not any(v > 1e-8 for v in violations.values()),
            'violations': violations}
