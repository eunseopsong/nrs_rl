"""Experimental nonrotating spatial-dose policy and bounded local retracing.

No ROS dependency. The fast model is a Preston/servo surrogate, not Isaac
physics. Force and feed are continuous actions; dwell is their measured
residence-time consequence. A third action requests one bounded local return.
"""
from pathlib import Path
import importlib.util
import math
import numpy as np
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'source/nrs_rl/nrs_rl').is_dir())
SURFACE_FILE = Path(__file__).with_name('preston_surface.py')
spec = importlib.util.spec_from_file_location('spatial_preston_surface', SURFACE_FILE)
surface_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(surface_module)
PlanarPrestonSurface = surface_module.PlanarPrestonSurface
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


class SignedSpeedLimiter:
    """Signed extension of the existing slew-plus-low-pass limiter."""
    def __init__(self, dt=.008):
        self.dt = dt
        self.beta = min(1., 160.*dt/(2.*16.))
        self.reset()

    def reset(self):
        self.slew = self.speed = self.acceleration = self.jerk = 0.

    def step(self, requested, enabled=True):
        if not enabled or not math.isfinite(requested):
            self.reset()
            return 0.
        self.slew += np.clip(np.clip(requested, -3., 9.)-self.slew, -16.*self.dt, 16.*self.dt)
        previous, acceleration = self.speed, self.acceleration
        self.speed += self.beta*(self.slew-self.speed)
        self.acceleration = (self.speed-previous)/self.dt
        self.jerk = (self.acceleration-acceleration)/self.dt
        return float(self.speed)


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


class DoseGeometry:
    """Fixed ROI and pressure footprints, shared by training and evaluation."""
    def __init__(self, reference=REFERENCE, *, cell_mm=2., path_step_mm=.5, path_length=None):
        import h5py
        with h5py.File(reference) as h:
            positions = np.asarray(h['position'][:, :3], dtype=float)
        arc = np.r_[0., np.linalg.norm(np.diff(positions, axis=0), axis=1).cumsum()]
        if path_length is not None:
            end = min(len(positions), np.searchsorted(arc, path_length)+1)
            positions, arc = positions[:end], arc[:end]
        self.positions, self.length = positions, float(arc[-1])
        self.surface = PlanarPrestonSurface(positions, tool_diameter_mm=30., cell_size_mm=cell_mm)
        self.arc = np.linspace(0., self.length, math.ceil(self.length/path_step_mm)+1)
        self.ds = np.diff(self.arc)
        midpoint = (self.arc[:-1]+self.arc[1:])*.5
        xyz = np.stack([np.interp(midpoint, arc, positions[:, i]) for i in range(3)], axis=1)
        self.roi_flat = np.flatnonzero(self.surface.roi.ravel())
        inverse = np.full(self.surface.roi.size, -1, dtype=int)
        inverse[self.roi_flat] = np.arange(len(self.roi_flat))
        footprints = []
        for point in xyz:
            center = (point-self.surface.origin) @ self.surface.basis.T
            rows, cols, _, _, w, norm = self.surface._footprint(center)
            indices = inverse[rows*self.surface.shape[1]+cols]
            valid = (indices >= 0) & (w > 0)
            footprints.append((indices[valid], w[valid]/norm))
        count = max(len(pair[0]) for pair in footprints)
        self.indices = np.zeros((len(midpoint), count), dtype=int)
        self.weights = np.zeros((len(midpoint), count), dtype=np.float32)
        dense = np.zeros((len(midpoint), len(self.roi_flat)), dtype=np.float32)
        for i, (indices, weights) in enumerate(footprints):
            self.indices[i, :len(indices)] = indices
            self.weights[i, :len(indices)] = weights
            dense[i, indices] = weights*20.*self.ds[i]
        self.nominal = dense.sum(axis=0)
        self.target_depth = float(self.nominal.mean())
        self.remaining = np.maximum(0., self.nominal[None, :]-np.cumsum(dense, axis=0))
        self.nominal_cv = float(self.nominal.std()/self.nominal.mean())
        local_nominal = (self.nominal[self.indices]*self.weights).sum(axis=1)/self.weights.sum(axis=1)
        self.path_mean_deficit = float(np.sum((1.-local_nominal/self.target_depth)*self.ds)/self.length)

    def index(self, cursor):
        return np.clip(np.searchsorted(self.arc, cursor, side='right')-1, 0, len(self.ds)-1)

    def features(self, depth, residence, cursor, frontier, force, velocity,
                 elapsed, reverse, target_force, allowed, tracking=None, shield=None):
        depth = np.atleast_2d(depth)
        n = len(depth)
        b = np.arange(n)[:, None]
        index = self.index(cursor)
        future = self.index(frontier)
        ids, weights = self.indices[index], self.weights[index]
        norm = weights.sum(axis=-1)
        def local_mean(values):
            return (values*weights).sum(axis=-1)/norm
        tail = self.remaining[future[:, None], ids]
        predicted = depth[b, ids]+tail
        deficit = 1.-local_mean(predicted)/self.target_depth
        backidx = self.index(np.maximum(0., np.asarray(cursor)-5.))
        backids, backweights = self.indices[backidx], self.weights[backidx]
        backpred = depth[b, backids]+self.remaining[future[:, None], backids]
        backdeficit = 1.-(backpred*backweights).sum(axis=-1)/backweights.sum(axis=-1)/self.target_depth
        obs = np.zeros((n, len(OBSERVATIONS)), dtype=np.float32)
        obs[:, 0] = np.asarray(cursor)/self.length
        obs[:, 1] = np.asarray(force)/20.
        obs[:, 2] = np.asarray(velocity)/6.
        obs[:, 3] = 1.-local_mean(self.nominal[ids])/self.target_depth-self.path_mean_deficit
        obs[:, 4] = deficit-self.path_mean_deficit
        obs[:, 5] = backdeficit-deficit
        obs[:, 6] = local_mean(residence[b, ids])/5.
        obs[:, 7] = 0. if tracking is None else np.asarray(tracking)/10.
        obs[:, 8] = np.asarray(elapsed)/(self.length/6.)
        obs[:, 9] = np.asarray(reverse)/(.05*self.length)
        obs[:, 10] = np.asarray(force) >= 1.5
        obs[:, 11] = 0. if shield is None else np.asarray(shield)
        obs[:, 12] = np.asarray(target_force)/20.
        obs[:, 13] = np.asarray(force)*np.abs(velocity)/120.-1.
        obs[:, 14] = allowed
        obs[:, 15] = local_mean(tail)/self.target_depth
        return np.clip(obs, -5., 5.)

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
