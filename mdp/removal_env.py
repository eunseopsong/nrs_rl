"""Approximate rotary-removal environment; actual Isaac validation is separate."""
import math
import numpy as np
import torch
from ..utils.rotary_geometry import FixedForceGeometry
from ..utils.removal_metrics import processing_mask
from .removal_action import advance_feed
from .removal_observation import removal_observation
from .removal_rewards import predicted_cv2, step_reward, terminal_penalty
from .removal_terminations import episode_ends

class RemovalEnv:
    """Vectorized 80 ms actions / 8 ms feed limiter with planar rotary removal.

    Features reproduce the existing 16D schema on a 2 mm grid. The training
    deposition uses a .5 mm reference-footprint table, one midpoint per 80 ms
    decision. Final Isaac measurements use actual TCP motion at 8 ms / .5 mm.
    """
    def __init__(self, count=64, device='cpu', seed=3101, randomize=True):
        self.count, self.device, self.randomize = count, torch.device(device), randomize
        self.rng = torch.Generator(device=self.device).manual_seed(seed)
        geometry = FixedForceGeometry()
        self.geometry = geometry
        self.length, self.cells = geometry.length, len(geometry.roi_flat)
        to = lambda x, dtype=torch.float32: torch.as_tensor(x, dtype=dtype, device=self.device)
        self.arc = to(geometry.arc)
        self.indices = to(geometry.indices, torch.long)
        self.weights = to(geometry.weights)
        self.nominal = to(geometry.nominal)
        self.remaining = to(geometry.remaining)
        self.target = geometry.target_depth
        self.path_deficit = geometry.path_mean_deficit
        self.mask = to(processing_mask(geometry.surface, geometry.positions).ravel()[geometry.roi_flat], torch.bool)
        self.baseline_processing_volume = self.nominal[self.mask].sum()*4.
        self.baseline_full_volume = self.nominal.sum()*4.
        self.batch = torch.arange(count, device=self.device)[:, None]
        reference_arc = np.r_[0., np.linalg.norm(np.diff(geometry.positions, axis=0), axis=1).cumsum()]
        positions = np.stack([np.interp(geometry.arc, reference_arc, geometry.positions[:, i]) for i in range(3)], axis=1)
        self.positions = to(positions)
        midpoints = .5*(positions[1:]+positions[:-1])
        cell_r, cell_c = np.unravel_index(geometry.roi_flat[geometry.indices], geometry.surface.shape)
        self.dx = to(geometry.surface.u[cell_c] - ((midpoints-geometry.surface.origin) @ geometry.surface.basis.T)[:, 0:1])
        self.dy = to(geometry.surface.v[cell_r] - ((midpoints-geometry.surface.origin) @ geometry.surface.basis.T)[:, 1:2])
        self.direction = to(np.diff(positions, axis=0)[:, :2]/np.maximum(np.linalg.norm(np.diff(positions, axis=0)[:, :2], axis=1, keepdims=True), 1.e-12))
        self.omega = 1000.*2.*math.pi/60.
        self.depth = torch.zeros((count, self.cells), device=self.device)
        self.processing = torch.zeros_like(self.depth)
        self.residence = torch.zeros_like(self.depth)
        self.history = torch.zeros((count, 4), device=self.device)
        for name in ('cursor', 'speed', 'slew', 'filtered', 'elapsed', 'previous_action', 'force', 'potential', 'episode_reward'):
            setattr(self, name, torch.zeros(count, device=self.device))
        self.force_scale = torch.ones(count, device=self.device)
        self.force_bias = torch.zeros(count, device=self.device)
        self.delay = torch.zeros(count, device=self.device, dtype=torch.long)
        self.reset(torch.arange(count, device=self.device))

    def reset(self, ids):
        for name in ('depth', 'processing', 'residence', 'history', 'cursor', 'speed', 'slew', 'filtered', 'elapsed', 'previous_action', 'episode_reward'):
            getattr(self, name)[ids] = 0.
        if self.randomize:
            self.force_scale[ids] = .9+.2*torch.rand(len(ids), generator=self.rng, device=self.device)
            self.force_bias[ids] = 1.5*torch.rand(len(ids), generator=self.rng, device=self.device)-.75
            self.delay[ids] = torch.randint(0, 4, (len(ids),), generator=self.rng, device=self.device)
        else:
            self.force_scale[ids] = 1.; self.force_bias[ids] = 0.; self.delay[ids] = 0
        self.force[ids] = 20.*self.force_scale[ids]+self.force_bias[ids]
        self.potential[ids] = self.predicted_cv2()[ids]

    def index(self, cursor):
        return torch.searchsorted(self.arc, cursor.contiguous(), right=True).sub(1).clamp(0, len(self.arc)-2)

    def predicted_cv2(self):
        return predicted_cv2(self)
    def observation(self):
        return removal_observation(self)
    def step(self, action, auto_reset=True):
        action, previous = advance_feed(self, action)
        index = self.index(.5*(self.cursor+previous))
        ids, weights = self.indices[index], self.weights[index]
        velocity = self.direction[index]*(self.cursor-previous)[:, None]/.08
        relative = torch.sqrt((velocity[:, :1]-self.omega*self.dy[index]).square()
                              +(velocity[:, 1:2]+self.omega*self.dx[index]).square())
        values = self.force[:, None]*weights*relative*.08
        self.depth.scatter_add_(1, ids, values)
        self.processing.scatter_add_(1, ids, values*(self.elapsed >= 2.-1.e-5)[:, None])
        self.residence.scatter_add_(1, ids, (weights > 0).float()*.08)
        self.elapsed += .08
        potential = self.predicted_cv2()
        reward = step_reward(self.potential, potential, action, self.previous_action)
        self.previous_action = action.clone()
        self.potential = potential
        completed, done = episode_ends(self.cursor, self.length, self.elapsed)
        pvalues = self.processing[:, self.mask]
        cv = pvalues.std(1, unbiased=False)/pvalues.mean(1).clamp_min(1.e-6)
        volume_ratio = pvalues.sum(1)*4./self.baseline_processing_volume
        full_ratio = self.depth.sum(1)*4./self.baseline_full_volume
        reward -= terminal_penalty(done, completed, volume_ratio, full_ratio)
        self.episode_reward += reward
        info = {'done': done.clone(), 'completed': completed.clone(), 'cv': cv.clone(),
                'volume_ratio': volume_ratio.clone(), 'full_ratio': full_ratio.clone(),
                'seconds': self.elapsed.clone(), 'episode_reward': self.episode_reward.clone()}
        if auto_reset:
            ids_done = torch.nonzero(done).flatten()
            if len(ids_done):
                self.reset(ids_done)
        return self.observation(), reward, done, info

