"""A finer reward surface without changing the deployed 2 mm observation schema."""
import numpy as np
import torch

from ..utils.removal_metrics import processing_mask
from ..utils.rotary_geometry import FixedForceGeometry


class RemovalRewardSurface:
    """Reference-path removal only; independent Isaac validation is still required.

    Uses the same 0.5 mm path-footprint table and one deposition per 80 ms as
    the original surrogate, with a configurable surface cell size. Forces and
    motion come from the existing environment, not a second plant model.
    """
    def __init__(self, count, device, cell_mm):
        if cell_mm not in (.5, 1.):
            raise ValueError('Fine reward cells must be 0.5 or 1 mm')
        self.device = torch.device(device)
        self.geometry = g = FixedForceGeometry(cell_mm=cell_mm)
        self.area, self.length = g.surface.area, g.length
        to = lambda x, dtype=torch.float32: torch.as_tensor(x, dtype=dtype, device=self.device)
        self.arc = to(g.arc)
        self.indices = to(g.indices, torch.long)
        self.weights = to(g.weights)
        self.remaining = to(g.remaining)
        self.mask = to(processing_mask(g.surface, g.positions).ravel()[g.roi_flat], torch.bool)
        nominal = to(g.nominal)
        self.baseline_processing_volume = nominal[self.mask].sum() * self.area
        self.baseline_full_volume = nominal.sum() * self.area
        reference_arc = np.r_[0., np.linalg.norm(np.diff(g.positions, axis=0), axis=1).cumsum()]
        positions = np.stack([np.interp(g.arc, reference_arc, g.positions[:, i]) for i in range(3)], axis=1)
        midpoints = .5 * (positions[1:] + positions[:-1])
        centers = (midpoints - g.surface.origin) @ g.surface.basis.T
        rows, cols = np.unravel_index(g.roi_flat[g.indices], g.surface.shape)
        self.dx = to(g.surface.u[cols] - centers[:, 0:1])
        self.dy = to(g.surface.v[rows] - centers[:, 1:2])
        tangent = np.diff(positions, axis=0)
        direction = tangent / np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1.e-12)
        self.direction = to(direction @ g.surface.basis.T)
        self.omega = g.surface.omega
        self.depth = torch.zeros((count, len(g.roi_flat)), device=self.device)
        self.processing = torch.zeros_like(self.depth)

    def reset(self, ids):
        self.depth[ids] = 0.
        self.processing[ids] = 0.

    def index(self, cursor):
        return torch.searchsorted(self.arc, cursor.contiguous(), right=True).sub(1).clamp(0, len(self.arc)-2)

    def deposit(self, previous, cursor, force, elapsed):
        index = self.index(.5 * (cursor + previous))
        velocity = self.direction[index] * (cursor - previous)[:, None] / .08
        relative = torch.sqrt((velocity[:, :1] - self.omega * self.dy[index]).square()
                              + (velocity[:, 1:2] + self.omega * self.dx[index]).square())
        values = force[:, None] * self.weights[index] * relative * .08
        self.depth.scatter_add_(1, self.indices[index], values)
        self.processing.scatter_add_(1, self.indices[index], values * (elapsed >= 2.-1.e-5)[:, None])

    def predicted_cv2(self, cursor):
        remaining = self.remaining[self.index(cursor)]
        remaining = torch.where((cursor >= self.length)[:, None], torch.zeros_like(remaining), remaining)
        values = (self.processing + remaining)[:, self.mask]
        return values.square().mean(1) / values.mean(1).clamp_min(1.e-6).square() - 1.

    def metrics(self):
        values = self.processing[:, self.mask]
        cv = values.std(1, unbiased=False) / values.mean(1).clamp_min(1.e-6)
        volume_ratio = values.sum(1) * self.area / self.baseline_processing_volume
        full_ratio = self.depth.sum(1) * self.area / self.baseline_full_volume
        return cv, volume_ratio, full_ratio
