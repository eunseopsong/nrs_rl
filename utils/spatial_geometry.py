"""Shared reference-path geometry and 16D spatial observations; no policy optimizer."""
import math
import numpy as np
from ..paths import ROOT, REFERENCE
from .preston_surface import PlanarPrestonSurface

OBSERVATIONS = ['progress', 'force_over_20', 'speed_over_6',
                'centered_nominal_depth_deficit', 'centered_predicted_final_deficit',
                'behind_minus_current_deficit', 'local_residence_over_5s',
                'tracking_error_over_10mm', 'elapsed_over_nominal',
                'reverse_budget_fraction', 'contact', 'shield',
                'target_force_over_20', 'rate_error', 'return_allowed',
                'remaining_nominal_depth_over_target']


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

