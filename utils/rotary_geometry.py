"""Virtual rotary removal and its normalized observation geometry."""
import numpy as np
from .spatial_geometry import DoseGeometry, OBSERVATIONS, PlanarPrestonSurface
from ..paths import REFERENCE

SPINDLE_RPM = 1000.

def rotating_surface(reference, cell_mm=.5, rpm=SPINDLE_RPM):
    return PlanarPrestonSurface(reference, cell_size_mm=cell_mm,
                               velocity_model='rotating_disk', spindle_rpm=rpm)


class FixedForceGeometry(DoseGeometry):
    """Same spatial observations, normalized to the rotating 20 N/6 mm/s baseline."""
    def __init__(self, reference=REFERENCE, *, cell_mm=2.):
        super().__init__(reference, cell_mm=cell_mm)
        self.surface = rotating_surface(self.positions, cell_mm)
        arc = np.r_[0., np.linalg.norm(np.diff(self.positions, axis=0), axis=1).cumsum()]
        xyz = np.stack([np.interp(self.arc, arc, self.positions[:, j]) for j in range(3)], axis=1)
        dense = np.zeros((len(self.ds), len(self.roi_flat)), dtype=np.float32)
        inverse = np.full(self.surface.roi.size, -1, dtype=int)
        inverse[self.roi_flat] = np.arange(len(self.roi_flat))
        for i, midpoint in enumerate((xyz[1:] + xyz[:-1])*.5):
            direction = xyz[i+1]-xyz[i]
            velocity = 6.*direction/max(np.linalg.norm(direction), 1.e-12)
            center = (midpoint-self.surface.origin) @ self.surface.basis.T
            rows, cols, dx, dy, w, norm = self.surface._footprint(center)
            ids = inverse[rows*self.surface.shape[1]+cols]
            feed = velocity @ self.surface.basis.T
            relative = np.hypot(feed[0]-self.surface.omega*dy, feed[1]+self.surface.omega*dx)
            keep = (ids >= 0) & (w > 0)
            dense[i, ids[keep]] = 20.*w[keep]/norm*relative[keep]*self.ds[i]/6.
        self.nominal = dense.sum(axis=0)
        self.target_depth = float(self.nominal.mean())
        self.remaining = np.maximum(0., self.nominal[None, :]-np.cumsum(dense, axis=0))
        self.nominal_cv = float(self.nominal.std()/self.nominal.mean())
        local = (self.nominal[self.indices]*self.weights).sum(axis=1)/self.weights.sum(axis=1)
        self.path_mean_deficit = float(np.sum((1.-local/self.target_depth)*self.ds)/self.length)

