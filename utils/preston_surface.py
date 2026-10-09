"""Planar, equal-area Preston depth integration from measured motion.

dh(x)/K = p(x) * |v_tool(x) - v_surface(x)| * dt.

``tcp`` retains the experiment's F * measured TCP sliding-speed model exactly:
integrating the full pressure footprint gives dV/K = F*v*dt. ``rotating_disk``
adds the local spindle velocity and requires a supplied RPM. Neither model is
a calibrated depth prediction. The contact diameter/pressure shape are explicit
assumptions, not inferred from the nominal tool diameter or supply voltage.

All comparisons use one reference-path-derived ROI, including uncovered cells.
There is no visited-cell selection, signal smoothing, or per-rollout ROI fit.
"""

from __future__ import annotations

import math
import numpy as np


class PlanarPrestonSurface:
    def __init__(self, reference_xyz_mm, normal=(0., 0., 1.), *,
                 tool_diameter_mm=30., contact_diameter_mm=None, cell_size_mm=2.,
                 pressure_profile="uniform", velocity_model="tcp", spindle_rpm=None):
        self.reference = np.asarray(reference_xyz_mm, dtype=np.float64)
        self.normal = np.asarray(normal, dtype=np.float64)
        if (self.reference.ndim != 2 or self.reference.shape[1] != 3
                or len(self.reference) < 2 or not np.isfinite(self.reference).all()
                or self.normal.shape != (3,) or not np.isfinite(self.normal).all()
                or np.linalg.norm(self.normal) < 1e-9):
            raise ValueError("Finite reference XYZ points and a nonzero normal are required")
        self.normal /= np.linalg.norm(self.normal)
        if contact_diameter_mm is None:
            contact_diameter_mm = tool_diameter_mm
        if (not all(math.isfinite(v) and v > 0 for v in
                    (tool_diameter_mm, contact_diameter_mm, cell_size_mm))
                or contact_diameter_mm > tool_diameter_mm
                or cell_size_mm > contact_diameter_mm / 4):
            raise ValueError("Require 0 < contact diameter <= tool diameter and >=4 cells across contact")
        if pressure_profile not in ("uniform", "hertz"):
            raise ValueError("pressure_profile must be uniform or hertz")
        if velocity_model not in ("tcp", "rotating_disk"):
            raise ValueError("velocity_model must be tcp or rotating_disk")
        if spindle_rpm is not None and (not math.isfinite(spindle_rpm) or spindle_rpm < 0):
            raise ValueError("spindle_rpm must be finite and nonnegative")
        if velocity_model == "rotating_disk" and spindle_rpm is None:
            raise ValueError("rotating_disk requires measured/supplied RPM; voltage is not RPM")
        self.radius = contact_diameter_mm / 2
        self.cell = float(cell_size_mm)
        self.area = self.cell**2
        self.profile = pressure_profile
        self.model = velocity_model
        self.omega = 0. if velocity_model == "tcp" else spindle_rpm * 2 * math.pi / 60
        self.origin = self.reference[0].copy()
        axis = np.eye(3)[np.argmin(np.abs(self.normal))]
        axis -= (axis @ self.normal) * self.normal
        axis /= np.linalg.norm(axis)
        self.basis = np.stack((axis, np.cross(self.normal, axis)))
        relative = self.reference - self.origin
        if np.max(np.abs(relative @ self.normal)) > 0.1:
            raise ValueError("Planar Preston integration requires a planar path (0.1 mm tolerance)")
        path = relative @ self.basis.T
        pad = self.radius + 2 * self.cell
        low = np.floor((path.min(axis=0) - pad) / self.cell) * self.cell
        high = np.ceil((path.max(axis=0) + pad) / self.cell) * self.cell
        self.u = np.arange(low[0], high[0] + self.cell / 2, self.cell)
        self.v = np.arange(low[1], high[1] + self.cell / 2, self.cell)
        self.shape = (len(self.v), len(self.u))
        self.roi = np.zeros(self.shape, dtype=bool)
        # Sweep a fixed footprint along the reference before seeing any run.
        # Resampling bounds discretization even for sparse trajectory files.
        for start, end in zip(path[:-1], path[1:]):
            count = max(1, math.ceil(np.linalg.norm(end - start) / (self.cell / 2)))
            for fraction in np.arange(count) / count:
                rows, cols, _, _, weights, _ = self._footprint(start + fraction * (end - start))
                self.roi[rows, cols] |= weights > 0
        rows, cols, _, _, weights, _ = self._footprint(path[-1])
        self.roi[rows, cols] |= weights > 0
        self.metadata = {
            "law": "dh/K = p * relative_sliding_speed * dt",
            "velocity_model": velocity_model, "tool_diameter_mm": tool_diameter_mm,
            "contact_diameter_mm": contact_diameter_mm, "pressure_profile": pressure_profile,
            "contact_assumption": "fixed circular footprint; full face unless contact diameter supplied",
            "spindle_rpm": spindle_rpm, "cell_size_mm": self.cell,
            "depth_units": "h/K (uncalibrated)", "volume_units": "V/K (uncalibrated)",
            "roi": "reference swept contact footprint, all equal-area cells including zeros",
            "plane_origin_mm": self.origin.tolist(), "plane_basis": self.basis.tolist(),
            "plane_normal": self.normal.tolist(),
        }
        self.reset()

    def _footprint(self, center):
        # Use the complete stencil for pressure normalization, even when a
        # measured TCP leaves the grid. Never redistribute off-grid removal.
        c0 = math.ceil((center[0] - self.radius - self.u[0]) / self.cell)
        c1 = math.floor((center[0] + self.radius - self.u[0]) / self.cell)
        r0 = math.ceil((center[1] - self.radius - self.v[0]) / self.cell)
        r1 = math.floor((center[1] + self.radius - self.v[0]) / self.cell)
        rows, cols = np.meshgrid(np.arange(r0, r1 + 1), np.arange(c0, c1 + 1), indexing="ij")
        dx = self.u[0] + cols * self.cell - center[0]
        dy = self.v[0] + rows * self.cell - center[1]
        radius2 = (dx**2 + dy**2) / self.radius**2
        weights = (radius2 <= 1).astype(float)
        if self.profile == "hertz":
            weights *= np.sqrt(np.maximum(0., 1 - radius2))
        normalizer = float(weights.sum()) * self.area
        valid = (rows >= 0) & (rows < self.shape[0]) & (cols >= 0) & (cols < self.shape[1])
        return rows[valid], cols[valid], dx[valid], dy[valid], weights[valid], normalizer

    def reset(self):
        self.depth = np.zeros(self.shape, dtype=np.float64)
        self.integrated_volume = 0.
        self.full_footprint_volume = 0.
        self.processing_time = 0.

    def deposit(self, xyz_mm, tangent_velocity_mm_s, force_n, dt_s, *, contact=True):
        position = np.asarray(xyz_mm, dtype=np.float64)
        velocity = np.asarray(tangent_velocity_mm_s, dtype=np.float64)
        if (position.shape != (3,) or velocity.shape != (3,)
                or not np.isfinite(position).all() or not np.isfinite(velocity).all()
                or not math.isfinite(force_n) or force_n < 0
                or not math.isfinite(dt_s) or dt_s <= 0):
            raise ValueError("Finite position, velocity, nonnegative force and positive dt required")
        self.processing_time += dt_s
        if not contact or force_n == 0:
            return
        center = (position - self.origin) @ self.basis.T
        feed = velocity @ self.basis.T
        rows, cols, dx, dy, weights, normalizer = self._footprint(center)
        relative_speed = np.hypot(feed[0] - self.omega * dy, feed[1] + self.omega * dx)
        increment = force_n * weights / normalizer * relative_speed * dt_s
        self.depth[rows, cols] += increment
        self.integrated_volume += float(increment.sum()) * self.area
        if self.model == "tcp":
            self.full_footprint_volume += force_n * np.linalg.norm(feed) * dt_s

    def metrics(self):
        depth = self.depth[self.roi]
        mean = float(depth.mean())
        volume = depth * self.area
        cv = float(depth.std() / mean) if mean > 0 else None
        return {
            "spatial_depth_cv": cv, "equal_cell_volume_cv": cv,
            "mean_depth_over_k": mean,
            "std_depth_over_k": float(depth.std()),
            "roi_volume_over_k": float(volume.sum()),
            "grid_volume_over_k": float(self.depth.sum()) * self.area,
            "outside_roi_volume_over_k": float(self.depth[~self.roi].sum()) * self.area,
            "roi_cells": int(self.roi.sum()), "roi_area_mm2": int(self.roi.sum()) * self.area,
            "zero_depth_fraction": float(np.mean(depth == 0)),
            "processing_time_s": self.processing_time,
            "tcp_full_footprint_volume_over_k": self.full_footprint_volume if self.model == "tcp" else None,
        }

    def integrate_trace(self, trace):
        """Use raw 125 Hz observations; include all active zero-removal ticks.

        The logged speed is authoritative. Finite differences provide direction
        only, so the volume integral retains the action term's speed convention.
        A midpoint footprint reduces discretization along moving samples.
        """
        self.reset()
        time = np.asarray(trace["time_s"], dtype=float)
        xyz = np.asarray(trace["tcp_pose"], dtype=float)[:, :3]
        speed = np.asarray(trace["measured_speed_mm_s"], dtype=float)
        force = np.asarray(trace["normal_force_n"], dtype=float)
        active = np.asarray(trace["polishing_active"]) > 0
        fault = np.asarray(trace.get("safety_fault_reason", np.zeros(len(time)))) != 0
        if len(time) < 2 or not np.allclose(np.diff(time), .008, rtol=1e-4, atol=1e-7):
            raise ValueError("Expected an unbroken 125 Hz control trace")
        displacement = np.diff(xyz, axis=0, prepend=xyz[:1])
        tangent = displacement - (displacement @ self.normal)[:, None] * self.normal
        length = np.linalg.norm(tangent, axis=1)
        velocity = tangent / np.maximum(length[:, None], 1e-12) * speed[:, None]
        if np.any(active & (speed > 1e-6) & (length < 1e-12)):
            raise ValueError("A moving active sample has no recoverable tangent direction")
        midpoint = xyz - .5 * displacement
        # Original removal=0 also denotes acquisition/dropout/fault. For an
        # explicit spinning tool, stationary contact still removes material.
        contact = (force >= 1.5) & ~fault
        for i in np.flatnonzero(active):
            self.deposit(midpoint[i], velocity[i], force[i], .008, contact=bool(contact[i]))
        return self.metrics()

    def save(self, path):
        np.savez_compressed(path, depth_over_k=self.depth, roi=self.roi,
                            u_mm=self.u, v_mm=self.v, cell_area_mm2=self.area,
                            origin_mm=self.origin, basis=self.basis)
