#!/usr/bin/env python3
"""Conditional Preston-depth comparison without inventing a spindle RPM.

Assume a constant spindle rate whose local sliding contribution dominates TCP
translation: dh/(K*omega) = p*r*dt. Feed still changes measured dwell time.
This is a model assumption, not a calibrated prediction or measured RPM.
"""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT as _REPO_ROOT
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = _REPO_ROOT
TASK = ROOT / 'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl'
spec = importlib.util.spec_from_file_location('preston_surface_base', TASK / 'utils/preston_surface.py')
preston = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preston)


class RotationDominatedSurface(preston.PlanarPrestonSurface):
    def __init__(self, reference_xyz_mm, normal=(0., 0., 1.), **kwargs):
        if 'spindle_rpm' in kwargs or 'velocity_model' in kwargs:
            raise ValueError('This normalized limit does not take a spindle RPM')
        super().__init__(reference_xyz_mm, normal, **kwargs)
        # Unit angular rate implements normalization by omega; it is not a
        # chosen operating speed. Translational sliding is omitted below.
        self.omega = 1.
        self.model = 'rotation_dominated_normalized'
        self.metadata.update(
            law='dh/(K*omega) = p * radius_from_spindle_axis * dt',
            velocity_model=self.model, spindle_rpm=None,
            assumption='constant spindle angular speed; rotation dominates relative sliding',
            depth_units='h/(K*omega), uncalibrated', volume_units='V/(K*omega), uncalibrated',
            physical_depth_validated=False,
        )

    def deposit(self, xyz_mm, tangent_velocity_mm_s, force_n, dt_s, *, contact=True):
        velocity = np.asarray(tangent_velocity_mm_s)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise ValueError('Finite 3D tangent velocity required')
        super().deposit(xyz_mm, np.zeros(3), force_n, dt_s, contact=contact)

    def metrics(self):
        original = super().metrics()
        return {key.replace('_over_k', '_over_k_omega'): value for key, value in original.items()}

    def save(self, path):
        np.savez_compressed(path, depth_over_k_omega=self.depth, roi=self.roi,
                            u_mm=self.u, v_mm=self.v, cell_area_mm2=self.area,
                            origin_mm=self.origin, basis=self.basis)


def analyze(evaluation, output):
    import h5py

    source = json.loads((evaluation / 'summary.json').read_text())
    with h5py.File(TASK / 'datasets/cmd_continue9D_flat.h5') as stream:
        path = stream['position'][:, :3]
    arc = np.r_[0., np.linalg.norm(np.diff(path, axis=0), axis=1).cumsum()]
    end = int(np.argmin(np.abs(arc-source['path_length_mm']))) + 1
    if abs(arc[end-1]-source['path_length_mm']) > .1:
        raise ValueError('Recorded path length does not match the reference')
    normal = source['preston_surface_model']['plane_normal']
    contact_diameter = source['preston_surface_model']['contact_diameter_mm']
    surface = RotationDominatedSurface(path[:end], normal,
        tool_diameter_mm=source['physics_tool_diameter_mm'], contact_diameter_mm=contact_diameter,
        cell_size_mm=source['preston_surface_model']['cell_size_mm'])
    output.mkdir(parents=True, exist_ok=True)
    report = {'source': str(evaluation.resolve()), 'model': surface.metadata, 'candidates': {}}
    names = source['candidates'] if 'candidates' in source else ('adaptive', 'constant')
    maps = {}
    for name in names:
        with np.load(evaluation / f'{name}_trace.npz') as trace:
            report['candidates'][name] = surface.integrate_trace(trace)
        surface.save(output / f'{name}_surface.npz')
        maps[name] = surface.depth.copy()
    baseline_name = 'constant_a' if 'constant_a' in names else 'constant'
    baseline = report['candidates'][baseline_name]
    for name, metrics in report['candidates'].items():
        metrics['comparison_to_constant'] = {
            'spatial_cv_improvement_fraction': 1-metrics['spatial_depth_cv']/baseline['spatial_depth_cv'],
            'mean_depth_ratio': metrics['mean_depth_over_k_omega']/baseline['mean_depth_over_k_omega'],
            'uncovered_area_not_increased': metrics['zero_depth_fraction'] <= baseline['zero_depth_fraction'],
        }
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(maps)
    columns = min(4, len(names))
    fig, axes = plt.subplots(int(np.ceil(len(names)/columns)), columns,
                             figsize=(5*columns, 4.5*int(np.ceil(len(names)/columns))),
                             squeeze=False, layout='constrained')
    maximum = max(float(depth[surface.roi].max()) for depth in maps.values())
    for ax, name in zip(axes.flat, names):
        im = ax.imshow(np.where(surface.roi, maps[name], np.nan), origin='lower',
            extent=(surface.u[0], surface.u[-1], surface.v[0], surface.v[-1]),
            vmin=0, vmax=maximum, cmap='viridis')
        ax.set_title(f"{name}: depth CV {report['candidates'][name]['spatial_depth_cv']:.4f}")
        ax.set_xlabel('Plane u [mm]')
        ax.set_ylabel('Plane v [mm]')
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    fig.colorbar(im, ax=list(axes.flat), label='h/(K omega), assumed rotation-dominated Preston model')
    fig.savefig(output / 'comparison.png', dpi=160)
    plt.close(fig)
    print(json.dumps({name: {'cv': metrics['spatial_depth_cv'], **metrics['comparison_to_constant']}
                      for name, metrics in report['candidates'].items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluation', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    analyze(args.evaluation, args.output or args.evaluation / 'rotation_dominated_depth')
