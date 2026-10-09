#!/usr/bin/env python3
"""Reintegrate the October 5 ROS runs with spindle rotation excluded.

The saved Force-phase recording is authoritative for the measurement interval.
Raw controller rows are used only to diagnose clocks and command tracking.
No stationary, entry-transient or uncovered-ROI samples are removed.
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
import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

ROOT = _REPO_ROOT
TASK = ROOT / 'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl'
spec = importlib.util.spec_from_file_location('surface', TASK / 'utils/preston_surface.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Surface = module.PlanarPrestonSurface
RUNS = {
    'mode5': Path('/home/eunseop/evaluation/rl_best/d14_20261005_185437'),
    'mode6': Path('/home/eunseop/evaluation/rl_no_speed_modulation/d14_20261005_191400'),
}


def analyze(output):
    output.mkdir(parents=True, exist_ok=False)
    reference = TASK / 'datasets/cmd_continue9D_flat.h5'
    with h5py.File(reference) as stream:
        path = stream['position'][:, :3]
    surface = Surface(path, cell_size_mm=.5, velocity_model='tcp')
    tree = cKDTree((path-surface.origin) @ surface.basis.T)
    report = {'spindle_rpm': None, 'rotation_contribution': False,
              'physical_depth_validated': False, 'model': surface.metadata,
              'reference_path_sha256': hashlib.sha256(reference.read_bytes()).hexdigest(),
              'caveats': ['One run per mode; unequal initial home poses.',
                          'ROS recording is sampled and held; interval speed includes duplicate poses.',
                          'Uncalibrated F*distance proxy, not measured material volume.'], 'runs': {}}
    maps, signals = {}, {}
    for name, run in RUNS.items():
        source = next(run.glob('removal/session_*/recording.csv'))
        rows = np.genfromtxt(source, delimiter=',', names=True)
        t = rows['time_s']; xyz = np.column_stack([rows[k] for k in ('x_mm','y_mm','z_mm')])
        dt = np.diff(t, prepend=t[0]-.008)
        if (dt <= 0).any() or (dt > .1).any():
            raise ValueError('Nonmonotonic or missing measurement samples')
        dx = np.diff(xyz, axis=0, prepend=xyz[:1])
        dx -= (dx @ surface.normal)[:, None]*surface.normal
        velocity = dx/dt[:, None]
        speed = np.linalg.norm(velocity, axis=1)
        force = abs(rows['fz_N']); rate = force*speed*(force >= 1.5)
        surface.reset()
        for i in range(len(t)):
            surface.deposit(xyz[i]-.5*dx[i], velocity[i], force[i], dt[i], contact=force[i]>=1.5)
        metrics = surface.metrics()
        surface.save(output / f'{name}_tcp_surface.npz')
        maps[name] = surface.depth.copy()
        distance, _ = tree.query((xyz-surface.origin) @ surface.basis.T)
        raw = np.genfromtxt(run/'raw.csv', delimiter=',', names=True)
        active = (raw['active']>0) & (raw['done']==0)
        bounded = raw[active]
        raw_dt = np.diff(bounded['time'])
        clock_span = float(bounded['time'][-1]-bounded['time'][0])
        command_error = np.sqrt(sum((bounded[f'actual_tcp_{a}']-bounded[f'reference_{a}'])**2 for a in ('x','y')))
        metrics.update(recording=str(source), recording_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            samples=len(t), force_mean_n=float(force.mean()), force_peak_n=float(force.max()),
            force_rmse_n=float(np.sqrt(np.mean((force-20.)**2))),
            force_above_32n_fraction=float(np.mean(force>32.)),
            raw_tcp_rate_cv=float(rate.std()/rate.mean()), raw_tcp_mean_rate=float(rate.mean()),
            measured_tangent_distance_mm=float(np.linalg.norm(dx, axis=1).sum()),
            speed_mean_mm_s=float(speed.mean()), speed_p99_mm_s=float(np.quantile(speed,.99)),
            zero_displacement_fraction=float(np.mean(speed<1e-8)),
            path_cross_track_rmse_mm=float(np.sqrt(np.mean(distance**2))),
            path_cross_track_p95_mm=float(np.quantile(distance,.95)),
            endpoint_tangent_error_mm=float(np.linalg.norm((xyz[-1]-path[-1]) @ surface.basis.T)),
            controller_samples=int(active.sum()), controller_nominal_duration_s=int(active.sum())*.008,
            controller_clock_span_s=clock_span,
            controller_repeated_clock_fraction=float(np.mean(raw_dt==0)),
            controller_clock_backwards_count=int(np.sum(raw_dt<0)),
            controller_clock_to_fixed_step_ratio=clock_span/(int(active.sum())*.008),
            reference_tangent_tracking_rmse_mm=float(np.sqrt(np.mean(command_error**2))))
        report['runs'][name] = metrics
        signals[name] = (t-t[0],force,speed,distance)
    a,b = report['runs']['mode5'],report['runs']['mode6']
    report['mode5_vs_mode6'] = {
        'spatial_cv_improvement_percent': 100*(1-a['spatial_depth_cv']/b['spatial_depth_cv']),
        'roi_volume_change_percent': 100*(a['roi_volume_over_k']/b['roi_volume_over_k']-1),
        'raw_tcp_rate_cv_improvement_percent': 100*(1-a['raw_tcp_rate_cv']/b['raw_tcp_rate_cv']),
    }
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(12,8),layout='constrained')
    maximum=max(float(x.max()) for x in maps.values())
    for col,name in enumerate(('mode6','mode5')):
        z=np.where(surface.roi,maps[name],np.nan)
        view=axes[0,col].imshow(z,origin='lower',vmin=0,vmax=maximum,extent=[surface.u[0],surface.u[-1],surface.v[0],surface.v[-1]])
        m=report['runs'][name]
        axes[0,col].set_title(f"{name}: TCP-only depth CV {m['spatial_depth_cv']:.4f}")
        axes[0,col].set_xlabel('u [mm]'); axes[0,col].set_ylabel('v [mm]')
        t,f,v,e=signals[name]
        axes[1,0].plot(t,f,label=name,alpha=.8,lw=.7)
        axes[1,1].plot(t,e,label=name,alpha=.8,lw=.7)
    fig.colorbar(view,ax=axes[0,:],label='Depth / K [uncalibrated, RPM excluded]')
    axes[1,0].set(xlabel='Simulation time [s]',ylabel='Normal force [N]')
    axes[1,1].set(xlabel='Simulation time [s]',ylabel='Distance to reference path [mm]')
    for ax in axes[1,:]: ax.legend(); ax.grid(alpha=.2)
    fig.savefig(output/'diagnosis.png',dpi=170); plt.close(fig)
    print(json.dumps({'comparison':report['mode5_vs_mode6'],'runs':report['runs']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    analyze(parser.parse_args().output)
