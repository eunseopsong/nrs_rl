"""Fixed-ROI removal from one atomic controller snapshot per 8 ms tick."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('paired_preston_surface',
    ROOT/'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/model_based/runtime/fixed_force_12p59_source/nrs_fixed_force_12p59/preston_surface.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
PlanarPrestonSurface = module.PlanarPrestonSurface

FIELDS = ['schema','epoch','sequence','simulation_time_s','cursor_mm','path_length_mm',
    'frontier_mm','reverse_mm','target_force_n','return_allowed','elapsed_s','shield','fault',
    'actual_speed_mm_s','tracking_error_mm','tcp_x_mm','tcp_y_mm','tcp_z_mm',
    'normal_x','normal_y','normal_z','normal_force_n','done']


class Recording:
    def __init__(self):
        self.rows = []
        self.error = None

    def append(self, values):
        v = np.asarray(values, dtype=float)
        if v.shape != (23,) or not np.isfinite(v).all() or v[0] != 2:
            raise ValueError('Expected finite 23D controller snapshot schema 2')
        if v[1] < 1 or v[1] != int(v[1]) or v[2] != int(v[2]):
            raise ValueError('Invalid epoch/sequence')
        if v[8] != 20. or v[7] != 0. or v[9] != 0.:
            raise ValueError('Fixed force or forward-only contract violated')
        if self.rows and self.rows[-1][22] == 1:
            return False  # End strictly at the first completed control tick.
        if not self.rows:
            if v[2] != 1 or abs(v[10]-.008) > 1.e-8:
                raise ValueError('Missed the first Force control tick')
        else:
            previous = self.rows[-1]
            if v[1] != previous[1] or v[2] != previous[2]+1:
                raise ValueError('Mixed epochs or missing/reordered control tick')
            if abs(v[10]-previous[10]-.008) > 1.e-8:
                raise ValueError('Control cadence differs from 8 ms')
            if v[3] < previous[3]-1.e-9:
                raise ValueError('Simulation clock moved backwards during Force')
        self.rows.append(v.copy())
        return True


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def summarize(rows, reference, output, *, interrupted=False, error=None, make_plots=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = np.asarray(rows, dtype=float)
    if not len(data):
        result = {'valid_completed_run': False, 'samples': 0, 'interrupted': interrupted,
                  'error': error or 'No controller samples'}
        write_json(output/'summary.json', result)
        return result
    reference = np.asarray(reference, dtype=float)[:, :3]
    surface = PlanarPrestonSurface(reference, cell_size_mm=.5, velocity_model='rotating_disk', spindle_rpm=1000.)
    roi = surface.roi.copy()
    arc = np.r_[0., np.linalg.norm(np.diff(reference, axis=0), axis=1).cumsum()]
    for position in reference[arc <= 18.]:
        rr, cc, _, _, weights, _ = surface._footprint((position-surface.origin)@surface.basis.T)
        roi[rr[weights>0], cc[weights>0]] = False
    time = (data[:, 2]-1)*.008  # Exact control index, not callback arrival time.
    # Match the float32 pose/midpoint arithmetic in the Isaac training recorder.
    # Float64 midpoint rounding can change cells on the disk footprint boundary.
    # The original double-precision telemetry remains available in telemetry.csv.
    xyz = data[:, 15:18].astype(np.float32)
    delta = np.diff(xyz, axis=0, prepend=xyz[:1])
    delta -= (delta@surface.normal)[:, None]*surface.normal
    distance = np.linalg.norm(delta, axis=1)
    speed = data[:,13].copy()
    velocity = delta / np.maximum(distance[:,None],1.e-12) * speed.astype(np.float32)[:,None]
    force = np.abs(data[:, 21])
    contact = (force >= 1.5) & (data[:, 12] == 0)
    rate = np.zeros(len(data))
    entry_depth = np.zeros(surface.shape)
    for i in range(len(data)):
        before = surface.integrated_volume
        surface.deposit(xyz[i]-.5*delta[i], velocity[i], force[i], .008, contact=bool(contact[i]))
        rate[i] = (surface.integrated_volume-before)/.008
        if time[i] < 2.:
            entry_depth[:] = surface.depth
    remaining = np.maximum(surface.depth-entry_depth, 0.)
    values = remaining[roi]
    keep = time >= 2.
    mean = float(values.mean())
    full = surface.metrics()
    processing = {'spatial_depth_cv': float(values.std()/mean) if mean > 0 else None,
        'roi_volume_over_k': float(values.sum()*surface.area),
        'mean_depth_over_k': mean, 'zero_depth_fraction': float(np.mean(values == 0)),
        'roi_cells': int(roi.sum()), 'processing_time_s': int(keep.sum())*.008,
        'force_mean_n': float(force[keep].mean()) if keep.any() else None,
        'distance_mm': float(np.sum(speed[keep])*.008),
        'raw_rate_cv': float(rate[keep].std()/rate[keep].mean()) if keep.any() and rate[keep].mean()>0 else None}
    completed = bool(data[-1,22] == 1 and abs(data[-1,4]-data[-1,5]) < 1.e-5)
    result = {'valid_completed_run': completed and not interrupted and error is None and not np.any(data[:,12]),
        'completed': completed, 'fault_samples': int(np.count_nonzero(data[:,12])),
        'interrupted': interrupted, 'error': error, 'samples': len(data),
        'epoch': int(data[0,1]), 'last_sequence': int(data[-1,2]),
        'recording_source': 'single controller telemetry snapshot; no independent latest-position/latest-force timer',
        'force_basis': 'absolute measured TCP-normal force in the same controller snapshot',
        'clock': 'control tick at 0.008 s; simulation timestamp also retained',
        'physical_depth_validated': False, 'model': surface.metadata,
        'target_force_n':20., 'virtual_spindle_rpm':1000., 'urdf_rotation_unchanged':True,
        'entry_protocol': {'seconds': 2., 'reference_footprint_mm': 18., 'zero_depth_cells_retained': True},
        'metrics': full, 'processing_metrics': processing}
    for name, depth, mask in [('surface',surface.depth,surface.roi),('processing_surface',remaining,roi)]:
        np.savez_compressed(output/f'{name}.npz', depth_over_k=depth, roi=mask,
            u_mm=surface.u, v_mm=surface.v, cell_area_mm2=surface.area,
            origin_mm=surface.origin, basis=surface.basis)
    np.savez_compressed(output/'derived_signals.npz', time_s=time, xyz_mm=xyz,
        force_n=force, tangent_speed_mm_s=speed, raw_rate_n_mm_s=rate)
    write_json(output/'summary.json', result)
    if make_plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        extent=[surface.u[0]+surface.origin[0],surface.u[-1]+surface.origin[0],
                surface.v[0]+surface.origin[1],surface.v[-1]+surface.origin[1]]
        fig,ax=plt.subplots(figsize=(7,6),layout='constrained')
        im=ax.imshow(np.where(roi,remaining,np.nan),origin='lower',extent=extent,cmap='viridis')
        ax.set(xlabel='X [mm]',ylabel='Y [mm]',title=f'Processing removal depth / K | CV {processing["spatial_depth_cv"]}')
        fig.colorbar(im,ax=ax,label='Uncalibrated depth / K')
        fig.savefig(output/'01_removal_heatmap.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(3,1,sharex=True,figsize=(11,7),layout='constrained')
        axes[0].plot(time,force,lw=.6,label='Measured');axes[0].plot(time,data[:,8],lw=.6,label='Target')
        axes[0].legend();axes[0].set_ylabel('Force [N]')
        axes[1].plot(time,speed,lw=.6);axes[1].set_ylabel('TCP speed [mm/s]')
        axes[2].plot(time,rate,lw=.6);axes[2].set_ylabel('Removal rate / K');axes[2].set_xlabel('Force control time [s]')
        for ax in axes:ax.axvspan(0,2,color='gray',alpha=.15)
        fig.savefig(output/'03_signals_subplot.png',dpi=150);plt.close(fig)
    return result
