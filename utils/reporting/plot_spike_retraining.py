"""Archive final matched retraining comparisons, including first contact and heatmaps."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    with np.load(path) as data:
        return dict(data)


def archive(output, baseline, candidate, result):
    output, baseline, candidate = map(Path, (output, baseline, candidate))
    output.mkdir(parents=True, exist_ok=True)
    runs = [('mode6', 'Mode 6', candidate, 'constant'),
            ('mode5_previous', 'Mode 5 previous', baseline, 'selected'),
            ('mode5_new', 'Mode 5 retrained', candidate, 'selected')]
    summaries = {}
    signals, surfaces = [], []
    for key, label, directory, name in runs:
        destination = output / key
        destination.mkdir(exist_ok=True)
        summary = json.loads((directory / 'spike_summary.json').read_text())['candidates'][name]
        summaries[key] = summary
        for suffix in ('trace.npz', 'precontrol_contact.npz', 'contact_surface.npz', 'spike_dose.npz'):
            source = directory / f'{name}_{suffix}'
            if source.exists():
                shutil.copy2(source, destination / suffix)
        (destination / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        active = load(directory / f'{name}_trace.npz')
        prefix_file = directory / f'{name}_precontrol_contact.npz'
        prefix = load(prefix_file) if prefix_file.exists() else None
        n = len(prefix['normal_force_n']) if prefix is not None else 0
        force = np.r_[prefix['normal_force_n'] if prefix else [], active['normal_force_n']]
        feed = np.r_[np.zeros(n), active['commanded_speed_mm_s']]
        rate = np.r_[prefix['raw_mrr_n_mm_s'] if prefix else [], active['raw_mrr_n_mm_s']]
        # Causal 128 ms display averaging; raw data are retained in the archive.
        averaged = np.convolve(rate, np.ones(16), 'full')[:len(rate)] / np.minimum(np.arange(len(rate))+1, 16)
        length = json.loads((directory / 'summary.json').read_text())['path_length_mm']
        progress = np.r_[np.zeros(n), active['cursor_mm'] / length * 100.]
        signals.append({'key': key, 'label': label, 'progress': progress, 'time': np.arange(len(force)) * .008,
                        'values': [force, feed, averaged], 'prefix': n})
        surfaces.append(load(directory / f'{name}_contact_surface.npz'))
    colors = [('#D6604D', '#4D4D4D', '#D6604D'),
              ('#999999', '#999999', '#999999'), ('#B2182B', '#1B7837', '#2166AC')]
    styles = ['--', ':', '-']
    fig, axes = plt.subplots(3, 2, figsize=(15, 8), gridspec_kw={'width_ratios': [3.2, 1.]})
    for row, ylabel in enumerate(['Measured normal force [N]', 'Scheduled feed [mm/s]',
                                  'Estimated removal rate / K\n128 ms trailing display mean']):
        for column in range(2):
            values = []
            for i, signal in enumerate(signals):
                x = signal['progress'] if column == 0 else signal['time']
                keep = (x > 0.) if column == 0 else (x <= 3.)
                y = signal['values'][row]
                axes[row, column].plot(x[keep], y[keep], color=colors[i][row], linestyle=styles[i],
                                       linewidth=1., label=signal['label'])
                values.append(y[keep])
            joined = np.concatenate(values)
            low, high = float(joined.min()), float(joined.max())
            pad = .04 * max(high - low, .01)
            axes[row, column].set_ylim(low - pad, high + pad)
            axes[row, column].set_xlim((0., 100.) if column == 0 else (0., 3.))
            axes[row, column].grid(alpha=.2)
        axes[row, 0].set_ylabel(ylabel)
        axes[row, 0].legend(loc='best', fontsize=8)
    axes[0, 0].set_title('Same path position; stationary initial contact shown at right')
    axes[0, 1].set_title('Initial contact: first 3 s')
    axes[-1, 0].set_xlabel('Path progress [%]')
    axes[-1, 1].set_xlabel('Time since first contact [s]')
    fig.suptitle('Headless Isaac | TCP 146 mm | fixed 20 N | virtual 1000 RPM\n'
                 'First measured contact (1.5 N) to completion; initial contact included, pre-contact excluded')
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(output / '01_spike_force_feed_removal.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout='constrained')
    vmax = max(float(s['depth_over_k'][s['roi']].max()) for s in surfaces)
    for ax, signal, surface in zip(axes, signals, surfaces):
        data = np.ma.masked_where(~surface['roi'], surface['depth_over_k'])
        u, v = surface['u_mm'], surface['v_mm']
        cell = float(np.sqrt(surface['cell_area_mm2']))
        im = ax.imshow(data, origin='lower', cmap='viridis', vmin=0., vmax=vmax,
                       extent=(u[0]-.5*cell, u[-1]+.5*cell, v[0]-.5*cell, v[-1]+.5*cell))
        row = summaries[signal['key']]
        ax.set(title=f"{signal['label']}\nCV={row['contact_cv']:.5f} | {row['time_s']:.3f} s",
               xlabel='Plane U [mm]', ylabel='Plane V [mm]')
    fig.colorbar(im, ax=axes, label='Estimated depth / K (uncalibrated; common scale)', shrink=.8)
    fig.suptitle('Headless Isaac | first contact to completion | common reference ROI including zero-depth cells')
    fig.savefig(output / '02_spike_contact_depth_maps.png', dpi=180)
    plt.close(fig)
    (output / 'comparison.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    rows = []
    for key, row in summaries.items():
        rows.append({'name': key, **{field: row[field] for field in (
            'contact_cv', 'active_cv', 'processing_cv', 'contact_volume_over_k', 'time_s', 'force_peak_n')},
            'spike_abs_error_seconds': row['spike']['event_abs_error_seconds'],
            'spike_error_rms': row['spike']['event_error_rms'], 'spike_seconds': row['spike']['event_seconds']})
    with (output / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in output.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json'}
    (output / 'artifact_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
