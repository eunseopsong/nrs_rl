#!/usr/bin/env python3
"""Independently check exported figures/CSV against their original 125 Hz traces."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected, description):
    if not np.isclose(actual, expected, rtol=1e-9, atol=1e-9):
        raise ValueError(f'{description}: {actual} != {expected}')


def verify(output):
    manifest = json.loads((output/'00_export_manifest.json').read_text())
    report = json.loads((output/'00_frozen_report.json').read_text())
    source = Path(manifest['training_source'])
    for path, expected in manifest['input_sha256'].items():
        if digest(path) != expected:
            raise ValueError('Original data changed: '+path)
    for key in ('checkpoint', 'previous_checkpoint'):
        if digest(report[key]) != report[key+'_sha256']:
            raise ValueError('Checkpoint identity changed: '+key)
    with (output/'00_comparison_summary.csv').open() as file:
        records = list(csv.DictReader(file))
    traces, summary = {}, []
    for row in records:
        seed, episode, mode = int(row['seed']), int(row['episode']), row['mode']
        original = source/f'held_out_{seed}'/f'{mode}_trace.npz'
        exported = output/f'ep{episode}'/('08_control_trace.npz' if mode == 'learned'
                                        else f'comparison_traces/{mode}_trace.npz')
        if digest(original) != digest(exported):
            raise ValueError('Exported trace differs: '+str(exported))
        with np.load(exported) as data:
            active = data['polishing_active'] > 0
            rate = data['raw_mrr_n_mm_s'][active].astype(float)
            time = data['time_s'][active].astype(float)
        if not np.isfinite(rate).all() or np.any(rate < 0):
            raise ValueError('Invalid raw removal rate')
        if not np.allclose(np.diff(time), .008, atol=1e-9, rtol=0):
            raise ValueError('125 Hz trace has gaps')
        cv = float(rate.std()/rate.mean())
        close(cv, float(row['raw_rate_cv']), 'Raw CSV CV')
        close(len(rate)*.008, float(row['processing_time_s']), 'CSV duration')
        key = (seed, mode)
        if key in traces:
            raise ValueError('Duplicate comparison trace')
        traces[key] = rate
        summary.append({'seed': seed, 'mode': mode, 'samples': len(rate),
                        'zero_samples': int((rate == 0).sum()), 'raw_cv': cv,
                        'trace_sha256': digest(exported)})
    window_checks = {}
    for name in ('00_temporal_rate_heatmap.csv', '00_temporal_fluctuation_heatmap.csv'):
        totals = {key: [0, 0., 0.] for key in traces}
        windows = 0
        with (output/name).open() as file:
            for row in csv.DictReader(file):
                key = (int(row['seed']), row['mode'])
                rate = traces[key]
                start = int(np.ceil(float(row['window_start_s'])/.008-1e-7))
                end = int(np.ceil(float(row['window_end_s'])/.008-1e-7))
                part = rate[start:end]
                count = int(row['samples'])
                if len(part) != count:
                    raise ValueError('Window sample count differs')
                if not count:
                    if not (np.isnan(float(row['mean_rate_n_mm_s'])) and
                            np.isnan(float(row['relative_rms_from_run_mean']))):
                        raise ValueError('Unobserved window is not marked missing')
                    continue
                mean, rms = float(row['mean_rate_n_mm_s']), float(row['relative_rms_from_run_mean'])
                close(mean, part.mean(), 'Window mean')
                close(rms, np.sqrt(np.mean(((part-rate.mean())/rate.mean())**2)), 'Window RMS')
                close(float(row['raw_run_cv']), rate.std()/rate.mean(), 'Window raw CV')
                totals[key][0] += count
                totals[key][1] += count*mean
                totals[key][2] += count*rms*rms
                windows += 1
        for key, (count, total, squares) in totals.items():
            rate = traces[key]
            if count != len(rate):
                raise ValueError('Windows omit or duplicate samples')
            close(total/count, rate.mean(), 'Count-weighted window mean')
            close(np.sqrt(squares/count), rate.std()/rate.mean(), 'RMS conservation')
        window_checks[name] = {'nonempty_windows': windows, 'all_samples_retained': True,
                               'window_values_and_weighted_totals_match_raw': True}
    images = sorted(output.rglob('*.png'))
    for path in images:
        with Image.open(path) as image:
            image.verify()
    if len(traces) != 18 or len(images) != 37:
        raise ValueError('Incomplete export')
    result = {'passed': True, 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'output': str(output), 'traces': summary, 'png_files_verified': len(images),
              'window_checks': window_checks, 'source_and_checkpoint_hashes_match': True,
              'physical_material_volume_validated': False}
    (output/'00_independent_verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'output': str(output), 'passed': True, 'traces': len(traces), 'pngs': len(images)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, nargs='+')
    for directory in parser.parse_args().output:
        verify(directory.resolve())
