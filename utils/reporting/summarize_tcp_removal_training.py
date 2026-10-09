#!/usr/bin/env python3
"""Publish descriptive held-out results without selecting on validation seeds."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def summarize(root):
    state=json.loads((root/'result.json').read_text())
    if state['phase']!='complete' or len(state['validation'])!=3:
        raise ValueError('Wait for all three frozen held-out evaluations')
    rows=[]; per_seed=[]
    for seed in state['held_out_seeds']:
        summary=json.loads((root/f'held_out_{seed}/summary.json').read_text())
        candidates=summary['candidates']; group={}
        for name,metrics in candidates.items():
            surface=metrics['preston_surface']
            row={'seed':seed,'policy':name,'spatial_depth_cv':surface['spatial_depth_cv'],
                 'roi_volume_over_k':surface['roi_volume_over_k'],
                 'uncovered_fraction':surface['zero_depth_fraction'],
                 'raw_rate_cv':metrics['processing_rate_cv'],
                 'processing_time_s':metrics['processing_samples']*.008,
                 'cross_track_rmse_mm':metrics['cross_track_rmse_mm'],
                 'endpoint_error_mm':metrics['endpoint_tangential_error_mm'],
                 'force_rmse_n':metrics['force_rmse_n'],'force_peak_n':metrics['force_peak_n'],
                 'shield_fraction':metrics['shield_fraction'],'fault_fraction':metrics['fault_fraction']}
            rows.append(row); group[name]=row
        group['constant']={key:float(np.mean([group[x][key] for x in ('constant_a','constant_b')]))
                           for key in rows[-1] if key not in ('seed','policy')}
        per_seed.append(group)
    with (root/'validation_comparison.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    numeric=[key for key in rows[0] if key not in ('seed','policy')]
    stats={}
    for mode in ('constant','old_d14','learned'):
        members=('constant_a','constant_b') if mode=='constant' else (mode,)
        stats[mode]={key:{'mean':float(np.mean([s[m][key] for s in per_seed for m in members])),
                          'min':float(np.min([s[m][key] for s in per_seed for m in members])),
                          'max':float(np.max([s[m][key] for s in per_seed for m in members]))}
                     for key in numeric}
    gains=[c for seed in state['validation'] for c in seed['comparisons']]
    report={'phase':'complete','spindle_rpm':None,'rotation_contribution':False,
        'algorithm':state['algorithm'],'training_generations':len(state['generations']),
        'training_seeds':state['training_seeds'],'validation_seeds':state['held_out_seeds'],
        'checkpoint':state['checkpoint'],'checkpoint_sha256':hashlib.sha256(Path(state['checkpoint']).read_bytes()).hexdigest(),
        'learned_parameters':state['best_parameters'],'frozen_gates_passed':state['validation_passed'],
        'disposition':state['disposition'],'production_checkpoint_replaced':False,
        'statistics':stats,
        'spatial_cv_gain_percent_range_vs_both_baselines':[100*min(x['spatial_cv_gain'] for x in gains),100*max(x['spatial_cv_gain'] for x in gains)],
        'raw_rate_cv_gain_percent_range_vs_both_baselines':[100*min(x['raw_rate_cv_gain'] for x in gains),100*max(x['raw_rate_cv_gain'] for x in gains)],
        'validation_rejections':{str(x['seed']):x['rejections'] for x in state['validation']},
        'limits':['Uncalibrated modeled removal; no physical material-removal measurement.',
                  'Three held-out sensor/delay conditions on the original flat reference path.',
                  'Reference-cursor completion; actual endpoint settling is not simulated.',
                  'Newly corrected ROS timing has separate integration tests; GUI machining was not rerun.',
                  'No held-out result was used to revise coefficients or thresholds.']}
    (root/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    modes=['constant','old_d14','learned']; labels=['Mode 6','Old D14','Retrained']
    colors=['#3b6e9b','#d68b39','#389377']
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    metrics=[('spatial_depth_cv','Spatial depth CV (lower is better)'),
             ('raw_rate_cv','Raw removal-rate CV'),('roi_volume_over_k','ROI volume / K (uncalibrated)'),
             ('processing_time_s','Processing time [s]'),('cross_track_rmse_mm','Path error RMS [mm]'),
             ('force_peak_n','Peak normal force [N]')]
    for ax,(field,title) in zip(axes.flat,metrics):
        mean=np.array([stats[m][field]['mean'] for m in modes])
        low=np.array([stats[m][field]['min'] for m in modes])
        high=np.array([stats[m][field]['max'] for m in modes])
        ax.bar(labels,mean,color=colors,yerr=np.stack([mean-low,high-mean]),capsize=4)
        ax.set_title(title); ax.grid(axis='y',alpha=.2)
        for i,v in enumerate(mean): ax.annotate(f'{v:.3g}',(i,high[i]),ha='center',xytext=(0,7),textcoords='offset points')
        ax.set_ylim(0,float(high.max())*1.23)
    result='PASS' if state['validation_passed'] else 'FAIL — experimental checkpoint'
    fig.suptitle(f'RPM excluded | Frozen validation: {result}\n3 held-out seeds; mean and full trial range (two constant replicates)',fontsize=13)
    fig.savefig(root/'validation_comparison.png',dpi=170);plt.close(fig)
    # The representative seed is fixed by protocol order, not best performance.
    seed=state['held_out_seeds'][0]
    maps=[]
    for name in ('constant_a','old_d14','learned'):
        with np.load(root/f'held_out_{seed}/{name}_preston_surface.npz') as data: maps.append(dict(data))
    maximum=max(float(m['depth_over_k'][m['roi']].max()) for m in maps)
    fig,axes=plt.subplots(1,3,figsize=(13,5),layout='constrained')
    for ax,data,label in zip(axes,maps,labels):
        view=ax.imshow(np.where(data['roi'],data['depth_over_k'],np.nan),origin='lower',vmin=0,vmax=maximum,
                       extent=[data['u_mm'][0],data['u_mm'][-1],data['v_mm'][0],data['v_mm'][-1]])
        ax.set(title=label,xlabel='u [mm]',ylabel='v [mm]')
    fig.colorbar(view,ax=axes,label='Depth / K (uncalibrated)')
    fig.suptitle(f'Fixed reference ROI, including uncovered cells | held-out seed {seed}')
    fig.savefig(root/'validation_depth_maps.png',dpi=170);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('training_directory',type=Path)
    summarize(parser.parse_args().training_directory)
