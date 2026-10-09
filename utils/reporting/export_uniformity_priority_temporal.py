#!/usr/bin/env python3
"""Add time-domain diagnostics without confusing them with spatial uniformity."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np


def main(root):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/nrs_uniformity_priority_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result=json.loads((root/'result.json').read_text());export=Path(result['export_directory'])
    selected=result['selected'];names=['constant','incumbent',selected]
    labels=['Constant baseline','Previous GUI policy','New candidate']
    traces=[];row_labels=[];max_time=0.
    for seed in result['heldout_seeds']:
        for name,label in zip(names,labels):
            with np.load(root/f'held_out_{seed}'/f'{name}_trace.npz') as data:
                t=data['time_s'];rate=data['raw_mrr_n_mm_s'];keep=t>=2.
                traces.append((t[keep].copy(),rate[keep].copy()))
                max_time=max(max_time,float(t[-1]));row_labels.append(f'{seed}: {label}')
    edges=np.arange(2.,max_time+.081,.08);centers=(edges[:-1]+edges[1:])/2.
    raw=np.full((len(traces),len(centers)),np.nan);normalized=raw.copy();raw_cv=[]
    for row,(t,rate) in enumerate(traces):
        counts=np.histogram(t,bins=edges)[0];sums=np.histogram(t,bins=edges,weights=rate)[0]
        np.divide(sums,counts,out=raw[row],where=counts>0)
        normalized[row]=raw[row]/rate.mean();raw_cv.append(float(rate.std()/rate.mean()))
    fig,axes=plt.subplots(2,1,figsize=(14,8),sharex=True,constrained_layout=True)
    for ax,data,title,unit in zip(axes,[raw,normalized],
        ['Instantaneous removal rate (80 ms bin means)','Removal rate / each run mean (80 ms bin means)'],
        ['Removal rate / K (N mm/s)','Rate / mean rate']):
        cap=float(np.nanquantile(data,.995));cmap=plt.colormaps['magma'].copy();cmap.set_bad('#dddddd')
        im=ax.imshow(data,aspect='auto',origin='upper',extent=[edges[0],edges[-1],len(traces)-.5,-.5],
            vmin=0.,vmax=cap,cmap=cmap,interpolation='nearest')
        ax.set_yticks(np.arange(len(traces)),[f'{label}  [raw CV={cv:.3f}]' for label,cv in zip(row_labels,raw_cv)])
        ax.set_title(title);fig.colorbar(im,ax=ax,label=unit)
    axes[-1].set_xlabel('Time since Force entry (s); gray = completed, not zero removal')
    fig.suptitle('Temporal diagnostics only; candidate selection uses spatial depth CV')
    fig.savefig(export/'04_temporal_removal_heatmap.png',dpi=160);plt.close(fig)
    np.savez_compressed(export/'temporal_heatmap_data.npz',time_bin_edges_s=edges,
        raw_rate_bin_mean=raw,normalized_rate_bin_mean=normalized,raw_sample_cv=raw_cv,
        row_labels=np.asarray(row_labels))
    source=Path(result['training_dir'])/'learned_force_profiles.png'
    if source.exists():shutil.copyfile(source,export/'03_learned_force_profiles.png')
    audit=root/'verification.json'
    if audit.exists():shutil.copyfile(audit,export/'verification.json')
    manifest={str(p.relative_to(export)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in export.rglob('*') if p.is_file() and p.name!='manifest.json'}
    (export/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(export/'04_temporal_removal_heatmap.png')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    main(parser.parse_args().root.resolve())
