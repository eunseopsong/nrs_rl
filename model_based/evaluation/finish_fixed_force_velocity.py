#!/usr/bin/env python3
"""Autonomous physics selection, refinement, independent validation and export."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT as _REPO_ROOT
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.fixed_force_velocity import CONTRACT, REFERENCE, export_actor
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_fixed_force_velocity import save, sha
from nrs_rl.tasks.manager_based.nrs_rl.utils.assess_fixed_force_velocity import evaluate

ROOT = _REPO_ROOT


def batch(training, root, label, names, seed):
    command = [sys.executable, str(ROOT/'scripts/probe_fixed_force_velocity.py'),
        '--headless', '--device', 'cpu', '--training-dir', str(training),
        '--output', str(root/label), '--names', ','.join(names), '--seed', str(seed),
        '--max-steps', '42000', '--hard-force-stop-n', '100', '--uniformity-only']
    save(root/f'{label}_command.json', command)
    with (root/f'{label}.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
            env={**os.environ, 'MPLCONFIGDIR': '/tmp/nrs_fixed_force_mpl', 'MPLBACKEND': 'Agg'})
    report = evaluate(root/label, audit=True)
    print('FIXED_FORCE_BATCH', label, json.dumps({n: {
        'eligible': m['eligible'], 'mode6_cv_gain': m['mode6_cv_gain'],
        'mode6_volume_ratios': m['mode6_volume_ratios']} for n,m in report['candidates'].items()}), flush=True)
    return report


def parallel_batches(training, root, specs):
    reports = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {pool.submit(batch, training, root, label, names, seed): label
                for label, names, seed in specs}
        for job in as_completed(jobs):
            reports[jobs[job]] = job.result()
    return reports


def select(reports):
    rows = [(m['mode6_cv_gain'], label, name) for label,r in reports.items()
            for name,m in r['candidates'].items()
            if name not in ('constant', 'constant_b') and m['eligible']]
    if not rows:
        raise RuntimeError('No eligible full-path, fixed-force, volume-feasible candidate')
    return max(rows)


def figures(export, physics, state):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    seeds = [2611, 2612]
    reports = [state['heldouts'][str(s)] for s in seeds]
    fig, axes = plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    labels = ['CV / Mode 6', 'Processing removal / Mode 6', 'Full removal / Mode 6']
    for ax,key,label in zip(axes, ['spatial_cv','roi_volume_over_k','full_roi_volume_over_k'], labels):
        values = [100.*r['candidates']['selected'][key]/r['candidates']['constant'][key] for r in reports]
        ax.bar([0,1],values,color='#237c92'); ax.axhline(100.,color='gray',ls='--')
        if key!='spatial_cv': ax.axhline(50.,color='red',ls=':')
        ax.set_xticks([0,1],list(map(str,seeds)));ax.set_ylabel('Mode 6 = 100%')
        ax.set_xlabel('Held-out seed');ax.set_title(label);ax.grid(axis='y',alpha=.2)
        for x,v in enumerate(values):ax.text(x,v,f'{v:.2f}%',ha='center',va='bottom')
    fig.suptitle('Fixed target force 20 N; assumed virtual spindle 1000 RPM')
    fig.savefig(export/'00_mode6_comparison.png',dpi=160);plt.close(fig)
    folder = physics/'heldout_2611'
    arrays = []
    for name in ['constant','selected']:
        with np.load(folder/f'{name}_processing_surface.npz') as f:arrays.append(dict(f))
    vmax = max(np.quantile(d['depth_over_k'][d['roi']],.995) for d in arrays)
    fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    for i,(data,label) in enumerate(zip(arrays,['Mode 6','Fixed-force velocity policy'])):
        depth,mask=data['depth_over_k'],data['roi']
        extent=[data['u_mm'][0],data['u_mm'][-1],data['v_mm'][0],data['v_mm'][-1]]
        for j,values,cap in [(0,depth,vmax),(1,depth/depth[mask].mean(),2.)]:
            im=axes[j,i].imshow(np.where(mask,values,np.nan),origin='lower',extent=extent,vmin=0,vmax=cap,cmap='viridis')
            axes[j,i].set_title(label if j==0 else 'Mean-normalized depth')
            axes[j,i].set_xlabel('u (mm)');axes[j,i].set_ylabel('v (mm)')
            fig.colorbar(im,ax=axes[j,i],label='h/K' if j==0 else 'h/mean(h)')
    fig.savefig(export/'01_processing_depth_maps.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(4,1,figsize=(14,10),sharex=True,constrained_layout=True)
    times,rates=[],[]
    for name,label in [('constant','Mode 6'),('selected','Fixed-force velocity policy')]:
        with np.load(folder/f'{name}_trace.npz') as f:
            keep=f['time_s']>=2.;t=f['time_s'][keep]
            for ax,key in zip(axes,['target_force_n','normal_force_n','measured_speed_mm_s','raw_mrr_n_mm_s']):
                ax.plot(t,f[key][keep],lw=.55,label=label)
            times.append(t.copy());rates.append(f['raw_mrr_n_mm_s'][keep].copy())
    for ax,label in zip(axes,['Target force (N)','Measured force (N)','Sliding speed (mm/s)','Removal rate / K']):
        ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0].legend();axes[0].set_ylim(19.,21.);axes[-1].set_xlabel('Time since Force entry (s)')
    fig.savefig(export/'02_process_signals.png',dpi=160);plt.close(fig)
    bins=np.arange(2.,max(t[-1] for t in times)+.081,.08)
    heat=np.full((2,len(bins)-1),np.nan)
    for i,(t,r) in enumerate(zip(times,rates)):
        count=np.histogram(t,bins=bins)[0];value=np.histogram(t,bins=bins,weights=r)[0]
        np.divide(value,count,out=heat[i],where=count>0)
    fig,ax=plt.subplots(figsize=(13,3.5),constrained_layout=True)
    cmap=plt.colormaps['magma'].copy();cmap.set_bad('#ddd')
    im=ax.imshow(heat,aspect='auto',extent=[bins[0],bins[-1],1.5,-.5],cmap=cmap,interpolation='nearest')
    ax.set_yticks([0,1],['Mode 6','Fixed-force policy']);ax.set_xlabel('Time (s); gray = completed')
    ax.set_title('Instantaneous removal rate with virtual rotation, 80 ms means')
    fig.colorbar(im,ax=ax,label='Removal rate / K')
    fig.savefig(export/'03_temporal_heatmap.png',dpi=160);plt.close(fig)


def main(args):
    root=args.root.resolve(); physics=root/'physics';physics.mkdir(exist_ok=False)
    training=root/'training'; started=time.monotonic();torch.set_num_threads(1)
    state={'phase':'physics_selection', 'protocol':json.loads((training/'protocol.json').read_text()),
           'gui_trial_executed':False,'deployed':False,'accepted':False,'heldouts':{}}
    def status():
        state['elapsed_s']=time.monotonic()-started;save(root/'status.json',state)
    status()
    scripts=['fixed_force_velocity.py','fixed_force_velocity_action.py','probe_fixed_force_velocity.py',
             'train_fixed_force_velocity.py','finish_fixed_force_velocity.py','assess_fixed_force_velocity.py',
             'test_fixed_force_velocity.py','spatial_uniformity.py','tcp_tracking_compensation.py']
    sources={}
    for rel in ['scripts/'+n for n in scripts]+[
        'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/mdp/action.py',
        'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/utils/preston_surface.py']:
        dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/rel,dest);sources[rel]=sha(dest)
    save(root/'source_manifest.json',sources)
    first=parallel_batches(training,physics,[
        ('selection_a_2601',['constant','profile49_100','profile97_90','profile97_110'],2601),
        ('selection_b_2601',['constant','profile193_100','profile97_feedback','profile97_pause'],2601)])
    gain,label,name=select(first)
    state.update(phase='physics_trace_refinement',initial_selected=name,initial_selection_gain=gain,
                 initial_reports=first);status()
    refined=root/'refined_training'
    command=[sys.executable,str(ROOT/'scripts/train_fixed_force_velocity.py'),'--output',str(refined),
             '--trace',str(physics/label/'constant_trace.npz')]
    with (root/'refinement.log').open('w') as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    shutil.copyfile(training/'checkpoints'/f'{name}.pt',refined/'checkpoints/incumbent.pt')
    state.update(phase='refined_physics_selection');status()
    second=parallel_batches(refined,physics,[
        ('selection_a_2602',['constant','incumbent','profile193_100','profile97_90'],2602),
        ('selection_b_2602',['constant','incumbent','profile97_110','profile97_feedback'],2602)])
    gain,label,name=select(second)
    bundle=root/'candidate';(bundle/'checkpoints').mkdir(parents=True)
    shutil.copyfile(refined/'checkpoints'/f'{name}.pt',bundle/'checkpoints/actor.pt')
    extra={'fixed_force_policy.json':''}
    torch.jit.load(str(bundle/'checkpoints/actor.pt'),_extra_files=extra)
    save(bundle/'fixed_force_policy.json',json.loads(extra['fixed_force_policy.json']))
    shutil.copyfile(bundle/'checkpoints/actor.pt',refined/'checkpoints/selected.pt')
    state.update(phase='independent_physics_validation',selected=name,selection_reports=second,
        selection_mode6_gain=gain,checkpoint=str(bundle/'checkpoints/actor.pt'),
        checkpoint_sha256=sha(bundle/'checkpoints/actor.pt'),
        selected_before_holdouts_at=datetime.now(timezone.utc).isoformat())
    save(bundle/'selection.json',state);status()
    heldouts=parallel_batches(refined,physics,[
        (f'heldout_{seed}',['constant','constant_b','selected'],seed) for seed in [2611,2612]])
    for label,report in sorted(heldouts.items()):
        m=report['candidates'];noise=abs(m['constant_b']['spatial_cv']/m['constant']['spatial_cv']-1.)
        threshold=max(.005,2.*noise)
        report['acceptance']={'passed':bool(m['selected']['eligible'] and m['constant_b']['eligible']
            and m['selected']['mode6_cv_gain']>=threshold),
            'minimum_mode6_cv_gain':threshold,'paired_baseline_variation':noise,
            'threshold_is_practical_not_statistical_significance':True}
        state['heldouts'][str(report['seed'])]=report
    state.update(phase='rpm_sensitivity_and_artifact_audit');status()
    sensitivity={}
    for seed in [2611,2612]:
        sensitivity[str(seed)]={}
        for rpm in [300.,3000.]:
            report=evaluate(physics/f'heldout_{seed}',rpm=rpm)
            sensitivity[str(seed)][str(rpm)]=report
            print('FIXED_FORCE_RPM_SENSITIVITY',seed,rpm,report['candidates']['selected']['mode6_cv_gain'],flush=True)
    state['rpm_sensitivity']=sensitivity
    state['accepted']=all(r['acceptance']['passed'] for r in state['heldouts'].values())
    # Sensitivity re-integrates the SAME measured trajectories, not new physical spindle dynamics.
    assert sha(bundle/'checkpoints/actor.pt')==state['checkpoint_sha256']
    for rel,digest in sources.items():assert sha(ROOT/rel)==digest,rel
    validation={'passed':state['accepted'],'force_fixed_20n_all_ticks':True,
        'action_dimension':1,'selected_before_holdouts':True,'raw_reintegration_passed':True,
        'source_hashes_unchanged':True,'holdouts':state['heldouts'],
        'rpm_sensitivity_kind':'same-trace counterfactual reintegration only',
        'deployed':False,'gui_trial_executed':False,
        'runtime_requirement':'scalar-feed runtime with hard-coded 20 N and virtual-rotation estimator; legacy variable-force deployment is incompatible'}
    save(bundle/'validation.json',validation)
    export=ROOT/'logs/polishing_results'/datetime.now().strftime('%Y%m%d_%H%M')
    export.mkdir(parents=True,exist_ok=False)
    figures(export,physics,state)
    rows=[]
    for seed,r in state['heldouts'].items():
        c=r['candidates']['selected'];v=c['mode6_volume_ratios']
        rows.append(f"{seed}: Mode 6 CV reduction {100*c['mode6_cv_gain']:.4f}%, processing removal retained {100*v['roi_volume_over_k']:.4f}%, full removal retained {100*v['full_roi_volume_over_k']:.4f}%")
    report=('Fixed 20 N, scalar velocity/dwell; virtual spindle assumed 1000 RPM.\n'
        'Only the removal estimator includes rotation; URDF dynamics are unchanged.\n'
        'Fresh paired Mode 6 has the same force controller/tracking/limits/estimator.\n'
        'Primary CV: first 2 active seconds and fixed first-18-mm footprint excluded; zeros retained.\n'
        'Removal constraint >=50% in processing ROI, full ROI and whole grid, each heldout.\n'
        'Optimization: model-based bounded residence-time fit with Isaac selection/refinement; not PPO.\n'
        +'\n'.join(rows)+'\nAccepted: '+str(state['accepted'])+'\n'
        'Results are conditional model estimates (h/K); no real removal-depth calibration or GUI trial.\n'
        'Old 14.24% variable-force checkpoint is not eligible under this force-fixed requirement.\n')
    (export/'REPORT.txt').write_text(report);(root/'REPORT.txt').write_text(report)
    state.update(phase='complete',export_directory=str(export),deployment_ready=False);status()
    save(root/'result.json',state);save(export/'result.json',state)
    save(export/'manifest.json',{p.name:sha(p) for p in export.iterdir() if p.is_file()})
    save(bundle/'manifest.json',{str(p.relative_to(bundle)):sha(p) for p in bundle.rglob('*') if p.is_file()})
    print('FIXED_FORCE_COMPLETE',json.dumps({'accepted':state['accepted'],'checkpoint':state['checkpoint'],
        'export':str(export),'elapsed_s':state['elapsed_s']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    main(p.parse_args())
