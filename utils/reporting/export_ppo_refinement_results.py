"""Audit and plot a continued-PPO study, preserving the requested PPO save."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT
from nrs_rl.tasks.manager_based.nrs_rl.utils.removal_metrics import save, sha
from nrs_rl.tasks.manager_based.nrs_rl.mdp.evaluation.refine_fixed_force_ppo import preserve
from nrs_rl.tasks.manager_based.nrs_rl.mdp.evaluation.verify_fixed_force_ppo import verify


def read(path):
    return json.loads(Path(path).read_text())


def finish(root):
    root = Path(root).resolve()
    torch.set_num_threads(1)
    proof = verify(root)
    assert proof['passed']
    preserve(root)
    result = read(root/'result.json')
    report = read(root/'final_report.json')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    plots = ROOT/'logs/polishing_results'/timestamp
    plots.mkdir(exist_ok=False)
    rows = []
    for seed, heldout in result['heldouts'].items():
        for name, m in heldout['candidates'].items():
            rows.append({'seed': seed, 'model': name, 'spatial_cv': m['spatial_cv'],
                'cv_improvement_percent_vs_mode6': 100.*m['mode6_cv_gain'],
                'processing_volume_percent_vs_mode6': 100.*m['mode6_volume_ratios']['roi_volume_over_k'],
                'full_volume_percent_vs_mode6': 100.*m['mode6_volume_ratios']['full_roi_volume_over_k'],
                'whole_volume_percent_vs_mode6': 100.*m['mode6_volume_ratios']['whole_grid_volume_over_k'],
                'time_s': m['time_s'], 'eligible': m['eligible']})
    with (plots/'metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({'font.size':10,'figure.dpi':130})
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for folder in sorted(root.glob('training_seed*')):
        if not folder.is_dir():continue
        protocol=read(folder/'protocol.json')
        evaluations=[read(p) for p in sorted(folder.glob('evaluation_*.json'))]
        steps=[v['update'] for v in evaluations]
        label=f"seed {protocol['seed']}, GAE lambda {protocol['gae_lambda']}"
        axes[0].plot(steps,[v['cv_improvement_percent_vs_mode6'] for v in evaluations],label=label)
        axes[1].plot(steps,[100.*np.mean([r['full_ratio'] for r in v['ppo']]) /
                           np.mean([r['full_ratio'] for r in v['mode6']]) for v in evaluations],label=label)
        logs=np.genfromtxt(folder/'ppo_metrics.csv',delimiter=',',names=True)
        axes[2].plot(logs['update'],logs['actor_total_change_l2'],label=label)
    for ax in axes:ax.set_xlabel('Additional PPO update');ax.grid(alpha=.25)
    axes[0].set_ylabel('Spatial CV improvement vs Mode 6 (%)')
    axes[1].set_ylabel('Full removal vs Mode 6 (%)');axes[1].axhline(50,color='red',ls='--')
    axes[2].set_ylabel('Actor weight change from saved PPO (L2)')
    axes[0].legend(fontsize=8)
    fig.suptitle('PPO continuation: approximate training environment, not Isaac validation')
    fig.savefig(plots/'01_ppo_learning_curves.png',bbox_inches='tight');plt.close(fig)
    names=['constant','ppo_saved','selected'];labels=['Mode 6','Saved PPO','New PPO']
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    colors=['#828282','#e69f00','#0072b2']
    for ax,key,ylabel in zip(axes,['cv_improvement_percent_vs_mode6','full_volume_percent_vs_mode6','time_s'],
                           ['Spatial CV improvement vs Mode 6 (%)','Full removal vs Mode 6 (%)','Processing run duration (s)']):
        values=[[r[key] for r in rows if r['model']==name] for name in names]
        ax.bar(labels,[np.mean(v) for v in values],color=colors,alpha=.75)
        for i,v in enumerate(values):
            ax.scatter([i]*len(v),v,color='black',s=15)
            ax.text(i,max(v),f'{np.mean(v):.2f}',ha='center',va='bottom')
        ax.set_ylabel(ylabel);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.25)
    axes[1].axhline(50,color='red',ls='--')
    axes[1].text(2.35,52,'50% minimum',color='red',ha='right',fontsize=8)
    fig.suptitle('Independent Isaac heldouts; all performance ratios use matched Mode 6')
    fig.savefig(plots/'02_isaac_mode6_comparison.png',bbox_inches='tight');plt.close(fig)
    first_seed=next(iter(result['heldouts']))
    folder=root/'physics'/f'heldout_{first_seed}'
    surfaces=[]
    for name in names:
        with np.load(folder/f'{name}_processing_surface.npz') as f:surfaces.append(dict(f))
    baseline_mean=surfaces[0]['depth_over_k'][surfaces[0]['roi']].mean()
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,name,label,s in zip(axes,names,labels,surfaces):
        data=np.where(s['roi'],s['depth_over_k']/baseline_mean,np.nan)
        im=ax.imshow(data,origin='lower',extent=[s['u_mm'][0],s['u_mm'][-1],s['v_mm'][0],s['v_mm'][-1]],vmin=0,vmax=2.,cmap='viridis',aspect='equal')
        ax.set_title(label);ax.set_xlabel('u (mm)');ax.set_ylabel('v (mm)')
    fig.colorbar(im,ax=axes,label='Depth / Mode 6 mean depth (shared scale)',shrink=.8,extend='max')
    fig.suptitle('Final spatial removal; first 2 active seconds and initial 18 mm footprint excluded')
    fig.savefig(plots/'03_isaac_processing_heatmaps.png',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True,layout='constrained')
    for name,label,color in zip(names,labels,colors):
        with np.load(folder/f'{name}_trace.npz') as f:t=dict(f)
        active=t['time_s']>=2.
        x=t['time_s'][active]
        for ax,key in zip(axes,['measured_speed_mm_s','normal_force_n','raw_mrr_n_mm_s']):
            ax.plot(x[::10],t[key][active][::10],label=label,color=color,alpha=.8,lw=.8)
    for ax,label in zip(axes,['Measured feed (mm/s)','Measured normal force (N)','Instantaneous removal rate / K']):
        ax.set_ylabel(label);ax.grid(alpha=.25)
    axes[1].axhline(20.,ls='--',color='black',lw=.7);axes[0].legend(ncol=3)
    axes[2].set_xlabel('Active time (s)')
    fig.suptitle('Force target is 20 N throughout; virtual spindle is fixed at 1000 RPM')
    fig.savefig(plots/'04_isaac_force_feed_removal_rate.png',bbox_inches='tight');plt.close(fig)
    report.update(numerical_audit_passed=True,plots_directory=str(plots),
                  independent_numerical_audit=str(root/'independent_numerical_audit.json'))
    save(root/'final_report.json',report)
    save(root/'candidate/validation.json',report)
    save(plots/'report.json',report)
    training=root/result['selected'].replace('ppo_','training_')
    snapshot=root/'candidate/package_source_snapshot'
    if not snapshot.exists():shutil.copytree(training/'package_source_snapshot',snapshot)
    candidate=root/'candidate'
    save(candidate/'manifest.json',{str(p.relative_to(candidate)):sha(p) for p in candidate.rglob('*')
         if p.is_file() and p!=candidate/'manifest.json'})
    save(root/'artifact_summary.json',{'plots_directory':str(plots),'report':str(root/'final_report.json'),
                                     'protected_save_preserved':True,'numerical_audit_passed':True})
    print(json.dumps({'accepted':report['accepted'],'comparison':report['same_batch_comparison'],
                      'plots_directory':str(plots),'checkpoint':report['checkpoint']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    finish(p.parse_args().root)
