"""Export the verified PPO candidate and separate surrogate / Isaac figures."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT as _REPO_ROOT
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from nrs_rl.tasks.manager_based.nrs_rl.agents.fixed_force_ppo import sha
from nrs_rl.tasks.manager_based.nrs_rl.evaluation.verify_fixed_force_ppo import verify

ROOT=_REPO_ROOT


def plots(root,output,result):
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2})
    def save(fig,name):
        fig.savefig(output/name,dpi=160,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for seed in [3101,3102]:
        folder=root/f'training_seed{seed}'
        log=np.genfromtxt(folder/'ppo_metrics.csv',delimiter=',',names=True)
        evaluations=[json.loads(p.read_text()) for p in sorted(folder.glob('evaluation_*.json'))]
        label=f'PPO seed {seed}'
        axes[0,0].plot([x['update'] for x in evaluations],
                       [x['cv_improvement_percent_vs_mode6'] for x in evaluations],marker='o',label=label)
        axes[0,1].plot(log['update'],log['actor_total_change_l2'],label=label)
        axes[1,0].plot(log['update'],log['value_loss'],label=label)
        axes[1,1].plot(log['update'],log['approx_kl'],label=label)
    for ax,title,ylabel in zip(axes.flat,['Training simulator CV gain','Actor weight change','Critic loss','Policy update size'],
                               ['CV reduction vs Mode 6 [%]','L2 distance from initialization','Value loss','Approximate KL']):
        ax.set(title=title,xlabel='PPO update',ylabel=ylabel);ax.legend()
    fig.suptitle('Actual neural PPO training | 2,097,152 transitions per seed | surrogate results only')
    save(fig,'01_ppo_learning_curves.png')
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    colors={'constant':'#777777','saved':'#4d8ba8','selected':'#d07834'}
    labels={'constant':'Mode 6','saved':'Frozen profile','selected':'PPO'}
    seeds=sorted(result['heldouts'])
    for ax,key,title in zip(axes,['spatial_cv','roi_volume_over_k','full_roi_volume_over_k'],
                             ['Spatial depth CV','Processing removal','Full-ROI removal']):
        for j,name in enumerate(['constant','saved','selected']):
            values=[100.*result['heldouts'][s]['candidates'][name][key]/result['heldouts'][s]['candidates']['constant'][key] for s in seeds]
            positions=np.arange(len(seeds))+(j-1)*.25
            bars=ax.bar(positions,values,width=.24,color=colors[name],label=labels[name])
            ax.bar_label(bars,fmt='%.1f',padding=2,fontsize=8)
        ax.set_xticks(range(len(seeds)),seeds);ax.set(xlabel='Independent Isaac seed',ylabel='Mode 6 = 100%',title=title)
        ax.axhline(100.,color='gray',ls='--',lw=.7);ax.set_ylim(0,ax.get_ylim()[1]*1.18)
    axes[0].legend(fontsize=8)
    fig.suptitle('Isaac validation | fixed 20 N and virtual 1000 RPM | entry-separated common ROI')
    save(fig,'02_isaac_mode6_comparison.png')
    folder=root/'physics'/f'heldout_{seeds[0]}'
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    arrays=[]
    for name in ['constant','saved','selected']:
        with np.load(folder/f'{name}_processing_surface.npz') as f:arrays.append(dict(f))
    normalized=[a['depth_over_k']/a['depth_over_k'][a['roi']].mean() for a in arrays]
    cap=max(float(np.max(a[m['roi']])) for a,m in zip(normalized,arrays))
    for ax,name,grid,values in zip(axes,['constant','saved','selected'],arrays,normalized):
        extent=[grid['u_mm'][0],grid['u_mm'][-1],grid['v_mm'][0],grid['v_mm'][-1]]
        im=ax.imshow(np.where(grid['roi'],values,np.nan),origin='lower',extent=extent,vmin=0,vmax=cap,cmap='viridis')
        ax.grid(False);ax.set(title=labels[name],xlabel='Plane u [mm]',ylabel='Plane v [mm]')
        fig.colorbar(im,ax=ax,label='Depth / own ROI mean',fraction=.045)
    fig.suptitle('Completed spatial removal | identical ROI / normalized color scale | first heldout')
    save(fig,'03_isaac_processing_heatmaps.png')
    fig,axes=plt.subplots(3,1,figsize=(12,7),sharex=True,layout='constrained')
    for name in ['constant','saved','selected']:
        with np.load(folder/f'{name}_trace.npz') as f:t=dict(f)
        keep=t['time_s']>=2.;time=t['time_s'][keep]
        for ax,key in zip(axes,['target_force_n','normal_force_n','commanded_speed_mm_s']):
            ax.plot(time,t[key][keep],color=colors[name],lw=.65,label=labels[name])
    for ax,label in zip(axes,['Target force [N]','Measured force [N]','Applied feed [mm/s]']):
        ax.set_ylabel(label);ax.legend(loc='upper right')
    axes[0].set_ylim(19,21);axes[-1].set_xlabel('Active simulation time [s]')
    save(fig,'04_isaac_force_and_feed.png')


def main(root):
    root=root.resolve();audit=verify(root)
    result=json.loads((root/'result.json').read_text())
    output=ROOT/'logs/polishing_results'/datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M')
    output.mkdir(parents=True,exist_ok=False)
    candidate=root/'candidate';(candidate/'checkpoints').mkdir(parents=True,exist_ok=False)
    shutil.copy2(result['checkpoint'],candidate/'checkpoints/actor.pt')
    extra={'fixed_force_policy.json':''};torch.jit.load(str(candidate/'checkpoints/actor.pt'),_extra_files=extra)
    contract=json.loads(extra['fixed_force_policy.json'])
    (candidate/'fixed_force_policy.json').write_text(json.dumps(contract,indent=2)+'\n')
    seed=contract['seed'];update=contract['ppo_updates']
    resume=root/f'training_seed{seed}'/'checkpoints'/f'training_{update:04d}.pt'
    shutil.copy2(resume,candidate/'checkpoints/ppo_training_state.pt')
    rows=[]
    for seed,report in result['heldouts'].items():
        b=report['candidates']['constant']
        for name,m in report['candidates'].items():
            rows.append({'seed':seed,'model':name,'spatial_cv':m['spatial_cv'],
             'mode6_cv_improvement_percent':100.*m['mode6_cv_gain'],
             'mode6_processing_volume_retention_percent':100.*m['mode6_volume_ratios']['roi_volume_over_k'],
             'mode6_full_volume_retention_percent':100.*m['mode6_volume_ratios']['full_roi_volume_over_k'],
             'mode6_whole_grid_volume_retention_percent':100.*m['mode6_volume_ratios']['whole_grid_volume_over_k'],
             'mode6_time_percent':100.*m['time_s']/b['time_s'],'time_s':m['time_s'],'eligible':m['eligible']})
    ppo=[r for r in rows if r['model']=='selected']
    metrics={key:float(np.mean([r[key] for r in ppo])) for key in ppo[0]
             if key.startswith('mode6_') or key=='time_s'}
    report={'accepted':result['accepted'],'algorithm':'PPO','selected':result['selected'],
        'checkpoint':str(candidate/'checkpoints/actor.pt'),'checkpoint_sha256':sha(candidate/'checkpoints/actor.pt'),
        'ppo_training_state':str(candidate/'checkpoints/ppo_training_state.pt'),
        'training_environment':'rotary-removal and digital-command-response model, not Isaac physics',
        'validation_environment':'two independent full-path Isaac robot/contact rollouts',
        'mean_heldout_metrics':metrics,'per_seed':ppo,'all_comparisons_relative_to_matched_mode6':True,
        'entry_seconds':2.,'entry_reference_footprint_mm':18.,'force_target_n':20.,'virtual_spindle_rpm':1000.,
        'action_dimension':1,'frozen_baseline_replaced':False,'gui_deployment_prepared':False,
        'gui_trial_executed':False,'numerical_audit_passed':audit['passed'],
        'plots_directory':str(output),'limitations':['Removal is an uncalibrated rotating-disk model estimate.',
            'Two heldout seeds are limited repeatability evidence, not hardware validation.',
            'PPO training is in the disclosed approximate process model; Isaac is an independent validation environment.']}
    for path in [root/'final_report.json',candidate/'validation.json',output/'report.json']:
        path.write_text(json.dumps(report,indent=2)+'\n')
    with (output/'mode6_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plots(root,output,result)
    text=['PPO 기반 가공량 균일도 학습 / 독립 Isaac 검증',
          '고정 힘 20 N, 가상 회전 1000 RPM, 스칼라 전진 속도와 연동 체류시간. 기존 고정 모델 유지.',
          f'독립 검증 채택: {report["accepted"]}',
          f'공간 CV 평균 개선: {metrics["mode6_cv_improvement_percent"]:.4f}% (mode 6 대비)',
          f'전체 제거량 평균 유지: {metrics["mode6_full_volume_retention_percent"]:.4f}% (mode 6 대비)',
          f'가공 시간 평균: {metrics["mode6_time_percent"]:.4f}% (mode 6 대비)',
          'PPO 학습: 가공량/명령응답 근사 환경. 검증: 학습에 쓰지 않은 Isaac 전체 로봇/접촉 물리.',
          'GUI 로봇 실증 및 PPO 배포 런타임 전환은 아직 수행하지 않았다.',str(candidate/'checkpoints/actor.pt')]
    (output/'REPORT.txt').write_text('\n'.join(text)+'\n')
    (root/'REPORT.txt').write_text('\n'.join(text)+'\n')
    manifest={str(p.relative_to(candidate)):sha(p) for p in candidate.rglob('*') if p.is_file()}
    (candidate/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    import argparse
    torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
