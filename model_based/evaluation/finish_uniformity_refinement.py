#!/usr/bin/env python3
"""Protect the saved model, select on a new seed, and reject regressions."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import time
import h5py
import numpy as np
import torch
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.spatial_uniformity import ROOT,PlanarPrestonSurface
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_uniformity_priority import save,sha,processing_mask
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_uniformity_refinement import SAVED
from nrs_rl.tasks.manager_based.nrs_rl.model_based.evaluation.finish_uniformity_priority import run_batch
from nrs_rl.tasks.manager_based.nrs_rl.model_based.evaluation.uniformity_save_guard import assess_save_guard

NEW=['fine_static','dynamic_8n','dynamic_4n']
NAMES=['constant','saved']+NEW
HOLDOUTS=[2211,2212]


def audit_batch(directory):
    full=json.loads((directory/'summary.json').read_text())
    summary=json.loads((directory/'processing_summary.json').read_text())
    with h5py.File(directory/'path.h5') as h:path=h['position'][:,:3]
    surface=PlanarPrestonSurface(path,cell_size_mm=.5);mask=processing_mask(surface,path);checks=[]
    for index,name in enumerate(full['names']):
        surface.reset();metrics=summary['candidates'][name]
        with np.load(directory/f'{name}_trace.npz') as trace:
            keep=trace['time_s']>=2.;force=trace['normal_force_n'];fault=trace['safety_fault_reason']
            assert np.allclose(np.diff(trace['time_s']),.008,atol=1.e-10)
            assert np.isfinite(force).all()
            target=trace['target_force_n'];low,high=full['force_bounds_n'][index]
            assert target.min()>=low-1.e-5 and target.max()<=high+1.e-5
            assert np.max(np.abs(np.diff(target)))<=.04001
            assert np.isclose(keep.sum()*.008,metrics['time_s'])
            assert np.isclose(np.mean(fault[keep]!=0),metrics['fault_fraction'])
            assert np.allclose(trace['raw_mrr_n_mm_s'],np.where((force>=1.5)&(fault==0),force*trace['measured_speed_mm_s'],0.))
            assert metrics['return_events']==0 and metrics['reverse_mm']==0.
            for i in np.flatnonzero(keep):
                surface.deposit(trace['deposition_midpoint_mm'][i],trace['tangent_velocity_mm_s'][i],
                    float(force[i]),.008,contact=bool(force[i]>=1.5 and fault[i]==0))
            rows=len(force)
        with np.load(directory/f'{name}_processing_surface.npz') as stored:
            assert np.array_equal(mask,stored['roi'])
            assert np.allclose(surface.depth,stored['depth_over_k'],atol=1.e-10,rtol=1.e-10)
            values=surface.depth[mask]
            assert np.isclose(values.std()/values.mean(),metrics['spatial_cv'],rtol=1.e-10)
            assert np.isclose(values.sum()*surface.area,metrics['roi_volume_over_k'],rtol=1.e-10)
        checks.append({'batch':directory.name,'policy':name,'samples':rows,'passed':True})
        print('REFINEMENT_AUDITED',directory.name,name,flush=True)
    return checks


def figures(export,root,selected,accepted):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['constant','saved',selected];labels=['Mode 6','Saved checkpoint','Refined candidate']
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    colors=['#777777','#dc8a32','#247ba0']
    for ax,key,title in zip(axes,['spatial_cv','roi_volume_over_k','time_s'],
        ['Spatial CV / Mode 6 (lower is better)','Removal / Mode 6 (diagnostic)','Process time / Mode 6 (diagnostic)']):
        for i,(name,label,color) in enumerate(zip(names,labels,colors)):
            values=[]
            for seed in HOLDOUTS:
                data=json.loads((root/f'held_out_{seed}/processing_summary.json').read_text())['candidates']
                values.append(100.*data[name][key]/data['constant'][key])
            ax.bar(np.arange(2)+(i-1)*.24,values,width=.24,label=label,color=color)
        ax.axhline(100.,color='gray',ls='--',lw=.7);ax.set_xticks([0,1],[str(x) for x in HOLDOUTS])
        ax.set_xlabel('Independent Isaac seed');ax.set_ylabel('Mode 6 = 100%');ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].axhline(88.86,color='black',ls=':',label='11.14% minimum CV gain');axes[0].legend(fontsize=8)
    fig.suptitle('Refined candidate '+('accepted' if accepted else 'REJECTED; saved checkpoint retained'))
    fig.savefig(export/'00_mode6_comparison.png',dpi=160);plt.close(fig)
    directory=root/f'held_out_{HOLDOUTS[0]}';arrays=[]
    data=json.loads((directory/'processing_summary.json').read_text())['candidates']
    for name in names:
        with np.load(directory/f'{name}_processing_surface.npz') as file:arrays.append(dict(file))
    vmax=max(np.quantile(a['depth_over_k'][a['roi']],.995) for a in arrays)
    fig,axes=plt.subplots(2,3,figsize=(14,9),constrained_layout=True)
    for i,(name,label,a) in enumerate(zip(names,labels,arrays)):
        depth=a['depth_over_k'];mask=a['roi'];mean=depth[mask].mean()
        extent=[a['u_mm'][0],a['u_mm'][-1],a['v_mm'][0],a['v_mm'][-1]]
        for row,values,cap in [(0,depth,vmax),(1,depth/mean,2.)]:
            im=axes[row,i].imshow(np.where(mask,values,np.nan),origin='lower',extent=extent,vmin=0,vmax=cap,cmap='viridis')
            axes[row,i].set_xlabel('Plane u (mm)');axes[row,i].set_ylabel('Plane v (mm)')
            if i==2:fig.colorbar(im,ax=axes[row,:],label='Depth / K' if row==0 else 'Depth / mean depth')
        gain=100*(1.-data[name]['spatial_cv']/data['constant']['spatial_cv'])
        axes[0,i].set_title(f'{label}\nMode 6 spatial CV gain {gain:.2f}%');axes[1,i].set_title('Mean-normalized depth')
    fig.savefig(export/'01_processing_depth_maps.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(14,9),sharex=True,constrained_layout=True)
    rates=[];times=[]
    for name,label in zip(names,labels):
        with np.load(directory/f'{name}_trace.npz') as f:
            keep=f['time_s']>=2.;t=f['time_s'][keep]
            for ax,key in zip(axes,['normal_force_n','measured_speed_mm_s','raw_mrr_n_mm_s']):
                ax.plot(t,f[key][keep],lw=.5,label=label)
            rates.append(f['raw_mrr_n_mm_s'][keep].copy());times.append(t.copy())
    for ax,label in zip(axes,['Contact force (N)','Actual speed (mm/s)','Instantaneous removal rate / K']):
        ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0].legend();axes[-1].set_xlabel('Time since Force entry (s)')
    fig.savefig(export/'02_process_signals.png',dpi=160);plt.close(fig)
    edges=np.arange(2.,max(t[-1] for t in times)+.081,.08)
    heat=np.full((3,len(edges)-1),np.nan)
    for i,(t,rate) in enumerate(zip(times,rates)):
        count=np.histogram(t,bins=edges)[0];total=np.histogram(t,bins=edges,weights=rate)[0]
        np.divide(total,count,out=heat[i],where=count>0)
    fig,ax=plt.subplots(figsize=(13,4),constrained_layout=True);cmap=plt.colormaps['magma'].copy();cmap.set_bad('#ddd')
    im=ax.imshow(heat,aspect='auto',extent=[edges[0],edges[-1],2.5,-.5],cmap=cmap,vmin=0,vmax=np.nanquantile(heat,.995),interpolation='nearest')
    ax.set_yticks([0,1,2],labels);ax.set_xlabel('Time since Force entry (s); gray = completed')
    ax.set_title('Instantaneous removal rate: temporal diagnostic, not the spatial selection metric')
    fig.colorbar(im,ax=ax,label='Removal rate / K; 80 ms bin mean')
    fig.savefig(export/'03_temporal_heatmap.png',dpi=160);plt.close(fig)


def main(args):
    training=args.training_dir.resolve();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();torch.set_num_threads(1)
    saved_digest=sha(SAVED/'candidate/checkpoints/actor.pt')
    assert saved_digest=='e31302e7a8edb84ef713f61b080a06e5ff44b5af5ad10dc3031fc26333dc77a6'
    state={'phase':'physics_selection','selection_seed':2201,'heldout_seeds':HOLDOUTS,
        'minimum_mode6_gain_each_holdout':.1114,'report_baseline':'Mode 6 only',
        'preserved_save':str(SAVED),'preserved_sha256':saved_digest,
        'candidate_names':NAMES,'deployed':False,'new_gui_experiment_executed':False,
        'holdouts_used_for_selection':False,'training_dir':str(training),'heldouts':[],
        'rejection_policy':'Discard newly generated rejected checkpoint files; retain reports and original save'}
    save(root/'status.json',state);save(root/'predeclared_protocol.json',state)
    source_files=['scripts/train_uniformity_refinement.py','scripts/finish_uniformity_refinement.py',
        'scripts/uniformity_save_guard.py','scripts/train_uniformity_priority.py','scripts/finish_uniformity_priority.py',
        'scripts/probe_spatial_uniformity.py','scripts/spatial_uniformity_action.py','scripts/spatial_uniformity.py',
        'scripts/assess_processing_uniformity.py','scripts/uniformity_priority_metrics.py',
        'scripts/tcp_tracking_compensation.py','scripts/test_uniformity_refinement.py',
        'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/mdp/action.py']
    source_hashes={}
    for rel in source_files:
        target=root/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/rel,target);source_hashes[rel]=sha(target)
    save(root/'source_manifest.json',source_hashes)
    checkpoint_hashes={name:sha(training/'checkpoints'/f'{name}.pt') for name in NAMES}
    save(root/'checkpoint_manifest.json',checkpoint_hashes)
    screen=run_batch(training,root,'selection_2201',2201,NAMES)
    eligible=[name for name in NEW if screen['candidates'][name]['eligible']]
    if not eligible:raise RuntimeError('No eligible candidate; preserve save and inspect selection logs')
    selected=min(eligible,key=lambda name:screen['candidates'][name]['spatial_cv'])
    bundle=root/'candidate';(bundle/'checkpoints').mkdir(parents=True)
    shutil.copyfile(training/'checkpoints'/f'{selected}.pt',bundle/'checkpoints/actor.pt')
    extra={'spatial_policy.json':''};torch.jit.load(str(bundle/'checkpoints/actor.pt'),_extra_files=extra)
    contract=json.loads(extra['spatial_policy.json']);save(bundle/'spatial_policy.json',contract)
    state.update(phase='independent_physics_validation',selected=selected,
        checkpoint_sha256=checkpoint_hashes[selected],runtime_contract=contract,
        selected_before_holdouts_at=datetime.now().astimezone().isoformat(),
        selection_mode6_gain=screen['candidates'][selected]['comparison']['gains']['spatial_cv_gain'])
    save(bundle/'selection.json',state);save(root/'status.json',state)
    print('REFINEMENT_SELECTED',json.dumps({'candidate':selected,'mode6_gain':state['selection_mode6_gain']}),flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs={pool.submit(run_batch,training,root,f'held_out_{seed}',seed,['constant','saved',selected]):seed for seed in HOLDOUTS}
        for job in as_completed(jobs):
            result=job.result();m=result['candidates']
            guard=assess_save_guard(m[selected],m['saved'],m['constant'])
            item={'seed':jobs[job],'candidate_metrics':m[selected],'preserved_metrics':m['saved'],
                'mode6_metrics':m['constant'],'guard':guard,'passed':guard['passed']}
            state['heldouts'].append(item);state['heldouts'].sort(key=lambda x:x['seed'])
            save(root/'status.json',state)
            print('REFINEMENT_HOLDOUT',json.dumps({'seed':jobs[job],**guard}),flush=True)
    for name,digest in checkpoint_hashes.items():assert sha(training/'checkpoints'/f'{name}.pt')==digest
    for rel,digest in source_hashes.items():assert sha(ROOT/rel)==digest,rel
    assert sha(SAVED/'candidate/checkpoints/actor.pt')==saved_digest
    accepted=all(x['passed'] for x in state['heldouts'])
    state.update(phase='artifact_verification',accepted=accepted);save(root/'status.json',state)
    audits=[]
    for label in ['selection_2201']+[f'held_out_{s}' for s in HOLDOUTS]:audits.extend(audit_batch(root/label))
    save(root/'verification.json',{'passed':True,'runs':audits,'preserved_checkpoint_unchanged':True})
    discarded=[]
    for name in NEW:
        if accepted and name==selected:continue
        path=training/'checkpoints'/f'{name}.pt'
        assert path.is_relative_to(training/'checkpoints') and sha(path)==checkpoint_hashes[name]
        path.unlink();discarded.append({'candidate':name,'sha256':checkpoint_hashes[name],
            'reason':'not selected' if name!=selected else 'failed independent save guard'})
    if not accepted:(bundle/'checkpoints/actor.pt').unlink()
    save(root/'discarded_checkpoints.json',discarded)
    export=ROOT/'logs/polishing_results'/datetime.now().strftime('%Y%m%d_%H%M');export.mkdir(parents=True,exist_ok=False)
    state.update(phase='complete',accepted=accepted,elapsed_s=time.monotonic()-started,
        disposition='new_candidate_accepted_for_later_GUI_integration' if accepted else 'candidate_discarded_preserve_11p14_save',
        retained_checkpoint=str(bundle/'checkpoints/actor.pt' if accepted else SAVED/'candidate/checkpoints/actor.pt'),
        discarded_checkpoints=discarded,export_directory=str(export))
    save(root/'result.json',state);save(root/'status.json',state);save(export/'result.json',state)
    rows=[]
    for label in ['selection_2201']+[f'held_out_{s}' for s in HOLDOUTS]:
        summary=json.loads((root/label/'processing_summary.json').read_text());save(export/f'{label}_summary.json',summary)
        baseline=summary['candidates']['constant']
        for name,m in summary['candidates'].items():
            rows.append({'seed':summary['seed'],'role':'selection' if label.startswith('selection') else 'holdout','policy':name,
                'spatial_cv':m['spatial_cv'],'mode6_cv_gain_percent':100*(1-m['spatial_cv']/baseline['spatial_cv']),
                'mode6_volume_change_percent':100*(m['roi_volume_over_k']/baseline['roi_volume_over_k']-1),
                'mode6_time_change_percent':100*(m['time_s']/baseline['time_s']-1),
                'completed':m['completed'],'fault_fraction':m['fault_fraction'],'eligible':m['eligible']})
    with (export/'mode6_comparison.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    figures(export,root,selected,accepted)
    report=['11.14% 세이브본 보존 후 추가 학습','모든 개선율/변화율의 비교 기준: 같은 조건의 Mode 6',
        f'선정 후보: {selected}',f'채택: {accepted}',
        '채택 조건: 독립 검증 각각 Mode 6 대비 공간 CV 11.14% 이상 개선, 동일 시드 세이브본보다 나아야 함.',
        '학습: 이전 GUI 및 selection 2101 로그로 접촉력 응답을 추정하고 97개 경로 위치의 힘 배분 최적화.',
        '이전 holdout 2111/2112는 학습 입력으로 사용하지 않음. 새 선정 2201, 독립 검증 2211/2212.',
        '비회전, 단일 경로, 국소 복귀 0회. 초기 2초/고정 첫 18 mm 접촉 영역 제외. 100 N hard stop 유지.',
        '총 제거량과 시간 유지 조건 없음. h/K, V/K는 교정되지 않은 모델 추정치.',
        'GUI 실증 및 현재 ROS launch 교체는 수행하지 않음.']
    for item in state['heldouts']:
        m=item['candidate_metrics'];b=item['mode6_metrics']
        report.append(f"seed {item['seed']}: Mode 6 대비 공간 CV 개선 {item['guard']['mode6_spatial_cv_gain']*100:.3f}%; 제거량 변화 {(m['roi_volume_over_k']/b['roi_volume_over_k']-1)*100:+.3f}%; 통과 {item['passed']}")
    report += ['보존 세이브본: '+str(SAVED/'candidate/checkpoints/actor.pt'),
        '이번 작업 후 유지할 체크포인트: '+state['retained_checkpoint'],
        '폐기한 신규 후보: '+', '.join(x['candidate'] for x in discarded),
        '원본 로그 재적분 검산: 11개 실행 통과.','상세 로그: '+str(root)]
    for path in [root/'REPORT.txt',export/'REPORT.txt']:path.write_text('\n'.join(report)+'\n')
    for name in ['verification.json','discarded_checkpoints.json']:shutil.copyfile(root/name,export/name)
    save(root/'exports.json',{'directory':str(export)})
    for directory in [export,bundle]:
        save(directory/'manifest.json',{str(p.relative_to(directory)):sha(p) for p in directory.rglob('*') if p.is_file() and p.name!='manifest.json'})
    print('REFINEMENT_COMPLETE',json.dumps({'accepted':accepted,'export':str(export),'checkpoint':state['retained_checkpoint']}),flush=True)


if __name__=='__main__':
    os.environ.setdefault('MPLCONFIGDIR','/tmp/nrs_uniformity_refinement_mpl')
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--training-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);main(parser.parse_args())
