#!/usr/bin/env python3
"""Select a uniformity policy on fresh Isaac data, freeze, then test holdouts."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.spatial_uniformity import ROOT
import nrs_rl.tasks.manager_based.nrs_rl.utils.assess_processing_uniformity as processing
from nrs_rl.tasks.manager_based.nrs_rl.utils.uniformity_priority_metrics import assess_uniformity

processing.assess=assess_uniformity
NAMES=['constant','incumbent','profile_robust','profile_wide','feedback']
NEW_NAMES=NAMES[2:]


def save(path,data):
    path=Path(path);temporary=path.with_suffix('.next.json')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_batch(training,root,label,seed,names):
    tracking={name:0. if name=='constant' else .6 for name in names}
    track=root/f'{label}_tracking.json';save(track,tracking)
    command=[sys.executable,str(ROOT/'scripts/probe_spatial_uniformity.py'),
        '--headless','--device','cpu','--training-dir',str(training),
        '--output',str(root/label),'--seed',str(seed),'--names',','.join(names),
        '--tracking-parameters',str(track),'--hard-force-stop-n','100','--uniformity-only']
    save(root/f'{label}_command.json',command)
    with (root/f'{label}.log').open('w') as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,
            env={**os.environ,'MPLCONFIGDIR':'/tmp/nrs_uniformity_priority_mpl','MPLBACKEND':'Agg'})
    full=json.loads((root/label/'summary.json').read_text())
    result=processing.evaluate(root/label)
    # Entry force magnitude is not an objective. A hard stop anywhere remains
    # a disqualifying controller fault, including within the entry interval.
    for name,metrics in result['candidates'].items():
        metrics['full_run_fault_fraction']=full['candidates'][name]['fault_fraction']
        metrics['full_run_force_peak_n']=full['candidates'][name]['force_peak_n']
        metrics['eligible']=(metrics['comparison']['passed']
            and metrics['full_run_fault_fraction']==0.
            and metrics['full_run_force_peak_n']<=100.)
    save(root/label/'processing_summary.json',result)
    return result


def figures(export,root,selected):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['constant','incumbent',selected];labels=['Constant baseline','Previous GUI policy','New candidate']
    directory=root/'held_out_2111'
    arrays=[]
    for name in names:
        with np.load(directory/f'{name}_processing_surface.npz') as f:arrays.append(dict(f))
    vmax=max(np.quantile(a['depth_over_k'][a['roi']],.995) for a in arrays)
    fig,axes=plt.subplots(2,3,figsize=(14,9),constrained_layout=True)
    for i,(name,label,a) in enumerate(zip(names,labels,arrays)):
        data=a['depth_over_k'];mask=a['roi'];mean=data[mask].mean()
        extent=[a['u_mm'][0],a['u_mm'][-1],a['v_mm'][0],a['v_mm'][-1]]
        for row,values,cap in [(0,data,vmax),(1,data/mean,2.)]:
            im=axes[row,i].imshow(np.where(mask,values,np.nan),origin='lower',extent=extent,
                vmin=0.,vmax=cap,cmap='viridis')
            axes[row,i].set_xlabel('Plane u (mm)');axes[row,i].set_ylabel('Plane v (mm)')
            if i==2:fig.colorbar(im,ax=axes[row,:],label='Depth / K' if row==0 else 'Depth / mean depth')
        axes[0,i].set_title(f'{label}\nSpatial CV {data[mask].std()/mean:.4f}')
        axes[1,i].set_title('Normalized for uniformity comparison')
    fig.savefig(export/'01_processing_depth_maps.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(4,1,figsize=(14,11),sharex=True,constrained_layout=True)
    for name,label in zip(names,labels):
        with np.load(directory/f'{name}_trace.npz') as f:
            t=f['time_s'];keep=t>=2.
            for ax,key in zip(axes,['normal_force_n','measured_speed_mm_s','raw_mrr_n_mm_s','cursor_mm']):
                ax.plot(t[keep],f[key][keep],lw=.55,label=label)
    for ax,label in zip(axes,['Contact force (N)','Sliding speed (mm/s)','Removal rate / K (N mm/s)','Reference progress (mm)']):
        ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0].legend();axes[-1].set_xlabel('Time since Force entry (s)')
    fig.savefig(export/'02_processing_signals.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    seeds=[2111,2112];colors=['#777777','#dc8a32','#247ba0']
    for ax,key,title in zip(axes,['spatial_cv','roi_volume_over_k','time_s'],
        ['Processing spatial CV (lower is better)','Total removal / K (diagnostic only)','Processing time (s; diagnostic only)']):
        for i,(name,label,color) in enumerate(zip(names,labels,colors)):
            vals=[json.loads((root/f'held_out_{seed}/processing_summary.json').read_text())['candidates'][name][key] for seed in seeds]
            ax.bar(np.arange(2)+(i-1)*.24,vals,width=.24,label=label,color=color)
        ax.set_xticks([0,1],[str(s) for s in seeds]);ax.set_xlabel('Independent Isaac seed');ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(export/'00_holdout_comparison.png',dpi=160);plt.close(fig)


def main(args):
    training=args.training_dir.resolve();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();torch.set_num_threads(1)
    state={'phase':'physics_selection','objective':'minimize processing spatial CV only',
        'total_removal_gate':False,'time_gate':False,'entry_peak_gate':False,'hard_force_stop_n':100.,
        'selection_seed':2101,'heldout_seeds':[2111,2112],
        'minimum_relative_cv_gain_over_incumbent_each_holdout':.01,
        'holdouts_used_for_selection':False,'new_gui_experiment_executed':False,'deployed':False,
        'training_dir':str(training),'candidate_names':NAMES,'heldouts':[]}
    save(root/'status.json',state);save(root/'predeclared_protocol.json',state)
    sources=['scripts/train_uniformity_priority.py','scripts/finish_uniformity_priority.py',
        'scripts/uniformity_priority_metrics.py','scripts/test_uniformity_priority.py',
        'scripts/probe_spatial_uniformity.py','scripts/spatial_uniformity_action.py',
        'scripts/spatial_uniformity.py','scripts/assess_processing_uniformity.py',
        'scripts/tcp_tracking_compensation.py',
        'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/mdp/action.py']
    hashes={}
    for rel in sources:
        target=root/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/rel,target);hashes[rel]=sha(target)
    save(root/'source_manifest.json',hashes)
    input_hashes={name:sha(training/'checkpoints'/f'{name}.pt') for name in NAMES}
    save(root/'checkpoint_manifest.json',input_hashes)
    screen=run_batch(training,root,'selection_2101',2101,NAMES)
    ranking={name:screen['candidates'][name] for name in NEW_NAMES}
    eligible=[name for name in NEW_NAMES if ranking[name]['eligible']]
    if not eligible:raise RuntimeError('No complete fault-free candidate; inspect selection_2101')
    selected=min(eligible,key=lambda name:ranking[name]['spatial_cv'])
    source=training/'checkpoints'/f'{selected}.pt'
    bundle=root/'candidate';(bundle/'checkpoints').mkdir(parents=True)
    shutil.copyfile(source,bundle/'checkpoints/actor.pt')
    extra={'spatial_policy.json':''};torch.jit.load(str(source),_extra_files=extra)
    contract=json.loads(extra['spatial_policy.json']);save(bundle/'spatial_policy.json',contract)
    state.update(phase='independent_physics_validation',selected=selected,selection_metrics=ranking,
        selection_gain_over_incumbent=1.-ranking[selected]['spatial_cv']/screen['candidates']['incumbent']['spatial_cv'],
        checkpoint_sha256=sha(source),runtime_contract=contract,
        selected_before_holdouts_at=datetime.now().astimezone().isoformat())
    save(bundle/'selection.json',state);save(root/'status.json',state)
    print('UNIFORMITY_SELECTED',json.dumps({k:state[k] for k in ['selected','selection_gain_over_incumbent','checkpoint_sha256']}),flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs={pool.submit(run_batch,training,root,f'held_out_{seed}',seed,['constant','incumbent',selected]):seed for seed in state['heldout_seeds']}
        for job in as_completed(jobs):
            data=job.result();metrics=data['candidates'][selected]
            versus_old=assess_uniformity(metrics,data['candidates']['incumbent'])
            versus_constant=assess_uniformity(metrics,data['candidates']['constant'])
            passed=(metrics['eligible'] and data['candidates']['incumbent']['eligible']
                and data['candidates']['constant']['eligible']
                and versus_old['gains']['spatial_cv_gain']>=.01
                and versus_constant['gains']['spatial_cv_gain']>0.)
            result={'seed':jobs[job],'metrics':metrics,'incumbent_comparison':versus_old,
                'constant_comparison':versus_constant,'passed':passed}
            state['heldouts'].append(result);state['heldouts'].sort(key=lambda x:x['seed'])
            save(root/'status.json',state);print('UNIFORMITY_HOLDOUT',json.dumps(result),flush=True)
    for name,digest in input_hashes.items():
        if sha(training/'checkpoints'/f'{name}.pt')!=digest:raise RuntimeError('Checkpoint changed during validation')
    for rel,digest in hashes.items():
        if sha(ROOT/rel)!=digest:raise RuntimeError(f'Evaluation source changed during validation: {rel}')
    passed=all(item['passed'] for item in state['heldouts'])
    state.update(phase='complete',validation_passed=passed,elapsed_s=time.monotonic()-started,
        disposition='isolated_candidate_requires_GUI_runtime_integration' if passed else 'keep_previous_GUI_policy')
    export=ROOT/'logs/polishing_results'/datetime.now().strftime('%Y%m%d_%H%M')
    export.mkdir(parents=True,exist_ok=False);state['export_directory']=str(export)
    save(root/'result.json',state);save(root/'status.json',state);save(export/'result.json',state)
    rows=[]
    for label in ['selection_2101','held_out_2111','held_out_2112']:
        summary=json.loads((root/label/'processing_summary.json').read_text())
        save(export/f'{label}_processing_summary.json',summary)
        for name,m in summary['candidates'].items():
            rows.append({'role':'selection' if label.startswith('selection') else 'holdout',
                'seed':summary['seed'],'policy':name,**{key:m[key] for key in ['spatial_cv',
                'roi_volume_over_k','time_s','rate_cv','force_mean_n','force_peak_n','zero_fraction',
                'return_events','reverse_mm','completed','fault_fraction','entry_force_peak_n','eligible']}})
    with (export/'comparison.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    figures(export,root,selected)
    report=['가공량 균일도 우선 학습 / 독립 Isaac 검증',f'선정 후보: {selected}',f'독립 검증 통과: {passed}',
        '선정 목적: 가공 진행 구간의 공간 깊이 CV 최소화. 총 제거량/평균 힘/시간 유지 보상 없음.',
        '고정 ROI: 첫 18 mm 접촉 footprint 제외, 첫 2초 누적 제거량 제외. 나머지 0깊이/정지/복귀 유지.',
        '기존 모델과 같은 MDK 및 tracking gain 0.6 사용. constant 비교군 tracking gain 0.',
        '학습: GUI 실험 접촉 궤적 기반 힘 프로파일 최적화 + CEM 피드백 파라미터 탐색. PPO 재학습 아님.',
        '후보 선정 Isaac seed 2101. 선정 후 ckpt 고정; 독립 검증 seed 2111/2112.',
        '새 GUI 실증은 아직 수행하지 않았으며 기존 실증용 모델을 교체하지 않음.',
        '비회전, 단일 경로. h/K 및 V/K는 교정되지 않은 Preston 모델 추정치.',
        '각 독립 검증에서 기존 모델 대비 CV 1% 이상 감소를 사전 통과 기준으로 설정.']
    for item in state['heldouts']:
        old=item['incumbent_comparison']['gains'];base=item['constant_comparison']['gains']
        report.append(f"seed {item['seed']}: CV {item['metrics']['spatial_cv']:.6f}; 기존 대비 개선 {old['spatial_cv_gain']*100:.3f}%; constant 대비 {base['spatial_cv_gain']*100:.3f}%; 기존 대비 V/K 변화 {(old['volume_ratio']-1)*100:+.3f}%")
    report += ['후보 force 범위: '+str(contract['force_bounds_n'])+' N',
        '실험 hard stop: 100 N 초과. 초반 힘 피크 자체는 CV 선정 점수에 미반영.',
        '체크포인트: '+str(bundle/'checkpoints/actor.pt'),'SHA256: '+state['checkpoint_sha256'],'상세 데이터: '+str(root)]
    for target in [export/'REPORT.txt',root/'REPORT.txt']:target.write_text('\n'.join(report)+'\n')
    save(root/'exports.json',{'directory':str(export)})
    for directory in [export,bundle]:
        manifest={str(p.relative_to(directory)):sha(p) for p in directory.rglob('*') if p.is_file() and p.name!='manifest.json'}
        save(directory/'manifest.json',manifest)
    print('UNIFORMITY_COMPLETE',json.dumps({'passed':passed,'export':str(export)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
