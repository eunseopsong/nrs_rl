#!/usr/bin/env python3
"""Select and independently validate policies with a 50% Mode 6 dose floor."""
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
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.spatial_uniformity import ROOT
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_uniformity_priority import save, sha
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_uniformity_volume50 import NEW, SELECTION_SEED, HOLDOUTS, SAVED, SAVED_SHA, PREVIOUS_ACTOR, PREVIOUS_SHA
from nrs_rl.tasks.manager_based.nrs_rl.model_based.evaluation.finish_uniformity_priority import run_batch
from nrs_rl.tasks.manager_based.nrs_rl.model_based.evaluation.finish_uniformity_refinement import audit_batch
from nrs_rl.tasks.manager_based.nrs_rl.model_based.evaluation.uniformity_volume_guard import assess_volume_guard

NAMES = ['constant', 'saved'] + NEW


def run_constrained(training, root, label, seed, names):
    result = run_batch(training, root, label, seed, names)
    full = json.loads((root / label / 'summary.json').read_text())['candidates']
    for name, m in result['candidates'].items():
        m['full_roi_volume_over_k'] = full[name]['roi_volume_over_k']
    for name, m in result['candidates'].items():
        m['volume_constraint'] = assess_volume_guard(m, result['candidates']['constant'])
    save(root / label / 'processing_summary.json', result)
    return result


def figures(export, root, selected, accepted, *, holdouts=None, selection_dirs=None, scatter_offsets=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    holdouts = HOLDOUTS if holdouts is None else holdouts
    selection_dirs = [f'selection_{SELECTION_SEED}'] if selection_dirs is None else selection_dirs
    names = ['constant', 'saved', selected]
    labels = ['Mode 6', 'Saved feasible model', 'Constrained candidate']
    colors = ['#777777','#dc8a32','#247ba0']
    batches = [json.loads((root / f'held_out_{s}/processing_summary.json').read_text())['candidates'] for s in holdouts]
    fig, axes = plt.subplots(1, 3, figsize=(14,4.5), constrained_layout=True)
    for ax, key, title in zip(axes, ['spatial_cv','roi_volume_over_k','full_roi_volume_over_k'],
            ['Spatial CV (lower is better)','Processing removal (floor: 50%)','Whole-path removal (floor: 50%)']):
        for i, (name,label,color) in enumerate(zip(names,labels,colors)):
            values = [100.*m[name][key]/m['constant'][key] for m in batches]
            ax.bar(np.arange(2)+(i-1)*.24, values, width=.24, label=label, color=color)
        ax.axhline(100., color='gray', ls='--', lw=.8)
        if key != 'spatial_cv': ax.axhline(50., color='red', ls=':', lw=1.5)
        ax.set_xticks([0,1], [str(s) for s in holdouts]); ax.set_xlabel('Independent Isaac seed')
        ax.set_ylabel('Mode 6 = 100%'); ax.set_title(title); ax.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Volume-constrained policy: ' + ('accepted' if accepted else 'rejected; save retained'))
    fig.savefig(export / '00_mode6_comparison.png', dpi=160); plt.close(fig)
    directory = root / f'held_out_{holdouts[0]}'
    arrays = []
    for name in names:
        with np.load(directory / f'{name}_processing_surface.npz') as file: arrays.append(dict(file))
    vmax = max(np.quantile(a['depth_over_k'][a['roi']],.995) for a in arrays)
    fig, axes = plt.subplots(2,3, figsize=(14,9), constrained_layout=True)
    for i,(name,label,a) in enumerate(zip(names,labels,arrays)):
        depth,mask = a['depth_over_k'],a['roi']; mean = depth[mask].mean()
        extent = [a['u_mm'][0],a['u_mm'][-1],a['v_mm'][0],a['v_mm'][-1]]
        for row,values,cap in [(0,depth,vmax),(1,depth/mean,2.)]:
            im = axes[row,i].imshow(np.where(mask,values,np.nan),origin='lower',extent=extent,vmin=0,vmax=cap,cmap='viridis')
            axes[row,i].set_xlabel('Plane u (mm)'); axes[row,i].set_ylabel('Plane v (mm)')
            if i == 2: fig.colorbar(im, ax=axes[row,:], label='Depth / K' if row == 0 else 'Depth / mean')
        gain = 100.*(1.-batches[0][name]['spatial_cv']/batches[0]['constant']['spatial_cv'])
        volume = 100.*batches[0][name]['roi_volume_over_k']/batches[0]['constant']['roi_volume_over_k']
        axes[0,i].set_title(f'{label}\nMode 6 CV gain {gain:.2f}%\nRemoval retained {volume:.2f}%', fontsize=11)
        axes[1,i].set_title('Mean-normalized depth')
    fig.savefig(export / '01_processing_depth_maps.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(3,1, figsize=(14,9), sharex=True, constrained_layout=True)
    times,rates = [],[]
    for name,label,color in zip(names,labels,colors):
        with np.load(directory / f'{name}_trace.npz') as f:
            keep = f['time_s'] >= 2.; t = f['time_s'][keep]
            for ax,key in zip(axes,['normal_force_n','measured_speed_mm_s','raw_mrr_n_mm_s']):
                ax.plot(t,f[key][keep],lw=.5,label=label,color=color)
            times.append(t.copy()); rates.append(f['raw_mrr_n_mm_s'][keep].copy())
    for ax,label in zip(axes,['Contact force (N)','Actual sliding speed (mm/s)','Removal rate / K']):
        ax.set_ylabel(label); ax.grid(alpha=.2)
    axes[0].legend(); axes[-1].set_xlabel('Time since Force entry (s)')
    fig.savefig(export / '02_process_signals.png',dpi=160); plt.close(fig)
    edges = np.arange(2.,max(t[-1] for t in times)+.081,.08)
    heat = np.full((3,len(edges)-1),np.nan)
    for i,(t,rate) in enumerate(zip(times,rates)):
        count = np.histogram(t,bins=edges)[0]; total = np.histogram(t,bins=edges,weights=rate)[0]
        np.divide(total,count,out=heat[i],where=count>0)
    fig,ax = plt.subplots(figsize=(13,4),constrained_layout=True)
    cmap = plt.colormaps['magma'].copy(); cmap.set_bad('#ddd')
    im = ax.imshow(heat,aspect='auto',extent=[edges[0],edges[-1],2.5,-.5],cmap=cmap,
                   vmin=0,vmax=np.nanquantile(heat,.995),interpolation='nearest')
    ax.set_yticks([0,1,2],labels); ax.set_xlabel('Time since Force entry (s); gray = completed')
    ax.set_title('Instantaneous removal rate; separate from spatial CV objective')
    fig.colorbar(im,ax=ax,label='Removal rate / K; 80 ms bin mean')
    fig.savefig(export / '03_temporal_heatmap.png',dpi=160); plt.close(fig)
    selection_rows = []
    for label in selection_dirs:
        selection = json.loads((root / label / 'processing_summary.json').read_text())['candidates']
        for name,m in selection.items():
            if name == 'constant' and selection_rows: continue
            selection_rows.append((name,100*m['roi_volume_over_k']/selection['constant']['roi_volume_over_k'],
                                   100*(1-m['spatial_cv']/selection['constant']['spatial_cv'])))
    fig,ax = plt.subplots(figsize=(9,6),constrained_layout=True)
    label_offsets = {'constant':(7,5), 'saved':(8,-22), 'profile97_55':(38,-35),
                     'profile97_62':(18,-10), 'profile145_55':(18,24), 'scaled_55':(-90,-5)}
    label_offsets.update(scatter_offsets or {})
    for name,v,gain in selection_rows:
        ax.scatter(v,gain,s=65,color='#247ba0' if name==selected else '#888888')
        ax.annotate(name,(v,gain),xytext=label_offsets.get(name,(6,5)),textcoords='offset points',
                    fontsize=8,arrowprops={'arrowstyle':'-', 'color':'#777777', 'lw':.6})
    ax.axvspan(0,50,color='#ffdddd'); ax.axvline(50,color='red',ls='--')
    ax.set_xlim(25,110); ax.set_ylim(min(-.5,min(row[2] for row in selection_rows)-.5),
                                   max(14.,max(row[2] for row in selection_rows)+1.2))
    ax.set_xlabel('Processing removal retained / Mode 6 (%)')
    ax.set_ylabel('Spatial CV reduction vs Mode 6 (%)'); ax.grid(alpha=.2)
    ax.set_title('Selection candidates: dose-uniformity tradeoff (not independent validation)')
    fig.savefig(export / '04_selection_tradeoff.png',dpi=160); plt.close(fig)


def main(args):
    training,root = args.training_dir.resolve(),args.output.resolve()
    root.mkdir(parents=True,exist_ok=False); started = time.monotonic(); torch.set_num_threads(1)
    assert sha(SAVED / 'candidate/checkpoints/actor.pt') == SAVED_SHA
    assert sha(PREVIOUS_ACTOR) == PREVIOUS_SHA
    state = {'phase':'physics_selection','selection_seed':SELECTION_SEED,'heldout_seeds':HOLDOUTS,
             'minimum_mode6_volume_ratio_each_run':.5,'volume_constraint_scopes':['processing_roi','full_roi'],
             'minimum_mode6_cv_gain_each_holdout':.1114,'report_baseline':'Mode 6 only',
             'preserved_feasible_checkpoint':str(SAVED / 'candidate/checkpoints/actor.pt'),
             'preserved_cv_only_checkpoint':str(PREVIOUS_ACTOR),'candidate_names':NAMES,
             'training_dir':str(training),'deployed':False,'new_gui_experiment_executed':False,
             'holdouts_used_for_selection':False,'heldouts':[],
             'selection_rule':'Lowest processing CV among valid new candidates satisfying both 50% removal gates',
             'acceptance_rule':'Both fresh holdouts satisfy 50% dose floors, Mode 6 CV gain >= 11.14%, improve saved feasible model',
             'rejection_policy':'Delete new unselected/rejected actor files only; retain source models, profiles and results'}
    save(root / 'status.json',state); save(root / 'predeclared_protocol.json',state)
    sources = ['scripts/'+name for name in [
        'train_uniformity_volume50.py','finish_uniformity_volume50.py','uniformity_volume_guard.py','test_uniformity_volume50.py',
        'train_uniformity_priority.py','train_uniformity_refinement.py','finish_uniformity_priority.py','finish_uniformity_refinement.py',
        'probe_spatial_uniformity.py','spatial_uniformity_action.py','spatial_uniformity.py','assess_processing_uniformity.py',
        'uniformity_priority_metrics.py','tcp_tracking_compensation.py']]
    sources += ['source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/mdp/action.py',
                'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/utils/preston_surface.py']
    hashes = {}
    for name in sources:
        target = root / 'source_snapshot' / name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT / name,target); hashes[name] = sha(target)
    save(root / 'source_manifest.json',hashes)
    checkpoints = {name:sha(training / 'checkpoints' / f'{name}.pt') for name in NAMES}
    save(root / 'checkpoint_manifest.json',checkpoints)
    screen = run_constrained(training,root,f'selection_{SELECTION_SEED}',SELECTION_SEED,NAMES)
    feasible = [name for name in NEW if screen['candidates'][name]['volume_constraint']['passed']]
    if not feasible:
        save(root / 'result.json',{**state,'phase':'selection_rejected','reason':'No candidate satisfies volume constraints'})
        raise RuntimeError('No feasible new candidate; inspect selection and retrain, saved models unchanged')
    selected = min(feasible,key=lambda name:screen['candidates'][name]['spatial_cv'])
    bundle = root / 'candidate'; (bundle / 'checkpoints').mkdir(parents=True)
    shutil.copyfile(training / 'checkpoints' / f'{selected}.pt',bundle / 'checkpoints/actor.pt')
    extra = {'spatial_policy.json':''}; torch.jit.load(str(bundle / 'checkpoints/actor.pt'),_extra_files=extra)
    contract = json.loads(extra['spatial_policy.json']); save(bundle / 'spatial_policy.json',contract)
    state.update(phase='independent_physics_validation',selected=selected,checkpoint_sha256=checkpoints[selected],
                 selected_before_holdouts_at=datetime.now().astimezone().isoformat(),runtime_contract=contract,
                 selection_guard=screen['candidates'][selected]['volume_constraint'])
    save(bundle / 'selection.json',state); save(root / 'status.json',state)
    print('VOLUME50_SELECTED',json.dumps({'candidate':selected,**state['selection_guard']}),flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {pool.submit(run_constrained,training,root,f'held_out_{s}',s,['constant','saved',selected]):s for s in HOLDOUTS}
        for job in as_completed(jobs):
            m = job.result()['candidates']; guard = assess_volume_guard(m[selected],m['constant'],m['saved'],minimum_gain=.1114)
            state['heldouts'].append({'seed':jobs[job],'candidate_metrics':m[selected],'mode6_metrics':m['constant'],
                                     'preserved_metrics':m['saved'],'guard':guard,'passed':guard['passed']})
            state['heldouts'].sort(key=lambda item:item['seed']); save(root / 'status.json',state)
            print('VOLUME50_HOLDOUT',json.dumps({'seed':jobs[job],**guard}),flush=True)
    for name,digest in checkpoints.items(): assert sha(training / 'checkpoints' / f'{name}.pt') == digest
    for name,digest in hashes.items(): assert sha(ROOT / name) == digest,name
    assert sha(SAVED / 'candidate/checkpoints/actor.pt') == SAVED_SHA
    assert sha(PREVIOUS_ACTOR) == PREVIOUS_SHA
    accepted = all(item['passed'] for item in state['heldouts'])
    state.update(phase='artifact_verification',accepted=accepted); save(root / 'status.json',state)
    audits = []
    labels = [f'selection_{SELECTION_SEED}']+[f'held_out_{s}' for s in HOLDOUTS]
    for label in labels: audits.extend(audit_batch(root / label))
    # Full-path sums are independently checked against their saved fine-grid maps.
    full_checks = []
    for label in labels:
        summary = json.loads((root / label / 'processing_summary.json').read_text())
        for name,m in summary['candidates'].items():
            with np.load(root / label / f'{name}_surface.npz') as f:
                volume = float(f['depth_over_k'][f['roi']].sum() * f['cell_area_mm2'])
                assert np.isclose(volume,m['full_roi_volume_over_k'],rtol=1.e-10)
            full_checks.append({'batch':label,'policy':name,'passed':True})
    save(root / 'verification.json',{'passed':True,'raw_reintegration':audits,'full_volume_checks':full_checks,
                                    'preserved_checkpoints_unchanged':True,'checkpoint_and_source_hashes_unchanged':True})
    discarded = []
    for name in NEW:
        if accepted and name == selected: continue
        path = training / 'checkpoints' / f'{name}.pt'
        assert path.is_relative_to(training / 'checkpoints') and sha(path) == checkpoints[name]
        reason = 'not selected; not independently validated' if name != selected else 'failed independent volume/save guard'
        path.unlink(); discarded.append({'candidate':name,'sha256':checkpoints[name],'reason':reason})
    if not accepted: (bundle / 'checkpoints/actor.pt').unlink()
    save(root / 'discarded_checkpoints.json',discarded)
    export = ROOT / 'logs/polishing_results' / datetime.now().strftime('%Y%m%d_%H%M')
    export.mkdir(parents=True,exist_ok=False)
    state.update(phase='complete',accepted=accepted,elapsed_s=time.monotonic()-started,
                 disposition='accepted_for_later_GUI_integration' if accepted else 'rejected_preserve_feasible_save',
                 retained_checkpoint=str(bundle / 'checkpoints/actor.pt' if accepted else SAVED / 'candidate/checkpoints/actor.pt'),
                 discarded_checkpoints=discarded,export_directory=str(export))
    for path in [root / 'result.json',root / 'status.json',export / 'result.json']: save(path,state)
    rows = []
    for label in labels:
        summary = json.loads((root / label / 'processing_summary.json').read_text()); save(export / f'{label}_summary.json',summary)
        baseline = summary['candidates']['constant']
        for name,m in summary['candidates'].items():
            rows.append({'seed':summary['seed'],'role':'selection' if label.startswith('selection') else 'holdout',
                         'policy':name,'spatial_cv':m['spatial_cv'],
                         'mode6_cv_gain_percent':100*(1-m['spatial_cv']/baseline['spatial_cv']),
                         'mode6_processing_volume_change_percent':100*(m['roi_volume_over_k']/baseline['roi_volume_over_k']-1),
                         'mode6_full_volume_change_percent':100*(m['full_roi_volume_over_k']/baseline['full_roi_volume_over_k']-1),
                         'volume_floor_passed':m['volume_constraint']['passed'],'completed':m['completed'],
                         'fault_fraction':m['fault_fraction']})
    with (export / 'mode6_comparison.csv').open('w',newline='') as file:
        writer = csv.DictWriter(file,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    figures(export,root,selected,accepted)
    report = ['총가공량 유지 제약을 포함한 공간 균일도 학습', '비교 기준: 같은 조건의 Mode 6만 사용',
        f'선정 후보: {selected}', f'독립 검증 채택: {accepted}',
        '목표: 가공 구간 공간 깊이 CV 최소화; 가공 구간 및 전체 경로 제거량 모두 Mode 6의 50% 이상.',
        '학습: GUI 및 이전 selection 2201의 접촉 궤적/힘 응답을 이용한 제약 최적화. PPO 재학습 아님.',
        '학습 하한 55%/62% 후보, 97/145개 힘 프로파일 절점, 이전 균일도 모델의 힘 크기 재보정 후보 비교.',
        '이전 holdout 2111/2112/2211/2212는 학습에 사용하지 않음. 새 선정 2301 후 고정, 독립 검증 2311/2312.',
        '비회전, 단일 전진 경로, 별도 재가공 없음. 초기 2초/고정 첫 18 mm footprint 제외 후 CV 평가.',
        '100 N hard stop, 제어기 fault 없는 완료 실행만 채택. 초반 힘 피크를 CV 보상에 넣지 않음.',
        '가공량은 교정되지 않은 모델 추정 V/K이며 실제 물리 제거량 측정치가 아님.',
        'GUI 실증/ROS launch 교체는 아직 수행하지 않음. 현재 런타임에 새 ckpt를 직접 끼우면 안 됨.']
    for item in state['heldouts']:
        g = item['guard']
        report.append(f"seed {item['seed']}: Mode 6 대비 CV 개선 {100*g['mode6_spatial_cv_gain']:.3f}%; 가공 구간 제거량 변화 {100*(g['mode6_volume_ratio']-1):+.3f}%; 전체 제거량 변화 {100*(g['mode6_full_volume_ratio']-1):+.3f}%; 통과 {item['passed']}")
    report += ['원본 로그 재적분 검산 통과 실행 수: '+str(len(audits)),
               '유지 체크포인트: '+state['retained_checkpoint'],'상세 로그: '+str(root)]
    for path in [root / 'REPORT.txt',export / 'REPORT.txt']: path.write_text('\n'.join(report)+'\n')
    for name in ['verification.json','discarded_checkpoints.json']: shutil.copyfile(root / name,export / name)
    save(root / 'exports.json',{'directory':str(export)})
    for directory in [export,bundle]:
        save(directory / 'manifest.json',{str(p.relative_to(directory)):sha(p) for p in directory.rglob('*')
                                         if p.is_file() and p.name != 'manifest.json'})
    print('VOLUME50_COMPLETE',json.dumps({'accepted':accepted,'export':str(export),'checkpoint':state['retained_checkpoint']}),flush=True)


if __name__ == '__main__':
    os.environ.setdefault('MPLCONFIGDIR','/tmp/nrs_uniformity_volume50_mpl')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
