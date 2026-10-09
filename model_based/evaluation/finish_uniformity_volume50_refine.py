#!/usr/bin/env python3
"""Fresh paired Isaac selection, saved-model guard, and best-save promotion."""
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
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.spatial_uniformity import ROOT
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_uniformity_priority import save,sha
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_uniformity_volume50_refine import SAVED,SAVED_SHA,NEW,SELECTION_SEED,HOLDOUTS,MINIMUM_ADVANTAGE
from nrs_rl.tasks.manager_based.nrs_rl.model_based.evaluation.finish_uniformity_volume50 import run_constrained,figures
from nrs_rl.tasks.manager_based.nrs_rl.model_based.evaluation.finish_uniformity_refinement import audit_batch
from nrs_rl.tasks.manager_based.nrs_rl.model_based.evaluation.uniformity_volume_guard import assess_volume_guard

SELECTION_GROUPS = {'selection_a_2401':['constant']+NEW[:2], 'selection_b_2401':['constant']+NEW[2:]}


def refinement_guard(candidate,mode6,saved):
    guard=assess_volume_guard(candidate,mode6,saved,minimum_gain=.1114)
    gains=[guard['mode6_spatial_cv_gain'],guard['saved_mode6_spatial_cv_gain']]
    guard['checks']['improvement_exceeds_predeclared_margin']=(all(v is not None for v in gains)
        and gains[0]-gains[1]>=MINIMUM_ADVANTAGE)
    guard['minimum_mode6_normalized_advantage']=MINIMUM_ADVANTAGE
    guard['passed']=all(guard['checks'].values())
    return guard


def check_preserved():
    for name,digest in json.loads((SAVED/'save_manifest.json').read_text()).items():assert sha(SAVED/name)==digest
    assert sha(SAVED/'candidate/checkpoints/actor.pt')==SAVED_SHA


def promote(root,result):
    """Only independently accepted artifacts may replace the training reference."""
    assert result['accepted'] and all(item['passed'] for item in result['heldouts'])
    pointer=ROOT/'logs/experiments/uniformity_saves/CURRENT.json'
    assert json.loads(pointer.read_text())['sha256']==SAVED_SHA,'Current reference changed during this run'
    maximum=max(h['guard']['mode6_spatial_cv_gain'] for h in result['heldouts'])*100
    stamp=datetime.now().strftime('%Y%m%d_%H%M')
    target=pointer.parent/f"mode6_{maximum:.2f}_volume50_{stamp}".replace('.','p')
    target.mkdir(exist_ok=False)
    shutil.copytree(root/'candidate',target/'candidate')
    for name in ['result.json','verification.json','predeclared_protocol.json','REPORT.txt']:
        shutil.copyfile(root/name,target/name)
    save(target/'SAVE.json',{'source':str(root),'sha256':result['checkpoint_sha256'],
         'reason':'Validated improvement over current feasible save on both held-out seeds',
         'previous_save':str(SAVED),'minimum_mode6_volume_ratio':.5,'created_at_utc':datetime.now(timezone.utc).isoformat()})
    save(target/'save_manifest.json',{str(p.relative_to(target)):sha(p) for p in target.rglob('*')
         if p.is_file() and p.name!='save_manifest.json'})
    for name,digest in json.loads((target/'save_manifest.json').read_text()).items():assert sha(target/name)==digest
    save(pointer,{'save_directory':str(target),'checkpoint':str(target/'candidate/checkpoints/actor.pt'),
                  'sha256':result['checkpoint_sha256'],'previous_save':str(SAVED),'minimum_mode6_volume_ratio':.5,
                  'updated_at_utc':datetime.now(timezone.utc).isoformat(),
                  'purpose':'Current reference for training and comparison; does not change ROS/GUI deployment'})
    return target


def main(args):
    training=args.training_dir.resolve();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);started=time.monotonic();check_preserved()
    state={'phase':'physics_selection','selection_seed':SELECTION_SEED,'selection_groups':SELECTION_GROUPS,
        'heldout_seeds':HOLDOUTS,'minimum_mode6_volume_ratio_each_run':.5,
        'volume_constraint_scopes':['processing_roi','full_roi'],'minimum_mode6_cv_gain_each_holdout':.1114,
        'minimum_mode6_normalized_advantage_over_save':MINIMUM_ADVANTAGE,
        'report_baseline':'Mode 6 only','preserved_save':str(SAVED),'preserved_sha256':SAVED_SHA,
        'training_dir':str(training),'candidate_names':['constant','saved']+NEW,
        'deployed':False,'new_gui_experiment_executed':False,'holdouts_used_for_selection':False,
        'selection_rule':'Maximize paired Mode 6 CV gain among volume-feasible new candidates; each selection group has its own Mode 6',
        'acceptance_rule':'Both independent runs: >=50% removal in both ROIs, CV gain >=11.14% vs Mode 6, >saved by 0.001 Mode 6-normalized CV',
        'heldouts':[],'rejection_policy':'Delete unselected or rejected new actors; preserve saved checkpoints and evidence'}
    save(root/'status.json',state);save(root/'predeclared_protocol.json',state)
    sources=['scripts/'+name for name in ['train_uniformity_volume50_refine.py','finish_uniformity_volume50_refine.py',
        'test_uniformity_volume50_refine.py','train_uniformity_volume50.py','finish_uniformity_volume50.py',
        'train_uniformity_refinement.py','finish_uniformity_refinement.py','train_uniformity_priority.py',
        'finish_uniformity_priority.py','uniformity_volume_guard.py','uniformity_priority_metrics.py',
        'assess_processing_uniformity.py','probe_spatial_uniformity.py','spatial_uniformity.py',
        'spatial_uniformity_action.py','tcp_tracking_compensation.py']]
    sources+=['source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/mdp/action.py',
              'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/utils/preston_surface.py']
    hashes={}
    for rel in sources:
        dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/rel,dest);hashes[rel]=sha(dest)
    save(root/'source_manifest.json',hashes)
    checkpoints={name:sha(training/'checkpoints'/f'{name}.pt') for name in state['candidate_names']}
    save(root/'checkpoint_manifest.json',checkpoints)
    screens={}
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs={pool.submit(run_constrained,training,root,label,SELECTION_SEED,names):label
              for label,names in SELECTION_GROUPS.items()}
        for job in as_completed(jobs):
            screens[jobs[job]]=job.result()
            print('REFINE_SELECTION_GROUP',jobs[job],flush=True)
    feasible=[]
    selection_metrics={}
    for label,screen in screens.items():
        for name,m in screen['candidates'].items():
            if name=='constant':continue
            selection_metrics[name]={'batch':label,'metrics':m}
            if m['volume_constraint']['passed']:feasible.append(name)
    save(root/'selection_metrics.json',selection_metrics)
    if not feasible:
        state.update(phase='selection_rejected',accepted=False,reason='No volume-feasible candidate')
        save(root/'result.json',state);save(root/'status.json',state)
        raise RuntimeError('No volume-feasible candidate; saved model preserved, inspect and retrain')
    selected=max(feasible,key=lambda name:selection_metrics[name]['metrics']['volume_constraint']['mode6_spatial_cv_gain'])
    bundle=root/'candidate';(bundle/'checkpoints').mkdir(parents=True)
    shutil.copyfile(training/'checkpoints'/f'{selected}.pt',bundle/'checkpoints/actor.pt')
    extra={'spatial_policy.json':''};torch.jit.load(str(bundle/'checkpoints/actor.pt'),_extra_files=extra)
    contract=json.loads(extra['spatial_policy.json']);save(bundle/'spatial_policy.json',contract)
    state.update(phase='independent_physics_validation',selected=selected,checkpoint_sha256=checkpoints[selected],
        runtime_contract=contract,selected_before_holdouts_at=datetime.now().astimezone().isoformat(),
        selection_metrics=selection_metrics[selected])
    save(bundle/'selection.json',state);save(root/'status.json',state)
    print('REFINE_SELECTED',json.dumps({'candidate':selected,'guard':selection_metrics[selected]['metrics']['volume_constraint']}),flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs={pool.submit(run_constrained,training,root,f'held_out_{s}',s,['constant','saved',selected]):s for s in HOLDOUTS}
        for job in as_completed(jobs):
            metrics=job.result()['candidates'];guard=refinement_guard(metrics[selected],metrics['constant'],metrics['saved'])
            state['heldouts'].append({'seed':jobs[job],'candidate_metrics':metrics[selected],
                'preserved_metrics':metrics['saved'],'mode6_metrics':metrics['constant'],'guard':guard,'passed':guard['passed']})
            state['heldouts'].sort(key=lambda item:item['seed']);save(root/'status.json',state)
            print('REFINE_HOLDOUT',json.dumps({'seed':jobs[job],**guard}),flush=True)
    accepted=all(h['passed'] for h in state['heldouts'])
    state.update(phase='artifact_verification',accepted=accepted);save(root/'status.json',state)
    for name,digest in checkpoints.items():assert sha(training/'checkpoints'/f'{name}.pt')==digest
    for rel,digest in hashes.items():assert sha(ROOT/rel)==digest,rel
    check_preserved()
    audits=[];full_checks=[];labels=list(SELECTION_GROUPS)+[f'held_out_{s}' for s in HOLDOUTS]
    for label in labels:
        audits.extend(audit_batch(root/label))
        m=json.loads((root/label/'processing_summary.json').read_text())['candidates']
        for name,metrics in m.items():
            with np.load(root/label/f'{name}_surface.npz') as data:
                volume=float(data['depth_over_k'][data['roi']].sum()*data['cell_area_mm2'])
                assert np.isclose(volume,metrics['full_roi_volume_over_k'],rtol=1.e-10)
            full_checks.append({'batch':label,'policy':name,'passed':True})
    save(root/'verification.json',{'passed':True,'raw_reintegration':audits,'full_volume_checks':full_checks,
        'preserved_checkpoint_unchanged':True,'checkpoint_and_source_hashes_unchanged':True})
    discarded=[]
    for name in NEW:
        if accepted and name==selected:continue
        path=training/'checkpoints'/f'{name}.pt';assert path.is_relative_to(training/'checkpoints') and sha(path)==checkpoints[name]
        path.unlink();discarded.append({'candidate':name,'sha256':checkpoints[name],
            'reason':'not selected; not independently validated' if name!=selected else 'failed independent volume/save guard'})
    if not accepted:(bundle/'checkpoints/actor.pt').unlink()
    save(root/'discarded_checkpoints.json',discarded)
    export=ROOT/'logs/polishing_results'/datetime.now().strftime('%Y%m%d_%H%M');export.mkdir(parents=True,exist_ok=False)
    state.update(phase='complete',accepted=accepted,elapsed_s=time.monotonic()-started,
        disposition='accepted_for_later_GUI_integration' if accepted else 'rejected_preserve_current_save',
        retained_checkpoint=str(bundle/'checkpoints/actor.pt' if accepted else SAVED/'candidate/checkpoints/actor.pt'),
        discarded_checkpoints=discarded,export_directory=str(export))
    for dest in [root/'result.json',root/'status.json',export/'result.json']:save(dest,state)
    rows=[]
    for label in labels:
        summary=json.loads((root/label/'processing_summary.json').read_text());save(export/f'{label}_summary.json',summary)
        baseline=summary['candidates']['constant']
        for name,m in summary['candidates'].items():
            rows.append({'batch':label,'seed':summary['seed'],'role':'selection' if label.startswith('selection') else 'holdout',
                'policy':name,'spatial_cv':m['spatial_cv'],'mode6_cv_gain_percent':100*(1-m['spatial_cv']/baseline['spatial_cv']),
                'mode6_processing_volume_change_percent':100*(m['roi_volume_over_k']/baseline['roi_volume_over_k']-1),
                'mode6_full_volume_change_percent':100*(m['full_roi_volume_over_k']/baseline['full_roi_volume_over_k']-1),
                'completed':m['completed'],'fault_fraction':m['fault_fraction'],
                'volume_floor_passed':m['volume_constraint']['passed']})
    with (export/'mode6_comparison.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    figures(export,root,selected,accepted,holdouts=HOLDOUTS,selection_dirs=list(SELECTION_GROUPS),
            scatter_offsets={'fine145_52':(-90,-10),'fine193_52':(20,25),'fine145_55':(45,-10),'timed193_52':(-90,-35)})
    validation={'validation_status':'passed_independent_headless_Isaac' if accepted else 'rejected',
        'checkpoint_sha256':checkpoints[selected],'comparison_baseline':'Mode 6 only','minimum_mode6_volume_ratio':.5,
        'heldouts':[{'seed':h['seed'],**h['guard']} for h in state['heldouts']],
        'raw_reintegration_runs':len(audits),'embedded_metadata_status_note':'Actor frozen before validation; this file records the subsequent outcome',
        'training_method':'model-based constrained force-profile and feed-timing optimization, not PPO',
        'control_comparison':{'mode6_tracking_gain':0.,'candidate_tracking_gain':.6,'inner_normal_force_weights_unchanged':True},
        'deployed':False,'current_GUI_runtime_compatible':False,'source_result':str(root/'result.json')}
    save(bundle/'validation.json',validation);save(export/'model_validation.json',validation)
    report=['총가공량 50% 유지 조건하의 추가 균일도 학습','모든 수치의 비교 기준: 같은 실행 조건의 Mode 6',
        '기준 세이브본: '+str(SAVED),f'선정 후보: {selected}',f'독립 검증 채택: {accepted}',
        '학습: 1 mm 제거량 격자, 최신 selection 2301의 힘 응답, 145/193개 힘 배분 절점.',
        '이전 holdout 2111/2112/2211/2212/2311/2312는 학습 입력으로 사용하지 않음.',
        '학습 제거량 하한 52% 및 55%, 힘/속도 결합을 조정한 체류시간 후보 포함. PPO 재학습 아님.',
        '선정: seed 2401의 두 독립 배치(각각 자체 Mode 6 비교군). 선정 후 고정, 검증 seed 2411/2412.',
        '채택: 두 독립 검증 모두 가공 구간/전체 제거량 >= Mode 6의 50%, Mode 6 CV 개선 >=11.14%,',
        '동일 조건 세이브본보다 Mode 6로 정규화한 CV가 0.001 이상 낮아야 함.',
        '비회전, 단일 경로, 국소 복귀 없음. 초기 2초/고정 첫 18 mm footprint 제외 후 공간 CV 평가.',
        '100 N hard stop 유지. 초기 힘 피크는 목적함수가 아님. 가공량은 미교정 V/K 모델 추정값.',
        'GUI 실증 및 ROS 실행 설정 변경은 수행하지 않음.']
    for item in state['heldouts']:
        g=item['guard'];report.append(f"seed {item['seed']}: Mode 6 대비 CV 개선 {100*g['mode6_spatial_cv_gain']:.3f}%; 가공 구간 제거량 변화 {100*(g['mode6_volume_ratio']-1):+.3f}%; 전체 제거량 변화 {100*(g['mode6_full_volume_ratio']-1):+.3f}%; 채택 조건 통과 {item['passed']}")
    report+=['원본 로그 재적분 검산: '+str(len(audits))+'개 실행 통과.','유지 체크포인트: '+state['retained_checkpoint'],
             '이전 세이브본 파일과 해시는 유지됨.','상세 로그: '+str(root)]
    for dest in [root/'REPORT.txt',export/'REPORT.txt']:dest.write_text('\n'.join(report)+'\n')
    for name in ['verification.json','discarded_checkpoints.json']:shutil.copyfile(root/name,export/name)
    for directory in [export,bundle]:
        save(directory/'manifest.json',{str(p.relative_to(directory)):sha(p) for p in directory.rglob('*')
            if p.is_file() and p.name!='manifest.json'})
    if accepted:
        promoted=promote(root,state)
        save(root/'save_promotion.json',{'new_save':str(promoted),'previous_save':str(SAVED),'checkpoint_sha256':checkpoints[selected]})
    else:assert json.loads((SAVED.parent/'CURRENT.json').read_text())['sha256']==SAVED_SHA
    save(root/'exports.json',{'directory':str(export)})
    print('VOLUME50_REFINE_COMPLETE',json.dumps({'accepted':accepted,'export':str(export),'checkpoint':state['retained_checkpoint']}),flush=True)


if __name__=='__main__':
    os.environ.setdefault('MPLCONFIGDIR','/tmp/nrs_uniformity_volume50_refine_mpl')
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--training-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);main(parser.parse_args())
