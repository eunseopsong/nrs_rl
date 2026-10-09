#!/usr/bin/env python3
"""Audit a spatial policy GUI run against the existing calibrated Mode 6 run."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nrs_rl.tasks.manager_based.nrs_rl.utils.reporting.compare_isaac_tcp_runs import ROOT, load_run, close, dump, sha, csv_rows

spec=importlib.util.spec_from_file_location('gui_spatial_surface',
    ROOT/'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/utils/preston_surface.py')
surface_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(surface_module)
PlanarPrestonSurface=surface_module.PlanarPrestonSurface
COLORS={'mode5':'#2166ac','mode6':'#d6604d'}
LABELS={'mode5':'Mode 5: spatial policy + tracking','mode6':'Mode 6: constant force/feed'}


def processing(run,output):
    surface=PlanarPrestonSurface(run['reference'][:,:3],cell_size_mm=.5)
    if not np.array_equal(surface.roi,run['surface']['roi']):raise ValueError('Reference ROI mismatch')
    path=run['reference'][:,:3]
    arc=np.r_[0.,np.linalg.norm(np.diff(path,axis=0),axis=1).cumsum()]
    roi=surface.roi.copy()
    for point in path[arc<=18.]:
        rows,cols,_,_,weights,_=surface._footprint((point-surface.origin)@surface.basis.T)
        roi[rows[weights>0],cols[weights>0]]=False
    xyz,t,dt=run['xyz'],run['t'],run['dt']
    delta=np.diff(xyz,axis=0,prepend=xyz[:1]);velocity=delta/dt[:,None]
    velocity-=(velocity@surface.normal)[:,None]*surface.normal
    entry=t<2.;keep=~entry
    for i in np.flatnonzero(entry):
        surface.deposit(xyz[i]-.5*delta[i],velocity[i],run['force'][i],dt[i],contact=run['force'][i]>=1.5)
    remaining=run['surface']['depth_over_k']-surface.depth
    if remaining.min() < -1.e-9:raise ValueError('Entry subtraction exceeds recorded removal')
    remaining=np.maximum(remaining,0.);values=remaining[roi]
    force=run['force'][keep];rate=run['rate'][keep];speed=run['speed'][keep]
    metrics={'spatial_depth_cv':float(values.std()/values.mean()),
        'roi_volume_over_k':float(values.sum()*surface.area),'mean_depth_over_k':float(values.mean()),
        'zero_depth_fraction':float(np.mean(values==0)), 'processing_time_s':float(dt[keep].sum()),
        'force_mean_n':float(force.mean()),'force_peak_n':float(force.max()),
        'raw_rate_cv':float(rate.std()/rate.mean()),'mean_speed_mm_s':float(speed.mean()),
        'distance_mm':float(np.dot(speed,dt[keep])),'roi_cells':int(roi.sum()),
        'entry_force_peak_n':float(run['force'][entry].max()),
        'entry_seconds':2.,'entry_reference_mm':18.}
    run['processing_metrics']=metrics;run['processing_depth']=remaining;run['processing_roi']=roi
    np.savez_compressed(output/run['name']/'processing_surface.npz',depth_over_k=remaining,roi=roi,
        u_mm=surface.u,v_mm=surface.v,origin_mm=surface.origin,cell_area_mm2=surface.area)
    # The stored recording integral was independently checked by load_run.
    # Reintegrate its entire depth map as an additional spatial audit.
    for i in np.flatnonzero(keep):
        surface.deposit(xyz[i]-.5*delta[i],velocity[i],run['force'][i],dt[i],contact=run['force'][i]>=1.5)
    close(surface.depth,run['surface']['depth_over_k'],'independent full spatial reintegration')
    run['audit']['full_depth_reintegrated_from_recording']=True
    return metrics


def policy_audit(run,output):
    source=Path(run['audit']['run'])/'policy.csv'
    policy=np.genfromtxt(source,delimiter=',',names=True)
    if any(not np.isfinite(policy[k]).all() for k in policy.dtype.names):raise ValueError('Nonfinite policy recording')
    if len(np.unique(policy['epoch']))!=1 or not np.all(np.diff(policy['sequence'])==1):
        raise ValueError('Missing or mixed policy sequences')
    if np.any(policy['fault']) or policy['done'][-1]!=1.:raise ValueError('Policy did not complete fault-free')
    close(np.diff(policy['elapsed_s']),np.full(len(policy)-1,.008),'policy control cadence')
    close(policy['cursor_mm'][-1],policy['path_length_mm'][-1],'policy path completion')
    if np.any(policy['target_force_n']<14.-1e-6) or np.any(policy['target_force_n']>24.+1e-6):
        raise ValueError('Force request outside actor bounds')
    if policy['missing_samples'].max()!=0:raise ValueError('Policy reports missing measurements')
    after=policy['elapsed_s']>=2.
    requested_speed=6.+3.*policy['feed_action']
    diagnostics={'samples':len(policy),'checkpoint_sha256':run['deployment']['checkpoint_sha256'],
        'missing_samples':int(policy['missing_samples'].max()),'fault_samples':int(np.count_nonzero(policy['fault'])),
        'force_target_min_max_n':[float(policy['target_force_n'].min()),float(policy['target_force_n'].max())],
        'requested_feed_min_max_mm_s':[float(requested_speed.min()),float(requested_speed.max())],
        'processing_force_tracking_rmse_n':float(np.sqrt(np.mean((policy['normal_force_n'][after]-policy['target_force_n'][after])**2))),
        'normal_force_peak_n':float(policy['normal_force_n'].max()),'reverse_cursor_mm':float(policy['reverse_mm'][-1]),
        'local_return_requested':bool(np.any(policy['return_action']>0)),
        'worker_latency_mean_max_ms':[float(policy['latency_ms'].mean()),float(policy['latency_ms'].max())],
        'normal_force_basis':'current TCP normal; removal recorder uses fixed plane Z',
        'policy_sha256':sha(source)}
    run['policy']=policy;shutil.copy2(source,output/run['name']/'policy.csv')
    return diagnostics


def plots(runs,output):
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2,'savefig.dpi':170})
    def save(fig,name):
        fig.savefig(output/name,bbox_inches='tight');plt.close(fig)
    fields=[('spatial_depth_cv','Spatial depth CV'),('roi_volume_over_k','Removal in processing ROI / K'),
        ('raw_rate_cv','Raw temporal rate CV'),('processing_time_s','Processing time [s]'),
        ('zero_depth_fraction','Zero-depth area fraction'),('force_mean_n','Mean normal force [N]')]
    fig,axes=plt.subplots(2,3,figsize=(12,6.8),layout='constrained')
    for ax,(key,title) in zip(axes.flat,fields):
        values=[r['processing_metrics'][key] for r in runs]
        bars=ax.bar(['Mode 5','Mode 6'],values,color=[COLORS[r['name']] for r in runs])
        ax.bar_label(bars,fmt='%.4f' if max(values)<2 else '%.2f',padding=3)
        ax.set(title=title,ylim=(0,max(values)*1.2))
    fig.suptitle('Isaac GUI | after first 2 s | fixed processing ROI | nonrotating tool\nOne run per mode; policy and tracking changes are both included')
    save(fig,'00_processing_comparison.png')
    for processing_view,name in [(True,'01_processing_depth_maps.png'),(False,'02_full_depth_maps.png')]:
        fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        depthkey='processing_depth' if processing_view else None
        maps=[r[depthkey] if depthkey else r['surface']['depth_over_k'] for r in runs]
        vmax=max(float(d.max()) for d in maps)
        for ax,r,depth in zip(axes,runs,maps):
            s=r['surface'];roi=r['processing_roi'] if processing_view else s['roi']
            metrics=r['processing_metrics'] if processing_view else r['metrics']
            extent=[s['u_mm'][0]+s['origin_mm'][0],s['u_mm'][-1]+s['origin_mm'][0],
                    s['v_mm'][0]+s['origin_mm'][1],s['v_mm'][-1]+s['origin_mm'][1]]
            im=ax.imshow(np.where(roi,depth,np.nan),origin='lower',extent=extent,vmin=0,vmax=vmax,cmap='viridis')
            ax.set(xlabel='X [mm]',ylabel='Y [mm]',title=f"{LABELS[r['name']]}\nCV {metrics['spatial_depth_cv']:.4f}")
            ax.grid(False);fig.colorbar(im,ax=ax,label='Depth / K (uncalibrated)',fraction=.046)
        fig.suptitle('Entry-separated spatial depth | identical ROI and color scale' if processing_view else 'Full spatial depth | entry included | identical ROI and color scale')
        save(fig,name)
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True,layout='constrained')
    for r in runs:
        color=COLORS[r['name']];label=LABELS[r['name']]
        axes[0].plot(r['t'],r['force'],color=color,lw=.6,alpha=.65,label=label)
        c=r['control'];idx=np.r_[np.diff(c['time'])>0,True];c=c[idx]
        axes[1].plot(c['time']-c['time'][0],6*c['action_scale'],color=color,lw=.7,label=label)
        # Display-only 128 ms averages; all metrics use raw samples.
        indices=np.arange(0,len(r['rate']),16)
        axes[2].plot(r['t'][indices],[r['rate'][i:i+16].mean() for i in indices],color=color,lw=.8,label=label)
    policy=runs[0]['policy'];pt=policy['elapsed_s']-policy['elapsed_s'][0]
    axes[0].plot(pt,policy['target_force_n'],'k--',lw=.7,label='Mode 5 target')
    for ax,label in zip(axes,['Normal force [N]','Applied path feed [mm/s]','F x speed [N mm/s]\n128 ms mean']):
        ax.set_ylabel(label);ax.axvspan(0,2,color='gray',alpha=.12);ax.legend(loc='upper right')
    axes[-1].set_xlabel('Simulation time since Force entry [s]')
    fig.suptitle('Measured force, applied feed and removal rate | entry shown separately')
    save(fig,'03_force_feed_rate.png')
    width=1.;bins=max(int(np.ceil(r['t'][-1]-2)) for r in runs)
    values=np.full((2,bins),np.nan);rows=[]
    for i,r in enumerate(runs):
        keep=r['t']>=2.;mean=r['rate'][keep].mean()
        for j in range(bins):
            mask=(r['t']>=j+2)&(r['t']<j+3)
            part=r['rate'][mask]
            if not len(part):continue
            rms=float(np.sqrt(np.mean(((part-mean)/mean)**2)));values[i,j]=rms
            rows.append({'mode':r['name'],'start_s':j+2,'end_s':j+3,'samples':len(part),'relative_rms':rms})
        selected=[row for row in rows if row['mode']==r['name']]
        close(np.sqrt(sum(row['samples']*row['relative_rms']**2 for row in selected)/sum(row['samples'] for row in selected)),
              r['processing_metrics']['raw_rate_cv'],'temporal heatmap RMS conservation')
    fig,ax=plt.subplots(figsize=(12,2.5),layout='constrained')
    im=ax.imshow(np.ma.masked_invalid(values),aspect='auto',interpolation='nearest',extent=[2,2+bins,1.5,-.5],
        vmin=0,vmax=float(np.nanmax(values)),cmap='magma')
    ax.set_yticks([0,1],[f"Mode 5 | CV {runs[0]['processing_metrics']['raw_rate_cv']:.4f}",
                       f"Mode 6 | CV {runs[1]['processing_metrics']['raw_rate_cv']:.4f}"])
    ax.set(xlabel='Simulation time since Force entry [s]',title='1 s removal-rate fluctuation | relative RMS from processing mean');ax.grid(False)
    fig.colorbar(im,ax=ax,label='Relative RMS');save(fig,'04_temporal_fluctuation.png')
    csv_rows(output/'temporal_fluctuation.csv',rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode5',type=Path,required=True)
    p.add_argument('--mode6',type=Path,required=True);p.add_argument('--output',type=Path)
    args=p.parse_args()
    output=(args.output or ROOT/'logs/polishing_results'/datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M')).resolve()
    output.mkdir(parents=True,exist_ok=False)
    a=load_run(args.mode5.resolve(),5,output);b=load_run(args.mode6.resolve(),6,output)
    if a['deployment']['observation_shape']!=[1,16] or a['deployment']['action_shape']!=[1,3]:
        raise ValueError('Mode 5 is not the new spatial policy')
    for key in ('trajectory_file','tangential_controller','backend','removal_model'):
        if a['deployment'][key]!=b['deployment'][key]:raise ValueError('Comparison setting differs: '+key)
    if a['summary']['model']!=b['summary']['model']:raise ValueError('Removal model differs')
    close(a['audit']['saved_home_mm_rad'],b['audit']['saved_home_mm_rad'],'initial home')
    for key in ('roi','u_mm','v_mm','cell_area_mm2','origin_mm','basis'):
        if not np.array_equal(a['surface'][key],b['surface'][key]):raise ValueError('Measurement grid differs: '+key)
    if a['audit']['reference_txt_sha256']!=b['audit']['reference_txt_sha256']:raise ValueError('Reference path changed')
    for r in (a,b):
        processing(r,output)
        r['diagnostics']['note']='Primary metrics use the fixed entry-separated processing ROI; full-window metrics are also retained. Legacy force_rmse_n is deviation from nominal 20 N, not variable-target tracking error.'
    if not np.array_equal(a['processing_roi'],b['processing_roi']):raise ValueError('Processing ROI differs')
    policy=policy_audit(a,output)
    fields=['spatial_depth_cv','roi_volume_over_k','raw_rate_cv','processing_time_s','zero_depth_fraction']
    changes={name:{k:100*(a[key][k]/b[key][k]-1) for k in fields}
        for name,key in [('full','metrics'),('processing','processing_metrics')]}
    report={'created_at':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        'comparison':'Spatial feed/force policy and tangent tracking vs existing calibrated Mode 6; one run each',
        'completed_and_saved_home_returned':True,'matched_path_model_grid_home_and_tangent_mdk':True,
        'checkpoint_sha256':a['deployment']['checkpoint_sha256'],'changes_percent_vs_mode6':changes,'policy':policy,
        'entry_protocol':{'seconds':2.,'reference_footprint_mm':18.,'entry_force_is_selection_gate':False,
            'full_roi_cells':int(a['surface']['roi'].sum()),'processing_roi_cells':int(a['processing_roi'].sum()),
            'zero_depth_cells_retained':True,'entry_depth_subtracted':True},
        'runs':{r['name']:{k:r[k] for k in ('metrics','processing_metrics','diagnostics','audit')} for r in (a,b)},
        'limits':['One GUI run per mode; no repeatability claim.',
            'Removal is an uncalibrated nonrotating Preston estimate, not measured material volume.',
            'Comparison includes the added tracking compensation as well as policy changes.',
            'Temporal rate CV includes repeated position samples; not directly equal to synchronized training CV.',
            'The GUI runtime hard force stop was changed to 100 N by user request; training guard was 40 N.']}
    gains=changes['processing'];report['processing_acceptance']={
        'spatial_gain_at_least_3_percent':gains['spatial_depth_cv']<=-3.,
        'removal_within_5_percent':abs(gains['roi_volume_over_k'])<=5.,
        'time_increase_at_most_5_percent':gains['processing_time_s']<=5.}
    dump(output/'comparison.json',report)
    # Different windows have different diagnostics: use a fixed metric schema.
    csv_rows(output/'comparison.csv',[{'mode':r['name'],'window':window,**{k:r[key][k] for k in fields}}
        for r in (a,b) for window,key in [('full','metrics'),('processing','processing_metrics')]])
    notes=['Isaac GUI 새 공간 균일도 모델 / 기존 Mode 6 비교',f'Mode 5: {args.mode5.resolve()}',f'Mode 6: {args.mode6.resolve()}',
        '경로 완료, 제거량 저장, 홈 복귀 확인. 새 16D/3D checkpoint 사용.',
        '가공 구간: 초기 2초 제거량을 빼고, 기준 첫 18 mm가 덮는 영역을 고정 제외. 0 깊이 셀 포함.',
        '공구 비회전, 동일 고정 ROI/초기 홈/경로/접선 MDK. 정책과 추종 보정의 합산 효과 비교.',
        '',*[f'{window}: '+', '.join(f'{k} {v:+.5f}%' for k,v in change.items()) for window,change in changes.items()],
        '',f'접촉력 목표 {policy["force_target_min_max_n"]} N, 이송 목표 {policy["requested_feed_min_max_mm_s"]} mm/s.',
        f'정책 샘플 {policy["samples"]}, 누락 {policy["missing_samples"]}, fault {policy["fault_samples"]}, 복귀 거리 {policy["reverse_cursor_mm"]} mm.',
        '균일도 개선과 총 제거량 감소를 함께 판단해야 한다. 제거량 허용 범위: Mode 6 대비 ±5%.',
        f'가공 구간 허용 조건: {report["processing_acceptance"]}',
        '제거량은 미교정 Preston V/K 추정치다. 각 1회 비교이며 반복 재현성은 확인하지 않았다.']
    (output/'comparison.txt').write_text('\n'.join(notes)+'\n')
    plots([a,b],output)
    dump(output/'artifact_manifest.json',{'files':{str(f.relative_to(output)):sha(f) for f in sorted(output.rglob('*')) if f.is_file()}})
    print(json.dumps({'output':str(output),'changes_percent':changes,'policy':policy,'processing_acceptance':report['processing_acceptance']},indent=2))


if __name__=='__main__':main()
