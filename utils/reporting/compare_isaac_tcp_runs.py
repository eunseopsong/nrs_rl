#!/usr/bin/env python3
"""Compare two completed ROS/Isaac runs on their full Force-mode recordings."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT as _REPO_ROOT
import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
from zoneinfo import ZoneInfo

import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = _REPO_ROOT
COLORS = {'mode5': '#2166ac', 'mode6': '#d6604d'}
LABELS = {'mode5': 'Mode 5: learned feed', 'mode6': 'Mode 6: constant feed'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def close(a, b, name):
    if not np.allclose(a, b, rtol=1e-9, atol=1e-8):
        raise ValueError('Saved data mismatch: '+name)


def command_log(run, mode):
    stamp = datetime.strptime(re.search(r'(\d{8}_\d{6})$',run.name)[1], '%Y%m%d_%H%M%S')
    base = Path.home()/'.ros/log'
    launch, = base.glob(stamp.strftime('%Y-%m-%d-%H-%M-%S-')+'*/launch.log')
    pid = re.search(r'singleArm_cmd-\d+.*process started with pid \[(\d+)\]', launch.read_text())[1]
    source, = base.glob('singleArm_cmd_'+pid+'_*.log')
    text = source.read_text()
    for expected in [f'Mode {mode} trajectory completed.',
                     '/polishing_removal_node/end response: success=true', 'TxtLoad home pose reached; Idling.']:
        if expected not in text:
            raise ValueError('Missing completion evidence: '+expected)
    home = re.search(r'TxtLoad home saved \(mm/rad\): ([^\n]+)', text)[1]
    return source, [float(x) for x in home.split()]


def load_run(run, mode, output):
    name = 'mode'+str(mode)
    folder = output/name
    folder.mkdir()
    dep = json.loads((run/'deployment.json').read_text())
    contract = dep.get('manifest', dep)
    spatial = dep.get('observation_shape') == [1,16] and dep.get('action_shape') == [1,3]
    if (dep['force_control_mode'] != mode or dep['backend'] != 'isaac'
            or dep['removal_model'] != 'tcp_fixed_roi' or contract['spindle_rpm'] not in (None,0.)
            or contract.get('rotation_contribution') is not False
            or (not spatial and contract['normal_force_target_n'] != 20.)):
        raise ValueError('Expected completed nonrotating Isaac runs on the 20 N reference path')
    sessions = list((run/'removal').glob('session_*'))
    if len(sessions) != 1:
        raise ValueError('Select a run with exactly one measurement session')
    session = sessions[0]
    summary = json.loads((session/'summary.json').read_text())
    data = np.loadtxt(session/'recording.csv', delimiter=',', skiprows=1)
    if not np.isfinite(data).all() or data.shape != (summary['recording']['samples'], 10):
        raise ValueError('Incomplete or nonfinite recording')
    t = data[:, 0]-data[0, 0]
    dt = np.r_[.008, np.diff(data[:, 0])]
    if not np.allclose(dt, .008, atol=1e-8, rtol=0):
        raise ValueError('Force recording is not continuous at 125 Hz')
    with np.load(session/'surface.npz') as file:
        surface = {k: file[k].copy() for k in file.files}
    normal = np.asarray(summary['model']['plane_normal'])
    xyz = data[:, 1:4]
    displacement = np.diff(xyz, axis=0, prepend=xyz[:1])
    displacement -= (displacement@normal)[:, None]*normal
    speed = np.linalg.norm(displacement, axis=1)/dt
    force = np.abs(data[:, 9])
    rate = force*speed*(force >= 1.5)
    depth = surface['depth_over_k'][surface['roi']]
    cell = float(surface['cell_area_mm2'])
    close(depth.sum()*cell, summary['metrics']['roi_volume_over_k'], 'ROI volume')
    close(depth.std()/depth.mean(), summary['metrics']['spatial_depth_cv'], 'spatial CV')
    close(np.dot(rate, dt), summary['metrics']['tcp_full_footprint_volume_over_k'], 'F-distance integral')
    close(dt.sum(), summary['metrics']['processing_time_s'], 'processing duration')
    reference = np.loadtxt(dep['trajectory_file'])
    if reference.shape != (17440, 9) or not np.allclose(reference[:, 6:9], [0., 0., 20.]):
        raise ValueError('Reference trajectory differs from the pinned 20 N path')
    xy = (xyz-surface['origin_mm'])@surface['basis'].T
    ref_xy = (reference[:, :3]-surface['origin_mm'])@surface['basis'].T
    distance, _ = cKDTree(ref_xy).query(xy)
    cmdlog, home = command_log(run, mode)
    blob = (run/'raw.csv').read_bytes()
    complete = blob[:blob.rfind(b'\n')+1]
    raw = np.genfromtxt(io.BytesIO(complete), delimiter=',', names=True)
    if any(not np.isfinite(raw[k]).all() for k in raw.dtype.names):
        raise ValueError('Nonfinite raw control data')
    if raw['trajectory_progress'].max() != 1 or not np.any(raw['done']>0):
        raise ValueError('Raw trajectory did not finish')
    bounded = raw[(raw['active']>0)&(raw['done']==0)]
    if mode == 6 and np.any(bounded['raw_ppo_action'] != 0):
        raise ValueError('Mode 6 applied a nonzero action')
    post_ramp = bounded['time']-bounded['time'][0] >= 2.
    scales = bounded['action_scale'][post_ramp]
    metrics = {**summary['metrics'], 'samples': len(data),
        'duration_span_s': float(t[-1]), 'force_mean_n': float(force.mean()),
        'force_peak_n': float(force.max()), 'force_rmse_n': float(np.sqrt(np.mean((force-20)**2))),
        'force_above_32n_s': float(dt[force>32].sum()),
        'raw_rate_cv': float(rate.std()/rate.mean()), 'raw_rate_mean_n_mm_s': float(rate.mean()),
        'force_cv': float(force.std()/force.mean()), 'tangent_speed_cv': float(speed.std()/speed.mean()),
        'mean_speed_mm_s': float(speed.mean()), 'peak_speed_mm_s': float(speed.max()),
        'zero_tangent_displacement_fraction': float(np.mean(speed<1e-8)),
        'measured_tangent_distance_mm': float(np.sum(speed*dt)),
        'reference_path_length_mm': float(np.linalg.norm(np.diff(reference[:, :3], axis=0), axis=1).sum()),
        'path_cross_track_rmse_mm': float(np.sqrt(np.mean(distance**2))),
        'endpoint_tangent_error_mm': float(np.linalg.norm((xyz[-1]-reference[-1, :3])@surface['basis'].T)),
        'reference_tracking_rmse_xy_mm': float(np.sqrt(np.mean(
            (bounded['actual_tcp_x']-bounded['reference_x'])**2+(bounded['actual_tcp_y']-bounded['reference_y'])**2)))}
    diagnostics = {'after_2s_force_rmse_n': float(np.sqrt(np.mean((force[t>=2]-20)**2))),
        'after_2s_force_peak_n': float(force[t>=2].max()),
        'first_2s_force_error_energy_fraction': float(np.sum((force[t<2]-20)**2)/np.sum((force-20)**2)),
        'action_scale_after_2s_p01_median_p99': np.quantile(scales, [.01, .5, .99]).tolist(),
        'processing_raw_samples': len(bounded), 'processing_raw_duplicate_times': int(np.sum(np.diff(bounded['time'])==0)),
        'policy_fresh_fraction': float(bounded['ppo_action_fresh'].mean()),
        'progression_allowed_fraction': float(bounded['ppo_progression_allowed'].mean()),
        'note': 'After-2s values are diagnostic only. All primary metrics retain every Force-phase sample.'}
    files = [run/'deployment.json', session/'recording.csv', session/'surface.npz', session/'summary.json', cmdlog]
    for source in files:
        shutil.copy2(source, folder/('command.log' if source == cmdlog else source.name))
    np.savez_compressed(folder/'processing_control.npz', **{k: bounded[k] for k in bounded.dtype.names})
    np.savez_compressed(folder/'derived_signals.npz', time_s=t, dt_s=dt, xyz_mm=xyz,
                        force_n=force, tangent_speed_mm_s=speed, raw_rate_n_mm_s=rate)
    audit = {'run': str(run), 'session': str(session), 'saved_home_mm_rad': home,
        'completion_and_home_return_logged': True, 'continuous_125hz_recording': True,
        'stored_surface_and_integral_reproduced': True,
        'source_sha256': {str(f): sha(f) for f in files}, 'reference_txt_sha256': sha(dep['trajectory_file']),
        'raw_snapshot_sha256': hashlib.sha256(complete).hexdigest(), 'raw_snapshot_complete_rows': len(raw),
        'raw_snapshot_trailing_partial_row_excluded': not blob.endswith(b'\n')}
    return dict(name=name, deployment=dep, summary=summary, metrics=metrics, diagnostics=diagnostics,
                audit=audit, surface=surface, t=t, dt=dt, force=force, speed=speed, rate=rate,
                reference=reference, xyz=xyz, control=bounded)


def csv_rows(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def historical_context(deployment, output):
    source = Path(deployment['bundle'])/'report.json'
    expected = deployment['manifest']['files']['report.json']
    if sha(source) != expected:
        raise ValueError('Historical training report differs from deployed bundle')
    history = json.loads(source.read_text())
    if history['checkpoint_sha256'] != deployment['checkpoint_sha256']:
        raise ValueError('Historical report refers to another checkpoint')
    shutil.copy2(source, output/'training_report.json')
    return {'source': str(source), 'sha256': expected,
            'comparison_environment': 'headless training validation, not these ROS trials',
            'paired_comparisons': history['paired_comparisons'],
            'attribution': history['attribution']}


def interpretation(report):
    runs = report['runs']
    scale = runs['mode5']['diagnostics']['action_scale_after_2s_p01_median_p99']
    history = report['historical_training_context']['paired_comparisons']
    original = history['learned_vs_old_d14']['raw_rate_cv_reduction_percent']['mean']
    isolated = history['learned_vs_calibrated_only']['raw_rate_cv_reduction_percent']['mean']
    repeated = [100*runs[n]['metrics']['zero_tangent_displacement_fraction'] for n in ['mode5', 'mode6']]
    entry = [100*runs[n]['diagnostics']['first_2s_force_error_energy_fraction'] for n in ['mode5', 'mode6']]
    return [
        'CV 차이가 작은 이유:',
        f'1. 두 모드는 학습된 MDK를 공유합니다. 이전 {original:.2f}% 시간 CV 감소는 원래 D14 대비 전체 제어 구성의 효과입니다.',
        f'   학습 검증에서 같은 MDK의 정속 대비 속도 정책만의 시간 CV 감소율은 {isolated:+.4f}%로, 추가 개선이 확인되지 않았습니다.',
        f'2. Mode 5 정책은 적용됐지만 진입 2초 이후 속도 배율 1/50/99백분위가 {scale[0]:.6f}/{scale[1]:.6f}/{scale[2]:.6f}입니다.',
        f'   중앙값은 정속 1.0 대비 {(scale[1]-1)*100:+.4f}% 차이로, 실제 적용되는 이송 명령이 거의 같습니다.',
        f'3. 접선 위치 변화가 없는 샘플은 Mode 5 {repeated[0]:.2f}%, Mode 6 {repeated[1]:.2f}%입니다.',
        '   위치 차분으로 구한 속도/제거율에는 샘플 유지와 실제 정지가 함께 반영될 수 있습니다. 이것만으로 물리적 원인을 확정하지 않습니다.',
        '4. RPM을 제외한 현재 모델의 제거율은 F×v입니다. 같은 경로와 일정한 힘에서는 시간 적분이 F×이동거리여서 속도만 바꾸는 가공량 효과가 제한됩니다.',
        f'힘 RMS 해석: 처음 2초가 전체 힘 제곱오차의 Mode 5 {entry[0]:.2f}%, Mode 6 {entry[1]:.2f}%를 차지합니다.',
        '두 모드 모두 진입 과도응답이 전체 힘 RMS를 지배합니다. 한 번씩의 실행 차이를 정책 효과로 확정하지 않습니다.',
    ]


def compare(args):
    output = (args.output or ROOT/'logs/polishing_results'/datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M')).resolve()
    output.mkdir(parents=True, exist_ok=False)
    a = load_run(args.mode5.resolve(), 5, output)
    b = load_run(args.mode6.resolve(), 6, output)
    for key in ['checkpoint_sha256', 'trajectory_file', 'tangential_controller', 'motion_binary', 'backend', 'removal_model']:
        if a['deployment'][key] != b['deployment'][key]:
            raise ValueError('Comparison setting differs: '+key)
    for key in ['reference_h5_sha256', 'motion_binary_sha256', 'normal_force_target_n', 'files']:
        if a['deployment']['manifest'][key] != b['deployment']['manifest'][key]:
            raise ValueError('Bundle differs: '+key)
    if a['summary']['model'] != b['summary']['model'] or a['audit']['saved_home_mm_rad'] != b['audit']['saved_home_mm_rad']:
        raise ValueError('Measurement model or initial home differs')
    for key in ['roi', 'u_mm', 'v_mm', 'cell_area_mm2', 'origin_mm', 'basis']:
        if not np.array_equal(a['surface'][key], b['surface'][key]):
            raise ValueError('Fixed measurement grid differs: '+key)
    fields = ['roi_volume_over_k', 'spatial_depth_cv', 'raw_rate_cv', 'force_rmse_n', 'force_mean_n',
              'force_peak_n', 'processing_time_s', 'reference_tracking_rmse_xy_mm', 'path_cross_track_rmse_mm']
    changes = {key: 100*(a['metrics'][key]/b['metrics'][key]-1) for key in fields}
    report = {'created_at': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        'comparison': 'Mode 5 learned feed vs Mode 6 constant feed; identical learned MDK',
        'run_count_per_mode': 1, 'matched_saved_home_and_configuration': True,
        'checkpoint_sha256': a['deployment']['checkpoint_sha256'],
        'mode5_percent_change_relative_to_mode6': changes,
        'runs': {x['name']: {'metrics': x['metrics'], 'diagnostics': x['diagnostics'], 'audit': x['audit']} for x in [a,b]},
        'limits': ['One run per mode; no statistical superiority or generalization claim.',
            'RPM excluded. F*measured TCP displacement and volume/K are uncalibrated material-removal proxies.',
            'All Force-mode samples, initial force transients, zero displacement and uncovered ROI cells are retained.',
            'Repeated recorded positions affect finite-difference rate CV; do not equate ROS sample timing with headless synchronized tracking.',
            'The earlier 35.01% CV reduction was a headless comparison with original D14, not this Mode 6 controller.',
            'Both runs exceed the 32 N headless peak-force threshold during entry.']}
    report['historical_training_context'] = historical_context(a['deployment'], output)
    report['interpretation'] = interpretation(report)
    dump(output/'comparison.json', report)
    csv_rows(output/'comparison.csv', [{'mode': x['name'], **x['metrics']} for x in [a,b]])
    make_plots([a,b], output)
    notes = [
        'Isaac Mode 5 / Mode 6 실증 비교',
        f'Mode 5: {args.mode5.resolve()}', f'Mode 6: {args.mode6.resolve()}',
        '첫 번째 이상 실행 tcp_20261007_130006은 제외했습니다.',
        '같은 초기 홈 자세, 20 N 경로, 학습된 MDK, RPM 제외 모델, 고정 ROI입니다. 각 모드 1회씩의 비교입니다.',
        '주요 평가는 가공 Force 구간 전체를 사용합니다. 진입부, 정지/반복 좌표, 0 제거율을 제외하지 않았습니다.', '',
        'Mode 6 대비 Mode 5 변화율(증가 + / 감소 -):',
        *[f'{key}: {value:+.6f}%' for key,value in changes.items()], '',
        '이번 한 쌍에서는 Mode 5의 가공량 및 가공 균일도 우위가 확인되지 않았고, 힘 RMS 오차는 더 작았습니다.',
        '이전 35.01%는 원래 D14를 상대로 한 headless 검증 결과입니다. 이번 Mode 6 대비 개선율을 뜻하지 않습니다.',
        '공간 CV와 시간 CV는 다른 지표입니다. 128 ms 평균은 표시용이며 시간 CV는 원시 125 Hz rate에서 계산합니다.',
        '1초 변동 히트맵은 각 주행 전체 평균 대비 상대 RMS입니다. 셀별 제곱값을 샘플 수로 가중 평균하고 제곱근을 취하면 전체 raw CV와 같습니다.',
        '진입부 과도 힘은 두 모드 모두 32 N 기준을 넘습니다. 2초 이후 값은 원인 진단용으로만 별도 기재했습니다.',
        '제거량은 F×측정 TCP 이동거리 기반 volume/K 추정치이며 실측 재료 체적(mm³)이 아닙니다.',
        '반복 위치 샘플은 샘플 유지 또는 실제 정지를 포함할 수 있습니다. ROS 원시 rate CV를 학습 환경 수치와 직접 등치하지 않습니다.',
        '', *report['interpretation'], '',
        '원본 데이터와 제어 노드는 수정하지 않았습니다. 각 모드 폴더의 CSV/NPZ/command.log 및 comparison.json에 근거를 보존했습니다.'
    ]
    (output/'comparison.txt').write_text('\n'.join(notes)+'\n')
    dump(output/'artifact_manifest.json', {'files': {str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    print(json.dumps({'output': str(output), 'changes': changes,
                      'diagnostics': {x['name']:x['diagnostics'] for x in [a,b]}}, indent=2))


def make_plots(runs, output):
    plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .2,
                         'savefig.dpi': 180, 'legend.fontsize': 9})

    def save(fig, name, footer):
        fig.text(.5, .015, footer, ha='center', va='bottom', fontsize=8)
        fig.tight_layout(rect=(0, .065, 1, .95))
        fig.savefig(output/name)
        plt.close(fig)

    fields = [('roi_volume_over_k','ROI removal / K'), ('spatial_depth_cv','Spatial depth CV'),
              ('raw_rate_cv','Raw temporal rate CV'), ('force_rmse_n','Force RMSE [N]'),
              ('processing_time_s','Processing time [s]'),
              ('reference_tracking_rmse_xy_mm','Reference XY tracking RMSE [mm]')]
    fig, axes = plt.subplots(2,3,figsize=(12,7))
    for ax,(key,title) in zip(axes.flat,fields):
        values = [x['metrics'][key] for x in runs]
        bars = ax.bar(['Mode 5','Mode 6'],values,color=[COLORS[x['name']] for x in runs])
        ax.bar_label(bars,fmt='%.4f' if max(values)<10 else '%.2f',padding=4)
        ax.set_title(title+(' (higher)' if key=='roi_volume_over_k' else ' (lower)'))
        ax.set_ylim(0,max(values)*1.2)
        ax.set_axisbelow(True)
    fig.suptitle('Isaac GUI comparison | same learned MDK, path and saved home | one run per mode')
    save(fig,'00_comparison_overview.png','Full Force-mode windows, including entry and zero-displacement samples. RPM excluded; uncalibrated proxy.')

    fig, axes = plt.subplots(1,2,figsize=(12,5.4))
    maximum = max(x['surface']['depth_over_k'].max() for x in runs)
    for ax,x in zip(axes,runs):
        s=x['surface'];z=np.where(s['roi'],s['depth_over_k'],np.nan)
        extent=[s['u_mm'][0]+s['origin_mm'][0],s['u_mm'][-1]+s['origin_mm'][0],
                s['v_mm'][0]+s['origin_mm'][1],s['v_mm'][-1]+s['origin_mm'][1]]
        im=ax.imshow(z,origin='lower',extent=extent,vmin=0,vmax=maximum,cmap='viridis')
        ax.set(xlabel='X [mm]',ylabel='Y [mm]',title=f"{LABELS[x['name']]} | CV {x['metrics']['spatial_depth_cv']:.4f}")
        ax.grid(False)
        fig.colorbar(im,ax=ax,label='Depth / K [uncalibrated]',fraction=.045,pad=.025)
    save(fig,'01_spatial_depth_comparison.png','Same fixed 66,254-cell ROI and color scale. All uncovered cells are retained. No spatial smoothing.')

    fig, axes=plt.subplots(4,1,figsize=(12,9),sharex=True)
    for x in runs:
        color=COLORS[x['name']]; label=LABELS[x['name']]
        for ax,field in zip(axes[:3],['force','speed','rate']):
            ax.plot(x['t'],x[field],color=color,lw=.55,alpha=.7,label=label)
        c=x['control']; axes[3].plot(c['time']-c['time'][0],c['action_scale'],color=color,lw=.7,label=label)
    axes[0].axhline(20,color='.2',ls='--',lw=.7)
    for ax,label in zip(axes,['Normal force\n[N]','Recorded TCP speed\n[mm/s]','Raw F x speed\n[N mm/s]','Applied feed scale']):
        ax.set_ylabel(label);ax.legend(loc='upper right')
    axes[3].set_xlabel('Simulation time since Force entry [s]')
    fig.suptitle('All raw signals | no trimming or smoothing')
    save(fig,'02_force_velocity_rate_policy.png','Recorded velocity is backward pose difference / simulation dt. Applied feed scale includes the common startup ramp.')

    fig, axes=plt.subplots(2,1,figsize=(12,5.6))
    for ax,(width,field,title) in zip(axes,[(.128,'mean','128 ms mean instantaneous rate [N mm/s]'),
                                         (1.,'rms','1 s relative RMS fluctuation from each full-run mean')]):
        step=int(round(width/.008));n=max((len(x['rate'])+step-1)//step for x in runs)
        values=np.full((2,n),np.nan);records=[]
        for i,x in enumerate(runs):
            mean=x['rate'].mean();squares=total=count=0.
            for j in range(n):
                part=x['rate'][j*step:(j+1)*step]
                if not len(part):continue
                avg=float(part.mean());rms=float(np.sqrt(np.mean(((part-mean)/mean)**2)))
                values[i,j]=avg if field=='mean' else rms
                records.append({'mode':x['name'],'window_start_s':j*width,'window_end_s':(j+1)*width,
                    'samples':len(part),'mean_rate_n_mm_s':avg,'relative_rms_from_run_mean':rms,
                    'full_raw_cv':x['metrics']['raw_rate_cv']})
                total+=len(part)*avg;squares+=len(part)*rms*rms;count+=len(part)
            if count != len(x['rate']):raise ValueError('Temporal bins lost samples')
            close(total/count,mean,'count-weighted binned mean')
            close(np.sqrt(squares/count),x['metrics']['raw_rate_cv'],'RMS conservation')
        image=ax.imshow(np.ma.masked_invalid(values),aspect='auto',origin='upper',interpolation='nearest',
                        extent=[0,n*width,1.5,-.5],vmin=0,vmax=float(np.nanmax(values)),cmap='magma')
        ax.set_yticks([0,1],[f"Mode 5 | CV {runs[0]['metrics']['raw_rate_cv']:.4f}",
                            f"Mode 6 | CV {runs[1]['metrics']['raw_rate_cv']:.4f}"])
        ax.set(title=title,xlabel='Simulation time since Force entry [s]');ax.grid(False)
        fig.colorbar(image,ax=ax,fraction=.025,pad=.025)
        csv_rows(output/f'03_temporal_{field}_{width:g}s.csv',records)
    save(fig,'03_temporal_rate_and_fluctuation.png','Means are display-only. CV uses every raw 125 Hz sample. Count-weighted RMS across 1 s cells exactly reproduces raw CV.')

    fig, axes=plt.subplots(1,2,figsize=(12,4.6))
    for x in runs:
        mask=x['t']<=3
        axes[0].plot(x['t'][mask],x['force'][mask],color=COLORS[x['name']],label=LABELS[x['name']])
        energy=(x['force']-20)**2*x['dt']
        axes[1].plot(x['t'],np.cumsum(energy),color=COLORS[x['name']],label=LABELS[x['name']])
    axes[0].axhline(20,color='.4',ls='--');axes[0].axhline(32,color='.4',ls=':',label='32 N headless threshold')
    axes[0].set(xlabel='Simulation time since Force entry [s]',ylabel='Normal force [N]',title='Entry force transient')
    axes[1].set(xlabel='Simulation time since Force entry [s]',ylabel='Integral (force - 20 N)^2 dt [N² s]',title='Cumulative force-error energy')
    for ax in axes:ax.legend()
    save(fig,'04_entry_force_and_error_energy.png','Entry is included in every primary metric. A lower full-run force RMSE may chiefly reflect a different entry transient.')

    fig, axes=plt.subplots(2,1,figsize=(12,6),sharex=True)
    for x in runs:
        c=x['control']; t=c['time']-c['time'][0]; mask=t>=2.
        for ax,field in zip(axes,['raw_ppo_action','action_scale']):
            ax.plot(t[mask],c[field][mask],lw=.7,color=COLORS[x['name']],label=LABELS[x['name']])
    axes[0].axhline(0,color='.4',ls='--',lw=.7)
    axes[1].axhline(1,color='.4',ls='--',lw=.7)
    axes[0].set_ylabel('Raw policy action')
    axes[1].set(ylabel='Applied feed scale',xlabel='Simulation time since Force entry [s]')
    for ax in axes:ax.legend()
    fig.suptitle('Policy and applied feed detail after 2 s | same learned MDK in both modes')
    save(fig,'05_policy_scale_detail.png','Diagnostic view excludes the first 2 s only to show policy scale. All primary metrics include the entire recording.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode5', type=Path, required=True)
    parser.add_argument('--mode6', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    compare(parser.parse_args())
