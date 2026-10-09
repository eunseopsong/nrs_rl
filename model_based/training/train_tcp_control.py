#!/usr/bin/env python3
"""Learn tangential MDK and feed feedback; independently validate the full stack.

This is a separate experiment from the frozen velocity-only study. It recovers
missing path coverage at the unchanged 20 N force target. Dose and travel caps
are tied to that target/path, so chatter cannot win by adding extra distance.
"""
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
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

ROOT=_REPO_ROOT
sys.path.insert(0,str(ROOT/'scripts/skrl'))
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.tcp_removal_actor import TcpRemovalActor
from nrs_rl.tasks.manager_based.nrs_rl.mdp.runtime.ppo_policy_node import SCHEDULER_CONTRACT, _build_agent, _validate_scheduler_contract

BASE=[6000.,2000.,0.,0.]
LOW=np.array([1500.,2000.,0.,0.]);HIGH=np.array([6000.,16000.,1.2,.35])
OLD=ROOT/'logs/experiments/preston_pad30_preview16_depth_search_20260914_090304/screen/checkpoints/actor.pt'
REFERENCE=ROOT/'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/datasets/cmd_continue9D_flat.h5'
GATES={'spatial_cv_gain_min':.05,'raw_rate_cv_ratio_max':1.05,'roi_volume_ratio_min':.95,
       'processing_time_ratio_max':1.05,'force_rmse_ratio_max':1.10,'force_peak_n_max':32.,
       'force_mean_n_max':21.,'cross_track_rmse_ratio_max':1.,'endpoint_error_ratio_max':1.,
       'full_volume_vs_nominal_path_max':1.03,'distance_vs_reference_max':1.03,
       'extra_uncovered_area_max':0.,'fault_fraction_max':0.,'shield_fraction_max':0.}
REWARD={'spatial_cv_gain':2.,'roi_volume_gain':1.,'coverage_gain':2.,
        'raw_rate_cv_gain':.25,'cross_track_gain':.2,'force_rmse_gain':.1,'time_gain':.1}


def save(path,value):
    path=Path(path);tmp=path.with_suffix('.next.json')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def assess(candidate,baselines,path_length,*,validation=False,check_endpoint=True):
    comparisons=[]; scores=[]; rejections=[]
    for index,base in enumerate(baselines):
        a,b=candidate['preston_surface'],base['preston_surface']
        if (a['roi_cells'],a['roi_area_mm2'])!=(b['roi_cells'],b['roi_area_mm2']):
            raise ValueError('Reference ROI changed')
        c={'spatial_cv_gain':1-a['spatial_depth_cv']/b['spatial_depth_cv'],
           'roi_volume_gain':a['roi_volume_over_k']/b['roi_volume_over_k']-1,
           'coverage_gain':b['zero_depth_fraction']-a['zero_depth_fraction'],
           'raw_rate_cv_gain':1-candidate['processing_rate_cv']/base['processing_rate_cv'],
           'cross_track_gain':1-candidate['cross_track_rmse_mm']/base['cross_track_rmse_mm'],
           'force_rmse_gain':1-candidate['force_rmse_n']/base['force_rmse_n'],
           'time_gain':1-candidate['processing_samples']/base['processing_samples']}
        checks={
            'roi_volume_ratio_min':(1+c['roi_volume_gain'],.95,'min'),
            'raw_rate_cv_ratio_max':(1-c['raw_rate_cv_gain'],1.05,'max'),
            'processing_time_ratio_max':(1-c['time_gain'],1.05,'max'),
            'force_rmse_ratio_max':(1-c['force_rmse_gain'],1.10,'max'),
            'force_peak_n_max':(candidate['force_peak_n']/32.,1.,'max'),
            'force_mean_n_max':(candidate['force_mean_n']/21.,1.,'max'),
            'cross_track_rmse_ratio_max':(1-c['cross_track_gain'],1.,'max'),
            'endpoint_error_ratio_max':(candidate['endpoint_tangential_error_mm']/base['endpoint_tangential_error_mm'],1.,'max'),
            'full_volume_vs_nominal_path_max':(a['tcp_full_footprint_volume_over_k']/(20.*path_length),1.03,'max'),
            'distance_vs_reference_max':(candidate['measured_path_length_mm']/path_length,1.03,'max'),
            'extra_uncovered_area_max':(-c['coverage_gain'],0.,'max')}
        if not check_endpoint:
            # A prefix-only search stops at an arbitrary cut in the trajectory.
            # Full-path training/validation always retains the endpoint gate.
            checks.pop('endpoint_error_ratio_max')
        if validation: checks['spatial_cv_gain_min']=(c['spatial_cv_gain'],.05,'min')
        violation=0.;failed=[]
        for key,(value,limit,direction) in checks.items():
            delta=(value-limit) if direction=='max' else (limit-value)
            if delta>1e-12: failed.append(key);violation+=delta
        if not candidate['completed'] or not base['completed']:
            failed.append('incomplete');violation+=10.
        if any(candidate[k]>0 or base[k]>0 for k in ('fault_fraction','shield_fraction')):
            failed.append('fault_or_shield');violation+=10.
        score=sum(REWARD[k]*c[k] for k in REWARD)-10.*violation
        if not np.isfinite(score): raise ValueError('Nonfinite score')
        scores.append(score);comparisons.append(c)
        rejections.extend(f'constant_{index}: {x}' for x in failed)
    return {'score':float(min(scores)),'passed':not rejections,'rejections':rejections,'comparisons':comparisons}


def batch(root,name,population,seed,checkpoint=None,old=False,signed_feedback=False):
    policies={key:[value[2],value[3],0.,0.,0.,0.] for key,value in population.items()}
    gains={key:value[:2] for key,value in population.items()}
    save(root/f'{name}_policies.json',policies);save(root/f'{name}_mdk.json',gains)
    command=[sys.executable,str(ROOT/'scripts/probe_tcp_control.py'),'--headless','--device','cpu',
        '--tcp-removal','--parameters-file',str(root/f'{name}_policies.json'),
        '--tangent-parameters',str(root/f'{name}_mdk.json'),'--path-length-mm','698',
        '--max-steps','24000','--seed',str(seed),'--surface-cell-mm','.5','--output',str(root/name)]
    if checkpoint:command+=['--candidate-checkpoint',str(checkpoint)]
    if old:command+=['--include-checkpoint',str(OLD)]
    if signed_feedback:command+=['--signed-force-feedback']
    save(root/f'{name}_command.json',command)
    with (root/f'{name}.log').open('w') as stream:
        subprocess.run(command,cwd=ROOT,env={**os.environ,'MPLCONFIGDIR':'/tmp/nrs_tcp_mpl','MPLBACKEND':'Agg'},
                       stdout=stream,stderr=subprocess.STDOUT,check=True)
    return json.loads((root/name/'summary.json').read_text())


def train(args):
    torch.set_num_threads(1)
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    state={'phase':'training','algorithm':'cem_joint_controller_and_feed_parameter_search',
        'started_at_utc':datetime.now(timezone.utc).isoformat(),'rotation_contribution':False,'spindle_rpm':None,
        'normal_force_target_n':20.,'normal_axis_controller_unchanged':True,'gates':GATES,'reward':REWARD,
        'parameter_names':['tangent_damping_ns_m','tangent_stiffness_n_m','inverse_force_gain','rate_gain'],
        'training_seeds':[109,110],'validation_seeds':[309,310,311],
        'validation_selection_used':False,'production_controller_reconfigured':False,
        'physical_depth_validated':False,'reference_path_sha256':hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        'study_reason':'Velocity-only policies do not remove the measured reference-to-command lag.',
        'dose_guard_reason':'Recover original path coverage at 20 N; cap total F*distance to 103% of nominal full-path dose.',
        'generations':[],'validation':[]}
    save(root/'protocol.json',state)
    files=['scripts/train_tcp_control.py','scripts/probe_tcp_control.py','scripts/calibrated_mode3.cpp',
           'scripts/build_calibrated_mode3.py','scripts/skrl/tcp_removal_actor.py','scripts/skrl/ppo_policy_node.py']
    manifest={}
    for rel in files:
        dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/rel,dest);manifest[rel]=hashlib.sha256(dest.read_bytes()).hexdigest()
    save(root/'source_manifest.json',manifest)
    def update(**values):
        state.update(values,updated_at_utc=datetime.now(timezone.utc).isoformat());save(root/'status.json',state)
        print('TCP_CONTROL_STATUS',state['phase'],state.get('current_batch',''),flush=True)
    population={'constant_a':BASE,'constant_b':BASE,'stiff':[6000.,8000.,0.,0.],
                'medium':[4000.,8000.,0.,0.],'fast':[2000.,8000.,0.,0.],
                'high_stiff':[4000.,12000.,0.,0.],'feedback':[4000.,8000.,.7,.15],
                'feedback_stiff':[3000.,12000.,.5,.1]}
    rng=np.random.default_rng(20261006);incumbent=BASE
    for generation,seed in enumerate(state['training_seeds']):
        name=f'generation_{generation}';update(current_batch=name)
        summary=batch(root,name,population,seed)
        metrics=summary['candidates'];baselines=[metrics[x] for x in ('constant_a','constant_b')]
        ranking={k:assess(v,baselines,summary['path_length_mm']) for k,v in metrics.items()}
        ordered=sorted(ranking,key=lambda key:ranking[key]['score'],reverse=True)
        incumbent=population[ordered[0]]
        state['generations'].append({'generation':generation,'seed':seed,'ranking':ranking,
                                    'winner':ordered[0],'parameters':incumbent})
        update(best_parameters=incumbent)
        normalized=(np.array([population[k] for k in ordered[:3]])-LOW)/(HIGH-LOW)
        mean=normalized.mean(0);std=np.maximum(normalized.std(0),.10)
        population={'constant_a':BASE,'constant_b':BASE,'incumbent':incumbent}
        for i,draw in enumerate(rng.normal(mean,std,(5,4))):
            population[f'proposal_{i}']=(LOW+np.clip(draw,0.,1.)*(HIGH-LOW)).tolist()
    tangent={'damping_ns_m':incumbent[0],'stiffness_n_m':incumbent[1]}
    feedback=[incumbent[2],incumbent[3],0.,0.,0.,0.]
    contract={**SCHEDULER_CONTRACT,'schema_version':3,'observation_size':14,'robot':'ur10_cb3',
        'control_period_s':.008,'policy_action_repeat':1,'policy_period_s':.008,
        'policy_backend':'torchscript_tcp_removal','objective':'preston_tcp_spatial_removal',
        'rotation_contribution':False,'spindle_rpm':None,'tool_diameter_mm':30.,'physics_tool_diameter_mm':30.,
        'training_algorithm':state['algorithm'],'learned_parameters':feedback,'tangential_controller':tangent,
        'requires_matching_controller_configuration':True,'reference_path_sha256':state['reference_path_sha256'],
        'validation_status':'candidate; consult result.json before integration'}
    checkpoint=root/'checkpoints/actor.pt';checkpoint.parent.mkdir();(root/'params').mkdir()
    save(root/'params/velocity_policy.json',contract)
    actor=TcpRemovalActor(feedback).eval()
    torch.jit.save(torch.jit.script(actor),str(checkpoint),_extra_files={'velocity_policy.json':json.dumps(contract,sort_keys=True)})
    agent,_=_build_agent(str(checkpoint));_validate_scheduler_contract(agent.velocity_contract,tangent)
    probe=torch.randn(1024,14);probe[:,10]=1.;probe[:,11]=0.
    error=float((actor(probe)-agent.act(probe)[0]).abs().max());assert error<1e-7
    rejected_legacy=False
    try:_validate_scheduler_contract(agent.velocity_contract)
    except ValueError:rejected_legacy=True
    if tangent!={'damping_ns_m':6000.,'stiffness_n_m':2000.}:assert rejected_legacy
    save(root/'export_validation.json',{'ok':True,'max_action_error':error,'legacy_config_rejected':rejected_legacy,
        'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
    update(phase='independent_validation',checkpoint=str(checkpoint))
    validation_population={'constant_a':BASE,'constant_b':BASE,'old_d14':BASE,
        'calibrated_only':[incumbent[0],incumbent[1],0.,0.],'learned':incumbent}
    def validate(seed):
        name=f'held_out_{seed}'
        summary=batch(root,name,validation_population,seed,checkpoint=checkpoint,old=True)
        m=summary['candidates'];baselines=[m[k] for k in ('constant_a','constant_b')]
        return {'seed':seed,**assess(m['learned'],baselines,summary['path_length_mm'],validation=True),
                'calibration_only':assess(m['calibrated_only'],baselines,summary['path_length_mm'],validation=True),
                'learned_vs_calibration':assess(m['learned'],[m['calibrated_only']],summary['path_length_mm'])}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(validate,seed):seed for seed in state['validation_seeds']}
        for future in as_completed(futures):
            state['validation'].append(future.result());state['validation'].sort(key=lambda v:v['seed']);update()
    passed=all(v['passed'] for v in state['validation'])
    update(phase='complete',validation_passed=passed,
           disposition='validated_simulator_candidate_requires_controller_integration' if passed else 'candidate_failed_frozen_gates')
    save(root/'result.json',state)
    print('TCP_CONTROL_COMPLETE',passed,str(root/'result.json'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    train(parser.parse_args())
