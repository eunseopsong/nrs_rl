#!/usr/bin/env python3
"""CEM training and frozen held-out validation of a no-spindle feed policy.

Whole-episode rewards use the fixed reference ROI including zero-depth cells.
Search does not see held-out seeds. A failed gate preserves the candidate as
experimental and never replaces a production checkpoint.
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

ROOT = _REPO_ROOT
sys.path.insert(0, str(ROOT/'scripts/skrl'))
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.tcp_removal_actor import TcpRemovalActor, PARAMETER_NAMES, LOW, HIGH
from nrs_rl.tasks.manager_based.nrs_rl.mdp.runtime.ppo_policy_node import SCHEDULER_CONTRACT, _build_agent, _validate_scheduler_contract

OLD = ROOT/'logs/experiments/preston_pad30_preview16_depth_search_20260914_090304/screen/checkpoints/actor.pt'
REFERENCE = ROOT/'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/datasets/cmd_continue9D_flat.h5'
ZERO = [0.]*6
# Freeze these before any candidate is evaluated. Positive rewards are
# dimensionless gains relative to paired constant controls.
REWARD = {'spatial_cv_gain':2., 'raw_rate_cv_gain':.35, 'cross_track_gain':.20,
          'force_rmse_gain':.10, 'cycle_time_gain':.10, 'roi_volume_gain':.25,
          'additional_jerk_penalty':.02, 'constraint_violation_penalty':10.}
GATES = {'minimum_spatial_cv_gain':.01, 'minimum_raw_rate_cv_gain':.01,
         'maximum_duration_ratio':1.05, 'minimum_roi_volume_ratio':.95,
         'maximum_roi_volume_ratio':1.05, 'maximum_force_peak_ratio':1.05,
         'maximum_force_rmse_ratio':1.05, 'maximum_tracking_rmse_ratio':1.05,
         'maximum_endpoint_error_ratio':1.05,
         'maximum_extra_zero_fraction':0., 'maximum_fault_fraction':0.,
         'maximum_shield_fraction':0.}


def compare(candidate, baseline):
    a,b = candidate['preston_surface'],baseline['preston_surface']
    if (a['roi_cells'],a['roi_area_mm2']) != (b['roi_cells'],b['roi_area_mm2']):
        raise ValueError('Comparison ROI must be identical')
    return {
        'spatial_cv_gain':1-a['spatial_depth_cv']/b['spatial_depth_cv'],
        'raw_rate_cv_gain':1-candidate['processing_rate_cv']/baseline['processing_rate_cv'],
        'cross_track_gain':1-candidate['cross_track_rmse_mm']/max(baseline['cross_track_rmse_mm'],1e-6),
        'force_rmse_gain':1-candidate['force_rmse_n']/max(baseline['force_rmse_n'],1e-6),
        'cycle_time_gain':1-candidate['processing_samples']/baseline['processing_samples'],
        'roi_volume_gain':a['roi_volume_over_k']/b['roi_volume_over_k']-1,
        'duration_ratio':candidate['processing_samples']/baseline['processing_samples'],
        'roi_volume_ratio':a['roi_volume_over_k']/b['roi_volume_over_k'],
        'force_peak_ratio':candidate['force_peak_n']/max(baseline['force_peak_n'],1e-6),
        'force_rmse_ratio':candidate['force_rmse_n']/max(baseline['force_rmse_n'],1e-6),
        'tracking_rmse_ratio':candidate['cross_track_rmse_mm']/max(baseline['cross_track_rmse_mm'],1e-6),
        'endpoint_error_ratio':candidate['endpoint_tangential_error_mm']/max(baseline['endpoint_tangential_error_mm'],1e-6),
        'extra_zero_fraction':a['zero_depth_fraction']-b['zero_depth_fraction'],
        'additional_jerk_penalty':max(0.,(candidate['command_jerk_rms_mm_s3']-baseline['command_jerk_rms_mm_s3'])/160.),
    }


def assess(candidate, baselines, *, gate=False):
    scores, comparisons, rejections = [], [], []
    for i,base in enumerate(baselines):
        c = compare(candidate,base)
        if not all(np.isfinite(x) for x in c.values()):
            raise ValueError('Nonfinite evaluation metric')
        failed=[]; violation=0.
        if not candidate['completed'] or not base['completed']:
            failed.append('incomplete path'); violation+=1.
        for key in ('fault_fraction','shield_fraction'):
            if candidate[key]>0. or base[key]>0.:
                failed.append(key); violation+=candidate[key]+base[key]+1.
        for metric,limit in GATES.items():
            if metric.startswith('minimum_'):
                name=metric.removeprefix('minimum_')
                # Improvement floors are validation gates, not hard search
                # boundaries that would flatten all initial rewards.
                if name.endswith('cv_gain') and not gate:
                    continue
                delta=limit-c[name]
            elif metric.startswith('maximum_'):
                name=metric.removeprefix('maximum_')
                if name in ('fault_fraction','shield_fraction'):
                    continue
                delta=c[name]-limit
            if delta>1e-12:
                failed.append(metric); violation+=delta
        reward=sum(REWARD[k]*c[k] for k in REWARD if k not in ('constraint_violation_penalty','additional_jerk_penalty'))
        reward-=REWARD['additional_jerk_penalty']*c['additional_jerk_penalty']
        reward-=REWARD['constraint_violation_penalty']*violation
        scores.append(reward); comparisons.append(c)
        rejections.extend(f'constant_{i}: {reason}' for reason in failed)
    return {'score':float(min(scores)), 'comparisons':comparisons,
            'passed':not rejections, 'rejections':rejections}


def save(path,data):
    temporary=path.with_suffix('.next.json')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False))
    temporary.replace(path)


def run_batch(output, name, candidates, seed, *, old=False, checkpoint=None):
    population=output/f'{name}_population.json'; save(population,candidates)
    command=[sys.executable,str(ROOT/'scripts/probe_velocity_feedback.py'),
             '--headless','--device','cpu','--tcp-removal','--parameters-file',str(population),
             '--path-length-mm','698','--max-steps','30000','--seed',str(seed),
             '--surface-cell-mm','.5','--output',str(output/name)]
    if old: command+=['--include-checkpoint',str(OLD)]
    if checkpoint: command+=['--candidate-checkpoint',str(checkpoint)]
    save(output/f'{name}_command.json',command)
    env=dict(os.environ,MPLBACKEND='Agg',MPLCONFIGDIR='/tmp/nrs_tcp_mpl')
    with (output/f'{name}.log').open('w') as stream:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{name} failed ({result.returncode}); inspect its log')
    return json.loads((output/name/'summary.json').read_text())


def train(args):
    torch.set_num_threads(1)
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=False)
    state={'phase':'training','started_at_utc':datetime.now(timezone.utc).isoformat(),
           'algorithm':'cem_episodic_parameter_search','parameter_names':PARAMETER_NAMES,
           'parameter_bounds':[LOW,HIGH], 'objective':'preston_tcp_spatial_removal',
           'rotation_contribution':False,'spindle_rpm':None,'reward':REWARD,'gates':GATES,
           'training_seeds':list(range(105,105+args.generations)), 'held_out_seeds':[205,206,207],
           'selection_uses_held_out_results':False,'physical_depth_validated':False,
           'source_reference_sha256':hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
           'old_checkpoint_sha256':hashlib.sha256(OLD.read_bytes()).hexdigest(),
           'production_checkpoint_replaced':False,'generations':[],'validation':[]}
    files=['scripts/train_tcp_removal.py','scripts/probe_velocity_feedback.py',
           'scripts/skrl/tcp_removal_actor.py','scripts/skrl/ppo_policy_node.py','scripts/skrl/velocity_actor.py']
    task='source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/'
    files += [task+x for x in ('nrs_rl_env_cfg.py','mdp/action.py','mdp/rewards.py','utils/preston_surface.py','utils/velocity_policy.py','assets/assets/robots/ur10_w_spindle.usda','assets/assets/robots/ur10e_w_spindle.py')]
    manifest={}
    for relative in files:
        dest=output/'source_snapshot'/relative; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,dest); manifest[relative]=hashlib.sha256(dest.read_bytes()).hexdigest()
    save(output/'source_manifest.json',manifest)
    save(output/'protocol.json',state)
    def update(**values):
        state.update(values,updated_at_utc=datetime.now(timezone.utc).isoformat())
        save(output/'status.json',state)
        print('TCP_TRAIN_STATUS',json.dumps({k:state[k] for k in ('phase','updated_at_utc','current_batch') if k in state}),flush=True)
    population={'constant_a':ZERO,'constant_b':ZERO,
                'corner_mild':[0.,0.,.15,0.,.03,0.],
                'corner_medium':[0.,0.,.35,0.,.08,0.],
                'corner_strong':[0.,0.,.55,0.,.16,0.],
                'mixed':[.3,.08,.3,.4,.1,.05],
                'force_rate':[.7,.15,0.,0.,0.,0.],
                'damped':[.1,0.,0.,0.,.05,.2]}
    generator=np.random.default_rng(20261005)
    low,high=np.array(LOW),np.array(HIGH)
    incumbent=ZERO
    for generation in range(args.generations):
        name=f'generation_{generation}'
        update(current_batch=name)
        summary=run_batch(output,name,population,105+generation)
        metrics=summary['candidates']; bases=[metrics[x] for x in ('constant_a','constant_b')]
        ranking={k:assess(v,bases) for k,v in metrics.items()}
        ordered=sorted(ranking,key=lambda k:ranking[k]['score'],reverse=True)
        incumbent=population[ordered[0]]
        state['generations'].append({'generation':generation,'seed':105+generation,
            'ranking':ranking,'winner':ordered[0],'selected_parameters':incumbent})
        update(best_parameters=incumbent,best_training_score=ranking[ordered[0]]['score'])
        elites=np.array([population[k] for k in ordered[:3]])
        mean=elites.mean(axis=0); std=np.maximum(elites.std(axis=0),.08*(high-low))
        population={'constant_a':ZERO,'constant_b':ZERO,'incumbent':incumbent}
        for j,p in enumerate(generator.normal(mean,std,(5,6))):
            population[f'proposal_{j}']=np.clip(p,low,high).tolist()
    spec=importlib.util.spec_from_file_location('velocity_policy',ROOT/task/'utils/velocity_policy.py')
    velocity=importlib.util.module_from_spec(spec); spec.loader.exec_module(velocity)
    contract={**SCHEDULER_CONTRACT, 'schema_version':3,'observation_size':14,
        'observation_fields':list(velocity.OBSERVATION_FIELDS),'robot':'ur10_cb3',
        'control_period_s':.008,'policy_action_repeat':1,'policy_period_s':.008,
        'policy_backend':'torchscript_tcp_removal','objective':'preston_tcp_spatial_removal',
        'training_algorithm':state['algorithm'],'rotation_contribution':False,
        'spindle_rpm':None,'tool_diameter_mm':30.,'physics_tool_diameter_mm':30.,
        'learned_parameters':incumbent,'training_seeds':state['training_seeds'],
        'reward':REWARD,'reference_path_sha256':state['source_reference_sha256'],
        'validation_status':'candidate; consult immutable validation result before deployment'}
    checkpoint=output/'checkpoints/actor.pt'; checkpoint.parent.mkdir()
    (output/'params').mkdir(); save(output/'params/velocity_policy.json',contract)
    actor=TcpRemovalActor(incumbent).eval(); scripted=torch.jit.script(actor)
    torch.jit.save(scripted,str(checkpoint),_extra_files={'velocity_policy.json':json.dumps(contract,sort_keys=True)})
    agent,_=_build_agent(str(checkpoint)); _validate_scheduler_contract(agent.velocity_contract)
    rng=torch.Generator().manual_seed(104)
    observations=torch.randn((1024,14),generator=rng); observations[:,10]=1.; observations[:,11]=0.
    error=float((actor(observations)-agent.act(observations)[-1]['mean_actions']).abs().max())
    if error>1e-7: raise RuntimeError('Export changed deterministic inference')
    save(output/'export_validation.json',{'max_abs_error':error,'probe_count':len(observations),
         'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
    update(phase='held_out_validation',checkpoint=str(checkpoint))
    for seed in state['held_out_seeds']:
        name=f'held_out_{seed}'; update(current_batch=name)
        summary=run_batch(output,name,{'constant_a':ZERO,'constant_b':ZERO,'old_d14':ZERO,'learned':ZERO},seed,old=True,checkpoint=checkpoint)
        metrics=summary['candidates']; bases=[metrics[x] for x in ('constant_a','constant_b')]
        result=assess(metrics['learned'],bases,gate=True)
        result.update(seed=seed,versus_old=compare(metrics['learned'],metrics['old_d14']),
                      old_versus_constant=[compare(metrics['old_d14'],b) for b in bases])
        state['validation'].append(result); update()
    passed=all(x['passed'] for x in state['validation'])
    update(phase='complete',validation_passed=passed,
           disposition='eligible_for_deployment_integration' if passed else 'experimental_candidate_rejected_by_frozen_gates')
    save(output/'result.json',state)
    print('TCP_RETRAIN_COMPLETE',str(output/'result.json'),'passed=',passed,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--generations',type=int,default=3)
    args=parser.parse_args()
    if args.generations<2: parser.error('At least two CEM generations are required')
    train(args)
