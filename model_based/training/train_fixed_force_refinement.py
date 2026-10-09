#!/usr/bin/env python3
"""Additional residence/feed training while preserving the validated save."""
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
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import torch

from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.fixed_force_velocity import REFERENCE, rotating_surface
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.fixed_force_refinement import export_actor
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_fixed_force_velocity import build_matrix, processing_mask, save, sha

ROOT = _REPO_ROOT
SAVED = ROOT/'logs/experiments/uniformity_saves/fixed20_virtual1000_velocity_20261009_0131'
SAVED_SHA = '1ed31e17c012d1f7e700926a3a6cc23580be845a2b17e0627b3aa280da625a11'
PREVIOUS = ROOT/'logs/experiments/fixed_force_rotation_20261009'
SELECTION_SEED = 2701
HOLDOUTS = [2711, 2712]


def check_preserved():
    for rel,digest in json.loads((SAVED/'save_manifest.json').read_text()).items():
        assert sha(SAVED/rel) == digest, rel
    assert sha(SAVED/'candidate/checkpoints/actor.pt') == SAVED_SHA


def replay_matrix(path, trace, knots, *, nominalize=False):
    if not nominalize:
        return build_matrix(path, knots, trace=trace)[0]
    surface=rotating_surface(path)
    mask=processing_mask(surface,path)
    inverse=np.full(mask.size,-1,dtype=int)
    inverse[mask.ravel()]=np.arange(mask.sum())
    length=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())
    with np.load(trace) as f:data=dict(f)
    assert np.all(data['target_force_n']==20.)
    # Cursor differences are the applied forward feed, including the limiter.
    ds=np.diff(data['cursor_mm'],prepend=data['cursor_mm'][0])
    assert ds.min()>=-1.e-5
    matrix=np.zeros((mask.sum(),knots))
    keep=(data['time_s']>=2.)&(data['normal_force_n']>=1.5)&(data['safety_fault_reason']==0)&(ds>0)
    for i in np.flatnonzero(keep):
        center=(data['deposition_midpoint_mm'][i]-surface.origin)@surface.basis.T
        rr,cc,dx,dy,w,norm=surface._footprint(center)
        ids=inverse[rr*surface.shape[1]+cc];valid=(ids>=0)&(w>0)
        feed=data['tangent_velocity_mm_s'][i]@surface.basis.T
        relative=np.hypot(feed[0]-surface.omega*dy,feed[1]+surface.omega*dx)
        # Nominalize the captured footprint to ds/6; the learned residence
        # multiplier then retimes it. This does not predict changed dynamics.
        value=data['normal_force_n'][i]*w[valid]/norm*relative[valid]*ds[i]/6.
        phase=np.clip(data['cursor_mm'][i]/length,0.,1.)*(knots-1)
        left=min(int(phase),knots-2);fraction=phase-left
        matrix[ids[valid],left]+=value*(1.-fraction)
        matrix[ids[valid],left+1]+=value*fraction
    return matrix


def fit(matrices, *, volume_ratio, minimum_speed, maximum_speed, knot_delta):
    knots=matrices[0].shape[1]
    stats=[]
    for matrix in matrices:
        a=matrix/matrix.sum(axis=1).mean()
        stats.append((a.T@a/len(a),a.mean(axis=0)))
    difference=np.diff(np.eye(knots),axis=0)
    smooth=difference.T@difference/knots
    def objective(x):
        losses=[];grads=[]
        for q,mean in stats:
            m=mean@x;square=x@q@x
            losses.append(square/m**2-1.)
            grads.append(2.*q@x/m**2-2.*square*mean/m**3)
        worst=int(np.argmax(losses))
        return (float(.75*np.mean(losses)+.25*losses[worst]+1.e-5*(x@smooth@x)),
                .75*np.mean(grads,axis=0)+.25*grads[worst]+2.e-5*smooth@x)
    mean=np.mean([item[1] for item in stats],axis=0)
    result=minimize(objective,np.full(knots,volume_ratio),jac=True,
        bounds=[(6./maximum_speed,6./minimum_speed)]*knots,
        constraints=[LinearConstraint(mean[None,:],volume_ratio,volume_ratio),
                     LinearConstraint(difference,-knot_delta,knot_delta)],
        method='SLSQP',options={'ftol':1.e-11,'maxiter':1200})
    if (not result.success or abs(mean@result.x-volume_ratio)>1.e-6
            or np.max(np.abs(np.diff(result.x)))>knot_delta+1.e-6):
        raise RuntimeError(str(result.message))
    predictions=[]
    for q,m in stats:
        cv=np.sqrt(max(0.,result.x@q@result.x/(m@result.x)**2-1.))
        base=np.sqrt(max(0.,np.ones(knots)@q@np.ones(knots)-1.))
        predictions.append({'cv':float(cv),'nominal_mode6_cv_gain':float(1.-cv/base),
                            'mode6_volume_ratio':float(m@result.x)})
    return result.x,{'solver_success':True,'iterations':int(result.nit),
                     'predictions':predictions,'qualification':'frozen-footprint model fit only; not fresh Isaac validation'}


def main(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);(root/'checkpoints').mkdir()
    torch.set_num_threads(1);check_preserved()
    protocol={
        'objective':'minimize processing spatial CV, subject to >=50% Mode 6 removal in all scopes',
        'target_force_n':20.,'policy_controls_force':False,'virtual_spindle_rpm':1000.,
        'urdf_rotation_unchanged':True,'second_pass':False,'local_reverse':False,
        'selection_seed':SELECTION_SEED,'heldout_seeds':HOLDOUTS,'holdouts_used_for_selection':False,
        'excluded_training_seeds':[2611,2612],
        'saved_checkpoint':str(SAVED/'candidate/checkpoints/actor.pt'),'saved_sha256':SAVED_SHA,
        'acceptance':'Both fresh heldouts: eligible and CV gain exceeds saved by >=0.005 in Mode6-normalized units; save is never overwritten by a rejected candidate',
        'minimum_mode6_normalized_cv_margin':.005,'minimum_mode6_volume_ratio':.5,
        'volume_scopes':['processing_roi','full_roi','whole_grid'],
        'entry_seconds':2.,'entry_reference_mm':18.,'hard_force_stop_n':100.,
        'maximum_feed_mm_s':12.,'minimum_moving_feed_mm_s':.75,
        'acceleration_limit_mm_s2':16.,'jerk_limit_mm_s3':160.,
        'physical_action_slew_mm_s2':18.,'tracking_gain_fixed':.6,
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'training_method':'constrained model-based residence optimization, not PPO',
        'report_baseline':'matched Mode 6 only','gui_deployment':False}
    save(root/'protocol.json',protocol)
    with h5py.File(REFERENCE) as f:path=f['position'][:,:3]
    length=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())
    traces={
        'mode6_2601':PREVIOUS/'physics/selection_b_2601/constant_trace.npz',
        'saved_2601':PREVIOUS/'physics/selection_b_2601/profile193_100_trace.npz'}
    save(root/'training_data_manifest.json',{str(p):sha(p) for p in traces.values()})
    # Wider speed authority, increased local residence, finer spatial sampling,
    # and command lookahead are separate candidates, under the same force/RPM.
    specs={
        'wide193_90':dict(knots=193,volume_ratio=.90,minimum_speed=1.5,maximum_speed=12.,knot_delta=.22,phase_lead_mm=0.,sources=['nominal','mode6_2601']),
        'wide193_110':dict(knots=193,volume_ratio=1.10,minimum_speed=1.5,maximum_speed=12.,knot_delta=.22,phase_lead_mm=0.,sources=['nominal','mode6_2601']),
        'dwell193_135':dict(knots=193,volume_ratio=1.35,minimum_speed=.75,maximum_speed=9.,knot_delta=.30,phase_lead_mm=0.,sources=['nominal','mode6_2601']),
        'dense289_110':dict(knots=289,volume_ratio=1.10,minimum_speed=1.5,maximum_speed=12.,knot_delta=.18,phase_lead_mm=0.,sources=['nominal','mode6_2601']),
        'robust193_110':dict(knots=193,volume_ratio=1.10,minimum_speed=1.5,maximum_speed=12.,knot_delta=.22,phase_lead_mm=0.,sources=['mode6_2601','saved_2601']),
        'lead193_110':dict(knots=193,volume_ratio=1.10,minimum_speed=1.5,maximum_speed=12.,knot_delta=.22,phase_lead_mm=.8,sources=['nominal','mode6_2601']),
    }
    save(root/'candidate_specs.json',specs)
    results={};cache={}
    for name,spec in specs.items():
        matrices=[]
        for source in spec['sources']:
            key=(source,spec['knots'])
            if key not in cache:
                if source=='nominal':cache[key]=build_matrix(path,spec['knots'])[0]
                else:cache[key]=replay_matrix(path,traces[source],spec['knots'],nominalize=(source=='saved_2601'))
            matrices.append(cache[key])
        params={k:spec[k] for k in ['volume_ratio','minimum_speed','maximum_speed','knot_delta']}
        if name=='lead193_110':
            q=np.array(results['wide193_110']['residence_profile'])
            metrics=results['wide193_110']['fit']
        else:q,metrics=fit(matrices,**params)
        contract=export_actor(root/'checkpoints'/f'{name}.pt',q.tolist(),
            minimum_speed=spec['minimum_speed'],maximum_speed=spec['maximum_speed'],
            phase_lead_mm=spec['phase_lead_mm'],path_length_mm=length,candidate=name,protocol=protocol)
        results[name]={'spec':spec,'fit':metrics,'residence_profile':q.tolist(),
                       'sha256':sha(root/'checkpoints'/f'{name}.pt')}
        save(root/'training_results.json',results)
        print('FIXED_FORCE_REFINEMENT_FIT',name,json.dumps(metrics),flush=True)
    shutil.copyfile(SAVED/'candidate/checkpoints/actor.pt',root/'checkpoints/saved.pt')
    shutil.copyfile(PREVIOUS/'training/checkpoints/constant.pt',root/'checkpoints/constant.pt')
    save(root/'checkpoint_manifest.json',{p.stem:sha(p) for p in (root/'checkpoints').glob('*.pt')})
    save(root/'status.json',{'phase':'trained_pending_physics','protocol':protocol,'candidates':list(specs)})
    check_preserved()
    print('FIXED_FORCE_REFINEMENT_TRAINED',str(root),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
