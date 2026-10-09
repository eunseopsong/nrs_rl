#!/usr/bin/env python3
"""Fit fine force profiles through an identified force-response model.

Only GUI training records and former physics SELECTION seed 2101 are read.
Former holdouts 2111/2112 are retained solely as historical save evidence.
The linear replay assumes fixed footprints; new Isaac trials remain required.
"""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import time
import numpy as np
from scipy import sparse
from scipy.ndimage import uniform_filter1d
from scipy.optimize import minimize,LinearConstraint
from scipy.signal import lfilter
import torch
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.spatial_uniformity import ROOT,DoseGeometry
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_uniformity_priority import ProfileActor,processing_mask,replay_sources,replay_matrix,fit_profile,save,sha

OLD=ROOT/'logs/experiments/uniformity_priority_20261007'
SAVED=ROOT/'logs/experiments/uniformity_saves/mode6_11p14_20261007'


def basis_at(cursor,length,knots):
    phase=np.clip(cursor/length,0.,1.)*(knots-1)
    index=np.minimum(phase.astype(int),knots-2);fraction=phase-index
    basis=np.zeros((len(cursor),knots));row=np.arange(len(cursor))
    basis[row,index]=1.-fraction;basis[row,index+1]=fraction
    return basis


def force_filter(values,time_s,delay_s,tau_s):
    values=np.asarray(values,dtype=float)
    if values.ndim==1:values=values[:,None]
    delayed=np.column_stack([np.interp(time_s-delay_s,time_s,values[:,i]) for i in range(values.shape[1])])
    if tau_s<=0.:return delayed
    dt=float(np.median(np.diff(time_s)));a=np.exp(-dt/tau_s)
    return lfilter([1.-a],[1.,-a],delayed,axis=0,zi=(a*delayed[0])[None,:])[0]


def identify_force_response(data):
    t=data['time'];force=uniform_filter1d(data['force'],11)
    target=data['target'];keep=(t>=3.)&(t<t[-1]-1.)
    if np.std(target[keep])<.5:raise ValueError('Identification requires varying target force')
    rows=[]
    for delay in np.arange(0.,.641,.04):
        for tau in np.arange(.04,1.001,.04):
            filtered=force_filter(target,t,float(delay),float(tau))[:,0]
            design=np.c_[filtered[keep],np.ones(keep.sum())]
            gain,bias=np.linalg.lstsq(design,force[keep],rcond=None)[0]
            if not .5<=gain<=1.5 or not -5.<=bias<=5.:continue
            error=float(np.sqrt(np.mean((design@[gain,bias]-force[keep])**2)))
            rows.append(dict(rmse_n=error,delay_s=float(delay),tau_s=float(tau),gain=float(gain),bias_n=float(bias)))
    if not rows:raise ValueError('No bounded force-response fit')
    return min(rows,key=lambda x:x['rmse_n'])


def physics_source(name):
    file=OLD/'physics/selection_2101'/f'{name}_trace.npz'
    with np.load(file) as d:
        data=dict(time=d['time_s'].copy(),midpoint=d['deposition_midpoint_mm'].copy(),
            distance=np.linalg.norm(d['tangent_velocity_mm_s'],axis=1)*.008,
            force=d['normal_force_n'].copy(),target=d['target_force_n'].copy(),cursor=d['cursor_mm'].copy())
    return data,file


def deposition_operator(geometry,data):
    surface=geometry.surface;mask=processing_mask(surface,geometry.positions)
    inverse=np.full(mask.size,-1,dtype=int);inverse[mask.ravel()]=np.arange(mask.sum())
    rows=[];cols=[];values=[]
    for i in np.flatnonzero((data['time']>=2.)&(data['force']>=1.5)&(data['distance']>0.)):
        rr,cc,_,_,weight,norm=surface._footprint((data['midpoint'][i]-surface.origin)@surface.basis.T)
        ids=inverse[rr*surface.shape[1]+cc];keep=(ids>=0)&(weight>0.)
        rows.append(ids[keep]);cols.append(np.full(keep.sum(),i));values.append(weight[keep]/norm*data['distance'][i])
    return sparse.csr_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),
        shape=(mask.sum(),len(data['time'])))


def affine_objective(profile,statistics,smooth):
    f=np.r_[profile,1.];losses=[];gradients=[]
    for q,m in statistics:
        mean=m@f;second=f@q@f
        losses.append(second/mean**2-1.)
        gradients.append((2*q@f/mean**2-2*second*m/mean**3)[:-1])
    worst=int(np.argmax(losses))
    loss=.75*np.mean(losses)+.25*losses[worst]+1.e-5*(profile@smooth@profile)
    gradient=.75*np.mean(gradients,axis=0)+.25*gradients[worst]+2.e-5*smooth@profile
    return float(loss),gradient


def fit_dynamic(matrices,offsets,initial,bounds,max_delta=3.5):
    n=len(initial);stats=[]
    for matrix,offset in zip(matrices,offsets):
        a=np.column_stack([matrix,offset]);stats.append((a.T@a/len(a),a.mean(axis=0)))
    d2=np.diff(np.eye(n),n=2,axis=0);smooth=d2.T@d2/(n-2)
    constraint=LinearConstraint(np.diff(np.eye(n),axis=0),-max_delta,max_delta)
    fits=[]
    for start in [initial,np.full(n,12.),np.full(n,18.)]:
        opt=minimize(affine_objective,np.clip(start,*bounds),args=(stats,smooth),jac=True,
            bounds=[bounds]*n,constraints=[constraint],method='SLSQP',options={'maxiter':1500,'ftol':1.e-11})
        if not np.isfinite(opt.x).all() or np.max(np.abs(np.diff(opt.x)))>max_delta+1.e-5:
            raise ValueError('Invalid learned force profile')
        fits.append(opt)
    best=min(fits,key=lambda x:x.fun)
    return best.x,dict(objective=float(best.fun),solver_success=bool(best.success),iterations=int(best.nit),
        message=str(best.message),predicted_cv=[float(np.std(a@best.x+b)/np.mean(a@best.x+b)) for a,b in zip(matrices,offsets)])


def main(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);(root/'checkpoints').mkdir()
    torch.set_num_threads(1);started=time.monotonic();geometry=DoseGeometry(cell_mm=2.)
    source=SAVED/'candidate/checkpoints/actor.pt';assert sha(source)=='e31302e7a8edb84ef713f61b080a06e5ff44b5af5ad10dc3031fc26333dc77a6'
    extra={'spatial_policy.json':''};torch.jit.load(str(source),_extra_files=extra)
    old=json.loads(extra['spatial_policy.json']);knots=97
    protocol={**old,'algorithm':'fine_force_profile_optimization_through_identified_force_dynamics',
        'stage':'retraining_from_GUI_and_selection_logs','validation_status':'pending_fresh_physics',
        'source_save_sha256':sha(source),'physics_training_seed':2201,'physics_holdout_seeds':[2211,2212],
        'training_seeds':[],'started_at_utc':datetime.now(timezone.utc).isoformat(),
        'report_comparison_baseline':'Mode 6 only','minimum_mode6_cv_gain_each_holdout':.1114,
        'must_not_regress_against_preserved_checkpoint_on_same_seed':True,
        'profile_knots':knots,'total_removal_is_selection_gate':False,'deployed':False,
        'training_data_excludes_previous_holdouts':[2111,2112]}
    for key in ['force_profile_n','candidate','previous_checkpoint_sha256']:
        protocol.pop(key,None)
    save(root/'protocol.json',protocol)
    latest,file=physics_source('profile_wide');response=identify_force_response(latest)
    save(root/'force_response_identification.json',{'training_file':str(file),'fit':response,
        'note':'First-order plus transport delay; linear replay approximation, not a physics validation'})
    print('IDENTIFIED_FORCE_RESPONSE',json.dumps(response),flush=True)
    sources=list(replay_sources())
    sources.append(('selected_2101',latest,[file]))
    sources.append(('robust_2101',*physics_source('profile_robust')))
    # Normalize the last source file container to the replay-source convention.
    sources[-1]=(sources[-1][0],sources[-1][1],[sources[-1][2]])
    static_matrices=[];dynamic_matrices=[];offsets=[];manifest={};audit=[]
    for name,data,files in sources:
        for file in files:manifest[str(file)]=sha(file)
        matrix,actual=replay_matrix(geometry,data,knots,0.)
        static_matrices.append(matrix)
        operator=deposition_operator(geometry,data)
        # A small delay variation covers timing mismatch across native/ROS execution.
        for delay_delta in [-.08,.08]:
            basis=basis_at(data['cursor'],geometry.length,knots)
            filtered=force_filter(basis,data['time'],max(0.,response['delay_s']+delay_delta),response['tau_s'])
            dynamic_matrices.append(operator@filtered*response['gain'])
            offsets.append(np.asarray(operator@np.full(len(data['time']),response['bias_n'])))
        audit.append({'source':name,'samples':len(data['time']),'observed_replay_cv':float(actual.std()/actual.mean()),
            'roi_cells':len(actual)})
        print('REPLAY_BUILT',name,flush=True)
    save(root/'source_data_manifest.json',manifest);save(root/'replay_audit.json',audit)
    initial=np.interp(np.linspace(0,1,knots),np.linspace(0,1,len(old['force_profile_n'])),old['force_profile_n'])
    fine,fit_fine=fit_profile(static_matrices,args.seed,bounds=(8.,30.),max_knot_delta=3.5)
    print('PROFILE_FIT','fine_static',fit_fine,flush=True)
    dynamic,fit_8=fit_dynamic(dynamic_matrices,offsets,initial,(8.,30.))
    print('PROFILE_FIT','dynamic_8n',fit_8,flush=True)
    contrast,fit_4=fit_dynamic(dynamic_matrices,offsets,initial,(4.,30.))
    print('PROFILE_FIT','dynamic_4n',fit_4,flush=True)
    definitions={'fine_static':(fine,8.,12.,fit_fine),'dynamic_8n':(dynamic,8.,12.,fit_8),
        'dynamic_4n':(contrast,4.,16.,fit_4)}
    save(root/'training_results.json',{name:{'profile_n':p.tolist(),**fit} for name,(p,low,scale,fit) in definitions.items()})
    shutil.copyfile(OLD/'training_final/checkpoints/constant.pt',root/'checkpoints/constant.pt')
    shutil.copyfile(source,root/'checkpoints/saved.pt')
    rng=np.random.default_rng(args.seed);probe=torch.tensor(rng.uniform(-5,5,(2048,16)),dtype=torch.float32);exports={}
    for name,(profile,lower,scale,fit) in definitions.items():
        actor=ProfileActor(profile,old['feed_parameters'],force_min=lower,force_max=30.,force_scale=scale)
        meta={**protocol,'candidate':name,'force_profile_n':profile.tolist(),
            'force_bounds_n':[lower,30.],'force_action_mapping':{'center_n':20.,'scale_n':scale},
            'action_mapping':['6+3*a0 mm/s',f'20+{scale:g}*a1 N','local return disabled'],
            'identified_force_response':response if name.startswith('dynamic') else None,
            'maximum_adjacent_profile_delta_n':3.5}
        checkpoint=root/'checkpoints'/f'{name}.pt'
        torch.jit.save(torch.jit.script(actor),str(checkpoint),_extra_files={'spatial_policy.json':json.dumps(meta)})
        loaded=torch.jit.load(str(checkpoint)).eval()
        with torch.inference_mode():actual=loaded(probe);expected=actor(probe)
        torch.testing.assert_close(actual,expected)
        assert torch.isfinite(actual).all() and actual.abs().max()<=1.000001
        force=20.+scale*actual[:,1];assert force.min()>=lower-1.e-5 and force.max()<=30.+1.e-5
        exports[name]={'sha256':sha(checkpoint),'parity_max_error':float((actual-expected).abs().max())}
    for name in ['constant','saved']:exports[name]={'sha256':sha(root/'checkpoints'/f'{name}.pt')}
    save(root/'export_validation.json',exports);save(root/'tracking.json',{n:0. if n=='constant' else .6 for n in exports})
    for name in ['train_uniformity_refinement.py','train_uniformity_priority.py']:
        dest=root/'source_snapshot'/name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(ROOT/'scripts'/name,dest)
    save(root/'status.json',{'phase':'training_complete_pending_physics','elapsed_s':time.monotonic()-started,
        'exports':exports,'comparison_baseline':'Mode 6 only'})
    print('REFINEMENT_TRAINING_COMPLETE',str(root),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=2026100721);main(parser.parse_args())
