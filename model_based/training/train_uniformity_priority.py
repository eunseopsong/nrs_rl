#!/usr/bin/env python3
"""Uniformity-only retraining from GUI footprints and episodic policy search.

Logged-trajectory dose optimization is a replay surrogate, not a claim that
counterfactual forces reproduce the same robot motion. Fresh Isaac rollouts
must select and validate these candidates before any deployment change.
"""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import torch
import nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.spatial_uniformity as spatial
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_spatial_refinement import capped_action, CappedSpatialActor

ROOT=spatial.ROOT
INCUMBENT=ROOT/'logs/experiments/spatial_uniformity_20261007/refined_physics/candidate/checkpoints/actor.pt'
GUI=ROOT/'logs/polishing_results/20261007_1900'
PHYSICS=ROOT/'logs/experiments/spatial_uniformity_20261007/refined_physics/train_1701'


def save(path,data):
    path=Path(path);temporary=path.with_suffix('.next.json')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def processing_mask(surface,path):
    roi=surface.roi.copy();arc=np.r_[0.,np.linalg.norm(np.diff(path,axis=0),axis=1).cumsum()]
    for point in path[arc<=18.]:
        rows,cols,_,_,weights,_=surface._footprint((point-surface.origin)@surface.basis.T)
        roi[rows[weights>0],cols[weights>0]]=False
    return roi


def replay_sources():
    for name in ('mode5','mode6'):
        with np.load(GUI/name/'derived_signals.npz') as d, np.load(GUI/name/'processing_control.npz') as c:
            t=d['time_s'];ct=c['time']-c['time'][0]
            last=np.r_[np.diff(ct)>0,True]
            # Both recordings start at Force entry. Arrival-order alignment
            # uncertainty is covered by +/- 2 mm profile shifts below.
            cursor=np.interp(t,ct[last],c['trajectory_cursor'][last])
            target=np.interp(t,ct[last],c['target_fz'][last])
            xyz=d['xyz_mm'];delta=np.diff(xyz,axis=0,prepend=xyz[:1]);delta[:,2]=0.
            yield name,dict(time=t,midpoint=xyz-.5*delta,distance=np.linalg.norm(delta,axis=1),
                force=d['force_n'],target=np.abs(target),cursor=cursor),[GUI/name/'derived_signals.npz',GUI/name/'processing_control.npz']
    with np.load(PHYSICS/'learned_trace.npz') as d:
        yield 'headless_training',dict(time=d['time_s'],midpoint=d['deposition_midpoint_mm'],
            distance=np.linalg.norm(d['tangent_velocity_mm_s'],axis=1)*.008,
            force=d['normal_force_n'],target=d['target_force_n'],cursor=d['cursor_mm']),[PHYSICS/'learned_trace.npz']


def replay_matrix(geometry,data,knots,shift):
    surface=geometry.surface;mask=processing_mask(surface,geometry.positions)
    inverse=np.full(mask.size,-1,dtype=int);inverse[mask.ravel()]=np.arange(mask.sum())
    matrix=np.zeros((mask.sum(),knots));actual=np.zeros(mask.sum())
    for i in np.flatnonzero((data['time']>=2.)&(data['force']>=1.5)&(data['distance']>0)):
        rows,cols,_,_,w,norm=surface._footprint((data['midpoint'][i]-surface.origin)@surface.basis.T)
        ids=inverse[rows*surface.shape[1]+cols];keep=(ids>=0)&(w>0);ids=ids[keep]
        weight=w[keep]/norm*data['distance'][i]
        actual[ids]+=weight*data['force'][i]
        # Estimate steady force tracking gain without treating the entry
        # transient as a material/removal coefficient.
        gain=np.clip(data['force'][i]/max(data['target'][i],1.),.75,1.25)
        position=np.clip((data['cursor'][i]+shift)/geometry.length,0.,1.)*(knots-1)
        index=min(int(position),knots-2);fraction=position-index
        matrix[ids,index]+=weight*gain*(1.-fraction)
        matrix[ids,index+1]+=weight*gain*fraction
    return matrix,actual


class ProfileActor(torch.nn.Module):
    def __init__(self,profile,feed,online_gain=0.,force_min=14.,force_max=24.,force_scale=6.):
        super().__init__()
        self.register_buffer('profile',torch.tensor(profile,dtype=torch.float32))
        self.register_buffer('feed',torch.tensor(feed,dtype=torch.float32))
        self.online_gain=float(online_gain)
        self.force_min=float(force_min);self.force_max=float(force_max);self.force_scale=float(force_scale)

    def forward(self,obs):
        phase=torch.clamp(obs[:,0],0.,1.)*(self.profile.numel()-1)
        index=torch.clamp(phase.to(torch.long),0,self.profile.numel()-2)
        force=self.profile[index]+(phase-index)*(self.profile[index+1]-self.profile[index])
        force=torch.clamp(force+20.*self.online_gain*(obs[:,4]-obs[:,3]),self.force_min,self.force_max)
        p=self.feed
        speed=torch.clamp(6.*(1.+p[0]*(20./force-1.)-p[1]*obs[:,13]+p[2]-p[3]*obs[:,7]),3.,9.)
        return torch.stack(((speed-6.)/3.,(force-20.)/self.force_scale,torch.full_like(force,-1.)),dim=1)


def fit_profile(matrices,seed,bounds=(14.,24.),max_knot_delta=4.):
    n=matrices[0].shape[1];statistics=[]
    for a in matrices:
        statistics.append((a.T@a/len(a),a.mean(axis=0)))
    d2=np.diff(np.eye(n),n=2,axis=0);smooth=d2.T@d2/max(1,n-2)
    def objective(f):
        losses=[];gradients=[]
        for q,m in statistics:
            mean=m@f;second=f@q@f
            losses.append(second/mean**2-1.)
            gradients.append(2*q@f/mean**2-2*second*m/mean**3)
        # Optimize average CV squared with a small worst-scenario contribution.
        worst=int(np.argmax(losses));value=.75*np.mean(losses)+.25*losses[worst]
        gradient=.75*np.mean(gradients,axis=0)+.25*gradients[worst]
        return float(value+2.e-5*(f@smooth@f)),gradient+4.e-5*smooth@f
    difference=np.diff(np.eye(n),axis=0)
    constraint=LinearConstraint(difference,-max_knot_delta,max_knot_delta)
    rng=np.random.default_rng(seed);results=[]
    for initial in (np.full(n,20.),np.full(n,16.),rng.uniform(16.,22.,n)):
        opt=minimize(objective,initial,jac=True,bounds=[bounds]*n,
            constraints=[constraint],method='SLSQP',options={'maxiter':1000,'ftol':1.e-11})
        if not np.isfinite(opt.x).all() or np.max(np.abs(np.diff(opt.x)))>max_knot_delta+1.e-5:
            raise ValueError('Invalid optimized force profile')
        results.append(opt)
    result=min(results,key=lambda r:r.fun)
    return result.x,{'objective':float(result.fun),'solver_success':bool(result.success),
        'message':str(result.message),'iterations':int(result.nit),
        'predicted_cv_by_scenario':[float(np.std(a@result.x)/np.mean(a@result.x)) for a in matrices]}


def main(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);(root/'checkpoints').mkdir()
    torch.set_num_threads(1);started=time.monotonic();geometry=spatial.DoseGeometry(cell_mm=2.)
    extra={'spatial_policy.json':''};incumbent=torch.jit.load(str(INCUMBENT),_extra_files=extra).eval()
    previous=json.loads(extra['spatial_policy.json']);oldp=np.array(previous['parameters'])
    contract={**previous,'algorithm':'GUI_replay_CV_profile_optimization_and_CEM_feedback_search',
        'objective':'minimize processing spatial depth CV; no total-removal or mean-force penalty',
        'gates':{'completed':True,'no_controller_fault':True,'positive_finite_depth':True},
        'entry_seconds':2.,'entry_reference_footprint_mm':18.,'hard_force_stop_n':100.,
        'entry_peak_force_is_selection_gate':False,'total_removal_is_selection_gate':False,
        'training_seeds':[2121,2122],'surrogate_holdout_seeds':[],
        'target_depth_is_feature_reference_only':True,
        'physics_training_seed':2101,'physics_holdout_seeds':[2111,2112],
        'holdouts_used_for_selection':False,'deployed':False,'validation_status':'pending_fresh_physics',
        'training_gui_run':'spatial_20261007_184916','previous_checkpoint_sha256':sha(INCUMBENT),
        'started_at_utc':datetime.now(timezone.utc).isoformat()}
    contract.pop('refinement_basis',None);contract.pop('next_physics_training_seed',None);contract.pop('next_physics_holdout_seeds',None)
    save(root/'protocol.json',contract)
    matrix_groups={};audit={};sources={}
    for name,data,files in replay_sources():
        matrix_groups[name]=[]
        for shift in (-2.,0.,2.):
            matrix,actual=replay_matrix(geometry,data,49,shift)
            matrix_groups[name].append(matrix)
        audit[name]={'observed_replay_cv_2mm':float(actual.std()/actual.mean()),
            'samples':len(data['time']),'processing_roi_cells':len(actual)}
        sources.update({str(p):sha(p) for p in files})
        print('REPLAY_MATRIX',name,audit[name],flush=True)
    save(root/'source_data_manifest.json',sources);save(root/'replay_audit.json',audit)
    gui_profile,gui_fit=fit_profile(matrix_groups['mode5']+matrix_groups['mode6'],args.seed)
    robust_profile,robust_fit=fit_profile(sum(matrix_groups.values(),[]),args.seed+1)
    wide_profile,wide_fit=fit_profile(sum(matrix_groups.values(),[]),args.seed+2,bounds=(8.,30.),max_knot_delta=6.)
    save(root/'profile_training.json',{'gui':{'force_n':gui_profile.tolist(),**gui_fit},
        'robust':{'force_n':robust_profile.tolist(),**robust_fit},
        'wide':{'force_n':wide_profile.tolist(),**wide_fit},
        'limitation':'Replay assumes measured footprint paths persist under changed force; physics checks required'})
    print('PROFILE_TRAINED',json.dumps({'gui':gui_fit,'robust':robust_fit,'wide':wide_fit}),flush=True)
    # Reward is exclusively processing spatial uniformity. Keep the same
    # force/feed bounds and completed single-path requirement.
    spatial.action_numpy=capped_action
    low=np.array([0.,0.,-.25,0.,0.,-.12,0.,0.,.025])
    high=np.array([8.,3.,.20,1.5,.12,.12,.5,2.,.6])
    roi=processing_mask(geometry.surface,geometry.positions).ravel()[geometry.roi_flat]
    mean=np.clip((oldp-low)/(high-low),0,1);std=np.full(9,.20);best=oldp.copy()
    rng=np.random.default_rng(args.seed);best_score=float('-inf');history=[]
    for generation in range(args.generations):
        draws=np.clip(rng.normal(mean,std,(args.population-2,9)),0,1)
        population=np.vstack([oldp,best,low+draws*(high-low)])
        scores=[];cvs=[]
        for seed in (2121,2122):
            metrics,depth,_,_=geometry.fast_rollout(population,seed=seed)
            values=depth[:,roi];cv=values.std(axis=1)/values.mean(axis=1)
            score=-cv-10*np.array([not m['completed'] for m in metrics])
            scores.append(score);cvs.append(cv.tolist())
        score=np.min(scores,axis=0);order=np.argsort(score)[::-1];best=population[order[0]].copy()
        elites=(population[order[:max(4,args.population//5)]]-low)/(high-low)
        mean=.25*mean+.75*elites.mean(axis=0);std=np.maximum(.035,.25*std+.75*elites.std(axis=0))
        best_score=max(best_score,float(score[order[0]]))
        item={'generation':generation+1,'best_score':float(score[order[0]]),'best_parameters':best.tolist(),
            'training_cv':cvs,'elapsed_s':time.monotonic()-started}
        history.append(item);save(root/'status.json',{'phase':'CEM_training',**item})
        print('UNIFORMITY_CEM',json.dumps({k:v for k,v in item.items() if k!='training_cv'}),flush=True)
    save(root/'cem_history.json',history)
    constant=np.array([0.,0.,0.,0.,0.,0.,0.,0.,.6])
    models={'constant':CappedSpatialActor(constant),'incumbent':incumbent,
        'profile_gui':ProfileActor(gui_profile,oldp[[3,4,5,6]]),
        'profile_robust':ProfileActor(robust_profile,oldp[[3,4,5,6]]),
        'profile_wide':ProfileActor(wide_profile,oldp[[3,4,5,6]],force_min=8.,force_max=30.,force_scale=12.),
        'feedback':CappedSpatialActor(best)}
    exports={};probe=torch.tensor(rng.uniform(-5,5,(2048,16)),dtype=torch.float32)
    for name,actor in models.items():
        path=root/'checkpoints'/f'{name}.pt'
        if name=='incumbent':shutil.copyfile(INCUMBENT,path)
        else:
            meta={**contract,'candidate':name,'tracking_compensation':{'gain':0. if name=='constant' else .6,'tau_s':.08,'limit_mm':4.}}
            if name.startswith('profile_'):
                meta.update(force_profile_n=actor.profile.tolist(),profile_knots=49,profile_phase='reference cursor fraction',
                    feed_parameters=actor.feed.tolist(),return_action='disabled for profile candidates',
                    force_bounds_n=[actor.force_min,actor.force_max],
                    force_action_mapping={'center_n':20.,'scale_n':actor.force_scale})
                meta['action_mapping']=['6+3*a0 mm/s',f'20+{actor.force_scale:g}*a1 N','local return disabled']
                meta.pop('parameters',None);meta.pop('parameter_names',None)
            else:meta['parameters']=(constant if name=='constant' else best).tolist()
            torch.jit.save(torch.jit.script(actor),str(path),_extra_files={'spatial_policy.json':json.dumps(meta)})
        loaded=torch.jit.load(str(path)).eval()
        with torch.inference_mode():
            result=loaded(probe);expected=actor(probe)
        torch.testing.assert_close(result,expected)
        if not torch.isfinite(result).all() or result.abs().max()>1.000001:raise ValueError('Invalid actor output')
        scale=actor.force_scale if isinstance(actor,ProfileActor) else 6.
        cap=actor.force_max if isinstance(actor,ProfileActor) else 24.
        if torch.max(20.+scale*result[:,1])>cap+1.e-5:raise ValueError('Wrong force cap')
        exports[name]={'sha256':sha(path),'shape':list(result.shape),'parity_max_error':float((result-expected).abs().max())}
    save(root/'export_validation.json',exports)
    save(root/'tracking.json',{name:0. if name=='constant' else .6 for name in models})
    snapshot=root/'source_snapshot';snapshot.mkdir()
    for name in ('train_uniformity_priority.py','spatial_uniformity.py','train_spatial_refinement.py'):
        shutil.copyfile(ROOT/'scripts'/name,snapshot/name)
    save(root/'status.json',{'phase':'training_complete_pending_physics','elapsed_s':time.monotonic()-started,
        'candidate_names':list(models),'exports':exports,'total_removal_constraint':False})
    print('UNIFORMITY_TRAINING_COMPLETE',str(root),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--generations',type=int,default=16);p.add_argument('--population',type=int,default=48)
    p.add_argument('--seed',type=int,default=2026100719);main(p.parse_args())
