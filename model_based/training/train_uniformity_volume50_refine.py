#!/usr/bin/env python3
"""Refine the preserved dose-constrained policy using finer deposition grids.

Fit path force allocation through newly identified dynamics; one candidate
jointly changes force/feed coupling with a counterfactual time warp. Recorded
footprints remain a replay approximation, so paired Isaac validation is required.
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
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.spatial_uniformity import ROOT, DoseGeometry
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_uniformity_priority import ProfileActor, replay_sources, save, sha
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_uniformity_refinement import basis_at, deposition_operator, force_filter, identify_force_response
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_uniformity_volume50 import read_trace, fit_constrained

PREVIOUS = ROOT / 'logs/experiments/uniformity_volume50_20261008'
SAVED = ROOT / 'logs/experiments/uniformity_saves/mode6_13p00_volume50_20261008'
SAVED_SHA = 'f55651593363ce3a57e07d7ae1a311a74343a37ca19a5e92c3e7968ce4a5d998'
NEW = ['fine145_52','fine193_52','fine145_55','timed193_52']
SELECTION_SEED = 2401
HOLDOUTS = [2411,2412]
MINIMUM_ADVANTAGE = .001  # A margin in Mode 6-normalized CV, not a percent vs saved model.


def nominal_speed(force, feed):
    return np.clip(6.*(1.+feed[0]*(20./np.maximum(force,4.)-1.)+feed[2]),3.,9.)


def counterfactual_time(data, profile, length, old_feed, new_feed):
    """Preserve path order/distances; adjust dwell time using force/feed commands."""
    force = np.interp(data['cursor']/length,np.linspace(0,1,len(profile)),profile)
    scale = nominal_speed(data['target'],old_feed)/nominal_speed(force,new_feed)
    scale = np.clip(scale,.75,1.4)
    dt = np.diff(data['time'])*.5*(scale[:-1]+scale[1:])
    return np.r_[data['time'][0],data['time'][0]+np.cumsum(dt)]


def response_irregular(values,time_s,delay_s,tau_s):
    """First-order response on a nonuniform clock (causal exponential update)."""
    if np.any(np.diff(time_s)<=0.): raise ValueError('Time must be strictly increasing')
    delayed = np.column_stack([np.interp(time_s-delay_s,time_s,values[:,i]) for i in range(values.shape[1])])
    out = np.empty_like(delayed);out[0] = delayed[0]
    for i,dt in enumerate(np.diff(time_s),1):
        a = np.exp(-dt/tau_s);out[i] = a*out[i-1]+(1.-a)*delayed[i]
    return out


def main(args):
    root = args.output.resolve();root.mkdir(parents=True,exist_ok=False);(root/'checkpoints').mkdir()
    started = time.monotonic();torch.set_num_threads(1)
    source = SAVED/'candidate/checkpoints/actor.pt';assert sha(source)==SAVED_SHA
    for name,digest in json.loads((SAVED/'save_manifest.json').read_text()).items():assert sha(SAVED/name)==digest
    extra = {'spatial_policy.json':''};torch.jit.load(str(source),_extra_files=extra);old = json.loads(extra['spatial_policy.json'])
    geometry = DoseGeometry(cell_mm=1.)
    protocol = {**old,'candidate':None,'force_profile_n':None,
        'algorithm':'fine_grid_volume_constrained_profile_and_feed_timing_search',
        'stage':'learning_from_previous_GUI_and_selection_2301',
        'training_surface_cell_mm':1.,'evaluation_surface_cell_mm':.5,
        'minimum_mode6_volume_ratio':.5,'source_save_sha256':SAVED_SHA,'preserved_feasible_sha256':SAVED_SHA,
        'training_data_excludes_previous_holdouts':[2111,2112,2211,2212,2311,2312],
        'physics_training_seed':SELECTION_SEED,'physics_holdout_seeds':HOLDOUTS,
        'minimum_mode6_cv_gain_each_holdout':.1114,'minimum_mode6_normalized_advantage_over_save':MINIMUM_ADVANTAGE,
        'holdouts_used_for_selection':False,'validation_status':'pending_fresh_physics',
        'started_at_utc':datetime.now(timezone.utc).isoformat(),'deployed':False}
    protocol.pop('preserved_cv_only_sha256',None)
    save(root/'protocol.json',protocol)
    directory = PREVIOUS/'physics/selection_2301'
    sources = list(replay_sources())[:2]
    for name in ['profile145_55','profile97_62','scaled_55']:
        file = directory/f'{name}_trace.npz';sources.append((name,read_trace(file),[file]))
    responses = {name:identify_force_response(read_trace(directory/f'{name}_trace.npz')) for name in ['profile145_55','profile97_62']}
    save(root/'force_response_identification.json',responses);print('FORCE_RESPONSES',json.dumps(responses),flush=True)
    baseline_file = directory/'constant_trace.npz';baseline_data = read_trace(baseline_file)
    baselines = {'gui':float(np.sum(deposition_operator(geometry,sources[1][1])@sources[1][1]['force'])*geometry.surface.area),
                 'physics':float(np.sum(deposition_operator(geometry,baseline_data)@baseline_data['force'])*geometry.surface.area)}
    manifest = {str(baseline_file):sha(baseline_file)};operators = []
    for name,data,files in sources:
        manifest.update({str(p):sha(p) for p in files})
        operator = deposition_operator(geometry,data)
        operators.append((name,data,operator,baselines['gui' if name.startswith('mode') else 'physics']))
        print('FINE_OPERATOR',name,'cells',operator.shape[0],flush=True)
    save(root/'source_data_manifest.json',manifest);save(root/'paired_baselines_1mm.json',baselines)
    feed = np.array(old['feed_parameters']);slow_feed = feed.copy();slow_feed[0] *= .70
    initial_profile = np.array(old['force_profile_n'])*(.52/.55)
    caches = {};scenarios = {}
    for knots,timed in [(145,False),(193,False),(193,True)]:
        key = f'{knots}_'+('timed' if timed else 'recorded')
        matrices,offsets,volumes,scenarios[key] = [],[],[],[]
        for name,data,operator,baseline in operators:
            for response_name,response in responses.items():
                basis = basis_at(data['cursor'],geometry.length,knots)
                if timed:
                    clock = counterfactual_time(data,initial_profile,geometry.length,feed,slow_feed)
                    filtered = response_irregular(basis,clock,response['delay_s'],response['tau_s'])
                else:
                    filtered = force_filter(basis,data['time'],response['delay_s'],response['tau_s'])
                matrices.append(operator@filtered*response['gain'])
                offsets.append(np.asarray(operator@np.full(len(data['time']),response['bias_n'])))
                volumes.append(baseline);scenarios[key].append({'trace':name,'force_response':response_name})
        caches[key] = (matrices,offsets,volumes)
    save(root/'scenarios.json',scenarios)
    definitions = {}
    for name,knots,floor,timed in [('fine145_52',145,.52,False),('fine193_52',193,.52,False),
                                  ('fine145_55',145,.55,False),('timed193_52',193,.52,True)]:
        key = f'{knots}_'+('timed' if timed else 'recorded')
        initial = np.interp(np.linspace(0,1,knots),np.linspace(0,1,len(initial_profile)),initial_profile)
        profile,fit = fit_constrained(*caches[key],geometry.surface.area,initial,floor,3.5*96/(knots-1))
        parameters = slow_feed if timed else feed
        definitions[name] = (profile,parameters,fit)
        print('FINE_PROFILE_TRAINED',name,json.dumps(fit),flush=True)
    save(root/'training_results.json',{name:{'profile_n':p.tolist(),'feed_parameters':f.tolist(),**fit}
                                       for name,(p,f,fit) in definitions.items()})
    shutil.copyfile(PREVIOUS/'training/checkpoints/constant.pt',root/'checkpoints/constant.pt')
    shutil.copyfile(source,root/'checkpoints/saved.pt')
    rng = np.random.default_rng(args.seed);probe = torch.tensor(rng.uniform(-5,5,(2048,16)),dtype=torch.float32);exports = {}
    for name,(profile,parameters,fit) in definitions.items():
        actor = ProfileActor(profile,parameters,force_min=4.,force_max=30.,force_scale=16.)
        meta = {**protocol,'candidate':name,'force_profile_n':profile.tolist(),'profile_knots':len(profile),
                'feed_parameters':parameters.tolist(),'training_volume_floor':fit['training_volume_floor'],
                'identified_force_responses':responses,'counterfactual_clock':name.startswith('timed'),
                'maximum_adjacent_profile_delta_n':float(np.max(np.abs(np.diff(profile))))}
        checkpoint = root/'checkpoints'/f'{name}.pt'
        torch.jit.save(torch.jit.script(actor),str(checkpoint),_extra_files={'spatial_policy.json':json.dumps(meta)})
        loaded = torch.jit.load(str(checkpoint)).eval()
        with torch.inference_mode():actual,expected = loaded(probe),actor(probe)
        torch.testing.assert_close(actual,expected)
        assert torch.isfinite(actual).all() and actual.abs().max()<=1.000001 and (actual[:,2]==-1).all()
        exports[name] = {'sha256':sha(checkpoint),'parity_max_error':float((actual-expected).abs().max())}
    for name in ['constant','saved']:exports[name] = {'sha256':sha(root/'checkpoints'/f'{name}.pt')}
    save(root/'export_validation.json',exports)
    save(root/'tracking.json',{name:0. if name=='constant' else .6 for name in exports})
    snapshot = root/'source_snapshot';snapshot.mkdir()
    for name in ['train_uniformity_volume50_refine.py','train_uniformity_volume50.py','train_uniformity_refinement.py','train_uniformity_priority.py']:
        shutil.copyfile(ROOT/'scripts'/name,snapshot/name)
    save(root/'status.json',{'phase':'training_complete_pending_physics','elapsed_s':time.monotonic()-started,'exports':exports})
    print('VOLUME50_REFINE_TRAINING_COMPLETE',str(root),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=2026100806);main(parser.parse_args())
