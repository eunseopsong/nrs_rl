#!/usr/bin/env python3
"""Constrained force-profile learning from prior training/selection traces.

Minimize spatial depth CV while retaining at least a prescribed fraction of
paired Mode 6 removal in every replay scenario. This is model-based policy
optimization, not PPO; all candidates require fresh Isaac validation.
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
from scipy.optimize import LinearConstraint, minimize
import torch

from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.spatial_uniformity import ROOT, DoseGeometry
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_uniformity_priority import ProfileActor, replay_sources, save, sha
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_uniformity_refinement import (
    SAVED, affine_objective, basis_at, deposition_operator, force_filter,
    identify_force_response,
)

PREVIOUS = ROOT / 'logs/experiments/uniformity_refinement_20261007'
PREVIOUS_ACTOR = PREVIOUS / 'physics/candidate/checkpoints/actor.pt'
PREVIOUS_SHA = 'd0ef2db1c4376e7ff462b257067f8c8ed616da75484cf14e514deb8368ac1d50'
SAVED_SHA = 'e31302e7a8edb84ef713f61b080a06e5ff44b5af5ad10dc3031fc26333dc77a6'
NEW = ['profile97_55', 'profile97_62', 'profile145_55', 'scaled_55']
SELECTION_SEED = 2301
HOLDOUTS = [2311, 2312]


def read_trace(path):
    with np.load(path) as d:
        if 'safety_fault_reason' in d and np.any(d['safety_fault_reason'] != 0):
            raise ValueError(f'Faulted training trace: {path}')
        return dict(time=d['time_s'].copy(), midpoint=d['deposition_midpoint_mm'].copy(),
                    distance=np.linalg.norm(d['tangent_velocity_mm_s'], axis=1) * .008,
                    force=d['normal_force_n'].copy(), target=d['target_force_n'].copy(),
                    cursor=d['cursor_mm'].copy())


def fit_constrained(matrices, offsets, baseline_volumes, area, initial, floor, max_delta):
    """Analytic CV gradient with explicit affine volume lower bounds."""
    n = len(initial)
    statistics, volume_rows, volume_offsets = [], [], []
    for matrix, offset, baseline in zip(matrices, offsets, baseline_volumes):
        augmented = np.column_stack([matrix, offset])
        statistics.append((augmented.T @ augmented / len(augmented), augmented.mean(axis=0)))
        volume_rows.append(matrix.sum(axis=0) * area / baseline)
        volume_offsets.append(offset.sum() * area / baseline)
    volume_rows, volume_offsets = np.array(volume_rows), np.array(volume_offsets)
    d2 = np.diff(np.eye(n), n=2, axis=0)
    smooth = d2.T @ d2 / (n - 2)
    constraints = [LinearConstraint(np.diff(np.eye(n), axis=0), -max_delta, max_delta),
                   LinearConstraint(volume_rows, floor - volume_offsets, np.inf)]
    results = []
    for start in [initial, np.full(n, 20. * floor + 2.)]:
        result = minimize(affine_objective, np.clip(start, 4., 30.), args=(statistics, smooth),
                          jac=True, bounds=[(4., 30.)] * n, constraints=constraints,
                          method='SLSQP', options={'maxiter':1800, 'ftol':1.e-11})
        ratios = volume_rows @ result.x + volume_offsets
        if (result.success and np.isfinite(result.x).all() and ratios.min() >= floor - 1.e-7
                and np.max(np.abs(np.diff(result.x))) <= max_delta + 1.e-6):
            results.append(result)
    if not results:
        raise RuntimeError('No feasible converged volume-constrained force profile')
    best = min(results, key=lambda x: x.fun)
    return best.x, dict(objective=float(best.fun), solver_success=True, iterations=int(best.nit),
                       predicted_mode6_volume_ratios=(volume_rows @ best.x + volume_offsets).tolist(),
                       predicted_spatial_cv=[float(np.std(a @ best.x + b) / np.mean(a @ best.x + b))
                                             for a, b in zip(matrices, offsets)],
                       training_volume_floor=floor, maximum_adjacent_profile_delta_n=max_delta)


def preserve_current_best():
    destination = ROOT / 'logs/experiments/uniformity_saves/mode6_13p33_20261008'
    assert sha(PREVIOUS_ACTOR) == PREVIOUS_SHA
    assert sha(SAVED / 'candidate/checkpoints/actor.pt') == SAVED_SHA
    if not destination.exists():
        destination.mkdir(parents=True)
        shutil.copytree(PREVIOUS / 'physics/candidate', destination / 'candidate')
        for name in ['result.json', 'verification.json', 'predeclared_protocol.json']:
            shutil.copyfile(PREVIOUS / 'physics' / name, destination / name)
        save(destination / 'SAVE.json', {'source':str(PREVIOUS_ACTOR), 'sha256':PREVIOUS_SHA,
             'reason':'Preserved historical CV-only winner; fails new 50% removal constraint',
             'created_at_utc':datetime.now(timezone.utc).isoformat()})
        save(destination / 'save_manifest.json', {str(p.relative_to(destination)):sha(p)
             for p in destination.rglob('*') if p.is_file() and p.name != 'save_manifest.json'})
    for name, digest in json.loads((destination / 'save_manifest.json').read_text()).items():
        assert sha(destination / name) == digest
    return destination


def main(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'checkpoints').mkdir()
    torch.set_num_threads(1)
    started = time.monotonic()
    preserved = preserve_current_best()
    geometry = DoseGeometry(cell_mm=2.)
    extra = {'spatial_policy.json':''}
    torch.jit.load(str(PREVIOUS_ACTOR), _extra_files=extra)
    old = json.loads(extra['spatial_policy.json'])
    protocol = {**old, 'candidate':None, 'force_profile_n':None,
        'algorithm':'volume_constrained_force_profile_optimization',
        'objective':'minimize processing spatial depth CV subject to Mode 6 removal >= 50%',
        'stage':'training_from_GUI_and_previous_selection_traces',
        'validation_status':'pending_fresh_physics', 'started_at_utc':datetime.now(timezone.utc).isoformat(),
        'report_comparison_baseline':'Mode 6 only', 'minimum_mode6_volume_ratio':.5,
        'total_removal_is_selection_gate':True, 'total_removal_constraint_scopes':['processing_roi','full_roi'],
        'physics_training_seed':SELECTION_SEED, 'physics_holdout_seeds':HOLDOUTS,
        'training_data_excludes_previous_holdouts':[2111,2112,2211,2212],
        'minimum_mode6_cv_gain_each_holdout':.1114,
        'must_not_regress_against_preserved_checkpoint_on_same_seed':True,
        'preserved_feasible_sha256':SAVED_SHA, 'preserved_cv_only_sha256':PREVIOUS_SHA,
        'source_save_sha256':PREVIOUS_SHA, 'new_gui_experiment_executed':False, 'deployed':False}
    protocol.pop('identified_force_response', None)
    save(root / 'protocol.json', protocol)
    save(root / 'preserved_models.json', {'feasible':str(SAVED), 'cv_only':str(preserved)})

    selection = PREVIOUS / 'physics/selection_2201'
    sources = list(replay_sources())[:2]  # GUI records, with a paired Mode 6 baseline.
    for name in ['saved', 'dynamic_8n', 'dynamic_4n']:
        file = selection / f'{name}_trace.npz'
        sources.append((f'selection2201_{name}', read_trace(file), [file]))
    response_sources = ['dynamic_8n', 'dynamic_4n']
    responses = {name:identify_force_response(read_trace(selection / f'{name}_trace.npz'))
                 for name in response_sources}
    save(root / 'force_response_identification.json', responses)
    print('FORCE_RESPONSE', json.dumps(responses), flush=True)
    baseline_file = selection / 'constant_trace.npz'
    baseline_data = read_trace(baseline_file)
    mode6_gui = sources[1][1]
    baseline_volumes = {
        'gui':float(np.sum(deposition_operator(geometry, mode6_gui) @ mode6_gui['force']) * geometry.surface.area),
        'physics':float(np.sum(deposition_operator(geometry, baseline_data) @ baseline_data['force']) * geometry.surface.area)}
    manifest = {str(baseline_file):sha(baseline_file)}
    operators = []
    for name, data, files in sources:
        manifest.update({str(p):sha(p) for p in files})
        operator = deposition_operator(geometry, data)
        baseline = baseline_volumes['gui' if name.startswith('mode') else 'physics']
        operators.append((name, data, operator, baseline))
        print('OPERATOR_BUILT', name, flush=True)
    save(root / 'source_data_manifest.json', manifest)
    save(root / 'paired_baseline_volumes_2mm.json', baseline_volumes)

    caches = {}
    for knots in [97, 145]:
        matrices, offsets, baselines, scenarios = [], [], [], []
        for name, data, operator, baseline in operators:
            for response_name, response in responses.items():
                basis = basis_at(data['cursor'], geometry.length, knots)
                filtered = force_filter(basis, data['time'], response['delay_s'], response['tau_s'])
                matrices.append(operator @ filtered * response['gain'])
                offsets.append(np.asarray(operator @ np.full(len(data['time']), response['bias_n'])))
                baselines.append(baseline)
                scenarios.append({'trace':name, 'force_response':response_name})
        caches[knots] = (matrices, offsets, baselines)
        save(root / f'scenarios_{knots}.json', scenarios)

    definitions = {}
    for name, knots, floor in [('profile97_55',97,.55), ('profile97_62',97,.62), ('profile145_55',145,.55)]:
        initial = np.interp(np.linspace(0,1,knots), np.linspace(0,1,len(old['force_profile_n'])), old['force_profile_n']) * 1.65
        max_delta = 3.5 * 96 / (knots - 1)
        profile, fit = fit_constrained(*caches[knots], geometry.surface.area, initial, floor, max_delta)
        definitions[name] = (profile, fit)
        print('PROFILE_TRAINED', name, json.dumps(fit), flush=True)
    # A separately calibrated fixed-shape candidate tests whether preserving
    # the former winner's relative force allocation transfers at higher dose.
    matrices, offsets, baselines = caches[97]
    old_profile = np.array(old['force_profile_n'])
    scales = [(.55*b - o.sum()*geometry.surface.area) / ((a@old_profile).sum()*geometry.surface.area)
              for a,o,b in zip(matrices,offsets,baselines)]
    scale = max(scales)
    profile = old_profile * scale
    if profile.min() < 4. or profile.max() > 30.:
        raise ValueError('Scaled profile would require clipping; re-fit instead')
    definitions['scaled_55'] = (profile, {'algorithm':'one_parameter_constrained_profile_scaling', 'scale':float(scale),
        'training_volume_floor':.55,
        'predicted_mode6_volume_ratios':[float((a@profile+o).sum()*geometry.surface.area/b)
                                        for a,o,b in zip(matrices,offsets,baselines)],
        'note':'Runtime force slew remains 5 N/s; rate-limit effects require physics validation'})
    save(root / 'training_results.json', {name:{'profile_n':p.tolist(), **fit} for name,(p,fit) in definitions.items()})
    shutil.copyfile(PREVIOUS / 'training/checkpoints/constant.pt', root / 'checkpoints/constant.pt')
    shutil.copyfile(SAVED / 'candidate/checkpoints/actor.pt', root / 'checkpoints/saved.pt')
    rng = np.random.default_rng(args.seed)
    probe = torch.tensor(rng.uniform(-5,5,(2048,16)), dtype=torch.float32)
    exports = {}
    for name, (profile, fit) in definitions.items():
        actor = ProfileActor(profile, old['feed_parameters'], force_min=4., force_max=30., force_scale=16.)
        meta = {**protocol, 'candidate':name, 'force_profile_n':profile.tolist(), 'profile_knots':len(profile),
                'force_bounds_n':[4.,30.], 'force_action_mapping':{'center_n':20., 'scale_n':16.},
                'action_mapping':['6+3*a0 mm/s','20+16*a1 N','local return disabled'],
                'identified_force_responses':responses, 'training_volume_floor':fit['training_volume_floor'],
                'maximum_adjacent_profile_delta_n':float(np.max(np.abs(np.diff(profile))))}
        path = root / 'checkpoints' / f'{name}.pt'
        torch.jit.save(torch.jit.script(actor), str(path), _extra_files={'spatial_policy.json':json.dumps(meta)})
        loaded = torch.jit.load(str(path)).eval()
        with torch.inference_mode():
            actual, expected = loaded(probe), actor(probe)
        torch.testing.assert_close(actual, expected)
        assert torch.isfinite(actual).all() and actual.abs().max() <= 1.000001
        assert (actual[:,2] == -1).all()
        exports[name] = {'sha256':sha(path), 'parity_max_error':float((actual-expected).abs().max())}
    for name in ['constant','saved']:
        exports[name] = {'sha256':sha(root / 'checkpoints' / f'{name}.pt')}
    save(root / 'export_validation.json', exports)
    save(root / 'tracking.json', {name:0. if name == 'constant' else .6 for name in exports})
    snapshot = root / 'source_snapshot'; snapshot.mkdir()
    for name in ['train_uniformity_volume50.py','train_uniformity_refinement.py','train_uniformity_priority.py']:
        shutil.copyfile(ROOT / 'scripts' / name, snapshot / name)
    save(root / 'status.json', {'phase':'training_complete_pending_physics', 'elapsed_s':time.monotonic()-started,
                               'exports':exports, 'comparison_baseline':'Mode 6 only'})
    print('VOLUME50_TRAINING_COMPLETE', str(root), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026100801)
    main(parser.parse_args())
