#!/usr/bin/env python3
"""Preserve the 12.15% save and tune residence profiles with command dynamics."""
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
from scipy.optimize import minimize, minimize_scalar
from scipy import sparse
import osqp
import torch

from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.fixed_force_velocity import REFERENCE
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.fixed_force_dynamics import export_actor, rollout_residence, replay_command_response
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_fixed_force_velocity import build_matrix, save, sha
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_fixed_force_refinement import replay_matrix

ROOT = _REPO_ROOT
SAVED = ROOT/'logs/experiments/uniformity_saves/fixed20_virtual1000_refined_20261009_0414'
SAVED_SHA = '041bcf79a30921461168210bd12e3ee191ec35b2662d6c4ff045bff957772b30'
PREVIOUS = ROOT/'logs/experiments/fixed_force_refinement_20261009'
SELECTION_SEED = 2801
HOLDOUTS = [2811, 2812]


def check_preserved():
    for directory in [SAVED, ROOT/'logs/experiments/uniformity_saves/fixed20_virtual1000_velocity_20261009_0131']:
        for rel, digest in json.loads((directory/'save_manifest.json').read_text()).items():
            assert sha(directory/rel) == digest, (directory, rel)
    assert sha(SAVED/'candidate/checkpoints/actor.pt') == SAVED_SHA


def statistics(matrix):
    a = matrix/matrix.sum(axis=1).mean()
    return a.T@a/len(a), a.mean(axis=0)


def fit_residence_qp(matrices, *, volume_ratio, minimum_speed, maximum_speed, knot_delta):
    """Convex variance fits with fixed mean removal in both response models.

    Fixed per-model means remove the nonlinear CV denominator. A small grid
    of convex weights approximates the previous mean/worst-CV objective.
    """
    assert len(matrices) == 2
    stats = [statistics(matrix) for matrix in matrices]
    knots = len(stats[0][1])
    delta = sparse.diags([-np.ones(knots-1), np.ones(knots-1)], [0, 1], shape=(knots-1, knots), format='csc')
    means = np.stack([x[1] for x in stats])
    constraints = sparse.vstack([sparse.csc_matrix(means), sparse.eye(knots, format='csc'), delta], format='csc')
    lower = np.r_[np.full(2, volume_ratio), np.full(knots, 6./maximum_speed), np.full(knots-1, -knot_delta)]
    upper = np.r_[np.full(2, volume_ratio), np.full(knots, 6./minimum_speed), np.full(knots-1, knot_delta)]
    smooth = (delta.T@delta).toarray()/knots
    covariances = [q-np.outer(mean, mean) for q, mean in stats]
    best = None
    warm = np.full(knots, volume_ratio)
    for weight in [.375, .5, .625]:
        hessian = (weight*covariances[0]+(1.-weight)*covariances[1])/volume_ratio**2
        hessian += 1.e-5*smooth+1.e-10*np.eye(knots)
        solver = osqp.OSQP()
        solver.setup(P=sparse.csc_matrix(np.triu(2.*hessian)), q=np.zeros(knots),
                     A=constraints, l=lower, u=upper, verbose=False,
                     eps_abs=2.e-8, eps_rel=2.e-8, max_iter=40000, polish=True)
        solver.warm_start(x=warm)
        result = solver.solve()
        if result.x is None: raise RuntimeError(f'QP failed: {result.info.status}')
        x = result.x
        residual = float(max(np.max(lower-constraints@x), np.max(constraints@x-upper), 0.))
        if result.info.status_val not in (1, 2) or residual > 2.e-6:
            raise RuntimeError(f'QP failed: {result.info.status}, residual {residual}')
        x = np.clip(x, 6./maximum_speed, 6./minimum_speed)
        losses = [float(x@q@x/(mean@x)**2-1.) for q, mean in stats]
        score = float(.75*np.mean(losses)+.25*max(losses)+1.e-5*x@smooth@x)
        candidate = (score, x, result.info.iter, weight, residual)
        if best is None or score < best[0]: best = candidate
        warm = x
    score, x, iterations, weight, residual = best
    predictions = []
    for q, mean in stats:
        cv = float(np.sqrt(max(0., x@q@x/(mean@x)**2-1.)))
        baseline = float(np.sqrt(max(0., np.ones(knots)@q@np.ones(knots)-1.)))
        predictions.append({'cv': cv, 'nominal_mode6_cv_gain': 1.-cv/baseline,
                            'mode6_volume_ratio': float(mean@x)})
    return x, {'solver_success': True, 'solver': 'OSQP convex fixed-mean variance fits',
               'iterations': int(iterations), 'selected_response_weight': weight,
               'constraint_max_residual': residual, 'score': score, 'predictions': predictions,
               'qualification': 'response-model fit only; not fresh Isaac performance'}


def response_score(profile, length, preview, stats):
    values = []
    for delay in [0, 3]:
        effective, seconds, completed = rollout_residence(profile, length, preview, delay, len(stats[0][1]))
        if not completed or not np.isfinite(effective).all():
            return 1.e6, []
        for square, mean in stats:
            volume = float(mean@effective)
            variance = float(effective@square@effective/volume**2-1.)
            penalty = 20.*max(0., .75-volume)**2+20.*max(0., volume-1.50)**2
            penalty += 10.*max(0., seconds-190.)**2
            values.append((variance, volume, seconds, penalty))
    loss = .75*np.mean([x[0] for x in values])+.25*max(x[0] for x in values)
    loss += max(x[3] for x in values)
    return float(loss), [{'predicted_spatial_cv': float(np.sqrt(max(x[0], 0.))),
                          'nominalized_volume_ratio': x[1], 'command_path_time_s': x[2]} for x in values]


def main(args):
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=False); (root/'checkpoints').mkdir()
    torch.set_num_threads(1); check_preserved()
    protocol = {'objective': 'minimize spatial processing CV with >=50% matched Mode6 volume',
        'target_force_n': 20., 'policy_controls_force': False, 'virtual_spindle_rpm': 1000.,
        'urdf_rotation_unchanged': True, 'second_pass': False, 'local_reverse': False,
        'selection_seed': SELECTION_SEED, 'heldout_seeds': HOLDOUTS, 'holdouts_used_for_selection': False,
        'excluded_training_seeds': [2611, 2612, 2711, 2712],
        'saved_checkpoint': str(SAVED/'candidate/checkpoints/actor.pt'), 'saved_sha256': SAVED_SHA,
        'minimum_mode6_normalized_cv_margin': .005, 'minimum_mode6_volume_ratio': .5,
        'acceptance': 'Both new paired heldouts: eligible and extra Mode6-normalized CV gain >=max(0.005,2*Mode6 repeat variation). Otherwise keep the 12.15% save.',
        'volume_scopes': ['processing_roi', 'full_roi', 'whole_grid'],
        'entry_seconds': 2., 'entry_reference_mm': 18., 'hard_force_stop_n': 100.,
        'maximum_feed_mm_s': 12., 'minimum_moving_feed_mm_s': 1.5,
        'acceleration_limit_mm_s2': 16., 'jerk_limit_mm_s3': 160.,
        'physical_action_slew_mm_s2': 18., 'tracking_gain_fixed': .6,
        'maximum_preview_mm': 2., 'lookahead_time_bounds_s': [0., .4],
        'training_method': 'convex fixed-mean variance residence fitting plus digital-command-response parameter optimization; not PPO',
        'response_model_limit': 'filters and reference-cursor dynamics only; contact response evaluated freshly in Isaac',
        'report_baseline': 'matched Mode6 only', 'gui_deployment': False,
        'created_at_utc': datetime.now(timezone.utc).isoformat()}
    save(root/'protocol.json', protocol)
    with h5py.File(REFERENCE) as f: path = f['position'][:, :3]
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    folder = PREVIOUS/'physics/selection_b_2701'
    traces = {'mode6': folder/'constant_trace.npz', 'saved': folder/'dense289_110_trace.npz'}
    save(root/'training_data_manifest.json', {str(p): sha(p) for p in traces.values()})
    with np.load(traces['saved']) as f:
        raw = 6.*(1.+f['raw_action'].ravel()); command = f['commanded_speed_mm_s'].ravel()
        filtered = 6.*(1.+f['filtered_action'].ravel()); t = f['time_s']
        estimated_request, estimated_command = replay_command_response(raw, 1, filtered[0])
        keep = t >= 2.
        error = float(np.max(np.abs(estimated_command[keep]-command[keep])))
        rmse = float(np.sqrt(np.mean((estimated_command[keep]-command[keep])**2)))
        if error > .01: raise RuntimeError(f'Command response surrogate mismatches recorded limiter: {error}')
        lag_errors = []
        for lag in range(76):
            ids = np.flatnonzero((t >= 3.) & (t < t[-1]-2.)); ids = ids[ids >= lag]
            lag_errors.append(float(np.mean((command[ids]-raw[ids-lag])**2)))
    save(root/'command_response_audit.json', {'passed': True, 'max_feed_error_mm_s': error,
        'feed_rmse_mm_s': rmse, 'best_constant_lag_s': int(np.argmin(lag_errors))*.008,
        'lag_is_an_empirical_summary_not_an_exact_delay_model': True})
    print('COMMAND_RESPONSE_AUDIT', error, rmse, flush=True)
    specs = {
        'dense433_125': {'knots': 433, 'volume': 1.25, 'delta': .24, 'sources': ['nominal', 'mode6']},
        'contrast289_135': {'knots': 289, 'volume': 1.35, 'delta': .36, 'sources': ['nominal', 'mode6']},
        'dense577_145': {'knots': 577, 'volume': 1.45, 'delta': .18, 'sources': ['nominal', 'mode6']},
        'replay433_125': {'knots': 433, 'volume': 1.25, 'delta': .24, 'sources': ['mode6', 'saved']},
        'preview_saved': {'kind': 'saved profile; fit bounded preview time using response surrogate'},
        'compensated433': {'kind': 'fit scale, contrast and bounded preview around dense433_125 using response surrogate'},
    }
    save(root/'candidate_specs.json', specs)
    results = {}; matrices = {}
    def matrix(source, knots):
        key = (source, knots)
        if key not in matrices:
            if source == 'nominal': matrices[key] = build_matrix(path, knots)[0]
            else: matrices[key] = replay_matrix(path, traces[source], knots, nominalize=(source == 'saved'))
        return matrices[key]
    def record(name, profile, preview, metrics):
        export_actor(root/'checkpoints'/f'{name}.pt', profile.tolist(), path_length_mm=length,
                     lookahead_time_s=float(preview), candidate=name, protocol=protocol)
        results[name] = {'spec': specs[name], 'residence_profile': profile.tolist(),
                         'lookahead_time_s': float(preview), 'fit': metrics,
                         'sha256': sha(root/'checkpoints'/f'{name}.pt')}
        save(root/'training_results.json', results)
        print('DYNAMICS_TRAINED', name, json.dumps(metrics), 'preview', preview, flush=True)
    for name in ['dense433_125', 'contrast289_135', 'dense577_145', 'replay433_125']:
        spec = specs[name]
        q, metrics = fit_residence_qp([matrix(s, spec['knots']) for s in spec['sources']],
            volume_ratio=spec['volume'], minimum_speed=1.5, maximum_speed=12., knot_delta=spec['delta'])
        record(name, q, 0., metrics)
    stats = [statistics(matrix(s, 433)) for s in ['nominal', 'mode6']]
    contract = json.loads((SAVED/'candidate/fixed_force_policy.json').read_text())
    saved_q = np.asarray(contract['residence_profile'], dtype=float)
    grid = np.linspace(0., .4, 21)
    scores = [response_score(saved_q, length, v, stats)[0] for v in grid]
    best = int(np.argmin(scores))
    opt = minimize_scalar(lambda value: response_score(saved_q, length, value, stats)[0],
                          bounds=(max(0., grid[best]-.03), min(.4, grid[best]+.03)),
                          method='bounded', options={'xatol': .0005})
    record('preview_saved', saved_q, opt.x,
           {'response_objective': float(opt.fun), 'surrogate_only': True,
            'prediction': response_score(saved_q, length, opt.x, stats)[1]})
    initial = np.asarray(results['dense433_125']['residence_profile'])
    def objective(x):
        q = np.clip(x[0]*initial**x[1], .5, 4.)
        return response_score(q, length, x[2], stats)[0]
    searches = []
    for start in [[1., 1., .1], [1.08, 1.2, .26]]:
        searches.append(minimize(objective, start, method='Powell', bounds=[(.9, 1.12), (.9, 1.4), (0., .4)],
                                options={'maxiter': 80, 'xtol': .0003, 'ftol': 1.e-7}))
    chosen = min(searches, key=lambda item: item.fun)
    q = np.clip(chosen.x[0]*initial**chosen.x[1], .5, 4.)
    record('compensated433', q, chosen.x[2], {'response_objective': float(chosen.fun),
           'parameters': chosen.x.tolist(), 'surrogate_only': True,
           'prediction': response_score(q, length, chosen.x[2], stats)[1]})
    shutil.copyfile(SAVED/'candidate/checkpoints/actor.pt', root/'checkpoints/saved.pt')
    shutil.copyfile(PREVIOUS/'training/checkpoints/constant.pt', root/'checkpoints/constant.pt')
    save(root/'checkpoint_manifest.json', {p.stem: sha(p) for p in (root/'checkpoints').glob('*.pt')})
    save(root/'status.json', {'phase': 'trained_pending_physics', 'candidates': list(specs)})
    check_preserved()
    print('DYNAMICS_TRAINING_COMPLETE', str(root), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
