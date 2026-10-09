#!/usr/bin/env python3
"""One final constrained force/feed search before fixing the selected model.

Use only historical GUI data and the previous selection batch. Fit requested
force -> measured force (including the existing slew limiter), residuals, and
counterfactual timing on the same 0.5 mm grid used for evaluation. Fixed replay
footprints and residual transfer are approximations, requiring fresh Isaac tests.
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
from scipy.ndimage import uniform_filter1d
import torch

from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.spatial_uniformity import ROOT, DoseGeometry
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_uniformity_priority import ProfileActor, replay_sources, save, sha
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_uniformity_refinement import basis_at, deposition_operator, force_filter, identify_force_response
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_uniformity_volume50 import read_trace, fit_constrained
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_uniformity_volume50_refine import nominal_speed, response_irregular

PREVIOUS = ROOT / 'logs/experiments/uniformity_volume50_refine_20261008'
SAVED = ROOT / 'logs/experiments/uniformity_saves/mode6_13p45_volume50_20261008_0900'
SAVED_SHA = 'aa85d7951b7a2934c4383ed494427702620fd7fe7dab6f5b5453a51c3efc34be'
SELECTION_SEED = 2501
HOLDOUTS = [2511, 2512]
MINIMUM_GAIN = .1445
MINIMUM_ADVANTAGE = .005  # Mode 6-normalized CV margin, never percent vs saved.
SPECS = {
    'recalibrated_515': dict(knots=193, residual_weight=0., feed_scale=1., physics_only=False),
    'residual_515': dict(knots=193, residual_weight=.65, feed_scale=1., physics_only=False),
    'slow_515': dict(knots=193, residual_weight=.65, feed_scale=.30, physics_only=False),
    'flatfeed_515': dict(knots=193, residual_weight=.65, feed_scale=0., physics_only=False),
    'physics_515': dict(knots=193, residual_weight=.65, feed_scale=.50, physics_only=True),
    'dense_515': dict(knots=289, residual_weight=.65, feed_scale=1., physics_only=False),
}
NEW = list(SPECS)


def check_preserved():
    for name, digest in json.loads((SAVED / 'save_manifest.json').read_text()).items():
        assert sha(SAVED / name) == digest, name
    assert sha(SAVED / 'candidate/checkpoints/actor.pt') == SAVED_SHA


def adjusted_clock(data, profile, length, old_feed, new_feed):
    force = np.interp(data['cursor'] / length, np.linspace(0., 1., len(profile)), profile)
    # Distance is held fixed: speed changes only the force-response clock.
    ratio = np.clip(nominal_speed(data['target'], old_feed) / nominal_speed(force, new_feed), .5, 2.)
    intervals = np.diff(data['time']) * .5 * (ratio[1:] + ratio[:-1])
    return np.r_[data['time'][0], data['time'][0] + np.cumsum(intervals)]


def bounded_residual(data, response):
    predicted = (force_filter(data['target'], data['time'], response['delay_s'], response['tau_s'])[:, 0]
                 * response['gain'] + response['bias_n'])
    residual = uniform_filter1d(data['force'] - predicted, size=41, mode='nearest')
    return np.clip(residual, -1.5, 1.5)


def main(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'checkpoints').mkdir()
    torch.set_num_threads(1)
    started = time.monotonic()
    check_preserved()
    source = SAVED / 'candidate/checkpoints/actor.pt'
    old = json.loads((SAVED / 'candidate/spatial_policy.json').read_text())
    protocol = {**old, 'candidate': None, 'force_profile_n': None,
        'algorithm': 'requested_force_response_residual_and_feed_timing_constrained_profile_optimization',
        'stage': 'final_attempt_from_GUI_and_selection_2401', 'training_surface_cell_mm': .5,
        'evaluation_surface_cell_mm': .5, 'source_save_sha256': SAVED_SHA,
        'preserved_feasible_sha256': SAVED_SHA, 'minimum_mode6_volume_ratio': .5,
        'training_data_excludes_previous_holdouts': [2111, 2112, 2211, 2212, 2311, 2312, 2411, 2412],
        'physics_training_seed': SELECTION_SEED, 'physics_holdout_seeds': HOLDOUTS,
        'minimum_mode6_cv_gain_each_holdout': MINIMUM_GAIN,
        'minimum_mode6_normalized_advantage_over_save': MINIMUM_ADVANTAGE,
        'holdouts_used_for_selection': False, 'validation_status': 'pending_fresh_physics',
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'deployed': False,
        'training_method': 'model-based constrained optimization, not PPO',
        'fallback': 'fix the preserved 13.45% model if the predeclared replacement threshold is not met'}
    save(root / 'protocol.json', protocol)
    save(root / 'candidate_specs.json', SPECS)
    geometry = DoseGeometry(cell_mm=.5)
    previous_fits = json.loads((PREVIOUS / 'training/training_results.json').read_text())
    physics = [('selection_a_2401', 'fine193_52'), ('selection_b_2401', 'fine145_55'),
               ('selection_b_2401', 'timed193_52')]
    traces, responses, manifest = {}, {}, {}
    for group, name in physics:
        file = PREVIOUS / 'physics' / group / f'{name}_trace.npz'
        data = read_trace(file)
        with np.load(file) as records:
            data['target'] = np.clip(20. + 16. * records['spatial_action'][:, 1], 4., 30.)
        response = identify_force_response(data)
        responses[name] = response
        traces[name] = (group, data, np.array(previous_fits[name]['feed_parameters']), response)
        manifest[str(file)] = sha(file)
        print('FINAL_FORCE_IDENTIFIED', name, json.dumps(response), flush=True)
    save(root / 'force_response_identification.json', responses)
    baselines = {}
    for group in sorted(set(item[0] for item in physics)):
        file = PREVIOUS / 'physics' / group / 'constant_trace.npz'
        data = read_trace(file)
        op = deposition_operator(geometry, data)
        baselines[group] = float(np.sum(op @ data['force']) * geometry.surface.area)
        manifest[str(file)] = sha(file)
    gui = list(replay_sources())[:2]
    op = deposition_operator(geometry, gui[1][1])
    baselines['gui'] = float(np.sum(op @ gui[1][1]['force']) * geometry.surface.area)
    operators = []
    for name, data, files in gui:
        manifest.update({str(p): sha(p) for p in files})
        # GUI feed differs from this candidate; retain its recorded clock as a
        # conservative geometry/force robustness scenario, without time warping.
        operators.append(dict(name=name, data=data, operator=deposition_operator(geometry, data),
                              baseline=baselines['gui'], old_feed=None, response=responses['timed193_52']))
    for name, (group, data, feed, response) in traces.items():
        operators.append(dict(name=name, data=data, operator=deposition_operator(geometry, data),
                              baseline=baselines[group], old_feed=feed, response=response))
    for item in operators:
        item['residual'] = bounded_residual(item['data'], item['response'])
        print('FINAL_OPERATOR', item['name'], item['operator'].shape, flush=True)
    save(root / 'source_data_manifest.json', manifest)
    save(root / 'paired_baselines_05mm.json', baselines)
    save(root / 'residual_diagnostics.json', {item['name']: {
        'rms_n': float(np.sqrt(np.mean(item['residual'][item['data']['time'] >= 2.] ** 2))),
        'max_abs_n': float(np.abs(item['residual']).max())} for item in operators})
    definitions = {}
    for name, spec in SPECS.items():
        knots = spec['knots']
        feed = np.array(old['feed_parameters'])
        feed[0] *= spec['feed_scale']
        profile = np.interp(np.linspace(0., 1., knots), np.linspace(0., 1., len(old['force_profile_n'])),
                            old['force_profile_n']) * (.515 / .52)
        scenarios = [item for item in operators if not spec['physics_only'] or item['old_feed'] is not None]
        iterations = []
        # Two alternating fits refresh the clock from the newly fitted profile.
        for iteration in range(2):
            matrices, offsets, volumes = [], [], []
            for item in scenarios:
                data, response = item['data'], item['response']
                clock = (data['time'] if item['old_feed'] is None else
                         adjusted_clock(data, profile, geometry.length, item['old_feed'], feed))
                basis = basis_at(data['cursor'], geometry.length, knots)
                filtered = response_irregular(basis, clock, response['delay_s'], response['tau_s'])
                matrices.append(item['operator'] @ filtered * response['gain'])
                offset_force = response['bias_n'] + spec['residual_weight'] * item['residual']
                offsets.append(np.asarray(item['operator'] @ offset_force))
                volumes.append(item['baseline'])
            profile, fit = fit_constrained(matrices, offsets, volumes, geometry.surface.area,
                profile, .515, 4.5 * geometry.length / (9. * (knots - 1)))
            iterations.append(fit)
            print('FINAL_PROFILE_FIT', name, iteration + 1, json.dumps(fit), flush=True)
        definitions[name] = dict(profile_n=profile.tolist(), feed_parameters=feed.tolist(),
            scenarios=[item['name'] for item in scenarios], alternating_fit_history=iterations, **spec, **fit)
        save(root / 'training_results.json', definitions)
    shutil.copyfile(PREVIOUS / 'training/checkpoints/constant.pt', root / 'checkpoints/constant.pt')
    shutil.copyfile(source, root / 'checkpoints/saved.pt')
    rng = np.random.default_rng(args.seed)
    probe = torch.tensor(rng.uniform(-5., 5., (2048, 16)), dtype=torch.float32)
    exports = {}
    for name, fit in definitions.items():
        actor = ProfileActor(fit['profile_n'], fit['feed_parameters'], force_min=4., force_max=30., force_scale=16.)
        meta = {**protocol, 'candidate': name, 'force_profile_n': fit['profile_n'],
            'feed_parameters': fit['feed_parameters'], 'profile_knots': len(fit['profile_n']),
            'training_volume_floor': .515, 'identified_force_responses': responses,
            'candidate_spec': SPECS[name], 'counterfactual_clock': True,
            'maximum_adjacent_profile_delta_n': float(np.abs(np.diff(fit['profile_n'])).max())}
        checkpoint = root / 'checkpoints' / f'{name}.pt'
        torch.jit.save(torch.jit.script(actor), str(checkpoint), _extra_files={'spatial_policy.json': json.dumps(meta)})
        loaded = torch.jit.load(str(checkpoint)).eval()
        with torch.inference_mode():
            actual, expected = loaded(probe), actor(probe)
        torch.testing.assert_close(actual, expected)
        assert torch.isfinite(actual).all() and actual.abs().max() <= 1.000001
        assert (actual[:, 2] == -1.).all()
        exports[name] = {'sha256': sha(checkpoint), 'parity_max_error': float((actual - expected).abs().max())}
    for name in ['constant', 'saved']:
        exports[name] = {'sha256': sha(root / 'checkpoints' / f'{name}.pt')}
    save(root / 'export_validation.json', exports)
    save(root / 'tracking.json', {name: 0. if name == 'constant' else .6 for name in exports})
    snapshot = root / 'source_snapshot'
    snapshot.mkdir()
    for name in ['train_uniformity_final_attempt.py', 'train_uniformity_volume50.py',
                 'train_uniformity_volume50_refine.py', 'train_uniformity_refinement.py', 'train_uniformity_priority.py']:
        shutil.copyfile(ROOT / 'scripts' / name, snapshot / name)
    save(root / 'source_manifest.json', {p.name: sha(p) for p in snapshot.iterdir()})
    for file, digest in manifest.items():
        assert sha(file) == digest
    check_preserved()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    for name, values in [('saved', old)] + list(definitions.items()):
        profile = np.array(values['force_profile_n'] if name == 'saved' else values['profile_n'])
        x = np.linspace(0., geometry.length, len(profile))
        axes[0].plot(x, profile, label=name, lw=.8)
        axes[1].plot(x, nominal_speed(profile, values['feed_parameters']), lw=.8)
    axes[0].set_ylabel('Requested contact force (N)')
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].set_ylabel('Nominal feed (mm/s)')
    axes[1].set_xlabel('Reference progress (mm)')
    fig.suptitle('Final attempt: learned force and nominal feed profiles; not validation results')
    fig.savefig(root / 'learned_profiles.png', dpi=150)
    plt.close(fig)
    save(root / 'status.json', {'phase': 'training_complete_pending_physics',
        'elapsed_s': time.monotonic() - started, 'exports': exports})
    print('FINAL_ATTEMPT_TRAINING_COMPLETE', str(root), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026100809)
    main(parser.parse_args())
