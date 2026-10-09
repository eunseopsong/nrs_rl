#!/usr/bin/env python3
"""Episodic parameter search for a 125 Hz, Preston-depth velocity policy.

An initial smooth dwell profile comes from explicit model optimization. CEM
then ranks bounded feedback parameters by realized simulation depth, with
duration, mean-depth and coverage constraints. It does not train a large MLP.
The depth model assumes constant RPM and rotation-dominated sliding. The user
specified cmd_continue9D_flat.h5 as the size/range reference on 2026-09-11;
the fixed ROI is that path swept by the 30 mm contact footprint.
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
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = _REPO_ROOT
sys.path.insert(0, str(ROOT / 'scripts/skrl'))
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.preston_dwell_actor import LOW, HIGH, PARAMETER_NAMES, export_dwell_actor
from nrs_rl.tasks.manager_based.nrs_rl.mdp.runtime.ppo_policy_node import SCHEDULER_CONTRACT
from nrs_rl.tasks.manager_based.nrs_rl.utils.reintegrate_depth_candidates import reintegrate


def assess_candidate(candidate, baselines):
    """Keep raw gains and rejection reasons distinct from the constrained score."""
    depth = candidate['rotation_depth']
    gain, reasons = [], []
    for index, baseline in enumerate(baselines):
        reference = baseline['rotation_depth']
        values = [depth['spatial_depth_cv'], reference['spatial_depth_cv'],
                  depth['mean_depth_over_k_omega'], reference['mean_depth_over_k_omega']]
        if (any(x is None or not np.isfinite(x) for x in values)
                or values[0] < 0 or any(x <= 0 for x in values[1:])):
            return {'score': -10., 'unconstrained_gain': None, 'rejections': ['invalid depth metrics']}
        checks = {
            'incomplete reference path': not candidate['completed'] or not baseline['completed'],
            'fault or shield activation': (candidate['fault_fraction'] != 0 or candidate['shield_fraction'] != 0
                                           or baseline['fault_fraction'] != 0 or baseline['shield_fraction'] != 0),
            'duration above 105%': candidate['processing_samples'] > 1.05*baseline['processing_samples'],
            'mean depth outside 95--105%': not .95 <= values[2]/values[3] <= 1.05,
            'increased zero-depth area': depth['zero_depth_fraction'] > reference['zero_depth_fraction'] + 1e-12,
        }
        reasons.extend(f'baseline_{index}: {reason}' for reason, failed in checks.items() if failed)
        gain.append(1-values[0]/values[1])
    raw_gain = float(min(gain))
    return {'score': -10. if reasons else raw_gain, 'unconstrained_gain': raw_gain, 'rejections': reasons}


def rank_candidate(candidate, baselines):
    return assess_candidate(candidate, baselines)['score']


def next_population(elites, incumbent, generator, size=8):
    """Update a bounded diagonal-Gaussian parameter search distribution."""
    low, high = np.asarray(LOW), np.asarray(HIGH)
    mean = np.mean(elites, axis=0)
    std = np.maximum(np.std(elites, axis=0), .08*(high-low))
    population = {'constant_a': [0., 0., 0., 0.], 'constant_b': [0., 0., 0., 0.],
                  'incumbent': list(incumbent)}
    for i, value in enumerate(generator.normal(mean, std, (size-3, 4))):
        population[f'proposal_{i}'] = np.clip(value, low, high).tolist()
    return population, mean.tolist(), std.tolist()


def train(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    profile = json.loads(args.profile.read_text())
    reference = ROOT / 'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/datasets/cmd_continue9D_flat.h5'
    if profile.get('reference_path_sha256') != hashlib.sha256(reference.read_bytes()).hexdigest():
        raise ValueError('Dwell profile must record the exact reference path hash')
    profile_path = output / 'initial_dwell_profile.json'
    shutil.copyfile(args.profile, profile_path)
    roi = {
        'confirmed_by_user': True, 'confirmation_date': '2026-09-11',
        'reference_dataset': str(reference.relative_to(ROOT)),
        'reference_path_sha256': profile['reference_path_sha256'],
        'definition': 'reference swept contact footprint, all equal-area cells including zeros',
        'tool_diameter_mm': 30., 'contact_diameter_mm': 30., 'cell_size_mm': args.surface_cell_mm,
    }
    if (profile['model']['roi'] != roi['definition']
            or profile['model']['contact_diameter_mm'] != roi['contact_diameter_mm']):
        raise ValueError('Initial profile must use the confirmed full swept 30 mm ROI')
    candidates = {
        'constant_a': [0., 0., 0., 0.], 'constant_b': [0., 0., 0., 0.],
        'profile_full': [1., 0., 0., 0.], 'profile_half': [.5, 0., 0., 0.],
        'force_speed': [0., 1., .25, 0.], 'mixed': [.7, .5, .2, 0.],
        'mild': [.3, .7, .1, 0.], 'profile_speed': [1., 0., .2, 0.],
    }
    if args.smoke:
        candidates = {key: candidates[key] for key in ('constant_a', 'constant_b', 'mixed')}
    state = {'phase': 'training' if not args.smoke else 'runtime_smoke',
             'algorithm': 'cem_episodic_parameter_search', 'runtime_smoke_only': args.smoke,
             'objective': 'rotation_dominated_spatial_preston_depth',
             'parameter_names': PARAMETER_NAMES, 'policy_hz': 125,
             'physical_depth_validated': False, 'roi_confirmed_by_user': True, 'reference_roi': roi,
             'started_at_utc': datetime.now(timezone.utc).isoformat(), 'seed': args.seed,
             'maximum_generations': 1 if args.smoke else args.generations,
             'initial_profile_cell_size_mm': profile['model']['cell_size_mm'],
             'reuse_initial_evaluation': str(args.reuse_initial_evaluation) if args.reuse_initial_evaluation else None,
             'promoted_to_robot': False, 'generations': [], 'best_score': 0.,
             'best_parameters': [0., 0., 0., 0.], 'profile_sha256': hashlib.sha256(profile_path.read_bytes()).hexdigest()}
    def save():
        state['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        temporary = output / 'status.next.json'
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(output / 'status.json')
        print(json.dumps(state), flush=True)
    save()
    generator = np.random.default_rng(args.seed)
    for generation in range(1 if args.smoke else args.generations):
        parameters_file = output / f'population_{generation}.json'
        parameters_file.write_text(json.dumps(candidates, indent=2))
        experiment = output / f'generation_{generation}'
        state.update(current_generation=generation, current_population=candidates,
                     progress_file=str(experiment / 'progress.json'))
        save()
        command = [sys.executable, str(ROOT / 'scripts/probe_velocity_feedback.py'),
            '--headless', '--device', 'cpu', '--path-length-mm',
            '10' if args.smoke else str(profile['cursor_knots_mm'][-1]),
            '--seed', str(args.seed), '--max-steps', '24000', '--output', str(experiment),
            '--surface-cell-mm', str(args.surface_cell_mm),
            '--parameters-file', str(parameters_file), '--dwell-profile', str(profile_path)]
        if generation == 0 and args.reuse_initial_evaluation:
            state.update(phase='reanalyzing_initial_population')
            save()
            summary = reintegrate(args.reuse_initial_evaluation, experiment, profile_path,
                                  candidates, args.seed, args.surface_cell_mm)
            state.update(phase='training')
        else:
            (output / f'command_{generation}.json').write_text(json.dumps(command, indent=2))
            with (output / f'generation_{generation}.log').open('w') as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                state.update(phase='error', failed_generation=generation)
                save()
                raise RuntimeError(f'Simulation failed; see {output / f"generation_{generation}.log"}')
            summary = json.loads((experiment / 'summary.json').read_text())
        metrics = summary['candidates']
        baselines = [metrics[key] for key in ('constant_a', 'constant_b')]
        assessments = {name: assess_candidate(metrics[name], baselines) for name in candidates}
        ranked = sorted(((assessments[name]['score'], name) for name in candidates), reverse=True)
        scores = {name: score for score, name in ranked}
        a, b = (x['rotation_depth']['spatial_depth_cv'] for x in baselines)
        noise_margin = max(.01, 2*abs(a/b-1))
        record = {'generation': generation, 'scores': scores, 'assessments': assessments,
                  'minimum_reliable_gain': noise_margin,
                  'reused_saved_dynamics': generation == 0 and bool(args.reuse_initial_evaluation),
                  'wall_seconds': summary['wall_seconds'], 'summary': str(experiment / 'summary.json')}
        state['generations'].append(record)
        if args.smoke:
            if not all(c['completed'] and c['fault_fraction'] == 0 and c['shield_fraction'] == 0
                       for c in metrics.values()):
                state.update(phase='error', runtime_smoke_passed=False)
                save()
                raise RuntimeError('Runtime smoke did not complete all candidates without guards')
            state.update(phase='finished', runtime_smoke_passed=True, learned_policy=False)
            save()
            return
        score, winner = ranked[0]
        if score > state['best_score']:
            state.update(best_score=score, best_parameters=candidates[winner], best_generation=generation)
        save()
        if score <= noise_margin:
            has_gain = any(v['unconstrained_gain'] is not None and v['unconstrained_gain'] > noise_margin
                           for v in assessments.values())
            state['stop_reason'] = ('Depth CV improved, but every improving candidate failed a recorded constraint'
                                    if has_gain else 'No depth CV gain above duplicate-baseline variability')
            break
        elites = [candidates[name] for value, name in ranked[:3] if value > -10]
        candidates, mean, std = next_population(elites, state['best_parameters'], generator)
        record['next_parameter_distribution'] = {'mean': mean, 'std': std}

    if state['best_score'] > 0 and 'stop_reason' not in state:
        # Export remains experimental until a held-out independent evaluation.
        spec = importlib.util.spec_from_file_location('velocity_schema', ROOT / 'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/utils/velocity_policy.py')
        schema = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(schema)
        contract = {**SCHEDULER_CONTRACT, 'schema_version': 3, 'observation_size': 14,
            'observation_fields': list(schema.OBSERVATION_FIELDS), 'robot': 'ur10_cb3',
            'control_period_s': .008, 'policy_action_repeat': 1, 'policy_period_s': .008,
            'policy_backend': 'torchscript_preston_depth',
            'objective': 'preston_rotation_dominated_depth', 'training_algorithm': state['algorithm'],
            'depth_model_assumption': 'constant RPM; rotation-dominated relative sliding',
            'physics_tool_diameter_mm': 30., 'tool_diameter_mm': 30., 'spindle_rpm': None,
            'reference_path_length_mm': profile['cursor_knots_mm'][-1],
            'reference_path_sha256': profile['reference_path_sha256'],
            'reference_roi': roi,
            'initial_profile_cell_size_mm': profile['model']['cell_size_mm'],
            'reference_profile_sha256': state['profile_sha256'], 'learned_parameters': state['best_parameters'],
            'validation_status': 'training_selection_only; independent validation required'}
        (output / 'params').mkdir()
        (output / 'params/velocity_policy.json').write_text(json.dumps(contract, indent=2))
        checkpoint = output / 'checkpoints/actor.pt'
        export_dwell_actor(profile, state['best_parameters'], checkpoint, contract)
        state.update(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    state.update(phase='finished', learned_policy='checkpoint' in state,
                 independent_validation_passed=False, finished_at_utc=datetime.now(timezone.utc).isoformat())
    save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--generations', type=int, default=2)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--surface-cell-mm', type=float, default=.5)
    parser.add_argument('--reuse-initial-evaluation', type=Path)
    args = parser.parse_args()
    if args.generations < 1:
        parser.error('generations must be positive')
    if not np.isfinite(args.surface_cell_mm) or not 0 < args.surface_cell_mm <= 2:
        parser.error('Use a positive depth cell size no coarser than 2 mm')
    if args.smoke and args.reuse_initial_evaluation:
        parser.error('Runtime smoke must execute new physics')
    train(args)
