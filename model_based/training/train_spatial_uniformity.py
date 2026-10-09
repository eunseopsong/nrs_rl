#!/usr/bin/env python3
"""Train bounded spatial feedback by CEM in a nonrotating dose surrogate.

Exported candidates must subsequently pass probe_spatial_uniformity.py. This
is episodic policy parameter learning, not PPO or a new G-Force checkpoint.
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
import torch
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.spatial_uniformity import (DoseGeometry, SpatialUniformityActor, LOW, HIGH,
    PARAMETERS, CONTRACT, REFERENCE, ROOT, assess, action_numpy)


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix('.next.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def main(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    model = DoseGeometry(cell_mm=2.)
    protocol = {**CONTRACT, 'stage': 'surrogate_policy_training',
        'algorithm': 'cem_spatial_feedback_parameter_learning', 'parameter_names': PARAMETERS,
        'surrogate_is_isaac_physics': False, 'training_seeds': [1201, 1202],
        'surrogate_holdout_seeds': [1401, 1402, 1403],
        'physics_training_seed': 1501, 'physics_holdout_seeds': [1601, 1602],
        'source_path_sha256': hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        'fixed_roi': model.surface.metadata, 'target_depth_over_k': model.target_depth,
        'path_length_mm': model.length, 'uniformity_metric': 'spatial_depth_cv',
        'gates': {'time_ratio_max': 1.05, 'volume_ratio': [.95, 1.05],
                  'force_mean_n_max': 21., 'force_peak_n_max': 32., 'distance_ratio_max': 1.10},
        'holdouts_used_for_selection': False, 'deployed': False,
        'started_at_utc': datetime.now(timezone.utc).isoformat()}
    save(root/'protocol.json', protocol)
    snapshots = ['scripts/spatial_uniformity.py', 'scripts/train_spatial_uniformity.py',
                 'scripts/probe_spatial_uniformity.py', 'scripts/spatial_uniformity_action.py',
                 'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/mdp/action.py']
    manifest = {}
    for rel in snapshots:
        source = ROOT/rel
        if not source.exists():
            continue
        target = root/'source_snapshot'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest[rel] = hashlib.sha256(target.read_bytes()).hexdigest()
    save(root/'source_manifest.json', manifest)
    baseline = np.array([0., 0., 0., 0., 0., 0., 0., 0., .6])
    incumbent = np.array([1.5, .1, 0., 1., .02, .02, .1, 0., .2])
    mean = (incumbent-LOW)/(HIGH-LOW)
    std = np.full(len(LOW), .22)
    rng = np.random.default_rng(args.seed)
    generations = []
    archive = {}
    started = time.monotonic()
    for generation in range(args.generations):
        draws = np.clip(rng.normal(mean, std, (args.population-4, len(LOW))), 0., 1.)
        population = np.vstack((baseline, incumbent,
            [2., 0., 0., 1., 0., .02, 0., 0., .2],
            [2., .3, 0., .8, 0., .05, 0., 2., .025], LOW+draws*(HIGH-LOW)))
        scores = []
        runs = []
        for seed in protocol['training_seeds']:
            metrics, _, _, _ = model.fast_rollout(population, seed=seed)
            ranking = [assess(m, metrics[0]) for m in metrics]
            scores.append([r['score'] for r in ranking])
            runs.append({'seed': seed, 'metrics': metrics, 'ranking': ranking})
        scores = np.min(scores, axis=0)
        order = np.argsort(scores)[::-1]
        incumbent = population[order[0]].copy()
        elites = (population[order[:max(3, args.population//5)]]-LOW)/(HIGH-LOW)
        mean = .25*mean+.75*elites.mean(axis=0)
        std = np.maximum(.04, .25*std+.75*elites.std(axis=0))
        for index in order[:8]:
            key = tuple(population[index].round(9))
            archive[key] = {'parameters': population[index].tolist(), 'score': float(scores[index])}
        generations.append({'generation': generation, 'best_score': float(scores[order[0]]),
                            'parameters': incumbent.tolist(), 'runs': runs})
        save(root/f'generation_{generation}.json', generations[-1])
        progress = {'phase': 'surrogate_training', 'generation': generation+1,
                    'total_generations': args.generations, 'elapsed_s': time.monotonic()-started,
                    'best_score': float(scores[order[0]]), 'best_parameters': incumbent.tolist()}
        save(root/'status.json', progress)
        print('SPATIAL_TRAIN', json.dumps(progress), flush=True)
    ordered = sorted(archive.values(), key=lambda x: x['score'], reverse=True)
    # Select from training scores only. The frozen no-return ablation shares
    # every parameter except its return-request gain.
    selected = {'constant': baseline.tolist(), 'learned': ordered[0]['parameters']}
    selected['learned_no_return'] = list(selected['learned'])
    selected['learned_no_return'][7] = 0.
    for rank, item in enumerate(ordered[1:3], 1):
        selected[f'alternative_{rank}'] = item['parameters']
    save(root/'candidates.json', selected)
    (root/'checkpoints').mkdir()
    export = {}
    probes = rng.uniform(-2., 2., (2048, 16)).astype(np.float32)
    for name, parameters in selected.items():
        actor = SpatialUniformityActor(parameters).eval()
        contract = {**protocol, 'parameters': parameters, 'validation_status': 'surrogate_candidate_only'}
        path = root/'checkpoints'/f'{name}.pt'
        torch.jit.save(torch.jit.script(actor), str(path), _extra_files={'spatial_policy.json': json.dumps(contract)})
        reloaded = torch.jit.load(str(path))
        with torch.no_grad():
            values = reloaded(torch.tensor(probes)).numpy()
        error = float(np.max(np.abs(values-action_numpy(probes, parameters))))
        if error > 1e-5:
            raise RuntimeError('Numpy/TorchScript policy mismatch')
        export[name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'parity_max_error': error}
    save(root/'export_validation.json', export)
    names = list(selected)
    heldouts = []
    for seed in protocol['surrogate_holdout_seeds']:
        metrics, depth, residence, trace = model.fast_rollout(list(selected.values()), seed=seed, record=True)
        heldouts.append({'seed': seed, 'candidates': {name: {'metrics': m, 'comparison': assess(m, metrics[0])}
                         for name, m in zip(names, metrics)}})
        np.savez_compressed(root/f'surrogate_heldout_{seed}.npz', depth=depth, residence=residence,
                            trace=trace, roi=model.surface.roi, names=np.array(names),
                            u_mm=model.surface.u, v_mm=model.surface.v)
    save(root/'surrogate_validation.json', heldouts)
    save(root/'status.json', {'phase': 'surrogate_complete_pending_isaac',
         'elapsed_s': time.monotonic()-started, 'parameters': selected['learned'],
         'new_checkpoint': str(root/'checkpoints/learned.pt'), 'deployed': False})
    print('SPATIAL_SURROGATE_COMPLETE', str(root), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--generations', type=int, default=8)
    parser.add_argument('--population', type=int, default=28)
    parser.add_argument('--seed', type=int, default=20261007)
    args = parser.parse_args()
    if args.generations < 1 or args.population < 8:
        parser.error('Need at least one generation and eight candidates')
    main(args)
