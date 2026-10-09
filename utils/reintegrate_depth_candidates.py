"""Reuse an initial population's saved dynamics with a finer depth grid.

The original experiment remains unchanged. This is measurement reanalysis,
not a new physics rollout or an independent validation.
"""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT as _REPO_ROOT
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time

import h5py
import numpy as np

from nrs_rl.tasks.manager_based.nrs_rl.utils.analyze_rotation_depth import RotationDominatedSurface, TASK

ROOT = _REPO_ROOT
DYNAMICS_FILES = (
    'scripts/skrl/preston_dwell_actor.py',
    'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/mdp/action.py',
    'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/mdp/rewards.py',
    'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/mdp/terminations.py',
    'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/nrs_rl_env_cfg.py',
    'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/utils/velocity_policy.py',
    'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/assets/assets/robots/ur10_w_spindle.usda',
    'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/assets/assets/robots/ur10e_w_spindle.py',
)


def validate_initial_population(summary, profile, profile_sha, reference_sha, candidates, seed):
    if (summary['seed'] != seed or not np.isclose(summary['dt_s'], .008)
            or not np.isclose(summary['physics_dt_s'], .002)
            or summary['source_path_sha256'] != reference_sha
            or summary['dwell_profile_sha256'] != profile_sha
            or summary['physics_tool_diameter_mm'] != 30.
            or summary['target_normal_force_n'] != 20.
            or not np.isclose(summary['path_length_mm'], profile['cursor_knots_mm'][-1], atol=.1)):
        raise ValueError('Saved population must match the full path, profile, seed and physical/control settings')
    recorded = {name: value['parameters'] for name, value in summary['candidates'].items()}
    if recorded != candidates:
        raise ValueError('Saved initial population has different policy coefficients')


def reintegrate(source, output, profile_path, candidates, seed, cell_size):
    started = time.monotonic()
    source = Path(source).resolve()
    output = Path(output)
    summary = json.loads((source / 'summary.json').read_text())
    profile = json.loads(profile_path.read_text())
    reference = TASK / 'datasets/cmd_continue9D_flat.h5'
    validate_initial_population(summary, profile,
        hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        hashlib.sha256(reference.read_bytes()).hexdigest(), candidates, seed)
    manifest_file = source.parents[1] / 'source_snapshot/manifest.json'
    manifest = json.loads(manifest_file.read_text())
    changed = [name for name in DYNAMICS_FILES
               if manifest.get(name) != hashlib.sha256((ROOT / name).read_bytes()).hexdigest()]
    if changed:
        raise ValueError(f'Saved dynamics/actor sources differ: {changed}')
    with h5py.File(reference) as stream:
        path = stream['position'][:, :3]
    surface = RotationDominatedSurface(path, summary['rotation_depth_model']['plane_normal'],
                                      tool_diameter_mm=30., cell_size_mm=cell_size)
    result = copy.deepcopy(summary)
    result.update(kind='saved_initial_population_depth_reanalysis',
                  original_rotation_depth_model=summary['rotation_depth_model'],
                  rotation_depth_model=surface.metadata,
                  source_evaluation=str(source), new_physics_rollouts=0,
                  original_rollout_wall_seconds=summary['wall_seconds'],
                  reused_trace_sha256={}, dynamics_source_manifest=str(manifest_file))
    output.mkdir(parents=True, exist_ok=False)
    for name in candidates:
        trace_path = source / f'{name}_trace.npz'
        with np.load(trace_path) as data:
            trace = dict(data)
        if int((trace['polishing_active'] > 0).sum()) != summary['candidates'][name]['processing_samples']:
            raise ValueError(f'{name}: saved processing sample count differs from trace')
        result['candidates'][name]['rotation_depth'] = surface.integrate_trace(trace)
        surface.save(output / f'{name}_rotation_surface.npz')
        shutil.copyfile(trace_path, output / trace_path.name)
        result['reused_trace_sha256'][name] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
        print('REINTEGRATED_DEPTH', name, json.dumps(result['candidates'][name]['rotation_depth']), flush=True)
    result['wall_seconds'] = time.monotonic()-started
    (output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    (output / 'progress.json').write_text(json.dumps({'phase': 'finished', 'new_physics_rollouts': 0,
                                                     'wall_seconds': result['wall_seconds']}, indent=2))
    return result
