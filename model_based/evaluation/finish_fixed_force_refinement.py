#!/usr/bin/env python3
"""Full-path selection, independent paired validation, and conditional promotion."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT as _REPO_ROOT
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from nrs_rl.tasks.manager_based.nrs_rl.utils.assess_fixed_force_velocity import evaluate
from nrs_rl.tasks.manager_based.nrs_rl.model_based.policies.fixed_force_refinement import check_contract
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_fixed_force_refinement import HOLDOUTS, SAVED, SAVED_SHA, SELECTION_SEED, check_preserved
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_fixed_force_velocity import save, sha
from nrs_rl.tasks.manager_based.nrs_rl.model_based.evaluation.verify_fixed_force_refinement import verify

ROOT = _REPO_ROOT


def batch(training, root, label, names, seed):
    command = [sys.executable, str(ROOT/'scripts/probe_fixed_force_refinement.py'),
        '--headless', '--device', 'cpu', '--training-dir', str(training),
        '--output', str(root/label), '--names', ','.join(names), '--seed', str(seed),
        '--max-steps', '42000', '--hard-force-stop-n', '100', '--uniformity-only']
    command_file = root/f'{label}_command.json'
    if command_file.exists():
        assert json.loads(command_file.read_text()) == command
    else:
        save(command_file, command)
    if not (root/label/'summary.json').exists():
        if (root/label).exists():
            raise RuntimeError(f'Incomplete batch retained for diagnosis: {root/label}')
        with (root/f'{label}.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                env={**os.environ, 'MPLCONFIGDIR': '/tmp/nrs_fixed_refinement_mpl', 'MPLBACKEND': 'Agg',
                     'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    source = json.loads((root/label/'summary.json').read_text())
    assert source['seed'] == seed and source['names'] == names
    for name, digest in source['checkpoint_sha256'].items():
        assert sha(training/'checkpoints'/f'{"constant" if name == "constant_b" else name}.pt') == digest
    assert source['checkpoint_sha256']['saved'] == SAVED_SHA
    report = evaluate(root/label, audit=True)
    print('REFINEMENT_BATCH', label, json.dumps({n: {
        'eligible': m['eligible'], 'mode6_cv_gain': m['mode6_cv_gain'],
        'mode6_volume_ratios': m['mode6_volume_ratios']} for n, m in report['candidates'].items()}), flush=True)
    return report


def parallel_batches(training, root, specs):
    reports = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {pool.submit(batch, training, root, label, names, seed): label
                for label, names, seed in specs}
        for job in as_completed(jobs):
            reports[jobs[job]] = job.result()
    return reports


def acceptance(report):
    m = report['candidates']
    noise = abs(m['constant_b']['spatial_cv']/m['constant']['spatial_cv']-1.)
    margin = max(.005, 2.*noise)
    improvement = m['selected']['mode6_cv_gain']-m['saved']['mode6_cv_gain']
    return {'passed': bool(all(x['eligible'] for x in m.values()) and improvement >= margin),
        'additional_mode6_normalized_cv_gain': improvement,
        'minimum_additional_mode6_normalized_gain': margin,
        'paired_mode6_repeat_variation': noise,
        'practical_threshold_not_statistical_significance': True,
        'volume_floor_all_three_scopes': .5}


def figures(export, physics, state):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['constant', 'saved', 'selected']
    labels = ['Mode 6', 'Saved model', 'New candidate']
    colors = ['#777777', '#4477aa', '#dd8844']
    reports = [state['heldouts'][str(s)] for s in HOLDOUTS]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, key, label in zip(axes, ['spatial_cv', 'roi_volume_over_k', 'full_roi_volume_over_k'],
                             ['Spatial CV', 'Processing removal', 'Full-ROI removal']):
        for j, name in enumerate(names):
            values = [100.*r['candidates'][name][key]/r['candidates']['constant'][key] for r in reports]
            positions = np.arange(2)+(j-1)*.25
            ax.bar(positions, values, width=.23, color=colors[j], label=labels[j])
            for x, value in zip(positions, values):
                ax.text(x, value+1., f'{value:.1f}', ha='center', fontsize=8)
        ax.axhline(100., color='gray', ls='--', lw=.8)
        if key != 'spatial_cv':
            ax.axhline(50., color='red', ls=':', lw=.8)
        ax.set_xticks([0, 1], list(map(str, HOLDOUTS))); ax.set_xlabel('Held-out seed')
        ax.set_title(label); ax.set_ylabel('Mode 6 = 100%'); ax.grid(axis='y', alpha=.15)
        ax.set_ylim(0, ax.get_ylim()[1]*1.10)
    axes[0].legend(fontsize=8)
    fig.suptitle('Fixed target force 20 N; assumed virtual rotation 1000 RPM')
    fig.savefig(export/'00_mode6_comparison.png', dpi=160); plt.close(fig)
    folder = physics/f'heldout_{HOLDOUTS[0]}'
    arrays = []
    for name in names:
        with np.load(folder/f'{name}_processing_surface.npz') as f:
            arrays.append(dict(f))
    cap = max(float(np.quantile(d['depth_over_k'][d['roi']], .995)) for d in arrays)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for i, (data, label) in enumerate(zip(arrays, labels)):
        depth, mask = data['depth_over_k'], data['roi']
        extent = [data['u_mm'][0], data['u_mm'][-1], data['v_mm'][0], data['v_mm'][-1]]
        for j, values, vmax in [(0, depth, cap), (1, depth/depth[mask].mean(), 2.)]:
            im = axes[j, i].imshow(np.where(mask, values, np.nan), origin='lower', extent=extent,
                                   vmin=0, vmax=vmax, cmap='viridis')
            axes[j, i].set_title(label if j == 0 else 'Depth / own mean')
            axes[j, i].set_xlabel('u (mm)'); axes[j, i].set_ylabel('v (mm)')
            fig.colorbar(im, ax=axes[j, i], label='h/K (uncalibrated)' if j == 0 else 'h/mean(h)')
    fig.suptitle('Common processing footprint; first 2 s and first 18 mm footprint excluded')
    fig.savefig(export/'01_processing_depth_maps.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True, constrained_layout=True)
    times, rates, traces = [], [], []
    for name, label, color in zip(names, labels, colors):
        with np.load(folder/f'{name}_trace.npz') as f:
            data = dict(f)
        traces.append(data); keep = data['time_s'] >= 2.; t = data['time_s'][keep]
        for ax, key in zip(axes, ['target_force_n', 'normal_force_n', 'measured_speed_mm_s', 'raw_mrr_n_mm_s']):
            ax.plot(t, data[key][keep], lw=.6, label=label, color=color, alpha=.8)
        times.append(t); rates.append(data['raw_mrr_n_mm_s'][keep])
    for ax, label in zip(axes, ['Target force (N)', 'Measured force (N)', 'TCP speed (mm/s)', 'Removal rate / K']):
        ax.set_ylabel(label); ax.grid(alpha=.2)
    axes[0].legend(); axes[0].set_ylim(19., 21.); axes[-1].set_xlabel('Time since Force entry (s)')
    fig.savefig(export/'02_process_signals.png', dpi=160); plt.close(fig)
    bins = np.arange(2., max(t[-1] for t in times)+.081, .08)
    heat = np.full((3, len(bins)-1), np.nan)
    for i, (t, r) in enumerate(zip(times, rates)):
        count = np.histogram(t, bins=bins)[0]; value = np.histogram(t, bins=bins, weights=r)[0]
        np.divide(value, count, out=heat[i], where=count > 0)
    fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)
    cmap = plt.colormaps['magma'].copy(); cmap.set_bad('#ddd')
    im = ax.imshow(heat, aspect='auto', extent=[bins[0], bins[-1], 2.5, -.5],
                   cmap=cmap, interpolation='nearest')
    ax.set_yticks(range(3), labels); ax.set_xlabel('Time (s); gray = completed')
    ax.set_title('Instantaneous removal rate; 80 ms mean, virtual rotation included')
    fig.colorbar(im, ax=ax, label='Removal rate / K')
    fig.savefig(export/'03_temporal_heatmap.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, constrained_layout=True)
    for data, label, color in zip(traces, labels, colors):
        axes[0].plot(data['cursor_mm'], data['commanded_speed_mm_s'], lw=.85, label=label, color=color)
        keep = data['commanded_speed_mm_s'] > .1
        axes[1].plot(data['cursor_mm'][keep], 1./data['commanded_speed_mm_s'][keep], lw=.85, color=color)
    axes[0].set_ylabel('Applied feed (mm/s)'); axes[0].legend()
    axes[1].set_ylabel('Residence per distance (s/mm)'); axes[1].set_xlabel('Reference path progress (mm)')
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(export/'04_feed_and_residence.png', dpi=160); plt.close(fig)


def write_tables(export, state):
    rows = []
    for seed, report in state['heldouts'].items():
        for name, m in report['candidates'].items():
            rows.append({'seed': seed, 'model': name, 'spatial_cv': m['spatial_cv'],
                'mode6_cv_reduction_percent': 100.*m['mode6_cv_gain'],
                **{f'mode6_{key}_percent': 100.*v for key, v in m['mode6_volume_ratios'].items()},
                'time_s': m['time_s'], 'target_force_min_n': m['force_target_min_n'],
                'target_force_max_n': m['force_target_max_n'], 'eligible': m['eligible']})
    with (export/'mode6_comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return rows


def promote(root, state, export):
    """Keep every prior save immutable, and change pointers only on acceptance."""
    pointers = ROOT/'logs/experiments/uniformity_saves'
    for name in ['CURRENT.json', 'FIXED.json']:
        assert sha(pointers/name) == state['previous_pointer_hashes'][name]
    check_preserved()
    if not state['accepted']:
        state.update(active_checkpoint=str(SAVED/'candidate/checkpoints/actor.pt'),
                     active_checkpoint_sha256=SAVED_SHA, previous_save_retained=True)
        return
    dest = pointers/f'fixed20_virtual1000_refined_{datetime.now():%Y%m%d_%H%M}'
    dest.mkdir(exist_ok=False)
    shutil.copytree(root/'candidate', dest/'candidate')
    for name in ['independent_numerical_audit.json', 'source_manifest.json', 'REPORT.txt']:
        shutil.copyfile(root/name, dest/name)
    shutil.copyfile(export/'mode6_comparison.csv', dest/'mode6_comparison.csv')
    save(dest/'validation_result.json', state)
    save(dest/'save_manifest.json', {str(p.relative_to(dest)): sha(p) for p in dest.rglob('*') if p.is_file()})
    checkpoint = dest/'candidate/checkpoints/actor.pt'
    assert sha(checkpoint) == state['checkpoint_sha256']
    pointer = {'status': 'validated_fixed_force_candidate', 'save_directory': str(dest),
        'checkpoint': str(checkpoint), 'sha256': sha(checkpoint),
        'saved_at_utc': datetime.now(timezone.utc).isoformat(), 'source_attempt': str(root/'result.json'),
        'original_candidate': state['selected'], 'previous_save_directory': str(SAVED),
        'previous_checkpoints_preserved': True, 'target_force_n': 20., 'policy_controls_force': False,
        'action_dimension': 1, 'assumed_virtual_spindle_rpm': 1000., 'minimum_mode6_volume_ratio': .5,
        'heldout_metrics': state['heldouts'], 'deployed': False, 'current_GUI_runtime_compatible': False,
        'contract': str(dest/'candidate/fixed_force_policy.json'),
        'validation': str(dest/'candidate/validation.json'),
        'comparison_baseline': 'fresh matched Mode 6: 20 N, 6 mm/s, same assumed virtual 1000 RPM',
        'runtime_requirement': 'scalar-feed runtime enforcing checkpoint speed mapping, fixed 20 N and rotating-disk estimator'}
    for name in ['CURRENT.json', 'FIXED.json']: save(pointers/name, pointer)
    requirements = ROOT/'deployment/POLICY_REQUIREMENTS.json'
    config = json.loads(requirements.read_text())
    assert config['active_new_checkpoint'] is None and not config['gui_runtime_ready']
    config.update(training_directory=str(root), validated_training_checkpoint=str(checkpoint),
                  checkpoint_sha256=sha(checkpoint), training_validation_passed=True)
    save(requirements, config)
    state.update(active_checkpoint=str(checkpoint), active_checkpoint_sha256=sha(checkpoint),
                 save_directory=str(dest), previous_save_retained=True)


def main(args):
    root = args.root.resolve(); training = root/'training'; physics = root/'physics'
    physics.mkdir(exist_ok=True); started = time.monotonic(); torch.set_num_threads(1)
    check_preserved()
    state = {'phase': 'physics_selection', 'protocol': json.loads((training/'protocol.json').read_text()),
        'gui_trial_executed': False, 'deployed': False, 'deployment_ready': False, 'accepted': False,
        'heldouts': {}, 'previous_save': str(SAVED), 'previous_save_sha256': SAVED_SHA}
    def status():
        state['elapsed_s'] = time.monotonic()-started; save(root/'status.json', state)
    status()
    scripts = ['fixed_force_refinement.py', 'fixed_force_refinement_action.py', 'probe_fixed_force_refinement.py',
        'train_fixed_force_refinement.py', 'finish_fixed_force_refinement.py', 'verify_fixed_force_refinement.py',
        'test_fixed_force_refinement.py', 'fixed_force_velocity.py', 'fixed_force_velocity_action.py',
        'assess_fixed_force_velocity.py', 'train_fixed_force_velocity.py', 'spatial_uniformity.py',
        'tcp_tracking_compensation.py']
    sources = {}
    for rel in ['scripts/'+n for n in scripts]+[
        'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/mdp/action.py',
        'source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl/utils/preston_surface.py']:
        dest = root/'source_snapshot'/rel; dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists(): assert sha(dest) == sha(ROOT/rel), rel
        else: shutil.copyfile(ROOT/rel, dest)
        sources[rel] = sha(dest)
    save(root/'source_manifest.json', sources)
    archive = root/'previous_selection_pointers'; archive.mkdir(exist_ok=True)
    state['previous_pointer_hashes'] = {}
    for name in ['CURRENT.json', 'FIXED.json']:
        source = ROOT/'logs/experiments/uniformity_saves'/name
        if (archive/name).exists(): assert sha(archive/name) == sha(source)
        else: shutil.copyfile(source, archive/name)
        state['previous_pointer_hashes'][name] = sha(source)
    # Predeclared acceptance noise control; duplicate Mode 6 affects thresholds,
    # never candidate ranking or checkpoint fitting.
    state['noise_rule'] = 'required extra Mode6-normalized CV gain=max(0.005,2*paired Mode6 variation)'
    status()
    reports = parallel_batches(training, physics, [
        ('selection_a_2701', ['constant', 'saved', 'wide193_90', 'wide193_110', 'dwell193_135'], SELECTION_SEED),
        ('selection_b_2701', ['constant', 'saved', 'dense289_110', 'robust193_110', 'lead193_110'], SELECTION_SEED)])
    rows = [(m['mode6_cv_gain'], label, name) for label, r in reports.items()
            for name, m in r['candidates'].items() if name not in ('constant', 'saved') and m['eligible']]
    if not rows:
        raise RuntimeError('No eligible new full-path candidate; previous save is preserved')
    gain, label, name = max(rows)
    bundle = root/'candidate'; (bundle/'checkpoints').mkdir(parents=True, exist_ok=True)
    checkpoint = bundle/'checkpoints/actor.pt'; source = training/'checkpoints'/f'{name}.pt'
    if checkpoint.exists(): assert sha(checkpoint) == sha(source)
    else: shutil.copyfile(source, checkpoint)
    extra = {'fixed_force_policy.json': ''}; torch.jit.load(str(checkpoint), _extra_files=extra)
    contract = json.loads(extra['fixed_force_policy.json']); check_contract(contract)
    save(bundle/'fixed_force_policy.json', contract)
    shutil.copyfile(checkpoint, training/'checkpoints/selected.pt')
    state.update(phase='independent_physics_validation', selected=name, selection_reports=reports,
        selection_label=label, selection_mode6_gain=gain, checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint), selected_before_holdouts_at=datetime.now(timezone.utc).isoformat())
    if (bundle/'selection.json').exists():
        prior = json.loads((bundle/'selection.json').read_text())
        assert prior['checkpoint_sha256'] == state['checkpoint_sha256']
        state['selected_before_holdouts_at'] = prior['selected_before_holdouts_at']
    else: save(bundle/'selection.json', state)
    status()
    heldouts = parallel_batches(training, physics, [
        (f'heldout_{seed}', ['constant', 'constant_b', 'saved', 'selected'], seed) for seed in HOLDOUTS])
    for report in heldouts.values():
        report['acceptance'] = acceptance(report)
        state['heldouts'][str(report['seed'])] = report
    state.update(phase='independent_numerical_audit'); status()
    state['numerical_audit'] = verify(root, state)
    state['accepted'] = all(r['acceptance']['passed'] for r in state['heldouts'].values())
    state['rpm_sensitivity'] = {}
    for seed in HOLDOUTS:
        state['rpm_sensitivity'][str(seed)] = {str(rpm): evaluate(physics/f'heldout_{seed}', rpm=rpm)
                                             for rpm in [300., 3000.]}
    state['rpm_sensitivity_kind'] = 'same-trace counterfactual reintegration; no new spindle dynamics'
    check_preserved(); assert sha(checkpoint) == state['checkpoint_sha256']
    for rel, digest in sources.items(): assert sha(ROOT/rel) == digest, rel
    validation = {'passed': state['accepted'], 'numerical_audit_passed': True,
        'heldouts': state['heldouts'], 'target_force_n': 20., 'action_dimension': 1,
        'source_hashes_unchanged': True, 'previous_save_preserved': True,
        'selected_before_holdouts': True, 'rpm_is_assumed': True, 'urdf_rotation_unchanged': True,
        'gui_trial_executed': False, 'deployed': False,
        'runtime_requirement': 'Dedicated scalar-feed runtime; old variable-force runtime is incompatible'}
    save(bundle/'validation.json', validation)
    export = ROOT/'logs/polishing_results'/datetime.now().strftime('%Y%m%d_%H%M')
    export.mkdir(parents=True, exist_ok=False)
    figures(export, physics, state); table = write_tables(export, state)
    lines = [
        'Fixed-force refinement: matched Mode 6 comparisons only.',
        'Target force fixed 20 N; scalar forward speed/residence; no reverse/second pass.',
        'Virtual spindle assumed 1000 RPM in removal estimator only. URDF dynamics unchanged.',
        'CV: spatial final removal depth, common .5-mm grid, zeros retained.',
        'First 2 active seconds and fixed first-18-mm reference footprint excluded.',
        'Volume floor >=50% of Mode 6 in processing ROI, full ROI, whole grid, every heldout.',
        'Training: constrained model-based residence optimization, not PPO.',
        'Two fresh heldouts are a limited repeatability check, not broad hardware validation.',
        f'New candidate: {name}; accepted: {state["accepted"]}.',
        f'Previous save SHA-256 unchanged: {SAVED_SHA}',
        'Removal values h/K are uncalibrated estimates. No GUI robot trial or deployment.',
    ]
    for row in table:
        if row['model'] not in ('saved', 'selected'): continue
        lines.append(f"Seed {row['seed']} {row['model']}: Mode6 CV reduction {row['mode6_cv_reduction_percent']:.4f}%; "
            f"processing removal {row['mode6_roi_volume_over_k_percent']:.4f}%; "
            f"full-ROI removal {row['mode6_full_roi_volume_over_k_percent']:.4f}%; "
            f"whole-grid removal {row['mode6_whole_grid_volume_over_k_percent']:.4f}% of Mode6.")
    report = '\n'.join(lines)+'\n'
    (export/'REPORT.txt').write_text(report); (root/'REPORT.txt').write_text(report)
    save(bundle/'manifest.json', {str(p.relative_to(bundle)): sha(p) for p in bundle.rglob('*') if p.is_file()})
    state.update(export_directory=str(export), phase='conditional_promotion'); status()
    promote(root, state, export)
    state.update(phase='complete'); status()
    save(root/'result.json', state); save(export/'result.json', state)
    save(export/'manifest.json', {p.name: sha(p) for p in export.iterdir() if p.is_file()})
    print('REFINEMENT_COMPLETE', json.dumps({'accepted': state['accepted'],
        'active_checkpoint': state['active_checkpoint'], 'export': str(export), 'elapsed_s': state['elapsed_s']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    arguments = parser.parse_args()
    try:
        main(arguments)
    except Exception as exc:
        save(arguments.root/'failure.json', {'error': str(exc), 'type': type(exc).__name__,
            'time_utc': datetime.now(timezone.utc).isoformat(), 'previous_save_not_deleted': True})
        raise
