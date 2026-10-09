"""Preserve a validated neural PPO actor and its resumable training state."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import torch

from nrs_rl.tasks.manager_based.nrs_rl.mdp.agents.ppo_network import PPOActorCritic
from nrs_rl.tasks.manager_based.nrs_rl.mdp.removal_config import check_ppo_or_baseline
from nrs_rl.tasks.manager_based.nrs_rl.utils.removal_metrics import sha, save
from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT


def validate_selection(report, user_selected=False):
    """A user may accept a small gain, while numerical/process checks remain mandatory."""
    if report['algorithm'] != 'PPO' or not report['numerical_audit_passed']:
        raise ValueError('Saving requires an audited neural PPO candidate')
    if not report['accepted']:
        if not user_selected:
            raise ValueError('Candidate missed automatic selection; explicit user selection is required')
        rows = [values['selected'] for values in report['per_seed'].values()]
        if len(rows) < 2 or not all(row['eligible'] for row in rows):
            raise ValueError('User selection cannot bypass full-path process eligibility')
        if not all(min(row['mode6_volume_ratios'].values()) >= .5 for row in rows):
            raise ValueError('User selection cannot bypass the Mode 6 removal floor')


def preserve(study, name, user_selected=False):
    study = Path(study).resolve()
    candidate = study / 'candidate'
    report = json.loads((candidate / 'validation.json').read_text())
    validate_selection(report, user_selected)
    source_manifest = json.loads((candidate / 'manifest.json').read_text())
    for relative, digest in source_manifest.items():
        assert sha(candidate / relative) == digest, relative
    extra = {'fixed_force_policy.json': ''}
    actor = torch.jit.load(str(candidate / 'checkpoints/actor.pt'), _extra_files=extra)
    contract = json.loads(extra['fixed_force_policy.json'])
    check_ppo_or_baseline(contract)
    assert contract['actor_kind'] == 'neural_ppo' and not contract['teacher_profile_used']
    state = torch.load(candidate / 'checkpoints/ppo_training_state.pt', weights_only=False, map_location='cpu')
    model = PPOActorCritic()
    model.load_state_dict(state['model'])
    torch.manual_seed(3219)
    with torch.no_grad():
        probes = torch.randn(4096, 16) * 2.
        torch.testing.assert_close(actor(probes), (.125 + .875 * model.actor(probes).tanh())[:, None], atol=0., rtol=0.)
    destination = ROOT / 'logs/experiments/uniformity_saves' / name
    destination.mkdir(exist_ok=False)
    shutil.copytree(candidate, destination / 'candidate')
    evidence = destination / 'evidence'
    evidence.mkdir()
    for filename in ['final_report.json', 'ppo_training_audit.json', 'independent_numerical_audit.json']:
        shutil.copy2(study / filename, evidence / filename)
    for relative, digest in source_manifest.items():
        assert sha(destination / 'candidate' / relative) == digest, relative
    record = {
        'status': 'user_saved_neural_ppo', 'algorithm': 'PPO',
        'saved_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_study': str(study), 'save_directory': str(destination),
        'checkpoint': str(destination / 'candidate/checkpoints/actor.pt'),
        'checkpoint_sha256': sha(destination / 'candidate/checkpoints/actor.pt'),
        'training_state': str(destination / 'candidate/checkpoints/ppo_training_state.pt'),
        'training_state_sha256': sha(destination / 'candidate/checkpoints/ppo_training_state.pt'),
        'validation': str(destination / 'candidate/validation.json'),
        'mean_heldout_metrics': report['mean_heldout_metrics'],
        'comparison_baseline': 'matched Mode 6 only',
        'automatic_acceptance_passed': bool(report['accepted']),
        'selection_authority': ('explicit_user_selection_of_eligible_candidate'
                                if user_selected else 'user_save_of_automatically_accepted_candidate'),
        'automatic_replacement_allowed': False, 'gui_deployed': False,
        'deployment_manifest_unchanged': True,
    }
    save(destination / 'SAVE.json', record)
    save(destination.parent / 'PPO_CURRENT.json', record)
    save(destination.parent / 'PPO_FIXED.json', record)
    return record


if __name__ == '__main__':
    torch.set_num_threads(1)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--user-selected', action='store_true',
                   help='Use only when the user explicitly selects a candidate below the practical gain threshold')
    args = p.parse_args()
    print(json.dumps(preserve(args.study, args.name, args.user_selected), indent=2))
