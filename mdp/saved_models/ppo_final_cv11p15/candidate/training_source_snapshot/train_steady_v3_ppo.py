#!/usr/bin/env python3
"""Run the frozen genuine PPO trainer with study-private environment/metadata hooks."""
from pathlib import Path
import sys
from steady_training_adapter import STUDY, MANIFEST, read, verify_frozen_inputs


def adapted_source():
    source = (STUDY / 'original_train_latest_force_ppo.py').read_text()
    changes = [
        ('args = p.parse_args()\n',
         'args = p.parse_args()\nfrom steady_training_adapter import prepare_training_args, training_metadata, decorate_actor_export, snapshot_private_sources, snapshot_initial_training_state, make_env_class\nprepare_training_args(args)\n'),
        ('from nrs_rl.tasks.manager_based.nrs_rl.mdp.isaac_latest_force_env import IsaacLatestForceEnv\n',
         'from nrs_rl.tasks.manager_based.nrs_rl.mdp.isaac_latest_force_env import IsaacLatestForceEnv as _BaseIsaacLatestForceEnv\nIsaacLatestForceEnv = make_env_class(_BaseIsaacLatestForceEnv)\nexport_actor = decorate_actor_export(export_actor)\n'),
        ("    save(output / 'protocol.json', protocol)\n",
         "    protocol.update(training_metadata(args))\n    save(output / 'protocol.json', protocol)\n"),
        ('    env = IsaacLatestForceEnv(args.num_envs, args.seed, args.native_dir,\n',
         '    snapshot_private_sources(source)\n    env = IsaacLatestForceEnv(args.num_envs, args.seed, args.native_dir,\n'),
        ("    initial_hash = export_actor(model, output / 'checkpoints/initial_actor.pt', {'ppo_updates': 0})\n",
         "    snapshot_initial_training_state(model, optimizer, output, args, parent)\n    initial_hash = export_actor(model, output / 'checkpoints/initial_actor.pt', {'ppo_updates': 0})\n"),
    ]
    for old, new in changes:
        if source.count(old) != 1:
            raise ValueError('Frozen trainer hook no longer unique: ' + old)
        source = source.replace(old, new, 1)
    return source


def main():
    manifest = verify_frozen_inputs()
    original = manifest['original_trainer']
    sys.path.insert(0, str(STUDY))
    # Preserve the original trainer __file__ for its exact source snapshot paths.
    namespace = {'__name__': '__main__', '__file__': original, '__package__': None}
    exec(compile(adapted_source(), original, 'exec'), namespace)


if __name__ == '__main__':
    main()
