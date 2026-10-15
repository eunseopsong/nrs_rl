"""Joint feed/RPM policy learning with measured Isaac contact and virtual rotation."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import time

from isaaclab.app import AppLauncher
from nrs_rl.tasks.manager_based.nrs_rl.mdp.robot_geometry import configure_training_robot
from nrs_rl.tasks.manager_based.nrs_rl.mdp.rpm_control import RPMConfig

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--parent', type=Path, required=True)
p.add_argument('--replay', type=Path, required=True)
p.add_argument('--native-dir', type=Path, required=True)
p.add_argument('--control-snapshot', type=Path, required=True)
p.add_argument('--gui-scene', type=Path, required=True)
p.add_argument('--seed', type=int, default=5210)
p.add_argument('--updates', type=int, default=6)
p.add_argument('--rollout-steps', type=int, default=10240)
p.add_argument('--initialization-kind', choices=['static', 'dynamic'], default='static')
p.add_argument('--warmstart-steps', type=int)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.warmstart_steps is None:
    args.warmstart_steps = 8000 if args.initialization_kind == 'dynamic' else 3500
if args.output.exists() or args.device != 'cpu':
    p.error('Use a new output directory and --device cpu')
robot = configure_training_robot(args.control_snapshot)
app = AppLauncher(args).app

import numpy as np
import torch
from nrs_rl.tasks.manager_based.nrs_rl.mdp.agents.feed_rpm_ppo import contract, export_actor
if args.initialization_kind == 'dynamic':
    from nrs_rl.tasks.manager_based.nrs_rl.mdp.agents.initialize_dynamic_feed_rpm import initialize
else:
    from nrs_rl.tasks.manager_based.nrs_rl.mdp.agents.initialize_feed_rpm import initialize
from nrs_rl.tasks.manager_based.nrs_rl.mdp.isaac_feed_rpm_env import IsaacFeedRPMEnv
from nrs_rl.tasks.manager_based.nrs_rl.utils.removal_metrics import save, sha


def learn():
    torch.set_num_threads(1)
    np.random.seed(args.seed)
    output = args.output.resolve()
    output.mkdir(parents=True)
    (output / 'checkpoints').mkdir()
    c = RPMConfig()
    model, initialization = initialize(args.parent, args.replay, args.seed, args.warmstart_steps)
    save(output / 'initialization.json', initialization)
    print('RPM_INITIALIZATION', json.dumps(initialization), flush=True)
    initial_actor = copy.deepcopy(model.actor).eval().requires_grad_(False)
    # New reward/value semantics: fit critic from scratch; no stale Adam moments.
    optimizer = torch.optim.Adam([
        {'params': model.actor.feed.parameters(), 'lr': 2.e-5},
        {'params': model.actor.rpm.parameters(), 'lr': 3.e-5},
        {'params': [model.log_std], 'lr': 1.e-5},
        {'params': model.critic.parameters(), 'lr': 3.e-4}], eps=1.e-5)
    protocol = {**contract(c), 'algorithm': 'PPO-Clip', 'robot_geometry': robot,
        'seed': args.seed, 'updates': args.updates, 'rollout_steps': args.rollout_steps,
        'num_envs': 4, 'physics_dt_s': .008, 'gamma': 1., 'gae_lambda': .97,
        'minibatch_size': 2048, 'epochs': 4, 'clip_range': .2, 'maximum_kl': .015,
        'critic_warmup_updates': 1, 'gradient_clipping': 'actor/critic independently, norm .5',
        'mean_rate_floor_ratio': .9, 'spatial_objective': 'full contact ROI including entry and uncovered cells',
        'reward': c.metadata(), 'parent': str(args.parent), 'parent_sha256': sha(args.parent),
        'initialization': initialization, 'evaluation_exploration': False,
        'feed_anchor_coefficient': .02,
        'controller_snapshot': str(args.control_snapshot), 'scene': str(args.gui_scene)}
    save(output / 'protocol.json', protocol)
    snapshot = output / 'source_snapshot'
    snapshot.mkdir()
    mdp = Path(__file__).parent.parent
    for source in [Path(__file__), Path(__file__).with_name('feed_rpm_ppo.py'),
                   Path(__file__).with_name('initialize_dynamic_feed_rpm.py' if args.initialization_kind == 'dynamic'
                                            else 'initialize_feed_rpm.py'), mdp / 'rpm_control.py',
                   mdp / 'isaac_feed_rpm_env.py', mdp / 'isaac_latest_force_env.py']:
        shutil.copy2(source, snapshot / source.name)
    export_actor(model, output / 'checkpoints/ppo_0000.pt', c,
                 {'ppo_updates': 0, 'initialization': initialization, 'robot_geometry': robot})
    env = IsaacFeedRPMEnv(4, args.seed, args.native_dir, args.control_snapshot, output,
                         c, args.gui_scene)
    shape = (args.rollout_steps, 4)
    observations = torch.zeros((*shape, 22))
    latent = torch.zeros((*shape, 2))
    logprobs, rewards, dones, values, advantages = [torch.zeros(shape) for _ in range(5)]
    started = last_report = time.monotonic()
    episodes = completed = actor_updates = 0
    try:
        for update in range(1, args.updates + 1):
            obs = env.reset_batch() if update > 1 else env.observation()
            rows = []
            warmup = update == 1
            with torch.no_grad():
                for t in range(args.rollout_steps):
                    observations[t] = obs
                    action, latent[t], logprobs[t], _, values[t] = model.action_value(obs)
                    obs, rewards[t], done, records = env.step(action)
                    dones[t] = done
                    if not all(torch.isfinite(x).all() for x in (obs, rewards[t], values[t])):
                        raise RuntimeError('Nonfinite Isaac rollout')
                    for row in records:
                        if row is not None:
                            rows.append(row)
                            episodes += 1
                            completed += int(row['completed'])
                    if time.monotonic() - last_report > 30.:
                        status = {'phase': 'rollout', 'update': update, 'step': t + 1,
                            'steps_per_update': args.rollout_steps, 'wall_seconds': time.monotonic() - started,
                            'completed_episodes': completed, 'episodes': episodes,
                            'cursor_mm': env.term.path_cursor_mm.tolist(),
                            'rpm': [s.rpm for s in env.spindles]}
                        save(output / 'status.json', status)
                        print('RPM_ROLLOUT', json.dumps(status), flush=True)
                        last_report = time.monotonic()
                next_value = model.critic(obs)
                gae = torch.zeros(4)
                for t in reversed(range(args.rollout_steps)):
                    nextv = next_value if t == args.rollout_steps - 1 else values[t + 1]
                    alive = 1. - dones[t]
                    gae = rewards[t] + nextv * alive - values[t] + .97 * alive * gae
                    advantages[t] = gae
                returns = advantages + values
                # Do not fit the actor on a rollout's unfinished suffix.
                valid = dones.flip(0).cumsum(0).flip(0) > 0
            data = [x.flatten(0, 1) for x in (observations, latent, logprobs,
                                             advantages, returns, valid)]
            before = torch.cat([x.detach().flatten() for x in model.actor.parameters()]).clone()
            old_mean = model.actor(data[0]).detach()
            old_std = model.log_std.detach().clamp(-5., -1.).exp().clone()
            losses = []
            stop = False
            for epoch in range(8 if warmup else 4):
                order = torch.randperm(len(data[0]))
                for start in range(0, len(order), 2048):
                    ids = order[start:start + 2048]
                    _, _, lp, entropy, value = model.action_value(data[0][ids], data[1][ids])
                    selected = data[5][ids]
                    actor_loss = torch.zeros(())
                    if not warmup and selected.any():
                        adv = data[3][ids][selected]
                        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1.e-8)
                        ratio = (lp[selected] - data[2][ids][selected]).exp()
                        actor_loss = torch.maximum(-adv * ratio, -adv * ratio.clamp(.8, 1.2)).mean()
                        # Preserve learned spatial coverage, allowing PPO to refine it.
                        current_feed = model.actor(data[0][ids])[:, 0].tanh()
                        with torch.no_grad():
                            reference_feed = initial_actor(data[0][ids])[:, 0].tanh()
                        actor_loss += .02 * (current_feed[selected] - reference_feed[selected]).square().mean()
                    critic_loss = .5 * (value - data[4][ids]).square().mean()
                    loss = actor_loss + .5 * critic_loss
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    actor_norm = torch.nn.utils.clip_grad_norm_([*model.actor.parameters(), model.log_std], .5)
                    critic_norm = torch.nn.utils.clip_grad_norm_(model.critic.parameters(), .5)
                    if not torch.isfinite(loss) or not torch.isfinite(actor_norm + critic_norm):
                        raise RuntimeError('Nonfinite PPO update')
                    optimizer.step()
                    with torch.no_grad():
                        model.log_std.clamp_(-5., -1.)
                        new_std = model.log_std.exp()
                        mean = model.actor(data[0][ids])
                        kl = (torch.log(new_std / old_std) +
                              (old_std.square() + (old_mean[ids] - mean).square()) /
                              (2. * new_std.square()) - .5).sum(-1).mean()
                    losses.append([float(actor_loss), float(critic_loss), float(kl)])
                    if not warmup and kl > .015:
                        stop = True
                        break
                if stop:
                    break
            after = torch.cat([x.detach().flatten() for x in model.actor.parameters()])
            delta = float((after - before).norm())
            if warmup:
                assert delta == 0.
            else:
                actor_updates += 1
            metrics = {'update': update, 'actor_updates': actor_updates, 'critic_only': warmup,
                'transitions': update * args.rollout_steps * 4, 'episodes': episodes,
                'completed_episodes': completed, 'actor_parameter_change_l2': delta,
                'actor_loss': float(np.mean(losses, axis=0)[0]),
                'critic_loss': float(np.mean(losses, axis=0)[1]), 'last_kl': float(kl),
                'eligible_transitions': int(valid.sum()), 'kl_stopped': stop,
                'mean_contact_cv': float(np.mean([r['contact_cv'] for r in rows])) if rows else None,
                'mean_temporal_rate_cv': float(np.mean([r['temporal_rate_cv'] for r in rows])) if rows else None,
                'mean_rate_ratio': float(np.mean([r['mean_rate_ratio'] for r in rows])) if rows else None,
                'wall_seconds': time.monotonic() - started}
            with (output / 'updates.jsonl').open('a') as stream:
                stream.write(json.dumps(metrics) + '\n')
            with (output / 'episodes.jsonl').open('a') as stream:
                for row in rows:
                    stream.write(json.dumps({'update': update, **row}) + '\n')
            save(output / 'status.json', {'phase': 'update', **metrics})
            print('RPM_PPO_UPDATE', json.dumps(metrics), flush=True)
            export_actor(model, output / f'checkpoints/ppo_{update:04d}.pt', c,
                         {**metrics, 'robot_geometry': robot, 'parent_sha256': protocol['parent_sha256'],
                          'initialization': initialization})
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'protocol': protocol, 'update': update, 'torch_rng_state': torch.get_rng_state()},
                       output / f'checkpoints/training_{update:04d}.pt')
        save(output / 'training_complete.json', {'completed': True, **metrics})
    finally:
        env.close()


try:
    learn()
finally:
    app.close()
