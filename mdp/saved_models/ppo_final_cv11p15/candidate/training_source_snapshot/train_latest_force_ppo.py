#!/usr/bin/env python3
"""Fine-tune genuine PPO with measured Isaac TCP removal and frozen latest force control."""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(next(p / 'source/nrs_rl' for p in Path(__file__).resolve().parents
                              if (p / 'source/nrs_rl/nrs_rl').is_dir())))

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import time
from datetime import datetime, timezone
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--native-dir', type=Path, required=True)
p.add_argument('--control-snapshot', type=Path, required=True)
p.add_argument('--gui-scene', type=Path)
p.add_argument('--resume', type=Path, required=True)
p.add_argument('--num-envs', type=int, default=4)
p.add_argument('--seed', type=int, default=4101)
p.add_argument('--rollout-steps', type=int, default=512)
p.add_argument('--updates', type=int, default=8)
p.add_argument('--save-interval', type=int, default=2)
p.add_argument('--learning-rate', type=float, default=1.e-4)
p.add_argument('--max-marginal-kl', type=float,
               help='Optional post-step old||new marginal Gaussian KL limit; disabled by default')
p.add_argument('--gae-lambda', type=float, default=.98)
p.add_argument('--entropy-coefficient', type=float, default=.0002)
p.add_argument('--reward-cell-mm', type=float, choices=[.5, 1., 2.], default=1.)
p.add_argument('--exploration-rho', type=float, default=0.)
p.add_argument('--exploration-log-std', type=float)
p.add_argument('--reward-forecast', type=Path)
p.add_argument('--reset-critic-output', action='store_true')
p.add_argument('--critic-potential-residual', action='store_true',
               help='Use V=100*removal potential+critic(obs); actor and reward stay unchanged')
p.add_argument('--critic-residual-bias', type=float, default=-22.0929677,
               help='Initial terminal-cost residual; default is mean of prior training warmup episodes')
p.add_argument('--nominal-sensors', action='store_true')
p.add_argument('--actor-complete-returns-only', action='store_true',
               help='Train actor only from transitions ending within this rollout; requires lambda=1')
p.add_argument('--restart-episodes-each-update', action='store_true',
               help='Restart training episodes and exploration history at each new rollout batch')
p.add_argument('--critic-warmup-updates', type=int, default=0)
p.add_argument('--critic-warmup-learning-rate', type=float, default=3.e-4)
p.add_argument('--critic-warmup-epochs', type=int, default=8)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.output.exists():
    p.error('Use a new output directory')
if args.device != 'cpu':
    p.error('This frozen native controller rollout currently requires --device cpu')
if not 0. <= args.exploration_rho <= .99:
    p.error('exploration-rho must be in [0, .99]')
if args.exploration_log_std is not None and not -2.5 <= args.exploration_log_std <= -.3:
    p.error('exploration-log-std must be in [-2.5, -.3]')
if args.actor_complete_returns_only and args.gae_lambda != 1.:
    p.error('Complete-return actor updates require --gae-lambda 1')
if not 0 <= args.critic_warmup_updates < args.updates:
    p.error('Need at least one actor update after nonnegative critic warmup')
if args.critic_warmup_learning_rate <= 0. or args.critic_warmup_epochs < 1:
    p.error('Critic warmup learning rate and epochs must be positive')
if not math.isfinite(args.critic_residual_bias):
    p.error('Critic residual bias must be finite')
if args.max_marginal_kl is not None and (not math.isfinite(args.max_marginal_kl)
                                        or args.max_marginal_kl <= 0.):
    p.error('Marginal KL limit must be finite and positive')
os.environ['Y2_CONTROL_SOURCE_DIR'] = str(args.control_snapshot.resolve())
os.environ['Y2_ROBOT_CONFIG'] = str(args.control_snapshot.resolve() / 'Y2RobMotion/config/ppo_fixed_force.yaml')
app = AppLauncher(args).app

import numpy as np
import torch
from nrs_rl.tasks.manager_based.nrs_rl.mdp.agents.latest_force_ppo import CONTRACT, PPOActorCritic, export_actor
from nrs_rl.tasks.manager_based.nrs_rl.mdp.agents.train_fixed_force_ppo import restore_training
from nrs_rl.tasks.manager_based.nrs_rl.mdp.agents.ppo_exploration import action_value
from nrs_rl.tasks.manager_based.nrs_rl.mdp.isaac_latest_force_env import IsaacLatestForceEnv
from nrs_rl.tasks.manager_based.nrs_rl.utils.removal_metrics import save, sha


def marginal_actor_kl(old_mean, old_std, new_mean, new_std):
    """Mean KL(old||new) of the latent marginal Gaussian, before tanh."""
    return (torch.log(new_std / old_std)
            + (old_std.square() + (old_mean - new_mean).square()) / (2. * new_std.square())
            - .5).mean()


def learn():
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'checkpoints').mkdir()
    model = PPOActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1.e-5)
    parent = restore_training(model, optimizer, args.resume, args.learning_rate)
    if args.exploration_log_std is not None:
        with torch.no_grad():
            model.log_std.fill_(args.exploration_log_std)
    resume_protocol = torch.load(args.resume, map_location='cpu', weights_only=False)['protocol']
    parent_residual = bool(resume_protocol.get('critic_potential_residual', False))
    critic_semantics_changed = parent_residual != args.critic_potential_residual
    critic_reset = args.reset_critic_output or critic_semantics_changed
    parent.update(parent_critic_potential_residual=parent_residual,
                  resumed_critic_potential_residual=args.critic_potential_residual,
                  critic_semantics_changed=critic_semantics_changed)
    if critic_reset:
        # A legacy V critic cannot be reused as U (or vice versa). Preserve its
        # feature layers, but reset the output and all critic Adam moments.
        critic_bias = args.critic_residual_bias if args.critic_potential_residual else 0.
        with torch.no_grad():
            model.critic.net[-1].weight.zero_()
            model.critic.net[-1].bias.fill_(critic_bias)
        for parameter in model.critic.parameters():
            optimizer.state.pop(parameter, None)
        parent.update(optimizer_moments_restored=False, actor_optimizer_moments_restored=True,
                      critic_output_reset=True, critic_optimizer_moments_restored=False,
                      critic_reset_reason=('value_semantics_changed' if critic_semantics_changed
                                           else 'explicit_reset'),
                      critic_output_initial_bias=critic_bias)
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    initial_hash = export_actor(model, output / 'checkpoints/initial_actor.pt', {'ppo_updates': 0})
    protocol = {**CONTRACT, 'algorithm': 'PPO-Clip', 'gamma': 1., 'gae_lambda': args.gae_lambda,
        'clip_range': .2, 'learning_rate': args.learning_rate, 'epochs': 4, 'minibatch_size': 1024,
        'max_grad_norm': .5, 'target_kl': .025, 'entropy_coefficient': args.entropy_coefficient,
        'max_marginal_kl': args.max_marginal_kl,
        'marginal_kl_reference': 'actor before this update on the same rollout observations',
        'marginal_kl_stop_scope': 'after each optimizer step on the current full minibatch; no rollback',
        'num_envs': args.num_envs, 'rollout_steps': args.rollout_steps, 'updates': args.updates,
        'seed': args.seed, 'device': 'cpu', 'reward_cell_mm': args.reward_cell_mm,
        'nominal_sensors': args.nominal_sensors,
        'actor_complete_returns_only': args.actor_complete_returns_only,
        'restart_episodes_each_update': args.restart_episodes_each_update,
        'rollout_boundary_reset_scope': ('training environments, removal history and exploration history only; '
                                        'model, optimizer and completed-episode counters are retained'),
        'critic_potential_residual': args.critic_potential_residual,
        'critic_value_expression': ('100*env.potential + critic(observation)'
                                    if args.critic_potential_residual else 'critic(observation)'),
        'critic_residual_bias_argument': args.critic_residual_bias,
        'critic_residual_initial_bias': (args.critic_residual_bias
                                        if args.critic_potential_residual and critic_reset else None),
        'critic_warmup': {'updates': args.critic_warmup_updates,
            'learning_rate': args.critic_warmup_learning_rate, 'epochs': args.critic_warmup_epochs},
        'exploration': {'rho': args.exploration_rho, 'initial_log_std': float(model.log_std.detach()),
            'log_std_override': args.exploration_log_std,
            'density': 'Autoregressive Gaussian; previous residual recomputed with each candidate actor',
            'evaluation_exploration': False},
        'parent_training': parent, 'initial_actor_sha256': initial_hash,
        'controller_snapshot': str(args.control_snapshot.resolve()),
        'native_validation': json.loads((args.native_dir / 'validation.json').read_text()),
        'scene_profile': (json.loads((args.gui_scene / 'manifest.json').read_text())
                          if args.gui_scene is not None else None),
        'reward_forecast': (json.loads(args.reward_forecast.with_name('calibration.json').read_text())
                            if args.reward_forecast is not None else None),
        'policy_scope': 'neural actor and critic only; no force, tracking or timing tuning',
        'candidate_selection': 'Separate matched Mode 6 Isaac validation required',
        'reward': {'cv_potential': '100*(previous predicted processing CV^2-current CV^2)',
            'volume_floor': .5, 'volume_reference': 'ideal nominal geometry during training only',
            'incomplete_penalty': 20., 'entry_seconds': 2., 'entry_footprint_mm': 18.},
        'started_at_utc': datetime.now(timezone.utc).isoformat()}
    save(output / 'protocol.json', protocol)
    source = output / 'source_snapshot'
    source.mkdir()
    for file in [Path(__file__), Path(__file__).with_name('latest_force_ppo.py'),
                 Path(__file__).with_name('ppo_exploration.py'),
                 Path(__file__).parent.parent / 'isaac_latest_force_env.py']:
        shutil.copy2(file, source / file.name)
    env = IsaacLatestForceEnv(args.num_envs, args.seed, args.native_dir,
                             args.control_snapshot, output, args.reward_cell_mm, args.gui_scene,
                             args.reward_forecast, args.nominal_sensors)
    shape = (args.rollout_steps, args.num_envs)
    observations = torch.zeros((*shape, 16))
    latent = torch.zeros(shape)
    logprobs = torch.zeros(shape)
    rewards = torch.zeros(shape)
    dones = torch.zeros(shape)
    values = torch.zeros(shape)
    advantages = torch.zeros(shape)
    potential_offsets = torch.zeros(shape)
    history_observations = torch.zeros((*shape, 16))
    history_latents = torch.zeros(shape)
    history_valid = torch.zeros(shape)
    previous_observation = torch.zeros(args.num_envs, 16)
    previous_latent = torch.zeros(args.num_envs)
    has_previous = torch.zeros(args.num_envs)
    obs = env.observation()
    started = time.monotonic()
    last_report = started
    episodes = completed = 0
    actor_updates = 0
    with (output / 'ppo_metrics.csv').open('w', newline='') as stream:
        writer = None
        try:
            for update in range(1, args.updates + 1):
                if args.restart_episodes_each_update and update > 1:
                    # Start a fresh training trajectory instead of carrying the
                    # discarded partial suffix into the next rollout batch.
                    obs = env.reset_batch()
                    previous_observation.zero_()
                    previous_latent.zero_()
                    has_previous.zero_()
                warmup = update <= args.critic_warmup_updates
                for group in optimizer.param_groups:
                    group['lr'] = args.critic_warmup_learning_rate if warmup else args.learning_rate
                rows = []
                with torch.no_grad():
                    for t in range(args.rollout_steps):
                        observations[t] = obs
                        history_observations[t] = previous_observation
                        history_latents[t] = previous_latent
                        history_valid[t] = has_previous
                        action, latent[t], logprobs[t], _, values[t] = action_value(
                            model, obs, previous_observation, previous_latent,
                            has_previous, args.exploration_rho)
                        if args.critic_potential_residual:
                            # Store Phi for this observation BEFORE the physics
                            # step, which can finish and reset an environment.
                            potential_offsets[t] = torch.as_tensor(100. * env.potential,
                                                                  dtype=values.dtype)
                            values[t] += potential_offsets[t]
                        previous_observation = obs.clone()
                        previous_latent = latent[t].clone()
                        obs, rewards[t], done, records = env.step(action)
                        has_previous = (~done).float()
                        dones[t] = done
                        if not all(torch.isfinite(x).all() for x in [obs, rewards[t], values[t]]):
                            raise RuntimeError('Nonfinite training rollout')
                        for row in records:
                            if row is not None:
                                rows.append(row)
                                episodes += 1
                                completed += int(row['completed'])
                        now = time.monotonic()
                        if now - last_report >= 30.:
                            progress = {'phase': 'isaac_ppo_rollout', 'update': update,
                                'step': t + 1, 'steps_per_update': args.rollout_steps,
                                'wall_seconds': now - started, 'episodes': episodes,
                                'completed_episodes': completed,
                                'cursor_mm': env.term.path_cursor_mm.tolist(),
                                'active_removal_samples': env.samples.tolist()}
                            save(output / 'status.json', progress)
                            print('PPO_ROLLOUT', json.dumps(progress), flush=True)
                            last_report = now
                    next_value = model.critic(obs)
                    if args.critic_potential_residual:
                        next_value += torch.as_tensor(100. * env.potential, dtype=next_value.dtype)
                    # The existing done mask removes the entire next V=Phi+U,
                    # including any Phi belonging to an automatically reset episode.
                    gae = torch.zeros(args.num_envs)
                    for t in reversed(range(args.rollout_steps)):
                        nextv = next_value if t == args.rollout_steps - 1 else values[t + 1]
                        alive = 1. - dones[t]
                        gae = rewards[t] + nextv * alive - values[t] + args.gae_lambda * alive * gae
                        advantages[t] = gae
                    returns = advantages + values
                    actor_valid = torch.ones_like(dones, dtype=torch.bool)
                    if args.actor_complete_returns_only:
                        # Exclude the unfinished suffix of each environment, whose
                        # return still depends on the critic at the batch boundary.
                        actor_valid = dones.flip(0).cumsum(0).flip(0) > 0
                data = (observations.flatten(0, 1), latent.flatten(), logprobs.flatten(),
                        advantages.flatten(), returns.flatten(), history_observations.flatten(0, 1),
                        history_latents.flatten(), history_valid.flatten(), actor_valid.flatten(),
                        potential_offsets.flatten())
                with torch.no_grad():
                    old_marginal_means = model.actor(data[0]).clone()
                    old_marginal_std = model.log_std.clamp(-2.5, -.3).exp().clone()
                before = torch.cat([v.detach().flatten() for v in model.actor.parameters()]).clone()
                batches = []
                stop = False
                marginal_kl_stopped = False
                for epoch in range(args.critic_warmup_epochs if warmup else 4):
                    order = torch.randperm(len(data[1]))
                    for offset in range(0, len(order), 1024):
                        ids = order[offset:offset + 1024]
                        _, _, lp, entropy, value = action_value(model, data[0][ids],
                            data[5][ids], data[6][ids], data[7][ids], args.exploration_rho, data[1][ids])
                        logratio = lp - data[2][ids]
                        ratio = logratio.exp()
                        valid = data[8][ids]
                        train_actor = not warmup and bool(valid.any())
                        if train_actor:
                            adv = data[3][ids][valid]
                            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1.e-8)
                            policy_loss = torch.maximum(-adv * ratio[valid],
                                -adv * ratio[valid].clamp(.8, 1.2)).mean()
                            actor_loss = policy_loss - args.entropy_coefficient * entropy[valid].mean()
                        else:
                            # No zero-valued actor gradient: Adam must not move
                            # actor weights via restored momentum during warmup.
                            policy_loss = actor_loss = torch.zeros(())
                        # Constant Phi has no gradients; only U is fitted. The
                        # target remains the original shaped return and penalties.
                        value = value + data[9][ids]
                        value_loss = .5 * (value - data[4][ids]).square().mean()
                        loss = actor_loss + .5 * value_loss
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
                        if not torch.isfinite(loss) or not torch.isfinite(grad):
                            raise RuntimeError('Nonfinite PPO update')
                        optimizer.step()
                        with torch.no_grad():
                            model.log_std.clamp_(-2.5, -.3)
                            kl = ((ratio[valid] - 1.) - logratio[valid]).mean() if train_actor else torch.zeros(())
                            marginal_kl = (marginal_actor_kl(
                                old_marginal_means[ids], old_marginal_std,
                                model.actor(data[0][ids]), model.log_std.clamp(-2.5, -.3).exp())
                                if train_actor else torch.zeros(()))
                            batches.append([float(policy_loss), float(value_loss), float(kl),
                                float(((ratio[valid] - 1.).abs() > .2).float().mean()) if train_actor else 0., float(grad)])
                        marginal_kl_stopped = (args.max_marginal_kl is not None
                                               and float(marginal_kl) > args.max_marginal_kl)
                        if kl > .025 or marginal_kl_stopped:
                            stop = True
                            break
                    if stop:
                        break
                after = torch.cat([v.detach().flatten() for v in model.actor.parameters()])
                if warmup:
                    torch.testing.assert_close(before, after, atol=0., rtol=0.)
                elif bool(actor_valid.any()):
                    actor_updates += 1
                with torch.no_grad():
                    if warmup:
                        full_marginal_kl = feed_change_mean = feed_change_max = 0.
                    else:
                        new_marginal_means = model.actor(data[0])
                        full_marginal_kl = float(marginal_actor_kl(
                            old_marginal_means, old_marginal_std, new_marginal_means,
                            model.log_std.clamp(-2.5, -.3).exp()))
                        feed_change = 5.25 * (new_marginal_means.tanh() - old_marginal_means.tanh()).abs()
                        feed_change_mean = float(feed_change.mean())
                        feed_change_max = float(feed_change.max())
                total_delta = float(torch.sqrt(sum((v.detach() - initial[k]).square().sum()
                    for k, v in model.state_dict().items() if k.startswith('actor.net.'))))
                mean = np.mean(batches, axis=0)
                metrics = {'update': update, 'transitions': update * args.rollout_steps * args.num_envs,
                    'actor_updates': actor_updates, 'critic_only_warmup': warmup,
                    'actor_eligible_transitions': int(actor_valid.sum()) if not warmup else 0,
                    'physics_ticks': update * args.rollout_steps * args.num_envs * 10,
                    'episodes': episodes, 'completed_episodes': completed,
                    'policy_loss': mean[0], 'value_loss': mean[1], 'approx_kl': mean[2],
                    'clip_fraction': mean[3], 'gradient_norm_before_clip': mean[4],
                    'marginal_actor_kl': full_marginal_kl,
                    'mean_abs_deterministic_feed_change_mm_s': feed_change_mean,
                    'max_abs_deterministic_feed_change_mm_s': feed_change_max,
                    'marginal_kl_early_stop': marginal_kl_stopped,
                    'actor_update_l2': float(torch.linalg.vector_norm(after - before)),
                    'actor_total_change_l2': total_delta, 'log_std': float(model.log_std.detach()),
                    'mean_reward': float(rewards.mean()), 'wall_seconds': time.monotonic() - started}
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(metrics))
                    writer.writeheader()
                writer.writerow(metrics)
                stream.flush()
                if rows:
                    with (output / 'episodes.jsonl').open('a') as row_stream:
                        for row in rows:
                            row_stream.write(json.dumps({'update': update, **row}) + '\n')
                save(output / 'status.json', {'phase': 'ppo_update', **metrics})
                print('PPO_UPDATE', json.dumps(metrics), flush=True)
                if update % args.save_interval == 0 or update == args.updates:
                    actor = output / 'checkpoints' / f'ppo_{update:04d}.pt'
                    digest = export_actor(model, actor, {'ppo_updates': actor_updates,
                        'rollout_batches': update,
                        'ppo_transitions': metrics['transitions'], 'seed': args.seed,
                        'initial_actor_sha256': initial_hash, 'parent_training': parent,
                        'trained_actor_total_change_l2': total_delta})
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'update': update, 'protocol': protocol, 'torch_rng_state': torch.get_rng_state()},
                        output / 'checkpoints' / f'training_{update:04d}.pt')
                    save(output / f'candidate_{update:04d}.json', {'actor': str(actor),
                        'sha256': digest, **metrics, 'independently_validated': False})
            save(output / 'training_complete.json', {'completed': True, **metrics,
                'independently_validated': False})
        finally:
            env.close()


try:
    learn()
finally:
    app.close()
