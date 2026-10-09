"""Train actual PPO from nominal-speed initialization, with an explicit surrogate label."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT as _REPO_ROOT
import argparse
import csv
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from nrs_rl.tasks.manager_based.nrs_rl.utils.source_snapshot import snapshot_sources
from nrs_rl.tasks.manager_based.nrs_rl.agents.fixed_force_ppo import CONTRACT, RemovalEnv, PPOActorCritic, export_actor, sha

ROOT = _REPO_ROOT


def dump(path, obj):
    path = Path(path)
    temp = path.with_suffix('.next.json')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


@torch.no_grad()
def evaluate(model, env, baseline=False):
    env.reset(torch.arange(env.count, device=env.device))
    ended = torch.zeros(env.count, dtype=torch.bool, device=env.device)
    rows = []
    obs = env.observation()
    for _ in range(4201):
        action = torch.zeros(env.count, device=env.device) if baseline else .125+.875*torch.tanh(model.actor(obs))
        obs, _, done, info = env.step(action, auto_reset=False)
        new = done & ~ended
        for i in torch.nonzero(new).flatten().tolist():
            rows.append({key: float(info[key][i]) for key in ['cv','volume_ratio','full_ratio','seconds','completed']})
        ended |= done
        if bool(ended.all()):
            break
    assert len(rows) == env.count
    return rows


def learn(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; run with GPU access or explicitly use --device cpu')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'checkpoints').mkdir()
    model = PPOActorCritic().to(args.device)
    env = RemovalEnv(args.num_envs, args.device, args.seed, True)
    evaluation = RemovalEnv(4, args.device, args.seed+100, False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1.e-5)
    baseline = evaluate(model, evaluation, baseline=True)
    initial_eval = evaluate(model, evaluation)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save({'model':initial, 'seed':args.seed}, output/'checkpoints/initial_training.pt')
    initial_hash = export_actor(model, output/'checkpoints/initial_actor.pt', {'ppo_updates':0,'seed':args.seed})
    settings = {**CONTRACT, 'algorithm':'PPO-Clip', 'gamma':1., 'gae_lambda':.98, 'clip_range':.2,
        'learning_rate':args.learning_rate, 'epochs':4, 'minibatch_size':2048,
        'max_grad_norm':.5, 'target_kl':.025, 'entropy_coefficient':.0002,
        'rollout_steps':args.rollout_steps, 'num_envs':args.num_envs, 'updates':args.updates,
        'seed':args.seed, 'device':args.device, 'reference_sha256':sha(CONTRACT_REFERENCE),
        'initialization':'random neural features, zero action output weights, nominal 6 mm/s mean; no teacher or profile',
        'reward':{'spatial_cv_potential': '100*(previous predicted-final processing CV^2 - current predicted-final processing CV^2)',
            'action_change': '-1e-5*(action-previous_action)^2', 'time': '-1e-4*dt',
            'terminal_volume_floor': '-100*max(0, 0.5-processing_volume_ratio, 0.5-full_volume_ratio)^2',
            'incomplete_timeout':'-20 at 336 s', 'entry_seconds':2., 'entry_footprint_mm':18.},
        'domain_randomization':{'measured_force_scale':[.9,1.1], 'measured_force_bias_n':[-.75,.75], 'action_delay_ticks':[0,3]},
        'surrogate_selection_seed':args.seed+100, 'isaac_selection_seed':3141, 'isaac_holdout_seeds':[3151,3152],
        'surrogate_validation_is_not_isaac':True, 'initial_actor_sha256':initial_hash,
        'started_at_utc':datetime.now(timezone.utc).isoformat()}
    dump(output/'protocol.json',settings)
    dump(output/'initial_evaluation.json',{'mode6':baseline,'untrained_actor':initial_eval})
    source_dir=output/'source_snapshot';source_dir.mkdir()
    for name in ['fixed_force_ppo.py','train_fixed_force_ppo.py']:
        shutil.copy2(ROOT/'scripts'/name,source_dir/name)
    snapshot_sources(output)
    print('PPO_INITIAL',json.dumps({'mode6_cv':baseline[0]['cv'],'initial_cv':initial_eval[0]['cv']}),flush=True)
    obs = env.observation()
    shape=(args.rollout_steps,args.num_envs)
    observations=torch.zeros((*shape,16),device=args.device)
    latents=torch.zeros(shape,device=args.device);logprobs=torch.zeros_like(latents)
    rewards=torch.zeros_like(latents);dones=torch.zeros_like(latents);values=torch.zeros_like(latents)
    advantages=torch.zeros_like(latents)
    best_gain=-math.inf;best_path=None;started=time.monotonic();episodes=0
    metrics_path=output/'ppo_metrics.csv'
    csv_stream=metrics_path.open('w',newline='');writer=None
    try:
        for update in range(1,args.updates+1):
            episode_rows=[]
            with torch.no_grad():
                for t in range(args.rollout_steps):
                    observations[t]=obs
                    action,latent,lp,_,value=model.action_value(obs)
                    latents[t]=latent;logprobs[t]=lp;values[t]=value
                    obs,reward,done,info=env.step(action)
                    rewards[t]=reward;dones[t]=done
                    ids=torch.nonzero(done).flatten()
                    episodes+=len(ids)
                    for i in ids.tolist():
                        episode_rows.append({'cv':float(info['cv'][i]), 'return':float(info['episode_reward'][i]),
                                             'completed':bool(info['completed'][i])})
                next_value=model.critic(obs)
                gae=torch.zeros(args.num_envs,device=args.device)
                for t in reversed(range(args.rollout_steps)):
                    nextv=next_value if t==args.rollout_steps-1 else values[t+1]
                    alive=1.-dones[t]
                    delta=rewards[t]+nextv*alive-values[t]
                    gae=delta+.98*alive*gae
                    advantages[t]=gae
                returns=advantages+values
            data=(observations.flatten(0,1),latents.flatten(),logprobs.flatten(),advantages.flatten(),returns.flatten())
            before=torch.cat([p.detach().flatten() for p in model.actor.parameters()]).clone()
            batches=[];stop=False
            for epoch in range(4):
                order=torch.randperm(len(data[1]),device=args.device)
                for offset in range(0,len(order),2048):
                    ids=order[offset:offset+2048]
                    _,_,newlp,entropy,newvalue=model.action_value(data[0][ids],data[1][ids])
                    logratio=newlp-data[2][ids];ratio=logratio.exp()
                    adv=data[3][ids];adv=(adv-adv.mean())/(adv.std()+1.e-8)
                    policy_loss=torch.maximum(-adv*ratio,-adv*ratio.clamp(.8,1.2)).mean()
                    value_loss=.5*(newvalue-data[4][ids]).square().mean()
                    loss=policy_loss+.5*value_loss-.0002*entropy.mean()
                    optimizer.zero_grad();loss.backward()
                    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),.5)
                    if not torch.isfinite(loss) or not torch.isfinite(grad):
                        raise RuntimeError('Nonfinite PPO update')
                    optimizer.step()
                    with torch.no_grad():
                        model.log_std.clamp_(-2.5,-.3)
                        kl=((ratio-1.)-logratio).mean()
                        batches.append([float(policy_loss),float(value_loss),float(kl),
                                        float(((ratio-1.).abs()>.2).float().mean()),float(grad)])
                    if kl>.025:
                        stop=True;break
                if stop:break
            after=torch.cat([p.detach().flatten() for p in model.actor.parameters()])
            actor_delta=float(torch.linalg.vector_norm(after-before))
            means=np.mean(batches,axis=0)
            total_delta=float(torch.sqrt(sum((p.detach().cpu()-initial[k]).square().sum()
                                              for k,p in model.state_dict().items() if k.startswith('actor.net.'))))
            metrics={'update':update,'transitions':update*args.rollout_steps*args.num_envs,
                'episodes':episodes,'policy_loss':means[0],'value_loss':means[1],
                'approx_kl':means[2],'clip_fraction':means[3],'gradient_norm_before_clip':means[4],
                'actor_update_l2':actor_delta,'actor_total_change_l2':total_delta,
                'log_std':float(model.log_std.detach()),'mean_step_reward':float(rewards.mean()),
                'wall_seconds':time.monotonic()-started}
            if writer is None:
                writer=csv.DictWriter(csv_stream,fieldnames=list(metrics));writer.writeheader()
            writer.writerow(metrics);csv_stream.flush()
            if episode_rows:
                with (output/'episodes.jsonl').open('a') as stream:
                    for row in episode_rows:stream.write(json.dumps({'update':update,**row})+'\n')
            dump(output/'status.json',{'phase':'ppo_training',**metrics,'target_updates':args.updates})
            print('PPO_UPDATE',json.dumps(metrics),flush=True)
            if update%args.eval_interval==0 or update==args.updates:
                result=evaluate(model,evaluation)
                gain=1.-np.mean([r['cv'] for r in result])/np.mean([r['cv'] for r in baseline])
                eligible=all(r['completed'] and min(r['volume_ratio'],r['full_ratio'])>=.5 for r in result)
                name=f'ppo_{update:04d}';path=output/'checkpoints'/f'{name}.pt'
                digest=export_actor(model,path,{'ppo_updates':update,'ppo_transitions':metrics['transitions'],
                    'seed':args.seed,'initial_actor_sha256':initial_hash,'trained_actor_total_change_l2':total_delta})
                torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'update':update,
                            'protocol':settings,'torch_rng_state':torch.get_rng_state()},output/'checkpoints'/f'training_{update:04d}.pt')
                record={'update':update,'cv_improvement_percent_vs_mode6':100.*gain,'eligible':eligible,
                        'mode6':baseline,'ppo':result,'actor_sha256':digest,'actor':str(path),
                        'scope':'surrogate only; not Isaac validation'}
                dump(output/f'evaluation_{update:04d}.json',record)
                print('PPO_SURROGATE_EVALUATION',json.dumps(record),flush=True)
                if eligible and gain>best_gain:
                    best_gain=gain;best_path=path
                    shutil.copy2(path,output/'checkpoints/selected_surrogate.pt')
                    dump(output/'selection_surrogate.json',record)
        if best_path is None:raise RuntimeError('No eligible surrogate-trained PPO candidate')
        dump(output/'training_complete.json',{'algorithm':'PPO','completed':True,'updates':args.updates,
             'transitions':args.updates*args.rollout_steps*args.num_envs,'best_actor':str(best_path),
             'selected_actor':str(output/'checkpoints/selected_surrogate.pt'),
             'surrogate_cv_improvement_percent_vs_mode6':100.*best_gain,'isaac_validated':False,
             'wall_seconds':time.monotonic()-started})
    finally:
        csv_stream.close()


if __name__=='__main__':
    from nrs_rl.tasks.manager_based.nrs_rl.utils.rotary_geometry import REFERENCE as CONTRACT_REFERENCE
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--num-envs',type=int,default=64);p.add_argument('--rollout-steps',type=int,default=512)
    p.add_argument('--updates',type=int,default=64);p.add_argument('--seed',type=int,default=3101)
    p.add_argument('--eval-interval',type=int,default=8);p.add_argument('--learning-rate',type=float,default=2.e-4)
    learn(p.parse_args())
