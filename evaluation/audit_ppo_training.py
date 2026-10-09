"""Verify saved PPO updates, learned actor/critic weights and deterministic exports."""
import argparse
import json
from pathlib import Path

if not __package__:
    import sys
    _root=next(p/'source/nrs_rl' for p in Path(__file__).resolve().parents
               if (p/'source/nrs_rl/nrs_rl').is_dir())
    sys.path.insert(0,str(_root))

import numpy as np
import torch
from nrs_rl.tasks.manager_based.nrs_rl.agents.fixed_force_ppo import PPOActorCritic, sha, check_ppo_or_baseline


def audit(root, seeds=(3101,3102)):
    torch.set_num_threads(1);root=Path(root).resolve();runs=[]
    for seed in seeds:
        folder=root/f'training_seed{seed}'
        selected=json.loads((folder/'selection_surrogate.json').read_text())
        done=json.loads((folder/'training_complete.json').read_text())
        assert done['completed'] and done['algorithm']=='PPO'
        update=selected['update'];checkpoint=Path(selected['actor'])
        initial=torch.load(folder/'checkpoints/initial_training.pt',map_location='cpu',weights_only=False)
        trained=torch.load(folder/f'checkpoints/training_{update:04d}.pt',map_location='cpu',weights_only=False)
        assert trained['update']==update and trained['optimizer']['state']
        extra={'fixed_force_policy.json':''}
        export=torch.jit.load(str(checkpoint),map_location='cpu',_extra_files=extra).eval()
        contract=json.loads(extra['fixed_force_policy.json']);check_ppo_or_baseline(contract)
        assert contract['actor_kind']=='neural_ppo' and not contract['teacher_profile_used']
        assert 'residence_profile' not in dict(export.named_buffers())
        count=sum(p.numel() for p in export.parameters());assert count==20865
        model=PPOActorCritic();model.load_state_dict(trained['model']);model.eval()
        torch.manual_seed(3199)
        with torch.no_grad():
            observations=torch.randn(4096,16)*2.
            expected=(.125+.875*torch.tanh(model.actor(observations))).unsqueeze(1)
            actual=export(observations)
        torch.testing.assert_close(actual,expected,atol=0.,rtol=0.)
        assert actual.shape==(4096,1) and torch.isfinite(actual).all()
        assert actual.min()>=-.750001 and actual.max()<=1.000001
        distances={prefix:float(torch.sqrt(sum((value-initial['model'][key]).square().sum()
                       for key,value in trained['model'].items() if key.startswith(prefix+'.net.'))))
                   for prefix in ['actor','critic']}
        assert min(distances.values())>0.
        logs=np.atleast_1d(np.genfromtxt(folder/'ppo_metrics.csv',delimiter=',',names=True))
        assert len(logs)==done['updates'] and np.all(logs['actor_update_l2']>0.)
        assert np.all(np.isfinite(logs['value_loss']))
        assert int(logs[-1]['transitions'])==done['transitions']
        snapshots=folder/'source_snapshot'
        source_hashes={p.name:sha(p) for p in snapshots.glob('*.py')}
        assert len(source_hashes)>=2
        package=folder/'package_source_snapshot'
        if package.exists():
            for relative,digest in json.loads((package/'manifest.json').read_text()).items():
                assert sha(package/relative)==digest
        runs.append({'seed':seed,'selected_update':update,'total_updates':done['updates'],
            'total_transitions':done['transitions'],'actor_parameters':count,
            'actor_weight_change_l2':distances['actor'],'critic_weight_change_l2':distances['critic'],
            'all_logged_actor_updates_nonzero':True,'optimizer_state_saved':True,
            'export_equals_trained_neural_policy':True,'scalar_export_probes':4096,
            'no_optimized_profile_embedded':True,'teacher_used':False,
            'actor_sha256':sha(checkpoint),'protocol_sha256':sha(folder/'protocol.json'),
            'log_sha256':sha(folder/'ppo_metrics.csv'),'source_hashes':source_hashes})
    return {'passed':True,'algorithm':'PPO','runs':runs,
            'algorithm_reference':'https://arxiv.org/abs/1707.06347',
            'scope':'Algorithm and numerical export audit; independent of Isaac performance'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=audit(args.root)
    output=args.output or args.root/'ppo_training_audit.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
