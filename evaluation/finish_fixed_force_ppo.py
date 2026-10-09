"""Select trained PPO actors in Isaac, then validate an unchanged actor on heldouts."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT as _REPO_ROOT
import argparse
from concurrent.futures import ThreadPoolExecutor
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
from nrs_rl.tasks.model_based.policies.fixed_force_dynamics import export_actor as export_constant_profile
from nrs_rl.tasks.manager_based.nrs_rl.agents.fixed_force_ppo import check_ppo_or_baseline, sha
from nrs_rl.tasks.model_based.training.train_fixed_force_velocity import save

ROOT=_REPO_ROOT
FROZEN_SHA='6d1dc47bc915b393880d2e6e380f44fbcb82e542de0fa12e4f02eb47c97147df'


def preserve(root):
    frozen=root/'frozen_baseline/actor.pt'
    assert sha(frozen)==FROZEN_SHA
    locked=json.loads((ROOT/'deployment/FIXED_MODEL.json').read_text())
    assert sha(locked['checkpoint'])==FROZEN_SHA
    assert not locked['automatic_replacement_allowed']


def batch(root, training, label, names, seed):
    output=root/'physics'/label
    command=[sys.executable,str(ROOT/'scripts/probe_fixed_force_ppo.py'),'--headless','--device','cpu',
        '--training-dir',str(training),'--output',str(output),'--names',','.join(names),
        '--seed',str(seed),'--max-steps','42000','--hard-force-stop-n','100','--uniformity-only']
    command_file=root/'physics'/f'{label}_command.json'
    if command_file.exists():assert json.loads(command_file.read_text())==command
    else:save(command_file,command)
    if not (output/'summary.json').exists():
        if output.exists():raise RuntimeError('Incomplete rollout retained: '+str(output))
        with (root/'physics'/f'{label}.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,
                env={**os.environ,'MPLCONFIGDIR':'/tmp/ppo_fixed_force_mpl',
                     'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    summary=json.loads((output/'summary.json').read_text())
    assert summary['seed']==seed and summary['names']==names
    for name in names:
        original='constant' if name=='constant_b' else name
        assert sha(training/'checkpoints'/f'{original}.pt')==summary['checkpoint_sha256'][name]
    assert summary['checkpoint_sha256']['saved']==FROZEN_SHA
    for values in summary['randomization'].values():assert all(v==values[0] for v in values)
    report=evaluate(output,audit=True)
    print('PPO_ISAAC_BATCH',label,json.dumps({k:{'eligible':v['eligible'],
        'cv_improvement_percent_vs_mode6':100.*v['mode6_cv_gain'],
        'volume_retention_percent_vs_mode6':{a:100.*b for a,b in v['mode6_volume_ratios'].items()}}
        for k,v in report['candidates'].items()}),flush=True)
    return report


def run(root):
    root=root.resolve();torch.set_num_threads(1);preserve(root)
    training=root/'physics_candidates';(training/'checkpoints').mkdir(parents=True,exist_ok=True)
    (root/'physics').mkdir(exist_ok=True)
    candidates=[]
    for seed in [3101,3102]:
        source=root/f'training_seed{seed}'
        if not (source/'training_complete.json').exists():raise RuntimeError('Training incomplete: '+str(source))
        picked=json.loads((source/'selection_surrogate.json').read_text())
        checkpoint=Path(picked['actor']);name=f'ppo_seed{seed}'
        destination=training/'checkpoints'/f'{name}.pt'
        if destination.exists():assert sha(destination)==sha(checkpoint)
        else:shutil.copy2(checkpoint,destination)
        extra={'fixed_force_policy.json':''};actor=torch.jit.load(str(destination),_extra_files=extra).eval()
        c=json.loads(extra['fixed_force_policy.json']);check_ppo_or_baseline(c)
        assert c['actor_kind']=='neural_ppo' and c['ppo_updates']>0 and c['trained_actor_total_change_l2']>0.
        assert not c['teacher_profile_used']
        candidates.append(name)
    shutil.copy2(root/'frozen_baseline/actor.pt',training/'checkpoints/saved.pt')
    c=json.loads((root/'frozen_baseline/fixed_force_policy.json').read_text())
    if not (training/'checkpoints/constant.pt').exists():
        export_constant_profile(training/'checkpoints/constant.pt',[1.,1.],path_length_mm=c['reference_length_mm'],
            lookahead_time_s=0.,candidate='constant_20N_6mmps_matched_mode6',
            protocol={'purpose':'untrained constant control; not a learned candidate'})
    state={'phase':'isaac_selection','algorithm':'PPO','frozen_baseline_preserved':True,
           'selection_seed':3141,'heldout_seeds':[3151,3152],
           'training_environment':'rotary-removal and command-response surrogate',
           'validation_environment':'Isaac full robot/contact dynamics','auto_deployment':False}
    save(root/'status.json',state)
    selection=batch(root,training,'selection_3141',['constant','saved',*candidates],3141)
    eligible=[(selection['candidates'][name]['mode6_cv_gain'],name) for name in candidates
              if selection['candidates'][name]['eligible']]
    if not eligible:raise RuntimeError('No eligible PPO candidate in independent Isaac selection')
    gain,name=max(eligible)
    checkpoint=training/'checkpoints/selected.pt'
    source=training/'checkpoints'/f'{name}.pt'
    if checkpoint.exists():assert sha(checkpoint)==sha(source)
    else:shutil.copy2(source,checkpoint)
    timestamp=datetime.now(timezone.utc).isoformat()
    if (root/'selection.json').exists():
        previous=json.loads((root/'selection.json').read_text());assert previous['checkpoint_sha256']==sha(checkpoint)
        timestamp=previous['selected_before_holdouts_at']
    selected={'selected':name,'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
        'selected_before_holdouts_at':timestamp,'selection_report':selection,
        'selection_cv_improvement_percent_vs_mode6':100.*gain}
    save(root/'selection.json',selected)
    state.update(phase='independent_isaac_holdouts',**selected);save(root/'status.json',state)
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks={seed:pool.submit(batch,root,training,f'heldout_{seed}',
                               ['constant','constant_b','saved','selected'],seed) for seed in [3151,3152]}
        heldouts={str(seed):task.result() for seed,task in tasks.items()}
    checks=[]
    for seed,report in heldouts.items():
        a=report['candidates'];noise=abs(a['constant_b']['spatial_cv']/a['constant']['spatial_cv']-1.)
        threshold=max(.005,2.*noise)
        passed=a['selected']['eligible'] and a['selected']['mode6_cv_gain']>=threshold
        checks.append(passed)
        report['acceptance']={'passed':passed,'minimum_cv_gain_vs_mode6':threshold,
            'mode6_repeat_cv_variation':noise,'volume_floor_all_three_scopes':.5,
            'practical_acceptance_not_statistical_significance':True}
    preserve(root)
    state.update(phase='complete',heldouts=heldouts,accepted=all(checks),gui_trial_executed=False,
                 active_baseline_checkpoint_unchanged=True,
                 candidate_is_ppo=True,selected_actor_sha256_unchanged=sha(checkpoint)==selected['checkpoint_sha256'])
    save(root/'result.json',state);save(root/'status.json',state)
    print('PPO_PIPELINE_COMPLETE',json.dumps({'accepted':state['accepted'],'checkpoint':str(checkpoint),
        'heldout_cv_gains_vs_mode6_percent':{k:100.*v['candidates']['selected']['mode6_cv_gain'] for k,v in heldouts.items()}}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--wait-training',action='store_true')
    args=parser.parse_args()
    if args.wait_training:
        deadline=time.monotonic()+600.
        while not all((args.root/f'training_seed{s}/training_complete.json').exists() for s in [3101,3102]):
            if time.monotonic()>deadline:raise RuntimeError('PPO training did not finish within the readiness window')
            time.sleep(5.)
    run(args.root)
